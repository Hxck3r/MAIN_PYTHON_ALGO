"""Tests for the `algo-trades` command line (offline, injected fake venue)."""
from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from marketdata import TradeStore
from marketdata.cli import main, parse_time

T = 1_700_000_000_000  # 2023-11-14T22:13:20Z

#: one contract in base units, as CCXT reports it per venue
CONTRACT_SIZES = {"binance": 1.0, "okx": 0.01, "bybit": 1.0}


def test_parse_time_reads_bare_dates_as_utc_and_passes_epoch_through():
    assert parse_time("2024-01-01") == pd.Timestamp("2024-01-01T00:00:00Z")
    assert parse_time("2024-01-01 05:30") == pd.Timestamp("2024-01-01T05:30:00Z")
    assert parse_time("1700000000000") == "1700000000000"
    assert parse_time("2024-01-01T00:00:00+03:00") == "2024-01-01T00:00:00+03:00"


class FakeClient:
    """Minimal CCXT stand-in: one page of trades, then a short page."""

    def __init__(self, exchange: str, contract_size: float | None = None, trades=1):
        self.exchange = exchange
        self.contract_size = CONTRACT_SIZES[exchange] if contract_size is None else contract_size
        self.trades = trades
        self.calls = 0

    def market(self, symbol):
        return {"id": "BTCUSDT", "contractSize": self.contract_size}

    def fetch_trades(self, symbol, since, limit):
        self.calls += 1
        if self.calls > 1 or not self.trades:
            return []
        return [
            {"id": f"{self.exchange}-{index}", "timestamp": since + index,
             "price": 42000.0, "amount": 2, "side": "buy"}
            for index in range(self.trades)
        ]

    def close(self):
        pass


def _factory(**kwargs):
    return lambda exchange: FakeClient(exchange, **kwargs)


def test_download_writes_every_exchange_with_detected_contract_sizes(tmp_path, capsys):
    code = main(
        ["download", "-e", "binance", "okx", "bybit", "--symbol", "BTC/USDT:USDT",
         "--since", str(T), "--until", str(T + 3_600_000), "--data-dir", str(tmp_path)],
        client_factory=_factory(trades=2),
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "okx: 2 rows written" in out and "contract_size=0.01" in out

    store = TradeStore(tmp_path)
    # OKX reports contracts: 2 contracts x 0.01 BTC must not be stored as 2 BTC.
    assert store.read("okx").iloc[0].quantity == pytest.approx(0.02)
    assert store.read("binance").iloc[0].quantity == pytest.approx(2.0)
    assert store.get_checkpoint("bybit", "BTC/USDT:USDT") is not None


def test_download_keeps_going_when_one_venue_fails(tmp_path, capsys):
    class Broken:
        def fetch_trades(self, symbol, since, limit):
            raise RuntimeError("venue down")

    def factory(exchange):
        return Broken() if exchange == "binance" else FakeClient(exchange)

    code = main(
        ["download", "-e", "binance", "bybit", "--since", str(T), "--until", str(T + 60_000),
         "--data-dir", str(tmp_path)],
        client_factory=factory,
    )
    assert code == 1
    assert "binance: FAILED (RuntimeError: venue down)" in capsys.readouterr().err
    assert len(TradeStore(tmp_path).read("bybit")) == 1


def test_download_does_not_detect_contract_size_for_base_quantities(tmp_path):
    client = FakeClient("okx")
    code = main(
        ["download", "-e", "okx", "--since", str(T), "--until", str(T + 60_000),
         "--quantity-source", "base", "--data-dir", str(tmp_path)],
        client_factory=lambda exchange: client,
    )
    assert code == 0
    assert TradeStore(tmp_path).read("okx").iloc[0].quantity == pytest.approx(2.0)


def test_download_warns_when_contract_size_changes_between_runs(tmp_path):
    args = ["download", "-e", "okx", "--since", str(T), "--until", str(T + 60_000),
            "--data-dir", str(tmp_path)]
    main(args, client_factory=_factory(trades=1))
    with pytest.warns(RuntimeWarning, match="contract_size changed"):
        main(args, client_factory=_factory(contract_size=1.0, trades=0))
    store = TradeStore(tmp_path)
    assert any(
        "contract_size changed" in issue["message"]
        for issue in store.manifest_data()["issues"]
    )


def test_footprint_command_aggregates_and_exports(tmp_path, capsys):
    main(["download", "-e", "binance", "okx", "--since", str(T), "--until", str(T + 3_600_000),
          "--data-dir", str(tmp_path)], client_factory=_factory(trades=3))
    out = tmp_path / "footprint.parquet"
    code = main(["footprint", "--price-step", "10", "--out", str(out),
                 "--data-dir", str(tmp_path)])
    assert code == 0
    assert "cells" in capsys.readouterr().out

    footprint = pd.read_parquet(out)
    assert {"combined_delta", "binance_covered", "okx_covered"} <= set(footprint.columns)
    # 3 trades x 2 BTC on Binance, plus 3 x 2 contracts x 0.01 BTC on OKX.
    assert footprint["combined_buy_qty"].sum() == pytest.approx(6.06)


def test_features_command_exports_bar_features(tmp_path):
    main(["download", "-e", "binance", "--since", str(T), "--until", str(T + 3_600_000),
          "--data-dir", str(tmp_path)], client_factory=_factory(trades=2))
    out = tmp_path / "features.csv"
    assert main(["features", "--price-step", "10", "--window", "5",
                 "--out", str(out), "--data-dir", str(tmp_path)]) == 0
    features = pd.read_csv(out)
    assert {"rolling_delta", "volume_imbalance", "cumulative_delta"} <= set(features.columns)
    assert features["rolling_delta"].iloc[0] == pytest.approx(4.0)


def test_commands_require_price_step_and_report_empty_store(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["footprint", "--data-dir", str(tmp_path)])
    assert error.value.code == 2

    assert main(["footprint", "--price-step", "10", "--data-dir", str(tmp_path)]) == 1
    assert "No trades" in capsys.readouterr().err


def test_status_lists_checkpoints_and_downloads(tmp_path, capsys):
    main(["download", "-e", "bybit", "--since", str(T), "--until", str(T + 60_000),
          "--data-dir", str(tmp_path)], client_factory=_factory(trades=1))
    assert main(["status", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "partitions: 1" in out
    assert "checkpoints:" in out and "bybit:BTC/USDT:USDT" in out
    assert "download" in out


@pytest.mark.skipif(
    importlib.util.find_spec("ccxt") is not None,
    reason="ccxt installed; the missing-dependency path cannot be exercised",
)
def test_download_without_ccxt_reports_the_install_extra(tmp_path, capsys):
    code = main(
        ["download", "-e", "bybit", "--since", str(T), "--until", str(T + 60_000),
         "--data-dir", str(tmp_path)]
    )
    assert code == 1
    assert "pip install -e '.[marketdata]'" in capsys.readouterr().err
