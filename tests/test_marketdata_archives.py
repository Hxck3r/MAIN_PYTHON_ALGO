"""Tests for the bulk-archive downloader (offline: the network is faked).

Everything here runs without touching an exchange: the archive files are built
in a temp directory in the exact shapes the venues publish, and ``fetch_unit``
is replaced. What matters and is asserted is the *parsing and windowing* logic --
the column sniffing, the aggressor side, the OKX contract scaling and the UTC+8
day offset -- because those are the parts that silently corrupt a footprint.
"""
from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from marketdata import archives
from marketdata.archives import (
    ArchiveProgress,
    DatasetRequest,
    ArchiveCancelled,
    archive_days,
    build_kline_units,
    build_units,
    day_range,
    download_archives,
    download_candles,
    format_event,
    ingest_file,
    ingest_kline_file,
    native_instrument,
    pandas_to_binance_interval,
    parse_day,
    parse_symbol,
)
from marketdata.store import TradeStore

DAY = "2024-01-02"
NEXT_DAY = "2024-01-03"


def _ms(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp() * 1000)


def _zip(path: Path, member: str, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, text)
    return path


# --------------------------------------------------------------------------- #
# Dates, symbols, URLs
# --------------------------------------------------------------------------- #
def test_day_range_is_half_open_and_validated():
    assert day_range("2024-01-01", "2024-01-04") == [
        "2024-01-01", "2024-01-02", "2024-01-03",
    ]
    with pytest.raises(ValueError):
        day_range("2024-01-04", "2024-01-04")
    with pytest.raises(ValueError):
        parse_day("02.01.2024")


def test_okx_archives_need_one_extra_day():
    """OKX names files by a UTC+8 day, so the last 8 UTC hours live in the next file."""
    days = day_range("2024-01-02", "2024-01-03")
    assert archive_days("binance", days, NEXT_DAY) == ["2024-01-02"]
    assert archive_days("bybit", days, NEXT_DAY) == ["2024-01-02"]
    assert archive_days("okx", days, NEXT_DAY) == ["2024-01-02", "2024-01-03"]


def test_symbol_and_instrument_naming():
    assert parse_symbol("BTC/USDT:USDT") == ("BTC", "USDT")
    assert native_instrument("binance", "BTC/USDT:USDT") == "BTCUSDT"
    assert native_instrument("bybit", "BTC/USDT:USDT") == "BTCUSDT"
    assert native_instrument("okx", "BTC/USDT:USDT") == "BTC-USDT-SWAP"
    assert native_instrument("okx", "BTC/USDT:USDT", override="BTC-USDC-SWAP") == "BTC-USDC-SWAP"
    with pytest.raises(ValueError):
        parse_symbol("BTCUSDT")


def test_archive_urls_match_the_published_layout():
    units = build_units("binance", "BTCUSDT", "aggTrades", [DAY])
    assert units[0].url.endswith("/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-02.zip")
    assert units[0].checksum_url == units[0].url + ".CHECKSUM"
    bybit = build_units("bybit", "BTCUSDT", "trades", [DAY])
    assert bybit[0].url == "https://public.bybit.com/trading/BTCUSDT/BTCUSDT2024-01-02.csv.gz"
    okx = build_units("okx", "BTC-USDT-SWAP", "trades", [DAY])
    assert okx[0].url.endswith("/20240102/BTC-USDT-SWAP-trades-2024-01-02.zip")
    with pytest.raises(ValueError):
        build_units("kraken", "BTCUSDT", "trades", [DAY])


def test_kline_units_group_days_into_months():
    days = day_range("2024-01-30", "2024-02-02")
    monthly = build_kline_units("BTCUSDT", "1m", days)
    assert [unit.label for unit in monthly] == ["2024-01", "2024-02"]
    assert monthly[0].url.endswith("/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip")
    daily = build_kline_units("BTCUSDT", "1m", days, monthly=False)
    assert len(daily) == 3 and daily[0].url.endswith("BTCUSDT-1m-2024-01-30.zip")


def test_pandas_frequencies_resolve_to_binance_intervals():
    assert pandas_to_binance_interval("1min") == "1m"
    assert pandas_to_binance_interval("60min") == "1h"
    assert pandas_to_binance_interval("1D") == "1d"
    assert pandas_to_binance_interval("7min") is None


# --------------------------------------------------------------------------- #
# Readers: header sniffing, side, contract scaling
# --------------------------------------------------------------------------- #
def test_binance_agg_trades_without_header_invert_is_buyer_maker(tmp_path):
    """A headerless aggTrades file: is_buyer_maker=True means the taker sold."""
    path = _zip(
        tmp_path / "agg.zip", "agg.csv",
        # id, price, qty, first, last, transact_time, is_buyer_maker
        f"1,42000.5,1.5,1,1,{_ms('2024-01-02T00:00:01Z')},false\n"
        f"2,42001.0,2.0,2,2,{_ms('2024-01-02T00:00:02Z')},true\n",
    )
    frame = pd.concat(list(archives.read_binance_trades(
        path, dataset="aggTrades", chunk_rows=10, symbol="BTC/USDT:USDT"
    )))
    assert list(frame["side"]) == ["buy", "sell"]
    assert frame["trade_id"].tolist() == ["agg:1", "agg:2"]
    assert frame["quantity"].tolist() == [1.5, 2.0]
    assert frame["timestamp"].dt.tz is not None
    assert set(frame["contract_type"]) == {"linear_perpetual"}
    assert set(frame["quantity_unit"]) == {"BTC"}


def test_binance_trades_with_header_is_read_by_name(tmp_path):
    """Recent Binance files carry a header; the same reader must accept both."""
    path = _zip(
        tmp_path / "trades.zip", "trades.csv",
        "id,price,qty,quote_qty,time,is_buyer_maker\n"
        f"9,42000.0,0.25,10500,{_ms('2024-01-02T00:00:03Z')},true\n",
    )
    frame = pd.concat(list(archives.read_binance_trades(
        path, dataset="trades", chunk_rows=10, symbol="BTC/USDT:USDT"
    )))
    assert frame["side"].tolist() == ["sell"]
    assert frame["trade_id"].tolist() == ["9"]      # trades keep the raw id


def test_bybit_size_is_base_asset_and_side_is_taker(tmp_path):
    path = tmp_path / "bybit.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(
            "timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional\n"
            f"{_ms('2024-01-02T00:00:01Z')/1000:.0f},BTCUSDT,Sell,0.5,42000.0,PlusTick,abc,21000,0.5,21000\n"
        )
    frame = pd.concat(list(archives.read_bybit_trades(
        path, chunk_rows=10, symbol="BTC/USDT:USDT"
    )))
    assert frame["side"].tolist() == ["sell"]
    assert frame["quantity"].tolist() == [0.5]
    assert frame["trade_id"].tolist() == ["abc"]


def test_okx_contracts_are_scaled_to_base_units(tmp_path):
    """OKX trades are quoted in contracts; a missed ctVal is a 100x error."""
    path = _zip(
        tmp_path / "okx.zip", "okx.csv",
        "instrument_name,trade_id,side,size,price,created_time\n"
        f"BTC-USDT-SWAP,77,buy,3,42000.0,{_ms('2024-01-02T00:00:04Z')}\n",
    )
    frame = pd.concat(list(archives.read_okx_trades(
        path, chunk_rows=10, symbol="BTC/USDT:USDT", contract_size=0.01
    )))
    assert frame["quantity"].tolist() == [pytest.approx(0.03)]
    assert frame["side"].tolist() == ["buy"]


def test_kline_reader_returns_ohlcv_with_taker_volume(tmp_path):
    path = _zip(
        tmp_path / "BTCUSDT-1m-2024-01.zip", "k.csv",
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
        f"{_ms('2024-01-02T00:00:00Z')},42000,42010,41990,42005,12.5,"
        f"{_ms('2024-01-02T00:00:59Z')},525000,80,7.5,315000,0\n",
    )
    frame = ingest_kline_file(path, since=pd.Timestamp("2024-01-02", tz="UTC"),
                              until=pd.Timestamp("2024-01-03", tz="UTC"))
    assert len(frame) == 1
    assert frame.index[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert frame["close"].iloc[0] == 42005.0
    assert frame["taker_buy_volume"].iloc[0] == 7.5
    # The window is half-open: a bar at the upper bound is not included.
    assert ingest_kline_file(
        path, since=pd.Timestamp("2024-01-02", tz="UTC"),
        until=pd.Timestamp("2024-01-02T00:00:00Z"),
    ).empty


# --------------------------------------------------------------------------- #
# Ingest: windowing, storage layout, idempotency
# --------------------------------------------------------------------------- #
def test_ingest_writes_one_partition_per_day_and_respects_the_window(tmp_path):
    store = TradeStore(tmp_path / "store")
    path = _zip(
        tmp_path / "agg.zip", "agg.csv",
        f"1,42000,1,1,1,{_ms('2024-01-02T23:59:59Z')},false\n"
        f"2,42000,1,2,2,{_ms('2024-01-03T00:00:01Z')},false\n",
    )
    written = ingest_file(
        path, exchange="binance", symbol="BTC/USDT:USDT", dataset="aggTrades",
        store=store, since=pd.Timestamp("2024-01-02", tz="UTC"),
        until=pd.Timestamp("2024-01-03", tz="UTC"),
    )
    assert written == 1                    # the row past `until` is dropped
    partition = archives.partition_path(store, "binance", "BTC/USDT:USDT", DAY)
    assert partition.is_file()
    assert not archives.partition_path(store, "binance", "BTC/USDT:USDT", NEXT_DAY).exists()
    stored = store.read()
    assert stored["exchange"].tolist() == ["binance"]
    assert stored["quantity_unit"].tolist() == ["BTC"]


def test_okx_extra_day_does_not_double_count(tmp_path):
    """Files that overhang the window are trimmed by timestamp, not by file date.

    An OKX file dated X holds UTC [X-1 16:00, X 16:00), so a 1-day window is
    covered by two files, each of which also carries rows outside it.
    """
    store = TradeStore(tmp_path / "store")
    # Rows as they appear in the files dated 2024-01-02 and 2024-01-03.
    rows = "\n".join(
        f"BTC-USDT-SWAP,{index},{side},{size},42000.0,{_ms(stamp)}"
        for index, (side, size, stamp) in enumerate([
            ("buy", 1, "2024-01-01T15:00:00Z"),    # file 01-02, before the window
            ("buy", 2, "2024-01-02T10:00:00Z"),    # file 01-02, inside
            ("sell", 4, "2024-01-02T23:00:00Z"),   # file 01-03, inside
            ("sell", 1, "2024-01-03T02:00:00Z"),   # file 01-03, past `until`
        ])
    )
    path = _zip(tmp_path / "okx.zip", "okx.csv",
                "instrument_name,trade_id,side,size,price,created_time\n" + rows + "\n")
    written = ingest_file(
        path, exchange="okx", symbol="BTC/USDT:USDT", dataset="trades", store=store,
        since=pd.Timestamp("2024-01-02", tz="UTC"),
        until=pd.Timestamp("2024-01-03", tz="UTC"), contract_size=0.01,
    )
    assert written == 2
    stored = store.read()
    assert stored["quantity"].tolist() == [pytest.approx(0.02), pytest.approx(0.04)]
    assert stored["timestamp"].min() == pd.Timestamp("2024-01-02T10:00:00Z")


# --------------------------------------------------------------------------- #
# The download loop, with the network replaced
# --------------------------------------------------------------------------- #
def _fetcher(rows: dict[str, str], calls: list[str]):
    """Stand-in for fetch_unit: writes a real zip for known days, else 404s."""
    def fetch_unit(unit, tmpdir, *, retries, timeout, checksum=True):
        calls.append(unit.label)
        if unit.label not in rows:
            return None
        return _zip(Path(tmpdir) / Path(unit.url).name, "data.csv", rows[unit.label])
    return fetch_unit


def _agg_rows(day: str, count: int = 2) -> str:
    """One bar's worth of aggTrades: seconds apart, distinct price levels."""
    start = pd.Timestamp(day, tz="UTC")
    return "".join(
        f"{index},{42000 + 10 * index},1,{index},{index},"
        f"{int((start + pd.Timedelta(seconds=index)).timestamp() * 1000)},false\n"
        for index in range(1, count + 1)
    )


def test_download_plans_and_writes_then_skips_what_is_already_there(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(archives, "fetch_unit", _fetcher({DAY: _agg_rows(DAY)}, calls))
    events: list[ArchiveProgress] = []
    store_dir = tmp_path / "store"

    first = download_archives(
        ["binance"], DAY, NEXT_DAY, data_dir=store_dir, workers=2, progress=events.append
    )
    assert first.rows["binance"] == 2
    assert calls == [DAY]
    assert any(event.event == "plan" for event in events)
    assert any(event.event == "exchange_done" for event in events)

    calls.clear()
    second = download_archives(["binance"], DAY, NEXT_DAY, data_dir=store_dir, workers=2)
    assert calls == []                    # recorded complete in archives_state.json
    assert second.skipped == 1
    assert second.rows.get("binance", 0) == 0

    calls.clear()
    download_archives(["binance"], DAY, NEXT_DAY, data_dir=store_dir, workers=2, force=True)
    assert calls == [DAY]                 # --force redoes the day
    assert len(TradeStore(store_dir).read()) == 2     # deduplicated, not doubled


def test_missing_archive_is_reported_not_failed(tmp_path, monkeypatch):
    calls: list[str] = []
    # T-1: today's file is not published yet.
    monkeypatch.setattr(archives, "fetch_unit", _fetcher({DAY: _agg_rows(DAY)}, calls))
    result = download_archives(
        ["binance"], DAY, "2024-01-04", data_dir=tmp_path / "store", workers=2
    )
    assert result.missing == [f"binance {NEXT_DAY}"]
    assert result.failures == []
    assert result.ok


def test_two_binance_datasets_cannot_share_a_day(tmp_path, monkeypatch):
    """aggTrades and trades are the same flow at two granularities: never mix."""
    calls: list[str] = []
    monkeypatch.setattr(
        archives, "fetch_unit",
        _fetcher({DAY: _agg_rows(DAY), NEXT_DAY: _agg_rows(NEXT_DAY)}, calls),
    )
    store_dir = tmp_path / "store"
    download_archives(["binance"], DAY, NEXT_DAY, data_dir=store_dir, workers=2)
    result = download_archives(
        ["binance"], DAY, NEXT_DAY, data_dir=store_dir, workers=2, dataset="trades"
    )
    assert result.failures and "double-count" in result.failures[0]


def test_cancellation_keeps_completed_days(tmp_path, monkeypatch):
    calls: list[str] = []
    rows = {DAY: _agg_rows(DAY), NEXT_DAY: _agg_rows(NEXT_DAY)}
    monkeypatch.setattr(archives, "fetch_unit", _fetcher(rows, calls))
    state = {"stop": False}

    def progress(event: ArchiveProgress) -> None:
        if event.event == "file":          # stop right after the first file
            state["stop"] = True

    result = download_archives(
        ["binance"], DAY, "2024-01-04", data_dir=tmp_path / "store", workers=2,
        progress=progress, should_stop=lambda: state["stop"],
    )
    assert result.cancelled
    assert result.rows["binance"] == 2
    assert len(TradeStore(tmp_path / "store").read()) == 2
    # The interrupted day is not marked complete, so a rerun picks it up.
    assert (tmp_path / "store" / "archives_state.json").is_file()


def test_dry_run_touches_nothing(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(archives, "fetch_unit", _fetcher({DAY: _agg_rows(DAY)}, calls))
    result = download_archives(
        ["binance"], DAY, NEXT_DAY, data_dir=tmp_path / "store", dry_run=True
    )
    assert calls == []
    assert result.planned and result.planned[0].endswith(".zip")
    assert not (tmp_path / "store" / "canonical").exists()


def test_non_btc_symbol_is_rejected_before_downloading(tmp_path):
    with pytest.raises(ValueError, match="BTC"):
        download_archives(["binance"], DAY, NEXT_DAY, symbol="ETH/USDT:USDT",
                          data_dir=tmp_path / "store")


def test_progress_events_are_formatted_for_the_console():
    assert format_event(ArchiveProgress(event="file", exchange="binance",
                                        label=DAY, rows=1234)) == "  binance 2024-01-02: 1,234 rows"
    assert format_event(ArchiveProgress(event="exchange_done", exchange="okx",
                                        rows=10, done=2)) == "okx: 10 rows in 2 file(s)"
    assert format_event(ArchiveProgress(event="error", exchange="okx",
                                        message="boom")) == "  ! okx : boom"
    # Per-day footprint lines are throttled: 25 of them would bury the log.
    assert format_event(ArchiveProgress(event="footprint", done=7, total=30)) is None
    assert "1/30" in format_event(ArchiveProgress(event="footprint", done=1, total=30))


# --------------------------------------------------------------------------- #
# Candles
# --------------------------------------------------------------------------- #
def test_download_candles_falls_back_to_daily_when_the_month_is_unpublished(monkeypatch):
    """The current month has no monthly file yet; daily files must fill it."""
    fetched: list[str] = []
    minute = _ms("2024-01-02T00:00:00Z")
    header = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
              "taker_buy_volume,taker_buy_quote_volume,ignore\n")
    monthly = "BTCUSDT-1m-2024-01.zip"
    daily = "BTCUSDT-1m-2024-01-02.zip"

    def fetch_unit(unit, tmpdir, *, retries, timeout, checksum=True):
        name = Path(unit.url).name
        fetched.append(name)
        if name == monthly:
            return None                     # not published
        if name == daily:
            return _zip(Path(tmpdir) / name, "k.csv", header + (
                f"{minute},42000,42010,41990,42005,12.5,{minute + 59_000},"
                "525000,80,7.5,315000,0\n"
            ))
        return None

    monkeypatch.setattr(archives, "fetch_unit", fetch_unit)
    frames = download_candles(DAY, NEXT_DAY, workers=1)
    assert len(frames) == 1
    assert frames.index[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert fetched == [monthly, daily]


def test_download_candles_merges_months_without_duplicates(monkeypatch):
    minute_a = _ms("2024-01-31T23:59:00Z")
    minute_b = _ms("2024-02-01T00:00:00Z")
    header = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
              "taker_buy_volume,taker_buy_quote_volume,ignore\n")

    def fetch_unit(unit, tmpdir, *, retries, timeout, checksum=True):
        minute = minute_a if unit.label == "2024-01" else minute_b
        return _zip(Path(tmpdir) / Path(unit.url).name, "k.csv", header + (
            f"{minute},1,2,0.5,1.5,10,{minute},10,1,5,5,0\n"
        ))

    monkeypatch.setattr(archives, "fetch_unit", fetch_unit)
    frames = download_candles("2024-01-31", "2024-02-02", workers=2)
    assert len(frames) == 2
    assert frames.index.is_monotonic_increasing and frames.index.is_unique
    assert {"open", "high", "low", "close", "volume"} <= set(frames.columns)


# --------------------------------------------------------------------------- #
# DatasetRequest
# --------------------------------------------------------------------------- #
def test_dataset_request_validates_and_maps_to_kwargs():
    request = DatasetRequest(since=DAY, until=NEXT_DAY)
    request.validate()
    kwargs = request.to_kwargs()
    assert kwargs["price_step"] == 10.0 and kwargs["candles_interval"] == "1m"
    without = DatasetRequest(since=DAY, until=NEXT_DAY, with_footprint=False,
                             with_candles=False)
    kwargs = without.to_kwargs()
    assert kwargs["price_step"] is None and kwargs["candles_interval"] is None


@pytest.mark.parametrize("overrides, message", [
    ({"exchanges": ()}, "биржу"),
    ({"since": NEXT_DAY, "until": DAY}, "раньше"),
    ({"price_step": 0.0}, "Шаг цены"),
    ({"candles_interval": "15m"}, "совпадать"),
    ({"exchanges": ("okx",)}, "Binance"),
])
def test_dataset_request_rejects_bad_input(overrides, message):
    fields = {"since": DAY, "until": NEXT_DAY, **overrides}
    with pytest.raises(ValueError, match=message):
        DatasetRequest(**fields).validate()


def test_dataset_download_without_network_still_reports_a_result(monkeypatch, tmp_path):
    """A run whose only day is unpublished must not raise, only report."""
    monkeypatch.setattr(archives, "fetch_unit",
                        _fetcher({}, []))
    request = DatasetRequest(since=DAY, until=NEXT_DAY, data_dir=str(tmp_path / "store"),
                             with_footprint=False, with_candles=False)
    result = request.download()
    assert result.missing == [f"binance {DAY}"]
    assert result.footprint is None and result.candles is None
    assert "not published" in result.summary()


def test_footprint_and_candles_are_built_from_a_real_download(tmp_path, monkeypatch):
    """End to end without the network: archives -> store -> footprint parquet."""
    day = "2024-01-02"
    rows = "".join(
        f"{index},{42000 + index},{index + 1},0,0,"
        f"{_ms(f'{day}T00:0{index}:00Z')},{'true' if index % 2 else 'false'}\n"
        for index in range(1, 6)
    )
    monkeypatch.setattr(archives, "fetch_unit", _fetcher({day: rows}, []))
    header = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
              "taker_buy_volume,taker_buy_quote_volume,ignore\n")
    minute = _ms(f"{day}T00:00:00Z")
    klines = header + f"{minute},42000,42010,41990,42005,12.5,{minute + 59_000},525000,80,7.5,315000,0\n"

    def fetch_unit(unit, tmpdir, *, retries, timeout, checksum=True):
        name = Path(unit.url).name
        return _zip(Path(tmpdir) / name, "data.csv",
                    klines if "klines" in unit.url else rows)

    monkeypatch.setattr(archives, "fetch_unit", fetch_unit)
    request = DatasetRequest(
        since=day, until="2024-01-03", data_dir=str(tmp_path / "store"),
        timeframe="1min", price_step=10.0, candles_interval="1m",
    )
    result = request.download()
    assert result.failures == []
    assert result.footprint_path and Path(result.footprint_path).is_file()
    assert result.candles_path and Path(result.candles_path).is_file()
    assert result.footprint.index.names == ["bar_start", "price_level"]
    assert "combined_delta" in result.footprint.columns
    assert len(result.candles) == 1
    assert "ticks" in result.summary()


def test_footprint_from_store_reads_day_partitions_in_order(tmp_path, monkeypatch):
    """The footprint is aggregated one day at a time, so memory stays at one day."""
    calls: list[str] = []
    rows = {"2024-01-02": _agg_rows("2024-01-02", 2), "2024-01-03": _agg_rows("2024-01-03", 3)}
    monkeypatch.setattr(archives, "fetch_unit", _fetcher(rows, calls))
    store_dir = tmp_path / "store"
    download_archives(["binance"], "2024-01-02", "2024-01-04", data_dir=store_dir, workers=2)
    days_seen: list[str] = []

    def progress(event: ArchiveProgress) -> None:
        if event.event == "footprint":
            days_seen.append(event.label)

    footprint = archives.footprint_from_store(
        TradeStore(store_dir), since="2024-01-02", until="2024-01-04",
        timeframe="1min", price_step=10.0, progress=progress,
    )
    assert days_seen == ["2024-01-02", "2024-01-03"]
    bars = footprint.index.get_level_values("bar_start").unique()
    assert len(bars) == 2
    assert footprint.index.is_monotonic_increasing
    assert footprint["combined_total_qty"].sum() == pytest.approx(5.0)
    assert footprint.index.names == ["bar_start", "price_level"]

    # A stop request is honoured instead of building a partial footprint.
    with pytest.raises(ArchiveCancelled):
        archives.footprint_from_store(
            TradeStore(store_dir), since="2024-01-02", until="2024-01-04",
            price_step=10.0, should_stop=lambda: True,
        )
