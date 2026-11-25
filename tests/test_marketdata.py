"""Offline tests for public-trade ingestion, storage and footprint aggregation."""
from __future__ import annotations

import pandas as pd
import pytest

from marketdata import (
    DownloadReport,
    SYMBOL,
    TRADE_COLUMNS,
    TradeStore,
    add_order_flow_features,
    aggregate_footprint,
    bar_features_from_footprint,
    build_footprint,
    download_trades,
    normalize_trades,
    validate_trades,
)

T = 1_700_000_000_000  # 2023-11-14T22:13:20Z


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def test_normalize_exchange_specific_trade_fields_to_utc_and_btc():
    binance = normalize_trades([{
        "id": "9", "timestamp": T, "price": "42000.5",
        "amount": "0.01", "side": "buy", "info": {"m": False},
    }], "binance")
    okx = normalize_trades([{
        "timestamp": T, "info": {
            "tradeId": "9", "px": "42000.5", "sz": "0.01", "side": "buy",
        }, "price": "42000.5", "amount": "0.01", "side": "buy",
    }], "okx")
    assert binance.iloc[0].timestamp == pd.Timestamp("2023-11-14T22:13:20Z")
    assert binance.iloc[0].side == "buy"
    assert okx.iloc[0].quantity == pytest.approx(0.01)
    assert okx.iloc[0].quantity_unit == "BTC"


@pytest.mark.parametrize(
    ("flag", "expected"),
    [(True, "sell"), (False, "buy"), ("1", "sell"), ("0", "buy"),
     ("True", "sell"), ("False", "buy"), ("yes", "sell"), ("no", "buy")],
)
def test_maker_flag_variants_map_to_aggressor_side(flag, expected):
    trades = normalize_trades([{
        "timestamp": T, "price": 2, "amount": 1, "info": {"m": flag}, "id": "1",
    }], "binance")
    assert trades.iloc[0].side == expected


def test_naive_timestamp_is_rejected_instead_of_assumed_utc():
    with pytest.raises(ValueError, match="timezone"):
        normalize_trades([{
            "timestamp": "2024-01-01T00:00:00", "price": 10,
            "amount": 1, "side": "buy",
        }], "bybit")


def test_aware_iso_timestamp_is_accepted():
    trades = normalize_trades([{
        "timestamp": "2023-11-14T22:13:20.000Z", "price": 10,
        "amount": 1, "side": "buy",
    }], "bybit")
    assert trades.iloc[0].timestamp == pd.Timestamp("2023-11-14T22:13:20Z")


def test_naive_top_level_timestamp_falls_back_to_info_timestamp():
    trades = normalize_trades([{
        "timestamp": "2024-01-01T00:00:00", "price": 10, "amount": 1,
        "side": "buy", "info": {"ts": T},
    }], "bybit")
    assert trades.iloc[0].timestamp == pd.Timestamp("2023-11-14T22:13:20Z")


# --------------------------------------------------------------------------- #
# Quantity units
# --------------------------------------------------------------------------- #
def test_venue_native_quantity_requires_contract_size():
    row = {"timestamp": T, "price": 10, "side": "buy", "info": {"qty": "3"}}
    with pytest.raises(ValueError, match="contract_size"):
        normalize_trades([row], "bybit")
    converted = normalize_trades([row], "bybit", contract_size=0.001)
    assert converted.iloc[0].quantity == pytest.approx(0.003)


def test_contract_size_makes_top_level_amount_contract_denominated():
    trades = normalize_trades([{
        "timestamp": T, "price": 10, "amount": 3, "side": "buy",
    }], "bybit", contract_size=0.001)
    assert trades.iloc[0].quantity == pytest.approx(0.003)


def test_quantity_source_base_disables_contract_conversion():
    trades = normalize_trades([{
        "timestamp": T, "price": 10, "amount": 3, "side": "buy",
    }], "bybit", contract_size=0.001, quantity_source="base")
    assert trades.iloc[0].quantity == pytest.approx(3.0)


def test_declared_contract_units_convert_to_base():
    trades = normalize_trades([{
        "timestamp": T, "price": 10, "amount": "3",
        "quantity_unit": "contracts", "side": "buy", "info": {"m": "False"},
    }], "bybit", contract_size=0.001)
    assert trades.iloc[0].quantity == pytest.approx(0.003)
    assert trades.iloc[0].side == "buy"


def test_unknown_quantity_source_is_rejected():
    with pytest.raises(ValueError, match="quantity_source"):
        normalize_trades([{
            "timestamp": T, "price": 10, "amount": 1, "side": "buy",
        }], "bybit", quantity_source="barrels")


def test_non_linear_contract_is_rejected():
    with pytest.raises(ValueError, match="non-linear"):
        normalize_trades([{
            "timestamp": T, "price": 10, "amount": 1,
            "side": "buy", "contractType": "inverse",
        }], "bybit")


# --------------------------------------------------------------------------- #
# Validation and deduplication
# --------------------------------------------------------------------------- #
def test_validation_deduplicates_missing_ids_by_trade_content():
    rows = normalize_trades([
        {"timestamp": T, "price": 10, "amount": 0.1, "side": "buy"},
        {"timestamp": T, "price": 10, "amount": 0.1, "side": "buy"},
    ], "binance")
    clean, report = validate_trades(rows)
    assert len(clean) == 1
    assert report.duplicates_removed == 1


def test_trade_id_deduplicates_even_with_different_source_identity():
    trades = normalize_trades([
        {"id": "stable", "sequence": "first", "timestamp": T,
         "price": 10, "amount": 0.1, "side": "buy"},
        {"id": "stable", "sequence": "retry", "timestamp": T,
         "price": 10, "amount": 0.1, "side": "buy"},
    ], "binance")
    clean, report = validate_trades(trades)
    assert len(clean) == 1
    assert report.duplicates_removed == 1


def test_missing_id_rows_with_distinct_source_identity_stay_distinct():
    trades = normalize_trades([
        {"sequence": "row-a", "timestamp": T, "price": 10, "amount": 0.1, "side": "buy"},
        {"sequence": "row-b", "timestamp": T, "price": 10, "amount": 0.1, "side": "buy"},
    ], "binance")
    clean, report = validate_trades(trades)
    assert len(clean) == 2
    assert report.duplicates_removed == 0


def test_same_trade_id_on_different_exchanges_is_not_deduplicated():
    raw = []
    for venue in ("binance", "okx"):
        raw += normalize_trades([
            {"id": "same", "timestamp": T, "price": 10, "amount": 0.1, "side": "buy"},
        ], venue).to_dict("records")
    clean, report = validate_trades(pd.DataFrame(raw))
    assert len(clean) == 2
    assert report.duplicates_removed == 0


def test_invalid_rows_are_removed_and_reported():
    raw = normalize_trades([
        {"id": "1", "timestamp": T, "price": 10, "amount": 0.1, "side": "buy"},
    ], "binance").to_dict("records")
    raw.append({**raw[0], "price": 0})
    clean, report = validate_trades(pd.DataFrame(raw))
    assert len(clean) == 1
    assert report.invalid_rows_removed == 1


# --------------------------------------------------------------------------- #
# Footprint
# --------------------------------------------------------------------------- #
def test_footprint_common_grid_separate_venues_and_coverage():
    # Two bars (1m apart), both printing into the same 42000 price bucket.
    frames = [normalize_trades([
        {"id": "1", "timestamp": T, "price": 42000.9, "amount": 0.2, "side": "buy"},
        {"id": "2", "timestamp": T + 70_000, "price": 42000.1,
         "amount": 0.1, "side": "sell"},
    ], venue) for venue in ("binance", "okx")]
    all_trades = pd.concat(frames, ignore_index=True)
    first = pd.Timestamp(T, unit="ms", tz="UTC").floor("1min")
    second = first + pd.Timedelta(minutes=1)

    foot = build_footprint(all_trades, price_step=1, exchange="binance")
    assert sorted(foot.index) == [(first, 42000.0), (second, 42000.0)]
    assert foot.loc[(first, 42000.0), "delta"] == pytest.approx(0.2)

    combined = aggregate_footprint(all_trades, price_step=1)
    up, down = combined.loc[(first, 42000.0)], combined.loc[(second, 42000.0)]
    assert up.binance_covered and up.okx_covered and not up.bybit_covered
    assert up.combined_buy_qty == pytest.approx(0.4)    # two venues, no double count
    assert up.combined_delta == pytest.approx(0.4)
    assert down.combined_delta == pytest.approx(-0.2)
    # A venue that traded nothing in a cell reports missing data, not zero.
    assert pd.isna(up.bybit_delta)
    assert up.binance_trade_count == 1


def test_footprint_uses_decimal_price_buckets():
    trades = normalize_trades([
        {"timestamp": T, "price": 0.3, "amount": 1, "side": "buy"},
    ], "binance")
    foot = build_footprint(trades, price_step=0.1)
    assert foot.index.get_level_values("price_level")[0] == pytest.approx(0.3)


def test_aggregate_footprint_on_empty_trades_keeps_schema():
    foot = aggregate_footprint(pd.DataFrame(columns=TRADE_COLUMNS), price_step=1)
    assert foot.empty
    assert "combined_delta" in foot.columns
    assert "binance_covered" in foot.columns


# --------------------------------------------------------------------------- #
# Order-flow features
# --------------------------------------------------------------------------- #
def test_order_flow_features_are_causal():
    index = pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC")
    frame = pd.DataFrame({
        "combined_delta": [1.0, 2.0, 100.0],
        "combined_buy_qty": [2.0, 3.0, 101.0],
        "combined_sell_qty": [1.0, 1.0, 1.0],
        "combined_total_qty": [3.0, 4.0, 102.0],
    }, index=index)
    features = add_order_flow_features(frame, window=2)
    assert features.loc[index[1], "rolling_delta"] == pytest.approx(3.0)
    assert features.loc[index[1], "cumulative_delta"] == pytest.approx(3.0)
    assert pd.isna(features.loc[index[0], "delta_zscore"])


def test_bar_features_do_not_sum_coverage_flags_as_volume():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01T00:00:00Z"), 10.0),
         (pd.Timestamp("2024-01-01T00:00:00Z"), 11.0)],
        names=["bar_start", "price_level"],
    )
    footprint = pd.DataFrame({
        "combined_delta": [1.0, 2.0],
        "combined_buy_qty": [2.0, 3.0],
        "combined_sell_qty": [1.0, 1.0],
        "combined_total_qty": [3.0, 4.0],
        "binance_has_data": [True, True],
        "binance_covered": [True, True],
        "bybit_has_data": [False, False],
        "bybit_covered": [False, False],
    }, index=index)
    features = bar_features_from_footprint(footprint)
    assert features.iloc[0].combined_total_qty == pytest.approx(7.0)
    assert features.iloc[0].combined_total_qty != 9.0   # coverage flags not added
    assert bool(features.iloc[0].binance_covered)
    assert not bool(features.iloc[0].bybit_covered)


def test_bar_features_work_for_single_venue_plain_columns():
    trades = normalize_trades([
        {"id": "1", "timestamp": T, "price": 10, "amount": 0.4, "side": "buy"},
        {"id": "2", "timestamp": T, "price": 11, "amount": 0.1, "side": "sell"},
    ], "binance")
    features = bar_features_from_footprint(build_footprint(trades, price_step=1))
    assert features.iloc[0].delta == pytest.approx(0.3)


def test_footprint_features_drive_a_strategy_end_to_end():
    """Trades -> footprint -> bar features -> engine -> executed orders."""
    from backtester import BacktestEngine
    from strategies import FootprintDeltaStrategy

    start = pd.Timestamp(T, unit="ms", tz="UTC").floor("1min")
    trades = normalize_trades([
        {"id": "1", "timestamp": int(start.timestamp() * 1000),
         "price": 100.5, "amount": 1.0, "side": "buy"},
        # Strong sell imbalance one minute later -> delta crosses zero.
        {"id": "2", "timestamp": int((start + pd.Timedelta(minutes=1)).timestamp() * 1000),
         "price": 100.5, "amount": 2.0, "side": "sell"},
    ], "binance")
    footprint = aggregate_footprint(trades, price_step=1)
    market_features = bar_features_from_footprint(footprint, window=20)
    assert market_features.loc[start, "rolling_delta"] == pytest.approx(1.0)
    assert market_features.iloc[1]["rolling_delta"] == pytest.approx(-1.0)

    index = pd.date_range(start, periods=6, freq="1min")
    candles = pd.DataFrame({
        "open": [100.0, 100.2, 100.4, 100.6, 100.8, 101.0],
        "high": [100.3, 100.5, 100.7, 100.9, 101.1, 101.3],
        "low": [99.8, 100.0, 100.2, 100.4, 100.6, 100.8],
        "close": [100.2, 100.4, 100.6, 100.8, 101.0, 101.2],
        "volume": [10.0] * 6,
    }, index=index)

    result = BacktestEngine(
        candles, FootprintDeltaStrategy, market_features=market_features,
        footprint=footprint, cash=10_000.0,
    ).run()

    tags = [order.tag for order in result.orders]
    assert "footprint_delta_buy" in tags
    assert "footprint_delta_sell" in tags
    # Features are known at the close of bar 0, so the market order fills at the
    # open of bar 1 -- no same-bar execution, no lookahead.
    entry = next(order for order in result.orders if order.tag == "footprint_delta_buy")
    assert entry.filled_bar == 1
    assert entry.filled_price == pytest.approx(candles["open"].iloc[1])
    assert result.footprint is footprint
    assert "rolling_delta" in result.market_features.columns


def test_heatmap_coordinate_edges_follow_irregular_gaps():
    from gui.chart_widget import _coordinate_edges

    edges = _coordinate_edges([0.0, 1.0, 10.0], 60.0)
    assert edges == pytest.approx([-0.5, 0.5, 5.5, 14.5])


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def _trades(exchange="binance", trade_id="1", price=10.0, ts=T):
    return normalize_trades([{
        "id": trade_id, "timestamp": ts, "price": price,
        "amount": 0.1, "side": "buy",
    }], exchange)


def test_store_partitions_reads_checkpoint_and_manifest(tmp_path):
    store = TradeStore(tmp_path)
    store.write(pd.concat([_trades("binance"), _trades("okx", price=20.0)],
                          ignore_index=True))
    store.set_checkpoint("binance", SYMBOL, 123)

    assert len(store.read(exchange="binance")) == 1
    assert len(store.read(exchange="okx")) == 1
    assert len(store.read()) == 2
    assert len(store.read(symbol="ETH/USDT:USDT")) == 0
    assert store.get_checkpoint("binance", SYMBOL) == 123
    assert (tmp_path / "manifest.json").exists()
    partitions = list(store.canonical.glob("exchange=*/symbol=*/date=*/trades.parquet"))
    assert len(partitions) == 2


def test_store_write_is_idempotent_for_repeated_pages(tmp_path):
    store = TradeStore(tmp_path)
    trades = _trades()
    store.write(trades)
    store.write(trades)
    assert len(store.read(exchange="binance")) == 1


def test_store_read_can_sort(tmp_path):
    store = TradeStore(tmp_path)
    store.write(pd.concat([_trades(ts=T + 1000), _trades(ts=T)], ignore_index=True))
    frame = store.read(sort=True)
    assert frame["timestamp"].is_monotonic_increasing


def test_raw_archive_survives_unprintable_values(tmp_path):
    store = TradeStore(tmp_path)
    path = store.save_raw("bybit", SYMBOL, T, [
        None,
        10**10000,
        {"timestamp": 10**10000, "info": {"m": float("nan")}},
        {"ok": 1},
    ])
    assert len(path.read_text().splitlines()) == 4


def test_store_records_issues_and_download_summaries(tmp_path):
    store = TradeStore(tmp_path)
    store.note_issue("bybit", SYMBOL, "test issue")
    store.record_download("bybit", SYMBOL, DownloadReport(pages=2, rows_written=5))
    import json
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["issues"][0]["message"] == "test issue"
    assert manifest["downloads"][0]["rows_written"] == 5


# --------------------------------------------------------------------------- #
# Downloader
# --------------------------------------------------------------------------- #
def test_download_preserves_same_millisecond_page_boundary(tmp_path):
    class Client:
        def __init__(self):
            self.calls = 0

        def fetch_trades(self, symbol, since, limit):
            self.calls += 1
            if self.calls == 1:
                return [{"id": "a", "timestamp": since, "price": 10,
                         "amount": 0.1, "side": "buy"}]
            if self.calls == 2:
                return [{"id": "b", "timestamp": since, "price": 11,
                         "amount": 0.2, "side": "sell"}]
            return [{"id": "b", "timestamp": since, "price": 11,
                     "amount": 0.2, "side": "sell"}]

    client, store = Client(), TradeStore(tmp_path)
    count = download_trades("bybit", T, T + 1, store, limit=1,
                            exchange_client=client)
    assert count == 2
    assert sorted(store.read(exchange="bybit")["trade_id"]) == ["a", "b"]


def test_download_end_boundary_page_stops_and_filters_output(tmp_path):
    class Client:
        def __init__(self):
            self.calls = 0

        def fetch_trades(self, symbol, since, limit):
            self.calls += 1
            return [
                {"id": "inside", "timestamp": since, "price": 10,
                 "amount": 0.1, "side": "buy"},
                {"id": "at-end", "timestamp": since + 1, "price": 11,
                 "amount": 0.1, "side": "sell"},
            ]

    client, store = Client(), TradeStore(tmp_path)
    count = download_trades("bybit", T, T + 1, store, limit=2,
                            exchange_client=client)
    assert count == 1
    assert client.calls == 1
    assert store.read(exchange="bybit").iloc[0].trade_id == "inside"


def test_download_accepts_iso_trade_timestamp(tmp_path):
    class Client:
        def fetch_trades(self, symbol, since, limit):
            return [{"id": "iso", "timestamp": "2023-11-14T22:13:20Z",
                     "price": 10, "amount": 0.1, "side": "buy"}]

    store = TradeStore(tmp_path)
    count = download_trades("bybit", "2023-11-14T22:13:20Z",
                            "2023-11-14T22:13:21Z", store, exchange_client=Client())
    assert count == 1
    assert len(store.read(exchange="bybit")) == 1
    assert len(store.read(exchange="binance")) == 0


def test_download_tolerates_non_mapping_and_extreme_rows(tmp_path):
    class Client:
        def fetch_trades(self, symbol, since, limit):
            return [None, {"id": "bad", "timestamp": 10**10000, "price": 1,
                           "amount": 1, "side": "buy"},
                    {"id": "good", "timestamp": since, "price": 10,
                     "amount": 0.1, "side": "buy"}]

    store = TradeStore(tmp_path)
    with pytest.warns(RuntimeWarning):
        count = download_trades("bybit", T, T + 1, store,
                                exchange_client=Client())
    assert count == 1
    assert len(store.read(exchange="bybit")) == 1
    assert list((tmp_path / "raw" / "bybit").rglob("*.jsonl"))


def test_download_archives_raw_page_even_if_every_row_is_bad(tmp_path):
    class Client:
        def fetch_trades(self, symbol, since, limit):
            return [
                {"id": "good", "timestamp": since, "price": 10,
                 "amount": 0.1, "side": "buy"},
                {"id": "bad", "timestamp": "not-a-time", "price": 10,
                 "amount": 0.1, "side": "buy"},
            ]

    store = TradeStore(tmp_path)
    with pytest.warns(RuntimeWarning, match="malformed"):
        count = download_trades("bybit", T, T + 1, store,
                                exchange_client=Client())
    assert count == 1
    raw_files = list((tmp_path / "raw" / "bybit").rglob("*.jsonl"))
    assert len(raw_files) == 1
    assert len(raw_files[0].read_text().splitlines()) == 2


def test_download_resumes_from_checkpoint_without_duplicates(tmp_path):
    class First:
        def fetch_trades(self, symbol, since, limit):
            return [{"id": "1", "timestamp": since, "price": 10,
                     "amount": 0.1, "side": "buy"}]

    class Second:
        def fetch_trades(self, symbol, since, limit):
            return [{"id": "2", "timestamp": since, "price": 11,
                     "amount": 0.2, "side": "sell"}]

    store = TradeStore(tmp_path)
    assert download_trades("bybit", T, T + 1000, store,
                           exchange_client=First()) == 1
    checkpoint = store.get_checkpoint("bybit", SYMBOL)
    assert checkpoint == T + 1
    assert download_trades("bybit", T, T + 1000, store,
                           exchange_client=Second()) == 1
    assert sorted(store.read(exchange="bybit")["trade_id"]) == ["1", "2"]


def test_download_reports_unpageable_bucket_in_strict_mode(tmp_path):
    class Client:
        def fetch_trades(self, symbol, since, limit):
            return [{"id": str(i), "timestamp": since, "price": 10 + i,
                     "amount": 0.1, "side": "buy"} for i in range(limit)]

    with pytest.raises(RuntimeError, match="Incomplete trade retrieval"):
        download_trades("bybit", T, T + 100, TradeStore(tmp_path), limit=2,
                        exchange_client=Client(), strict_pagination=True)


def test_download_warns_and_records_unpageable_bucket_by_default(tmp_path):
    class Client:
        def fetch_trades(self, symbol, since, limit):
            return [{"id": str(i), "timestamp": since, "price": 10 + i,
                     "amount": 0.1, "side": "buy"} for i in range(limit)]

    store = TradeStore(tmp_path)
    report = DownloadReport()
    with pytest.warns(RuntimeWarning, match="incomplete"):
        count = download_trades("bybit", T, T + 100, store, limit=2,
                                exchange_client=Client(), report=report)
    assert count == 2
    assert report.incomplete_buckets == [T]
    import json
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert any("incomplete" in issue["message"] for issue in manifest["issues"])


def test_download_report_counts_pages_and_rows(tmp_path):
    class Client:
        def fetch_trades(self, symbol, since, limit):
            return [{"id": "1", "timestamp": since, "price": 10,
                     "amount": 0.1, "side": "buy"}]

    report = DownloadReport()
    download_trades("bybit", T, T + 1, TradeStore(tmp_path),
                    exchange_client=Client(), report=report)
    assert report.pages == 1
    assert report.rows_fetched == 1
    assert report.rows_written == 1
    assert report.malformed_rows == 0


# --------------------------------------------------------------------------- #
# Within-bar (price cell) footprint features
# --------------------------------------------------------------------------- #
def _cell_footprint(bars: dict) -> pd.DataFrame:
    """Build a combined footprint from ``{"<bar iso>": {level: (buy, sell)}}``."""
    tuples, buy, sell = [], [], []
    for bar, cells in bars.items():
        for level, (cell_buy, cell_sell) in sorted(cells.items()):
            tuples.append((pd.Timestamp(bar), float(level)))
            buy.append(float(cell_buy))
            sell.append(float(cell_sell))
    index = pd.MultiIndex.from_tuples(tuples, names=["bar_start", "price_level"])
    return pd.DataFrame({
        "combined_buy_qty": buy,
        "combined_sell_qty": sell,
        "combined_delta": [b - s for b, s in zip(buy, sell)],
        "combined_total_qty": [b + s for b, s in zip(buy, sell)],
    }, index=index)


def test_cell_features_rebuild_poc_value_area_and_vwap():
    from marketdata import cell_order_flow_features

    footprint = _cell_footprint({
        "2024-01-02T00:00:00Z": {100: (0.5, 0.5), 101: (1, 1), 102: (5, 5),
                                 103: (1, 1), 104: (0.5, 0.5)},
    })
    cells = cell_order_flow_features(footprint)
    row = cells.iloc[0]

    assert row.poc_price == 102.0                    # 10 of 16 traded there
    assert row.poc_volume == pytest.approx(10.0)
    assert row.poc_share == pytest.approx(10.0 / 16.0)
    assert row.level_count == 5.0
    assert row.cell_volume == pytest.approx(16.0)
    assert row.bar_delta == pytest.approx(0.0)
    # 70% of 16 = 11.2: the first two-level step up (3) beats the step down (3)
    # on the tie, so the area is 102..104 and holds 12 of 16.
    assert row.value_area_low == 102.0
    assert row.value_area_high == 104.0
    assert row.value_area_share == pytest.approx(12.0 / 16.0)
    # A price level covers [level, level + step), so the bar's high is the top of
    # the highest traded bucket.
    assert row.bar_low_cell == 100.0
    assert row.bar_high_cell == 105.0
    assert row.bar_range_ticks == 5.0
    assert row.bar_vwap == pytest.approx(102.5)       # half-tick centred
    assert row.vwap_minus_poc == pytest.approx(0.5)
    assert pd.isna(cells["poc_migration"].iloc[0])     # nothing before it


def test_cell_features_count_diagonal_imbalance_and_sign_the_stack():
    from marketdata import cell_order_flow_features

    footprint = _cell_footprint({
        # Sellers rest 1 BTC at every level; buyers take >= 3x that at 101..103.
        "2024-01-02T00:00:00Z": {100: (0.1, 1), 101: (3.2, 1), 102: (3.1, 1),
                                 103: (3.5, 1)},
        # Mirror: sellers hit 101..103 through resting bids of 1 BTC, and the
        # quiet level below is balanced so it cannot read as a buy imbalance.
        "2024-01-02T00:01:00Z": {100: (1, 1), 101: (1, 3.2), 102: (1, 3.1),
                                 103: (1, 3.5)},
    })
    cells = cell_order_flow_features(footprint)

    buyers = cells.iloc[0]
    assert buyers.buy_imbalance_levels == 3.0
    assert buyers.sell_imbalance_levels == 0.0
    assert buyers.imbalance_stack == 3.0              # positive = buyers own it
    assert buyers.imbalance_skew == pytest.approx(1.0)

    sellers = cells.iloc[1]
    assert sellers.sell_imbalance_levels == 3.0
    assert sellers.buy_imbalance_levels == 0.0
    assert sellers.imbalance_stack == -3.0
    assert sellers.imbalance_skew == pytest.approx(-1.0)


def test_cell_features_read_absorption_from_the_heavy_flat_cell():
    from marketdata import cell_order_flow_features

    footprint = _cell_footprint({
        # One heavy cell with almost no net delta: aggressive selling met by bids.
        "2024-01-02T00:00:00Z": {100: (0.6, 0.0), 101: (0.6, 0.0),
                                 102: (5.0, 5.1), 103: (0.6, 0.0), 104: (0.6, 0.0)},
        # Mirror: aggressive buying absorbed by offers.
        "2024-01-02T00:01:00Z": {100: (0.0, 0.6), 101: (0.0, 0.6),
                                 102: (5.1, 5.0), 103: (0.0, 0.6), 104: (0.0, 0.6)},
        # No heavy flat cell at all: every cell is one-sided.
        "2024-01-02T00:02:00Z": {100: (0.6, 0.0), 101: (0.0, 0.6),
                                 102: (5.1, 0.0), 103: (0.0, 5.1), 104: (0.6, 0.0)},
    })
    cells = cell_order_flow_features(footprint)

    absorbed_bids = cells.iloc[0]
    assert absorbed_bids.absorption_price == 102.0
    assert absorbed_bids.absorption_volume == pytest.approx(10.1)
    assert absorbed_bids.absorption_signal == 1.0     # sellers absorbed -> bullish

    absorbed_offers = cells.iloc[1]
    assert absorbed_offers.absorption_signal == -1.0

    none_found = cells.iloc[2]
    assert none_found.absorption_signal == 0.0
    assert pd.isna(none_found.absorption_price)


def test_cell_features_delta_divergence_spans_bars_causally():
    from marketdata import cell_order_flow_features

    footprint = _cell_footprint({
        # bar 0: 100..101, delta 0
        "2024-01-02T00:00:00Z": {100: (1, 1), 101: (1, 1)},
        # bar 1: lower low, net buying -> bullish divergence
        "2024-01-02T00:01:00Z": {99: (2, 0.5), 100: (2, 0.5)},
        # bar 2: higher high, net selling -> bearish divergence
        "2024-01-02T00:02:00Z": {102: (0.5, 2), 103: (0.5, 2)},
        # bar 3: outside bar (new low and new high) with net buying -> +1
        "2024-01-02T00:03:00Z": {98: (1.5, 1), 105: (1.5, 1)},
    })
    cells = cell_order_flow_features(footprint)

    assert list(cells["delta_divergence"]) == [0.0, 1.0, -1.0, 1.0]
    # POC migration is the bar-over-bar change of the POC price; ties on volume
    # resolve to the lower level.
    assert list(cells["poc_migration"].iloc[1:]) == [99.0 - 100.0, 102.0 - 99.0,
                                                     98.0 - 102.0]


def test_cell_features_ignore_bars_without_traded_volume():
    from marketdata import bar_features_from_footprint, cell_order_flow_features

    footprint = _cell_footprint({
        "2024-01-02T00:00:00Z": {100: (1, 1)},
        "2024-01-02T00:01:00Z": {100: (0, 0)},          # venue row, nobody traded
    })
    cells = cell_order_flow_features(footprint)
    assert len(cells) == 1
    assert cells.index[0] == pd.Timestamp("2024-01-02T00:00:00Z")

    # The empty bar stays in the bar frame, but with no cell statistics: a level
    # nobody traded is not a level with zero buying.
    bars = bar_features_from_footprint(footprint)
    assert len(bars) == 2
    assert pd.isna(bars["poc_price"].iloc[1])
    assert bars["combined_total_qty"].iloc[1] == pytest.approx(0.0)


def test_cell_features_on_empty_or_plain_index():
    from marketdata import bar_features_from_footprint, cell_order_flow_features

    assert cell_order_flow_features(pd.DataFrame()).empty
    empty = pd.DataFrame(
        {"combined_total_qty": [], "combined_delta": [],
         "combined_buy_qty": [], "combined_sell_qty": []},
        index=pd.MultiIndex.from_arrays([[], []], names=["bar_start", "price_level"]),
    )
    assert cell_order_flow_features(empty).empty


def test_cell_features_validate_their_parameters():
    from marketdata import cell_order_flow_features

    footprint = _cell_footprint({"2024-01-02T00:00:00Z": {100: (1, 1)}})
    with pytest.raises(ValueError, match="imbalance_ratio"):
        cell_order_flow_features(footprint, imbalance_ratio=1.0)
    with pytest.raises(ValueError, match="value_area_share"):
        cell_order_flow_features(footprint, value_area_share=0.0)
    with pytest.raises(ValueError, match="absorption_quantile"):
        cell_order_flow_features(footprint, absorption_quantile=1.5)
    with pytest.raises(ValueError, match="absorption_delta_share"):
        cell_order_flow_features(footprint, absorption_delta_share=-0.1)
    with pytest.raises(ValueError, match="bar_start"):
        cell_order_flow_features(footprint.reset_index(drop=True))
    missing = footprint.drop(columns=["combined_buy_qty"])
    with pytest.raises(ValueError, match="no buy column"):
        cell_order_flow_features(missing)


def test_cell_features_do_not_read_future_bars():
    """The prefix of a run must reproduce the full run's features bar for bar."""
    from marketdata import bar_features_from_footprint, cell_order_flow_features

    bars = {
        "2024-01-02T00:00:00Z": {100: (1, 1), 101: (2, 3)},
        "2024-01-02T00:01:00Z": {99: (4, 0.5), 100: (1, 1)},
        "2024-01-02T00:02:00Z": {100: (0.5, 4), 101: (1, 1), 102: (0.5, 0.5)},
        "2024-01-02T00:03:00Z": {101: (5, 5), 102: (1, 1)},
        "2024-01-02T00:04:00Z": {100: (0.5, 0.5), 103: (6, 0.5)},
        "2024-01-02T00:05:00Z": {102: (1, 1), 99: (1, 1)},
    }
    footprint = _cell_footprint(bars)
    timestamps = [pd.Timestamp(bar) for bar in bars]
    full_cells = cell_order_flow_features(footprint, price_step=1.0)
    full_bars = bar_features_from_footprint(footprint, window=3)

    for cut in range(len(timestamps)):
        prefix = _cell_footprint(dict(list(bars.items())[:cut + 1]))
        prefix_cells = cell_order_flow_features(prefix, price_step=1.0)
        pd.testing.assert_frame_equal(
            prefix_cells, full_cells.loc[:timestamps[cut]], check_exact=True
        )
        prefix_bars = bar_features_from_footprint(prefix, window=3)
        pd.testing.assert_frame_equal(
            prefix_bars, full_bars.loc[:timestamps[cut]], check_exact=True
        )


def test_cell_features_are_skipped_without_a_price_level_index():
    from marketdata import bar_features_from_footprint

    footprint = _cell_footprint({
        "2024-01-02T00:00:00Z": {100: (1, 1), 101: (2, 2)},
    })
    bars = footprint.groupby(level="bar_start").sum()
    # A per-venue frame (bar_start, venue) has no price level to describe.
    per_venue = bars.copy()
    per_venue.index = pd.MultiIndex.from_product(
        [bars.index, ["binance"]], names=["bar_start", "venue"]
    )
    plain = bar_features_from_footprint(per_venue)
    assert "delta" in plain.columns
    assert "poc_price" not in plain.columns

    with_cells = bar_features_from_footprint(footprint, include_cells=False)
    assert "poc_price" not in with_cells.columns

    # A frame with no bar_start level at all is a mistake, not a silent skip.
    with pytest.raises(ValueError, match="bar_start"):
        bar_features_from_footprint(bars)


def test_cell_features_drive_the_imbalance_strategy():
    """A bar with stacked buy imbalance and extreme delta is bought."""
    from backtester import BacktestEngine
    from marketdata import bar_features_from_footprint
    from strategies import STRATEGIES

    footprint = _cell_footprint({
        # Two quiet bars keep the delta z-score defined; then buyers take three
        # consecutive levels three times over.
        "2024-01-02T00:00:00Z": {100: (1, 1), 101: (1, 1)},
        "2024-01-02T00:01:00Z": {100: (1, 1), 101: (1, 1)},
        "2024-01-02T00:02:00Z": {100: (1, 1), 101: (6, 0.5), 102: (6, 0.5),
                                 103: (6, 0.5)},
        "2024-01-02T00:03:00Z": {101: (1, 1), 102: (1, 1)},
        "2024-01-02T00:04:00Z": {101: (1, 1), 102: (1, 1)},
        "2024-01-02T00:05:00Z": {101: (1, 1), 102: (1, 1)},
    })
    features = bar_features_from_footprint(footprint, window=20)
    signal_bar = pd.Timestamp("2024-01-02T00:02:00Z")
    assert features.loc[signal_bar, "buy_imbalance_levels"] == 3.0
    assert features.loc[signal_bar, "imbalance_stack"] == 3.0
    assert features.loc[signal_bar, "delta_zscore"] > 1.0
    assert features.loc[signal_bar, "absorption_signal"] == 0.0

    index = pd.date_range("2024-01-02T00:00:00Z", periods=6, freq="1min")
    candles = pd.DataFrame({
        "open": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5],
        "high": [100.2, 100.3, 100.4, 100.5, 100.6, 100.7],
        "low": [99.9, 100.0, 100.1, 100.2, 100.3, 100.4],
        "close": [100.1, 100.2, 100.3, 100.4, 100.5, 100.6],
        "volume": [10.0] * 6,
    }, index=index)

    result = BacktestEngine(
        candles, STRATEGIES["Footprint imbalance + absorption (order flow)"],
        strategy_params={"window": 20, "imbalance_levels": 2, "delta_z": 1.0},
        market_features=features, footprint=footprint, cash=10_000.0,
    ).run()

    entry = next(order for order in result.orders
                 if order.tag == "footprint_imbalance_buy")
    # The signal is only known at that bar's close, so the fill is the next open.
    assert entry.filled_bar == 3
    assert entry.filled_price == pytest.approx(candles["open"].iloc[3])
    assert [p["name"] for p in result.indicators] == ["POC (footprint)"]


def test_imbalance_strategy_refuses_to_run_without_cell_features():
    from backtester import BacktestEngine
    from strategies import FootprintImbalanceStrategy

    index = pd.date_range("2024-01-02T00:00:00Z", periods=4, freq="1min")
    candles = pd.DataFrame({
        "open": [100.0] * 4, "high": [100.2] * 4, "low": [99.8] * 4,
        "close": [100.1] * 4, "volume": [1.0] * 4,
    }, index=index)
    with pytest.raises(ValueError, match="needs footprint features"):
        BacktestEngine(candles, FootprintImbalanceStrategy, cash=1_000.0).run()