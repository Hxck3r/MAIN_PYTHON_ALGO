"""Canonical public-trade schema and deterministic footprint aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

EXCHANGES = ("binance", "okx", "bybit")
SYMBOL = "BTC/USDT:USDT"
TRADE_COLUMNS = (
    "timestamp", "exchange", "symbol", "trade_id", "source_identity",
    "price", "quantity", "side", "contract_type", "quantity_unit",
)

#: units that mean "quantity is expressed in exchange contracts"
CONTRACT_UNITS = frozenset({
    "contract", "contracts", "contract_count", "contract_count_unit", "cont",
})
#: units that explicitly mean "quantity is already base-asset amount"
BASE_UNITS = frozenset({"base", "coin", "asset", "currency", "btc", "quantity"})

#: how to interpret the raw quantity field
QUANTITY_SOURCES = ("auto", "base", "contracts")


@dataclass
class TradeValidationReport:
    rows_input: int = 0
    rows_valid: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    issues: list[str] = field(default_factory=list)


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    """Convert epoch milliseconds or an offset-aware ISO value to UTC.

    Timezone-naive values are rejected instead of being silently interpreted as
    UTC: guessing a timezone can move trades into the wrong bar and corrupt the
    footprint. Callers that have a better timestamp field should fall back to it.
    """
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a timestamp: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            if not np.isfinite(float(value)):
                raise ValueError(f"Invalid trade timestamp: {value!r}")
            return pd.to_datetime(int(value), unit="ms", utc=True)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid trade timestamp: {value!r}") from exc
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("+-").isdigit():
            return _to_utc_timestamp(int(text))
    if isinstance(value, pd.Timestamp):
        stamp = value
    elif isinstance(value, datetime):
        stamp = pd.Timestamp(value)
    else:
        try:
            stamp = pd.to_datetime(value, errors="coerce")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid trade timestamp: {value!r}") from exc
    if not isinstance(stamp, pd.Timestamp) or pd.isna(stamp):
        raise ValueError(f"Invalid trade timestamp: {value!r}")
    if stamp.tzinfo is None:
        raise ValueError(
            f"Trade timestamp {value!r} has no timezone offset; refusing to guess UTC"
        )
    return stamp.tz_convert("UTC")


def trade_timestamp_candidates(trade: Mapping) -> tuple[Any, ...]:
    """Timestamp fields worth trying, most precise/authoritative first."""
    info = trade.get("info")
    if not isinstance(info, Mapping):
        info = {}
    return (
        trade.get("timestamp"),
        trade.get("datetime"),
        info.get("timestamp"),
        info.get("ts"),
        info.get("T"),
    )


def resolve_trade_timestamp(trade: Mapping) -> pd.Timestamp:
    """First usable timestamp among the known exchange fields."""
    errors: list[str] = []
    for candidate in trade_timestamp_candidates(trade):
        if candidate is None:
            continue
        try:
            return _to_utc_timestamp(candidate)
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(str(exc))
    detail = f" ({errors[-1]})" if errors else ""
    raise ValueError(f"Trade has no valid timestamp{detail}")


def _parse_flag(value: Any) -> bool | None:
    """Parse exchange boolean flags given as bools, 0/1 numbers or strings."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y", "t"):
        return True
    if text in ("false", "0", "no", "n", "f"):
        return False
    return None


def _resolve_side(trade: Mapping, info: Mapping) -> str | None:
    """Aggressor side: buy means the taker bought, sell means the taker sold."""
    raw_side = trade.get("side")
    if isinstance(raw_side, str) and raw_side.strip().lower() in ("buy", "sell"):
        return raw_side.strip().lower()
    # isBuyerMaker=True means the buyer was the maker, so the seller aggressed.
    maker = _parse_flag(info.get("isBuyerMaker", info.get("m")))
    if maker is not None:
        return "sell" if maker else "buy"
    direction = info.get("side", info.get("S"))
    if isinstance(direction, str) and direction.strip().lower() in ("buy", "sell"):
        return direction.strip().lower()
    return None


def _normalize_trade(
    trade: dict,
    exchange: str,
    symbol: str,
    contract_size: float | None,
    quantity_source: str = "auto",
) -> dict:
    if not isinstance(trade, Mapping):
        raise ValueError("Trade row must be a mapping")
    if quantity_source not in QUANTITY_SOURCES:
        raise ValueError(f"quantity_source must be one of {QUANTITY_SOURCES}")
    if contract_size is not None and not float(contract_size) > 0:
        raise ValueError("contract_size must be positive")

    info = trade.get("info") or {}
    if not isinstance(info, Mapping):
        info = {}
    timestamp = resolve_trade_timestamp(trade)

    trade_id = trade.get("id") or info.get("tradeId", info.get("id", info.get("execId")))
    source_identity = trade.get("sequence") or trade.get("seq") or trade.get("source_identity")
    if source_identity is None:
        source_identity = info.get("sequence", info.get("seq", info.get("tradeSeq")))

    amount = trade.get("amount")
    raw_unit = (
        trade.get("quantity_unit") or trade.get("amount_unit")
        or info.get("quantity_unit") or info.get("qtyUnit") or info.get("unit")
    )
    unit = str(raw_unit).strip().lower() if raw_unit is not None else ""
    venue_native_quantity = False
    if amount is None:
        # Venue-native qty/size/sz fields are contract-denominated by convention.
        amount = info.get("qty", info.get("size", info.get("sz")))
        venue_native_quantity = amount is not None
    if amount is None:
        raise ValueError("Trade has no quantity")

    if quantity_source == "base":
        is_contracts = False
    elif quantity_source == "contracts":
        is_contracts = True
    elif unit in BASE_UNITS:
        is_contracts = False
    else:
        # auto: declared contract units, venue-native quantity fields, or an
        # explicitly provided contract size all mean contract-denominated data.
        is_contracts = unit in CONTRACT_UNITS or venue_native_quantity or contract_size is not None
    if is_contracts:
        if contract_size is None:
            raise ValueError("Raw contract quantity requires contract_size")
        amount = float(amount) * float(contract_size)
    else:
        amount = float(amount)

    price = trade.get("price")
    if price is None:
        price = info.get("price", info.get("px", info.get("p")))
    side = _resolve_side(trade, info)
    market_type = trade.get("contractType") or info.get("contractType")
    if market_type and str(market_type).lower() not in ("linear", "swap", "perpetual"):
        raise ValueError(f"Unsupported non-linear contract: {market_type}")
    if price is None or side not in ("buy", "sell"):
        raise ValueError("Trade lacks price or aggressor side")
    return {
        "timestamp": timestamp, "exchange": exchange, "symbol": symbol,
        "trade_id": str(trade_id) if trade_id is not None else None,
        "source_identity": str(source_identity) if source_identity is not None else None,
        "price": float(price), "quantity": amount, "side": side,
        "contract_type": "linear_perpetual", "quantity_unit": "BTC",
    }


def normalize_trades(
    rows: Iterable[dict],
    exchange: str,
    symbol: str = SYMBOL,
    contract_size: float | None = None,
    quantity_source: str = "auto",
) -> pd.DataFrame:
    """Normalize CCXT trades to base-asset quantity.

    quantity_source controls how the raw quantity is interpreted:
      * ``auto``      -- contract units or a supplied contract_size mean
                         contracts; otherwise the amount is treated as base units;
      * ``base``      -- the amount is already expressed in the base asset;
      * ``contracts`` -- the amount is in contracts and must be multiplied by
                         contract_size.

    Direct normalization is strict: the first malformed row raises. Download
    callers use the same per-row helper while preserving the raw page.
    """
    exchange = exchange.lower()
    if exchange not in EXCHANGES:
        raise ValueError(f"Unsupported exchange: {exchange}")
    if contract_size is not None and contract_size <= 0:
        raise ValueError("contract_size must be positive")
    normalized = [
        _normalize_trade(trade, exchange, symbol, contract_size, quantity_source)
        for trade in rows
    ]
    return pd.DataFrame(normalized, columns=TRADE_COLUMNS)


def validate_trades(
    frame: pd.DataFrame, report: TradeValidationReport | None = None
) -> tuple[pd.DataFrame, TradeValidationReport]:
    report = report or TradeValidationReport()
    report.rows_input += len(frame)
    missing = set(TRADE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing canonical trade columns: {sorted(missing)}")
    out = frame.loc[:, TRADE_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    numeric = out[["price", "quantity"]].apply(pd.to_numeric, errors="coerce")
    finite_numeric = np.isfinite(numeric.to_numpy(dtype="float64"))
    invalid = (
        out["timestamp"].isna() | numeric.isna().any(axis=1)
        | ~finite_numeric.all(axis=1)
        | (numeric["price"] <= 0) | (numeric["quantity"] <= 0)
        | ~out["side"].isin(["buy", "sell"])
        | ~out["exchange"].isin(EXCHANGES)
        | (out["contract_type"] != "linear_perpetual")
        | (out["quantity_unit"] != "BTC")
    )
    report.invalid_rows_removed += int(invalid.sum())
    out = out.loc[~invalid].copy()
    out[["price", "quantity"]] = numeric.loc[~invalid].astype("float64")
    identity = out[["exchange", "symbol", "trade_id", "source_identity"]].copy()
    missing_id = identity["trade_id"].isna()
    # A stable exchange trade ID is the deduplication key on its own; the
    # source row/sequence identity only matters when the venue gives no ID.
    identity.loc[~missing_id, "source_identity"] = None
    # Content fallback removes exact retries. A source sequence/row identity is
    # included when available so legitimate identical prints remain distinct.
    if missing_id.any():
        identity.loc[missing_id, "trade_id"] = out.loc[missing_id].apply(
            lambda row: hashlib.sha256(json.dumps(
                [row.timestamp.isoformat(), row.exchange, row.symbol,
                 float(row.price), float(row.quantity), row.side,
                 row.source_identity], separators=(",", ":"),
            ).encode()).hexdigest(), axis=1,
        )
    duplicate = identity.duplicated()
    report.duplicates_removed += int(duplicate.sum())
    out = out.loc[~duplicate].sort_values(
        ["timestamp", "exchange", "trade_id", "source_identity"],
        kind="stable", na_position="last"
    ).reset_index(drop=True)
    report.rows_valid += len(out)
    return out, report


def build_footprint(
    trades: pd.DataFrame,
    timeframe: str = "1min",
    price_step: float = 1.0,
    exchange: str | None = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Aggregate aggressor volume by UTC bar and common price bucket.

    ``validate=False`` is an internal fast path for aggregate_footprint, which
    validates the whole frame once before slicing it per venue.
    """
    if price_step <= 0:
        raise ValueError("price_step must be positive")
    frame = validate_trades(trades)[0] if validate else trades
    if exchange is not None:
        frame = frame.loc[frame["exchange"] == exchange]
    if frame.empty:
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([], tz="UTC"), np.array([], dtype="float64")],
            names=["bar_start", "price_level"],
        )
        return pd.DataFrame(
            columns=["buy_qty", "sell_qty", "trade_count", "delta", "total_qty", "has_data"],
            index=empty_index,
        )
    frame = frame.copy()
    frame["bar"] = frame["timestamp"].dt.floor(timeframe)
    step = Decimal(str(price_step))
    # Decimal/integral tick arithmetic avoids float-floor artifacts such as
    # 0.3 / 0.1 landing in the 0.2 bucket.
    frame["price_level"] = frame["price"].map(
        lambda value: float((Decimal(str(value)) / step).to_integral_value(
            rounding=ROUND_FLOOR) * step)
    )
    frame["buy_qty"] = frame["quantity"].where(frame["side"] == "buy", 0.0)
    frame["sell_qty"] = frame["quantity"].where(frame["side"] == "sell", 0.0)
    keys = ["bar", "price_level"]
    result = frame.groupby(keys, sort=True).agg(
        buy_qty=("buy_qty", "sum"),
        sell_qty=("sell_qty", "sum"),
        trade_count=("trade_id", "size"),
    )
    result["delta"] = result["buy_qty"] - result["sell_qty"]
    result["total_qty"] = result["buy_qty"] + result["sell_qty"]
    result["has_data"] = True
    result.index.names = ["bar_start", "price_level"]
    return result


def aggregate_footprint(
    trades: pd.DataFrame, timeframe: str = "1min", price_step: float = 1.0
) -> pd.DataFrame:
    """Return separate raw BTC columns per exchange and combined raw sums."""
    if price_step <= 0:
        raise ValueError("price_step must be positive")
    valid = validate_trades(trades)[0]
    frames = []
    for exchange in EXCHANGES:
        part = build_footprint(
            valid, timeframe, price_step, exchange, validate=False
        )
        part = part.rename(columns={column: f"{exchange}_{column}" for column in part})
        frames.append(part)
    combined = pd.concat(frames, axis=1).sort_index()
    for exchange in EXCHANGES:
        covered = combined.get(f"{exchange}_has_data")
        if covered is None:
            covered = combined[f"{exchange}_trade_count"].notna()
        combined[f"{exchange}_covered"] = covered.fillna(False).astype(bool)
    for name in ("buy_qty", "sell_qty", "delta", "total_qty", "trade_count"):
        columns = [f"{exchange}_{name}" for exchange in EXCHANGES]
        combined[f"combined_{name}"] = combined[columns].sum(axis=1, min_count=1)
    return combined