"""Public-trade acquisition through optional CCXT with resumable pagination.

CCXT's generic ``fetch_trades(symbol, since, limit)`` paginates by timestamp.
When more trades share one millisecond than fit in a page, a timestamp-only
cursor cannot address the remainder. This module detects that condition and
reports it explicitly (or raises with ``strict_pagination=True``) instead of
silently returning partial data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
import math
import warnings
from typing import Any

import pandas as pd

from .core import (
    EXCHANGES, SYMBOL, TRADE_COLUMNS, _normalize_trade, _to_utc_timestamp,
    resolve_trade_timestamp, validate_trades,
)
from .store import TradeStore


CCXT_IDS = {"binance": "binanceusdm", "okx": "okx", "bybit": "bybit"}


@dataclass
class DownloadReport:
    """What a download actually did; inspect after a long ingest run."""

    pages: int = 0
    rows_fetched: int = 0
    rows_written: int = 0
    malformed_rows: int = 0
    contract_size: float | None = None
    incomplete_buckets: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def client_contract_size(client: Any, symbol: str) -> float | None:
    """Ask CCXT how many base units one contract holds.

    Returns ``None`` when the client cannot say (offline stub, unloaded market,
    spot instrument); the caller then treats the venue amount as base units.
    """
    getter = getattr(client, "market", None)
    if not callable(getter):
        return None
    try:
        market = getter(symbol)
    except Exception:  # noqa: BLE001 - metadata lookup is best-effort
        return None
    if not isinstance(market, Mapping):
        return None
    try:
        size = float(market.get("contractSize"))
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def _milliseconds(value: int | float | str | datetime) -> int:
    return int(_to_utc_timestamp(value).value // 1_000_000)


def _safe_text(value: Any) -> str:
    try:
        return str(value)
    except (ValueError, TypeError, OverflowError):
        return f"<{type(value).__name__}>"


def _row_identity(row: Mapping) -> str:
    """Identity used only to tell whether a re-fetched page differs."""
    info = row.get("info")
    if not isinstance(info, Mapping):
        info = {}
    trade_id = row.get("id") or info.get("tradeId", info.get("id", info.get("execId")))
    if trade_id is not None:
        return f"id:{_safe_text(trade_id)}"
    sequence = (
        row.get("sequence") or row.get("seq")
        or info.get("sequence", info.get("seq", info.get("tradeSeq")))
    )
    if sequence is not None:
        return f"seq:{_safe_text(sequence)}"
    content = [
        row.get("timestamp"), row.get("datetime"), row.get("price"),
        row.get("amount"), row.get("side"), info.get("px"), info.get("sz"),
    ]
    return "content:" + "|".join(_safe_text(item) for item in content)


def _row_ms(row: Mapping) -> int | None:
    try:
        return _milliseconds(resolve_trade_timestamp(row))
    except (TypeError, ValueError, OverflowError):
        return None


def _normalize_page(
    raw: list[Any],
    exchange: str,
    symbol: str,
    contract_size: float | None,
    quantity_source: str,
    start: int,
    end: int,
    page_cursor: int,
    report: DownloadReport,
) -> tuple[pd.DataFrame, int]:
    """Normalize a page row by row; malformed rows are skipped, never fatal."""
    rows = []
    for row in raw:
        try:
            rows.append(_normalize_trade(
                row, exchange, symbol, contract_size, quantity_source
            ))
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            report.malformed_rows += 1
            warnings.warn(
                f"Skipping malformed {exchange} trade in page at {page_cursor}: {exc}",
                RuntimeWarning, stacklevel=3,
            )
    normalized = validate_trades(pd.DataFrame(rows, columns=TRADE_COLUMNS))[0]
    stamps_ms = normalized["timestamp"].map(lambda value: value.value // 1_000_000)
    return normalized.loc[(stamps_ms >= start) & (stamps_ms < end)], len(rows)


def download_trades(
    exchange: str,
    since: int | str | datetime,
    until: int | str | datetime,
    store: TradeStore,
    symbol: str = SYMBOL,
    limit: int = 1000,
    exchange_client: Any = None,
    contract_size: float | None = None,
    quantity_source: str = "auto",
    report: DownloadReport | None = None,
    strict_pagination: bool = False,
    detect_contract_size: bool = True,
) -> int:
    """Fetch UTC ``[since, until)`` trades; archives each raw page before use.

    Returns the number of normalized rows accepted in this call. A page that is
    re-fetched at a millisecond boundary can repeat already-written trades;
    ``TradeStore.write`` deduplicates them, so the unique stored row count may
    be lower than the returned value.

    When ``contract_size`` is not given, it is read from the venue's market
    metadata (``detect_contract_size``); linear USDT perpetuals quote different
    contract sizes per venue (1 BTC on Binance/Bybit, 0.01 BTC on OKX), so the
    detected value is reported back in ``DownloadReport.contract_size``. Pass
    ``quantity_source="base"`` when the venue amount is already in BTC.

    ``strict_pagination=True`` turns the "more trades than fit in one page share
    a millisecond" condition into a ``RuntimeError`` instead of a warning.
    """
    report = report or DownloadReport()
    exchange = exchange.lower()
    if exchange not in EXCHANGES:
        raise ValueError(f"Unsupported exchange: {exchange}")
    if limit < 1:
        raise ValueError("limit must be positive")
    start, end = _milliseconds(since), _milliseconds(until)
    if start >= end:
        raise ValueError("since must be before until")

    owns_client = exchange_client is None
    if owns_client:
        try:
            import ccxt
        except ImportError as exc:
            raise RuntimeError(
                "Install market-data dependencies with "
                "pip install -e '.[marketdata]'"
            ) from exc
        client = getattr(ccxt, CCXT_IDS[exchange])({"enableRateLimit": True})
    else:
        client = exchange_client

    if quantity_source != "base" and contract_size is None and detect_contract_size:
        contract_size = client_contract_size(client, symbol)
    report.contract_size = contract_size
    # Quantities already written for this venue must keep the same meaning: a
    # contract size that changed between runs means the store mixes two units.
    previous_size = store.last_contract_size(exchange, symbol)
    if (
        contract_size is not None and previous_size is not None
        and not math.isclose(contract_size, previous_size, rel_tol=1e-9)
    ):
        message = (
            f"{exchange} {symbol}: contract_size changed from {previous_size:g} to "
            f"{contract_size:g}; previously stored quantities are in different units"
        )
        report.issues.append(message)
        store.note_issue(exchange, symbol, message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    stored_cursor = store.get_checkpoint(exchange, symbol)
    cursor = max(start, stored_cursor) if stored_cursor is not None else start
    total = 0
    previous_bucket_key: tuple[str, ...] | None = None
    bucket_pages = 0
    try:
        while cursor < end:
            page_cursor = cursor
            raw = client.fetch_trades(symbol, since=page_cursor, limit=limit)
            if not raw:
                break
            raw = list(raw)
            report.pages += 1
            report.rows_fetched += len(raw)
            # Archive raw first: the page stays auditable even if normalization
            # or persistence fails afterwards.
            store.save_raw(exchange, symbol, page_cursor, raw)

            normalized, _ = _normalize_page(
                raw, exchange, symbol, contract_size, quantity_source,
                start, end, page_cursor, report,
            )

            stamps = [
                ms for row in raw if isinstance(row, Mapping)
                for ms in [_row_ms(row)] if ms is not None
            ]
            if not stamps:
                message = (
                    f"{exchange} {symbol}: page at {page_cursor} has no usable timestamps"
                )
                report.issues.append(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                break
            latest = max(stamps)
            if latest < page_cursor:
                raise RuntimeError(
                    "Exchange pagination returned a timestamp before the cursor"
                )

            # A full page that starts and ends in the cursor millisecond cannot be
            # split by a timestamp cursor. Detect a repeated page and stop before
            # re-writing duplicates.
            full_same_bucket = len(raw) >= limit and latest == page_cursor
            bucket_key = None
            if full_same_bucket:
                bucket_pages += 1
                bucket_key = tuple(sorted(
                    _row_identity(row) for row in raw
                    if isinstance(row, Mapping) and _row_ms(row) == latest
                ))
                if bucket_key == previous_bucket_key or bucket_pages > 500:
                    message = (
                        f"{exchange} {symbol}: the exchange repeated the same page at "
                        f"timestamp {pd.Timestamp(latest, unit='ms', tz='UTC')}; "
                        "trades sharing that millisecond may be incomplete"
                    )
                    if strict_pagination:
                        raise RuntimeError(
                            "Incomplete trade retrieval: more trades share one "
                            f"millisecond than page size {limit}, and the exchange "
                            "offers no timestamp cursor to address the remainder. "
                            "Increase `limit` or use an exchange archive/ID-cursor "
                            f"loader. Timestamp: "
                            f"{pd.Timestamp(latest, unit='ms', tz='UTC')}"
                        )
                    report.incomplete_buckets.append(latest)
                    report.issues.append(message)
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    store.note_issue(exchange, symbol, message)
                    cursor = latest + 1
                    store.set_checkpoint(exchange, symbol, cursor)
                    break
            if not normalized.empty:
                store.write(
                    normalized,
                    {"exchange": exchange, "symbol": symbol, "time_unit": "UTC"},
                )
                total += len(normalized)
                report.rows_written += len(normalized)

            # No requested trade can be at/after `until`: the range is covered.
            if latest >= end:
                break
            if len(raw) < limit:
                # A short page means the exchange has nothing more at/after the
                # last timestamp; this also completes a tied boundary bucket.
                cursor = latest + 1
                store.set_checkpoint(exchange, symbol, cursor)
                break
            if latest > page_cursor:
                # Full page with later timestamps: re-request the boundary
                # millisecond inclusively so ties are not skipped.
                cursor = latest
                previous_bucket_key = None
                bucket_pages = 0
                store.set_checkpoint(exchange, symbol, cursor)
                continue
            previous_bucket_key = bucket_key
            # Stay on the same cursor to fetch the rest of this millisecond.
    finally:
        if owns_client:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - closing must not mask errors
                    pass
    if report.malformed_rows:
        warnings.warn(
            f"Skipped {report.malformed_rows} malformed {exchange} trade(s) "
            f"while downloading {symbol}",
            RuntimeWarning, stacklevel=2,
        )
    store.record_download(exchange, symbol, report)
    return total


__all__ = ["DownloadReport", "download_trades"]