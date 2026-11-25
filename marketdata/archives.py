"""Bulk exchange-archive downloader: trade ticks and candles.

``sources.download_trades`` pulls public trades through CCXT page by page, which is
fine for a recent window but cannot deliver multi-year tick history quickly and
cannot page past a timestamp cursor.  This module reads the venues' own bulk
archives instead: free, complete, and orders of magnitude faster.

  binance  https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip
  binance  https://data.binance.vision/data/futures/um/daily/trades/BTCUSDT/BTCUSDT-trades-2024-01-01.zip
  binance  https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip
  bybit    https://public.bybit.com/trading/BTCUSDT/BTCUSDT2024-01-01.csv.gz
  okx      https://www.okx.com/cdn/okex/traderecords/trades/daily/20240101/BTC-USDT-SWAP-trades-2024-01-01.zip

Trades are normalized into the canonical trade schema and written through
``store.TradeStore``, so the rest of the pipeline works on the result unchanged::

    from marketdata.archives import download_archives
    download_archives(["binance"], "2024-01-01", "2024-02-01", data_dir="marketdata_store")

Source semantics, verified against the live archives
----------------------------------------------------
=================  ==================  ============================  ===============
venue              size is             side is                       coverage
=================  ==================  ============================  ===============
binance um trades  base asset (BTC)    ``is_buyer_maker`` inverted   2019-09-08 ..
binance um aggTr.  base asset (BTC)    ``is_buyer_maker`` inverted   2019-12-31 ..
bybit (linear)     base asset (BTC)    ``side`` (already taker)      2020-03-25 ..
okx (SWAP)         **contracts**       ``side`` (already taker)      ~2021-10 ..
=================  ==================  ============================  ===============

Caveats worth knowing before trusting a footprint built from this
-----------------------------------------------------------------
* **OKX archive days are UTC+8, not UTC.**  The file dated X holds UTC
  ``[X-1 16:00, X 16:00)``, so a UTC day D is split across the files D and D+1.
  :func:`archive_days` therefore fetches one extra file past ``until`` for OKX and
  :func:`ingest_file` trims by timestamp; without that the last 8 hours of every
  UTC day would be lost silently.  Binance and Bybit archives are UTC calendar days.
* Older Binance Vision CSVs carry no header row, recent ones do; the reader sniffs.
* Binance daily archives are published up to T-1.
* ``aggTrades`` is the better Binance dataset for footprint work: same aggressor
  side and volume as ``trades``, far fewer rows and about a third of the bytes.
  The two are different granularities of the same flow, so mixing them for one
  venue/symbol/day would double-count volume; the state file blocks that.
* Candles come from Binance only -- Bybit and OKX publish no bulk kline archive.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .core import SYMBOL, TRADE_COLUMNS, aggregate_footprint
from .store import TradeStore

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EXCHANGES = ("binance", "bybit", "okx")
DATASETS = ("aggTrades", "trades")
DEFAULT_DATASET = "aggTrades"
DEFAULT_DATA_DIR = "marketdata_store"
DEFAULT_INTERVAL = "1m"
#: intervals offered by Binance Vision; ``1M`` is its spelling for a month
KLINE_INTERVALS = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")

BINANCE_ROOT = "https://data.binance.vision/data/futures"
BYBIT_ROOT = "https://public.bybit.com/trading"
OKX_ROOT = "https://www.okx.com/cdn/okex/traderecords/trades/daily"
OKX_INSTRUMENTS = "https://www.okx.com/api/v5/public/instruments"

USER_AGENT = "main-python-algo/0.1 (bulk archive fetcher)"
UTC = timezone.utc

#: column names used when an archive file has no header row
BINANCE_TRADES_COLUMNS = ("id", "price", "qty", "quote_qty", "time", "is_buyer_maker")
BINANCE_AGGTRADES_COLUMNS = (
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
)

_NORMALIZE = re.compile(r"[^a-z0-9]")

#: alias sets, keyed on a column name reduced by :func:`_key`
_ALIASES: dict[str, frozenset[str]] = {
    "id": frozenset({"id", "a", "aggtradeid", "tradeid", "trdmatchid", "execid"}),
    "ts": frozenset({"time", "transacttime", "createdtime", "timestamp", "ts"}),
    "price": frozenset({"price", "p"}),
    "qty": frozenset({"qty", "quantity", "q", "size", "volume", "amount"}),
    "maker": frozenset({"isbuyermaker", "m"}),
    "side": frozenset({"side", "s", "direction"}),
    "symbol": frozenset({"symbol", "instrumentname", "instrumentid", "instid"}),
    "open_time": frozenset({"opentime", "timestamp", "ts", "time"}),
    "open": frozenset({"open", "o"}),
    "high": frozenset({"high", "h"}),
    "low": frozenset({"low", "l"}),
    "close": frozenset({"close", "c"}),
    "volume": frozenset({"volume", "v", "qty"}),
    "taker_buy": frozenset({"takerbuyvolume", "takerbuybaseassetvolume", "takerbuy"}),
}


class ArchiveCancelled(Exception):
    """Raised internally when ``should_stop`` reports a user cancellation."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _key(name: str) -> str:
    return _NORMALIZE.sub("", str(name).strip().lstrip("﻿").lower())


def _pick(columns: Sequence[str], field_name: str, *, required: bool = True) -> str | None:
    """Resolve a logical field to the actual column name in this file."""
    wanted = _ALIASES[field_name]
    for column in columns:
        if _key(column) in wanted:
            return column
    if required:
        raise ValueError(
            f"None of the expected {field_name!r} columns {sorted(wanted)} present in "
            f"{list(columns)}"
        )
    return None


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype("str").str.strip().str.lower().isin(("true", "1", "t", "y", "yes"))


def _to_utc(series: pd.Series, unit: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if unit == "s":
        numeric = (numeric * 1000.0).round()
    return pd.to_datetime(numeric.astype("Int64"), unit="ms", utc=True, errors="coerce")


@contextlib.contextmanager
def _open_text(path: Path) -> Iterator[io.TextIOWrapper]:
    """Stream a .zip / .csv.gz / .csv payload as text without holding it in RAM."""
    suffix = path.suffix.lower()
    archive: zipfile.ZipFile | None = None
    if suffix == ".zip":
        archive = zipfile.ZipFile(path)
        member = next(
            (name for name in archive.namelist() if name.lower().endswith((".csv", ".txt"))),
            None,
        )
        if member is None:
            archive.close()
            raise ValueError(f"No CSV member inside {path.name}")
        stream = io.TextIOWrapper(archive.open(member), encoding="utf-8", newline="")
    elif suffix == ".gz":
        stream = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        stream = path.open("rt", encoding="utf-8", newline="")
    try:
        yield stream  # type: ignore[misc]
    finally:
        with contextlib.suppress(Exception):
            stream.close()
        if archive is not None:
            with contextlib.suppress(Exception):
                archive.close()


def _has_header(path: Path) -> bool:
    """Older Binance Vision CSVs have no header row; recent ones do. Sniff it."""
    with _open_text(path) as stream:
        line = stream.readline()
    token = line.split(",", 1)[0].strip().strip('"')
    return bool(token) and not token.lstrip("+-").isdigit()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, *, retries: int, timeout: float) -> bool:
    """Fetch ``url`` into ``dest``.  Returns False when the file is not published."""
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response, dest.open("wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            return True
        except HTTPError as exc:
            if exc.code == 404:
                return False
            if attempt < retries and exc.code in (403, 429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (URLError, TimeoutError, OSError):
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return False


def _get_text(url: str, *, retries: int = 3, timeout: float = 30.0) -> str:
    delay = 1.0
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"GET {url} failed: {last}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Instrument naming
# --------------------------------------------------------------------------- #
def parse_symbol(symbol: str) -> tuple[str, str]:
    """``BTC/USDT:USDT`` -> ``("BTC", "USDT")``."""
    body = symbol.split(":", 1)[0]
    base, _, quote = body.partition("/")
    if not base or not quote:
        raise ValueError(
            f"Symbol must look like BASE/QUOTE:SETTLE (e.g. BTC/USDT:USDT), got {symbol!r}"
        )
    return base.upper(), quote.upper()


def native_instrument(exchange: str, symbol: str, *, override: str | None = None) -> str:
    """Venue-native id for a CCXT-style symbol."""
    if override:
        return override
    base, quote = parse_symbol(symbol)
    if exchange == "okx":
        return f"{base}-{quote}-SWAP"
    return f"{base}{quote}"  # binance USD-M perpetual / bybit linear perpetual


def okx_contract_size(
    instrument: str, *, override: float | None = None, timeout: float = 30.0
) -> float:
    """Base units per contract.  OKX publishes trade size in contracts, not coins."""
    if override is not None:
        if override <= 0:
            raise ValueError("contract size must be positive")
        return float(override)
    url = f"{OKX_INSTRUMENTS}?{urlencode({'instType': 'SWAP', 'instId': instrument})}"
    try:
        payload = json.loads(_get_text(url, timeout=timeout))
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not read OKX instrument metadata for {instrument}: {exc}"
        ) from exc
    rows = payload.get("data") or []
    if payload.get("code") != "0" or not rows:
        raise RuntimeError(
            f"OKX returned no instrument metadata for {instrument!r}: {payload.get('msg')!r}"
        )
    row = rows[0]
    size = float(row["ctVal"]) * float(row.get("ctMult") or 1)
    if size <= 0:
        raise RuntimeError(f"OKX reported a non-positive contract value for {instrument}")
    return size


# --------------------------------------------------------------------------- #
# Download units
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Unit:
    """One archive file to fetch."""

    url: str
    label: str  # the UTC day (or month) this file covers
    checksum_url: str | None = None
    kind: str = "daily"  # daily | monthly


@dataclass
class ArchiveReport:
    """Shaped like ``sources.DownloadReport`` so ``algo-trades status`` shows it."""

    pages: int = 0
    rows_fetched: int = 0
    rows_written: int = 0
    malformed_rows: int = 0
    contract_size: float | None = None
    incomplete_buckets: list[int] = field(default_factory=list)


@dataclass
class ArchiveProgress:
    """One progress event, always delivered on the calling thread."""

    #: plan | file | missing | error | exchange_done | footprint | stage
    event: str
    exchange: str = ""
    label: str = ""
    done: int = 0
    total: int = 0
    rows: int = 0
    message: str = ""

    @property
    def fraction(self) -> float:
        return 0.0 if self.total <= 0 else min(1.0, self.done / self.total)


@dataclass
class ArchiveResult:
    """What a download run did; inspect after a long ingest."""

    rows: dict[str, int] = field(default_factory=dict)
    files: dict[str, int] = field(default_factory=dict)
    skipped: int = 0
    planned: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    cancelled: bool = False
    dry_run: bool = False
    store: str = ""

    @property
    def total_rows(self) -> int:
        return sum(self.rows.values())

    @property
    def ok(self) -> bool:
        return not self.failures and not self.cancelled


def archive_days(exchange: str, days: Sequence[str], until_day: str) -> list[str]:
    """Calendar days whose archive file must be fetched for the requested range.

    Binance and Bybit publish UTC calendar days, so the file set is the range
    itself.  OKX names files by a **UTC+8** day: the file dated X holds UTC
    ``[X-1 16:00, X 16:00)``, so UTC day D is split across files D and D+1 and one
    extra file past ``until`` is needed to cover the tail (16:00..24:00 UTC).
    Timestamp filtering in :func:`ingest_file` trims the overlap, so the extra
    file adds no duplicate volume.
    """
    if exchange == "okx":
        return [*days, until_day]
    return list(days)


def build_units(
    exchange: str, instrument: str, dataset: str, days: Sequence[str], *, market: str = "um"
) -> list[Unit]:
    """Trade-archive files for ``days``."""
    units: list[Unit] = []
    for day in days:
        if exchange == "binance":
            url = (
                f"{BINANCE_ROOT}/{market}/daily/{dataset}/{instrument}/"
                f"{instrument}-{dataset}-{day}.zip"
            )
            units.append(Unit(url, day, f"{url}.CHECKSUM"))
        elif exchange == "bybit":
            units.append(Unit(f"{BYBIT_ROOT}/{instrument}/{instrument}{day}.csv.gz", day))
        elif exchange == "okx":
            url = f"{OKX_ROOT}/{day.replace('-', '')}/{instrument}-trades-{day}.zip"
            units.append(Unit(url, day))
        else:
            raise ValueError(f"Unsupported exchange: {exchange!r}")
    return units


def build_kline_units(
    instrument: str, interval: str, days: Sequence[str], *, market: str = "um",
    monthly: bool = True,
) -> list[Unit]:
    """Candle-archive files: one monthly file per calendar month, or daily files."""
    root = f"{BINANCE_ROOT}/{market}"
    if not monthly:
        return [
            Unit(
                f"{root}/daily/klines/{instrument}/{interval}/"
                f"{instrument}-{interval}-{day}.zip",
                day, None, "daily",
            )
            for day in days
        ]
    months: dict[str, list[str]] = {}
    for day in days:
        months.setdefault(day[:7], []).append(day)
    units: list[Unit] = []
    for month in sorted(months):
        url = f"{root}/monthly/klines/{instrument}/{interval}/{instrument}-{interval}-{month}.zip"
        units.append(Unit(url, month, f"{url}.CHECKSUM", "monthly"))
    return units


def fetch_unit(
    unit: Unit, tmpdir: Path, *, retries: int, timeout: float, checksum: bool = True
) -> Path | None:
    """Download one archive file; ``None`` means "not published yet"."""
    dest = tmpdir / Path(unit.url).name
    if not _download(unit.url, dest, retries=retries, timeout=timeout):
        return None
    if checksum and unit.checksum_url:
        try:
            expected = _get_text(unit.checksum_url, timeout=timeout).split()[0].strip()
        except RuntimeError:
            expected = ""  # not every file ships a checksum
        if expected:
            actual = _sha256(dest)
            if actual != expected:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Checksum mismatch for {dest.name}: expected {expected}, got {actual}"
                )
    return dest


def _pending_units(
    pool: futures.ThreadPoolExecutor,
    units: Sequence[Unit],
    *,
    window: int,
    tmpdir: Path,
    retries: int,
    timeout: float,
    checksum: bool,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[tuple[Unit, Path | None]]:
    """Download with bounded prefetch; each temp file is handed over then deleted.

    Keeps at most ``window`` files on disk at once, so a year-long range never
    fills the disk with archives that have not been parsed yet.
    """
    queue = iter(units)
    pending: dict[futures.Future[Path | None], Unit] = {}

    def stopping() -> bool:
        return should_stop is not None and should_stop()

    def submit_next() -> bool:
        if stopping():
            return False
        unit = next(queue, None)
        if unit is None:
            return False
        pending[pool.submit(
            fetch_unit, unit, tmpdir, retries=retries, timeout=timeout, checksum=checksum
        )] = unit
        return True

    for _ in range(max(1, window)):
        if not submit_next():
            break
    try:
        while pending:
            done, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            for future in done:
                unit = pending.pop(future)
                try:
                    yield unit, future.result()
                except Exception as exc:  # noqa: BLE001 - one file must not stop the run
                    yield unit, exc  # type: ignore[misc]
            while len(pending) < max(1, window):
                if not submit_next():
                    break
            if stopping() and not pending:
                break
        if stopping():
            raise ArchiveCancelled
    finally:
        for future in pending:
            future.cancel()


# --------------------------------------------------------------------------- #
# Trade readers: archive file -> canonical frames
# --------------------------------------------------------------------------- #
def _canonical(
    exchange: str,
    symbol: str,
    timestamp: pd.Series,
    trade_id: pd.Series,
    price: pd.Series,
    quantity: pd.Series,
    side: pd.Series,
    *,
    id_prefix: str = "",
) -> pd.DataFrame:
    identifiers = trade_id.astype("str")
    frame = pd.DataFrame({
        "timestamp": timestamp,
        "exchange": exchange,
        "symbol": symbol,
        "trade_id": (id_prefix + identifiers) if id_prefix else identifiers,
        "source_identity": None,
        "price": pd.to_numeric(price, errors="coerce"),
        "quantity": pd.to_numeric(quantity, errors="coerce"),
        "side": side.astype("str").str.strip().str.lower(),
        "contract_type": "linear_perpetual",
        "quantity_unit": "BTC",
    })
    return frame.loc[:, list(TRADE_COLUMNS)]


def _chunks(path: Path, *, chunk_rows: int, names: Sequence[str] | None = None):
    """Yield raw string chunks; sniffing a header row is done by the caller's names."""
    header = _has_header(path)
    if not header and names is None:
        raise ValueError(f"{path.name} has no header row and no column names were given")
    with _open_text(path) as stream:
        yield from pd.read_csv(
            stream,
            header=0 if header else None,
            names=None if header else list(names or ()),
            chunksize=chunk_rows,
            dtype="str",
            keep_default_na=False,
        )


def read_binance_trades(
    path: Path, *, dataset: str, chunk_rows: int, symbol: str
) -> Iterator[pd.DataFrame]:
    names = BINANCE_TRADES_COLUMNS if dataset == "trades" else BINANCE_AGGTRADES_COLUMNS
    for chunk in _chunks(path, chunk_rows=chunk_rows, names=names):
        columns = list(chunk.columns)
        # is_buyer_maker=True means the buyer was the maker, so the taker sold.
        side = _to_bool(chunk[_pick(columns, "maker")]).map({True: "sell", False: "buy"})
        yield _canonical(
            "binance", symbol,
            _to_utc(chunk[_pick(columns, "ts")], "ms"),
            chunk[_pick(columns, "id")],
            chunk[_pick(columns, "price")],
            chunk[_pick(columns, "qty")],
            side,
            id_prefix="agg:" if dataset == "aggTrades" else "",
        ).reset_index(drop=True)


def read_bybit_trades(path: Path, *, chunk_rows: int, symbol: str) -> Iterator[pd.DataFrame]:
    for chunk in _chunks(path, chunk_rows=chunk_rows):
        columns = list(chunk.columns)
        yield _canonical(
            "bybit", symbol,
            _to_utc(chunk[_pick(columns, "ts")], "s"),
            chunk[_pick(columns, "id")],
            chunk[_pick(columns, "price")],
            chunk[_pick(columns, "qty")],
            chunk[_pick(columns, "side")],
        ).reset_index(drop=True)


def read_okx_trades(
    path: Path, *, chunk_rows: int, symbol: str, contract_size: float
) -> Iterator[pd.DataFrame]:
    for chunk in _chunks(path, chunk_rows=chunk_rows):
        columns = list(chunk.columns)
        size = pd.to_numeric(chunk[_pick(columns, "qty")], errors="coerce")
        yield _canonical(
            "okx", symbol,
            _to_utc(chunk[_pick(columns, "ts")], "ms"),
            chunk[_pick(columns, "id")],
            chunk[_pick(columns, "price")],
            size * float(contract_size),  # contracts -> BTC
            chunk[_pick(columns, "side")],
        ).reset_index(drop=True)


def trade_reader(
    exchange: str, *, dataset: str, chunk_rows: int, symbol: str, contract_size: float
) -> Callable[[Path], Iterator[pd.DataFrame]]:
    if exchange == "binance":
        return lambda path: read_binance_trades(
            path, dataset=dataset, chunk_rows=chunk_rows, symbol=symbol
        )
    if exchange == "bybit":
        return lambda path: read_bybit_trades(path, chunk_rows=chunk_rows, symbol=symbol)
    if exchange == "okx":
        return lambda path: read_okx_trades(
            path, chunk_rows=chunk_rows, symbol=symbol, contract_size=contract_size
        )
    raise ValueError(f"Unsupported exchange: {exchange!r}")


def read_binance_klines(path: Path, *, chunk_rows: int) -> Iterator[pd.DataFrame]:
    """Candle archive -> OHLCV chunk with a UTC ``DatetimeIndex``.

    Binance's ``taker_buy_volume`` is kept as ``taker_buy_volume``: it is bar-level
    order flow (aggressive buying) available without downloading any trades.
    """
    names = (
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    )
    for chunk in _chunks(path, chunk_rows=chunk_rows, names=names):
        columns = list(chunk.columns)
        frame = pd.DataFrame({
            "open": pd.to_numeric(chunk[_pick(columns, "open")], errors="coerce"),
            "high": pd.to_numeric(chunk[_pick(columns, "high")], errors="coerce"),
            "low": pd.to_numeric(chunk[_pick(columns, "low")], errors="coerce"),
            "close": pd.to_numeric(chunk[_pick(columns, "close")], errors="coerce"),
            "volume": pd.to_numeric(chunk[_pick(columns, "volume")], errors="coerce"),
        })
        taker = _pick(columns, "taker_buy", required=False)
        if taker is not None:
            frame["taker_buy_volume"] = pd.to_numeric(chunk[taker], errors="coerce")
        index = _to_utc(chunk[_pick(columns, "open_time")], "ms")
        frame.index = pd.DatetimeIndex(index)
        frame.index.name = "timestamp"
        yield frame.dropna(subset=["open", "high", "low", "close"])


# --------------------------------------------------------------------------- #
# Ingest into the trade store
# --------------------------------------------------------------------------- #
def _encode_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def partition_path(store: TradeStore, exchange: str, symbol: str, day: str) -> Path:
    return canonical_partition_path(store.root, exchange, symbol, day)


def canonical_partition_path(
    root: str | Path, exchange: str, symbol: str, day: str
) -> Path:
    """Where a day's trades land, without touching the filesystem.

    Used while planning, so that a dry run creates no directories at all.
    """
    return (
        Path(root) / "canonical"
        / f"exchange={exchange}"
        / f"symbol={_encode_symbol(symbol)}"
        / f"date={day}"
        / "trades.parquet"
    )


def _load_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def ingest_file(
    path: Path,
    *,
    exchange: str,
    symbol: str,
    dataset: str,
    store: TradeStore,
    since: pd.Timestamp,
    until: pd.Timestamp,
    chunk_rows: int = 1_000_000,
    contract_size: float = 1.0,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """Stream one trade archive into the store.  Returns rows written."""
    reader = trade_reader(
        exchange, dataset=dataset, chunk_rows=chunk_rows, symbol=symbol,
        contract_size=contract_size,
    )
    written = 0
    for frame in reader(path):
        if should_stop is not None and should_stop():
            raise ArchiveCancelled
        # The archive is a whole day; trim to the requested window and drop rows
        # the store schema would reject anyway (counted, not silently kept).
        frame = frame.dropna(subset=["timestamp", "price", "quantity"])
        frame = frame.loc[
            (frame["timestamp"] >= since) & (frame["timestamp"] < until)
            & (frame["price"] > 0) & (frame["quantity"] > 0)
            & frame["side"].isin(["buy", "sell"])
        ]
        if frame.empty:
            continue
        store.write(
            frame,
            {
                "exchange": exchange,
                "symbol": symbol,
                "time_unit": "UTC",
                "source": "exchange_archive",
                "archive_dataset": dataset,
            },
        )
        written += len(frame)
    return written


def ingest_kline_file(
    path: Path, *, since: pd.Timestamp, until: pd.Timestamp, chunk_rows: int = 200_000
) -> pd.DataFrame:
    """Read one candle archive into a UTC-indexed OHLCV frame trimmed to the window."""
    parts = [
        chunk for chunk in read_binance_klines(path, chunk_rows=chunk_rows)
        if not chunk.empty
    ]
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.concat(parts).sort_index()
    return frame.loc[(frame.index >= since) & (frame.index < until)]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _timestamps(since: str, until: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(parse_day(since), tz="UTC"),
        pd.Timestamp(parse_day(until), tz="UTC"),
    )


def _require_btc_store_symbol(symbol: str) -> None:
    base, _quote = parse_symbol(symbol)
    if base != "BTC":
        raise ValueError(
            f"The trade store schema is BTC-denominated (quantity_unit must be 'BTC'), so "
            f"{symbol!r} cannot be stored as-is. Use a BTC instrument, or relax the BTC "
            "check in marketdata/core.py:validate_trades first."
        )


def download_archives(
    exchanges: Sequence[str],
    since: str,
    until: str,
    *,
    symbol: str = SYMBOL,
    instrument: str | None = None,
    dataset: str = DEFAULT_DATASET,
    market: str = "um",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    contract_size: float | None = None,
    workers: int = 4,
    chunk_rows: int = 1_000_000,
    retries: int = 4,
    timeout: float = 120.0,
    checksum: bool = True,
    force: bool = False,
    dry_run: bool = False,
    progress: Callable[[ArchiveProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ArchiveResult:
    """Download trade archives into the canonical store for every venue.

    Days already ingested are skipped (``force=True`` redoes them).  A day whose
    archive is not published yet is reported in :attr:`ArchiveResult.missing`.
    Cancellation via ``should_stop`` leaves every completed day recorded, so the
    next call resumes instead of restarting.
    """
    unknown = [name for name in exchanges if name not in EXCHANGES]
    if unknown:
        raise ValueError(f"Unsupported exchange(s): {unknown}; expected {list(EXCHANGES)}")
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    _require_btc_store_symbol(symbol)

    since_ts, until_ts = _timestamps(since, until)
    if since_ts >= until_ts:
        raise ValueError("since must be strictly before until")
    days = day_range(parse_day(since), parse_day(until))
    until_day = parse_day(until)

    def emit(**kwargs: Any) -> None:
        if progress is not None:
            progress(ArchiveProgress(**kwargs))

    def stopping() -> bool:
        return should_stop is not None and should_stop()

    store: TradeStore | None = None
    state_path = Path(data_dir) / "archives_state.json"
    state = _load_state(state_path)
    result = ArchiveResult(store=str(data_dir), dry_run=dry_run)

    def build_job(exchange: str) -> tuple[str, str, float, list[Unit], list[Unit]]:
        native = native_instrument(exchange, symbol, override=instrument)
        # Only Binance publishes two trade streams.  Elsewhere the flag means
        # nothing, so record the single stream under one name: otherwise a later
        # run with a different ``dataset`` would trip the double-count guard.
        effective = dataset if exchange == "binance" else "trades"
        size = 1.0
        if exchange == "okx":
            size = okx_contract_size(native, override=contract_size, timeout=timeout)
        elif contract_size is not None:
            size = float(contract_size)
        units = build_units(
            exchange, native, effective, archive_days(exchange, days, until_day),
            market=market,
        )
        return native, effective, size, units, []

    jobs: dict[str, dict[str, Any]] = {}
    for exchange in exchanges:
        native, effective, size, units, _ = build_job(exchange)
        todo: list[Unit] = []
        for unit in units:
            record = state.get(f"{exchange}:{symbol}:{unit.label}")
            if not isinstance(record, dict) or not record.get("complete"):
                todo.append(unit)
                continue
            stored = record.get("dataset")
            # Only Binance offers two granularities of the same flow, so only
            # there can a mismatched dataset double-count volume.
            if exchange == "binance" and stored not in (None, effective):
                message = (
                    f"{exchange} {unit.label} is already stored from {stored!r}; adding "
                    f"{effective!r} would double-count volume. Use a separate data_dir, "
                    "or force=True to redo the day."
                )
                result.failures.append(message)
                emit(event="error", exchange=exchange, label=unit.label, message=message)
                continue
            path = canonical_partition_path(data_dir, exchange, symbol, unit.label)
            if force or not path.is_file():
                todo.append(unit)  # state says done but the partition is missing
            else:
                result.skipped += 1
        jobs[exchange] = {
            "instrument": native, "dataset": effective, "contract_size": size,
            "units": units, "todo": todo,
        }
        emit(
            event="plan", exchange=exchange, total=len(units), done=len(units) - len(todo),
            rows=0,
        )
        result.planned.extend(f"{exchange} {unit.label} {unit.url}" for unit in todo)

    if dry_run:
        return result
    if stopping():
        raise ArchiveCancelled
    if not any(job["todo"] for job in jobs.values()):
        return result

    # Created only now: planning and a dry run must leave the filesystem alone.
    store = TradeStore(data_dir)
    tmpdir = Path(tempfile.mkdtemp(prefix="algo_archives_"))
    try:
        for exchange in exchanges:
            job = jobs[exchange]
            todo: list[Unit] = job["todo"]
            if not todo:
                continue
            report = ArchiveReport(
                contract_size=job["contract_size"] if exchange == "okx" else None
            )
            written_total = 0
            units_done = 0
            try:
                with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    for unit, payload in _pending_units(
                        pool, todo, window=workers * 2, tmpdir=tmpdir, retries=retries,
                        timeout=timeout, checksum=checksum, should_stop=should_stop,
                    ):
                        if stopping():
                            # Prefetched files are dropped rather than parsed: a
                            # stop should stop, and the day stays unmarked so the
                            # next run resumes from it.
                            raise ArchiveCancelled
                        if isinstance(payload, Exception):
                            message = f"{exchange} {unit.label}: {payload}"
                            result.failures.append(message)
                            emit(event="error", exchange=exchange, label=unit.label,
                                 message=str(payload), done=units_done, total=len(todo),
                                 rows=written_total)
                            continue
                        if payload is None:
                            label = f"{exchange} {unit.label}"
                            result.missing.append(label)
                            emit(event="missing", exchange=exchange, label=unit.label,
                                 done=units_done, total=len(todo), rows=written_total)
                            continue
                        try:
                            written = ingest_file(
                                payload, exchange=exchange, symbol=symbol,
                                dataset=job["dataset"], store=store, since=since_ts,
                                until=until_ts, chunk_rows=chunk_rows,
                                contract_size=job["contract_size"],
                                should_stop=should_stop,
                            )
                        finally:
                            payload.unlink(missing_ok=True)
                        written_total += written
                        units_done += 1
                        report.pages += 1
                        report.rows_fetched += written
                        report.rows_written += written
                        # Only a fully ingested file is marked complete, so an
                        # interrupted run resumes instead of skipping a day.
                        state[f"{exchange}:{symbol}:{unit.label}"] = {
                            "dataset": job["dataset"],
                            "instrument": job["instrument"],
                            "rows": written,
                            "complete": True,
                            "at": datetime.now(UTC).isoformat(timespec="seconds"),
                        }
                        _save_state(state_path, state)
                        emit(event="file", exchange=exchange, label=unit.label,
                             done=units_done, total=len(todo), rows=written_total)
            except ArchiveCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the other venues going
                message = f"{exchange}: {type(exc).__name__}: {exc}"
                result.failures.append(message)
                emit(event="error", exchange=exchange, message=str(exc))
            finally:
                # Reported even when the run was cancelled: a partial ingest is
                # real data in the store, and hiding it would make the summary lie.
                result.rows[exchange] = written_total
                result.files[exchange] = units_done
                # Keep the archived ingest visible in the same manifest that
                # `algo-trades status` reads, so both download paths are auditable.
                if report.pages:
                    store.record_download(exchange, symbol, report)
                emit(event="exchange_done", exchange=exchange, done=units_done,
                     total=len(todo), rows=written_total)
    except ArchiveCancelled:
        result.cancelled = True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return result


def download_candles(
    since: str,
    until: str,
    *,
    symbol: str = SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    market: str = "um",
    instrument: str | None = None,
    workers: int = 4,
    retries: int = 4,
    timeout: float = 120.0,
    checksum: bool = True,
    monthly: bool = True,
    progress: Callable[[ArchiveProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Fetch Binance candles for ``[since, until)`` as a canonical OHLCV frame.

    Axes come from Binance only: Bybit and OKX publish no bulk kline archive.
    Monthly files are used when available (one request per month); a month whose
    monthly file is not published yet is filled from daily files, so a range
    ending today still works.  The returned frame carries ``taker_buy_volume``
    in addition to the five canonical columns.
    """
    if interval not in KLINE_INTERVALS:
        raise ValueError(f"interval must be one of {list(KLINE_INTERVALS)}")
    since_ts, until_ts = _timestamps(since, until)
    if since_ts >= until_ts:
        raise ValueError("since must be strictly before until")
    days = day_range(parse_day(since), parse_day(until))
    native = native_instrument("binance", symbol, override=instrument)

    def emit(**kwargs: Any) -> None:
        if progress is not None:
            progress(ArchiveProgress(**kwargs))

    def stopping() -> bool:
        return should_stop is not None and should_stop()

    frames: list[pd.DataFrame] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="algo_candles_"))
    try:
        groups = [build_kline_units(native, interval, days, market=market, monthly=monthly)]
        labels = ["klines"]
        for units, group in zip(groups, labels):
            emit(event="plan", exchange="binance", total=len(units), done=0,
                 message=f"свечи {interval} ({group})")
            missing_months: list[str] = []
            with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for unit, payload in _pending_units(
                    pool, units, window=workers * 2, tmpdir=tmpdir, retries=retries,
                    timeout=timeout, checksum=checksum, should_stop=should_stop,
                ):
                    if isinstance(payload, Exception):
                        raise RuntimeError(f"klines {unit.label}: {payload}")
                    if payload is None:
                        missing_months.append(unit.label)
                        emit(event="missing", exchange="binance", label=unit.label,
                             message="candle archive not published")
                        continue
                    try:
                        part = ingest_kline_file(payload, since=since_ts, until=until_ts)
                        frames.append(part)
                    finally:
                        payload.unlink(missing_ok=True)
                    emit(event="file", exchange="binance", label=unit.label,
                         done=len(frames), total=len(units), rows=len(part))
            # A month whose monthly file is absent (typically the current one) is
            # collected from daily files instead.
            fallback_days = [
                day for month in missing_months for day in days if day.startswith(month)
            ]
            if fallback_days:
                fallback = build_kline_units(
                    native, interval, fallback_days, market=market, monthly=False
                )
                emit(event="plan", exchange="binance", total=len(fallback),
                     message=f"daily fallback for {len(missing_months)} month(s)")
                with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    for unit, payload in _pending_units(
                        pool, fallback, window=workers * 2, tmpdir=tmpdir,
                        retries=retries, timeout=timeout, checksum=checksum,
                        should_stop=should_stop,
                    ):
                        if isinstance(payload, Exception):
                            raise RuntimeError(f"klines {unit.label}: {payload}")
                        if payload is None:
                            emit(event="missing", exchange="binance", label=unit.label)
                            continue
                        try:
                            part = ingest_kline_file(
                                payload, since=since_ts, until=until_ts
                            )
                            frames.append(part)
                        finally:
                            payload.unlink(missing_ok=True)
                        emit(event="file", exchange="binance", label=unit.label,
                             rows=len(part))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not frames:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "taker_buy_volume"]
        )
    frame = pd.concat([part for part in frames if not part.empty]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = "timestamp"
    return frame.loc[(frame.index >= since_ts) & (frame.index < until_ts)]


# --------------------------------------------------------------------------- #
# Ticks -> footprint -> candles: the whole dataset a backtest needs
# --------------------------------------------------------------------------- #
@dataclass
class DatasetResult:
    """Outcome of :func:`download_dataset`."""

    store: str = ""
    rows: dict[str, int] = field(default_factory=dict)
    files: dict[str, int] = field(default_factory=dict)
    skipped: int = 0
    footprint: pd.DataFrame | None = None
    footprint_path: str | None = None
    candles: pd.DataFrame | None = None
    candles_path: str | None = None
    missing: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    planned: list[str] = field(default_factory=list)
    cancelled: bool = False
    dry_run: bool = False

    @property
    def total_rows(self) -> int:
        return sum(self.rows.values())

    @property
    def ok(self) -> bool:
        return not self.failures and not self.cancelled

    def summary(self) -> str:
        """One-line human summary for a log pane or a console."""
        parts = [f"{sum(self.files.values())} archive file(s), {self.total_rows:,} ticks"]
        if self.skipped:
            parts.append(f"{self.skipped} file(s) already present")
        if self.footprint is not None:
            bars = self.footprint.index.get_level_values("bar_start").nunique()
            parts.append(f"footprint {len(self.footprint):,} cells / {bars:,} bars")
        if self.candles is not None:
            parts.append(f"{len(self.candles):,} candles")
        if self.missing:
            parts.append(f"{len(self.missing)} archive(s) not published")
        if self.failures:
            parts.append(f"{len(self.failures)} failure(s)")
        if self.cancelled:
            parts.append("cancelled")
        return ", ".join(parts)


#: Binance names intervals by duration; the footprint needs the pandas spelling.
#: Both are resolved through seconds so "1min", "60min" and "1h" all agree.
_SECONDS_TO_BINANCE = {
    60: "1m", 180: "3m", 300: "5m", 900: "15m", 1800: "30m", 3600: "1h",
    7200: "2h", 14400: "4h", 21600: "6h", 28800: "8h", 43200: "12h", 86400: "1d",
}


def pandas_to_binance_interval(timeframe: str) -> str | None:
    """``"1min"``/``"60min"``/``"1h"`` -> ``"1h"``; None when Binance has no such step."""
    try:
        seconds = int(pd.Timedelta(timeframe).total_seconds())
    except (ValueError, TypeError):
        return None
    return _SECONDS_TO_BINANCE.get(seconds) if seconds > 0 else None


@dataclass
class DatasetRequest:
    """A complete "give me this dataset" request -- what the GUI dialog builds.

    Keeps the twenty-odd knobs of :func:`download_dataset` in one validated
    object, so a GUI worker, a CLI call and a test can all describe the same job.
    """

    exchanges: tuple[str, ...] = ("binance",)
    symbol: str = SYMBOL
    since: str = ""
    until: str = ""
    dataset: str = DEFAULT_DATASET
    market: str = "um"
    data_dir: str = DEFAULT_DATA_DIR
    instrument: str | None = None
    contract_size: float | None = None
    workers: int = 4
    force: bool = False
    with_footprint: bool = True
    price_step: float | None = 10.0
    timeframe: str = "1min"
    with_candles: bool = True
    candles_interval: str | None = DEFAULT_INTERVAL
    chunk_rows: int = 1_000_000
    retries: int = 4
    timeout: float = 120.0
    checksum: bool = True
    dry_run: bool = False

    def validate(self) -> None:
        """Raise ValueError describing the first unusable field."""
        if not self.exchanges:
            raise ValueError("Выберите хотя бы одну биржу")
        unknown = [name for name in self.exchanges if name not in EXCHANGES]
        if unknown:
            raise ValueError(f"Неизвестные биржи: {unknown}; доступны {list(EXCHANGES)}")
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset: ожидается одно из {list(DATASETS)}")
        if self.workers < 1:
            raise ValueError("Потоков должно быть не меньше одного")
        parse_symbol(self.symbol)
        since, until = parse_day(self.since), parse_day(self.until)
        if since >= until:
            raise ValueError("Дата «с» должна быть раньше даты «по»")
        if self.with_footprint and not (self.price_step and self.price_step > 0):
            raise ValueError("Шаг цены для футпринта должен быть больше нуля")
        if self.with_candles:
            if self.candles_interval not in KLINE_INTERVALS:
                raise ValueError(
                    f"Интервал свечей: ожидается одно из {list(KLINE_INTERVALS)}"
                )
            if "binance" not in self.exchanges:
                # Candles come from Binance Vision only; a footprint can be built
                # from any venue, but the price axis of a backtest cannot.
                raise ValueError(
                    "Свечи публикует только Binance — включите её или снимите "
                    "галочку «свечи»"
                )
        expected = pandas_to_binance_interval(self.timeframe)
        if self.with_candles and self.candles_interval != expected:
            raise ValueError(
                f"Таймфрейм футпринта ({self.timeframe} -> {expected or '?'}) и интервал "
                f"свечей ({self.candles_interval}) должны совпадать, иначе признаки "
                "футпринта не совместятся с барами бэктеста"
            )

    def to_kwargs(self) -> dict[str, Any]:
        return dict(
            exchanges=list(self.exchanges), since=self.since, until=self.until,
            symbol=self.symbol, instrument=self.instrument, dataset=self.dataset,
            market=self.market, data_dir=self.data_dir, contract_size=self.contract_size,
            workers=self.workers, chunk_rows=self.chunk_rows, retries=self.retries,
            timeout=self.timeout, checksum=self.checksum, force=self.force,
            dry_run=self.dry_run,
            price_step=self.price_step if self.with_footprint else None,
            timeframe=self.timeframe,
            candles_interval=self.candles_interval if self.with_candles else None,
        )

    def download(
        self,
        *,
        progress: Callable[[ArchiveProgress], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> DatasetResult:
        self.validate()
        return download_dataset(**self.to_kwargs(), progress=progress, should_stop=should_stop)


def footprint_from_store(
    store: TradeStore,
    *,
    symbol: str = SYMBOL,
    since: str,
    until: str,
    timeframe: str = "1min",
    price_step: float = 10.0,
    progress: Callable[[ArchiveProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Aggregate stored ticks into a footprint, one UTC day at a time.

    Reading a day's Parquet partitions, aggregating, then releasing them keeps
    the peak memory at one day of ticks instead of the whole range -- a month of
    BTC trades is tens of millions of rows and would otherwise be aggregated in
    one allocation. Progress is reported per day.
    """
    if price_step <= 0:
        raise ValueError("price_step must be positive")
    days = day_range(parse_day(since), parse_day(until))
    since_ts, until_ts = _timestamps(since, until)

    def emit(**kwargs: Any) -> None:
        if progress is not None:
            progress(ArchiveProgress(**kwargs))

    parts: list[pd.DataFrame] = []
    for position, day in enumerate(days, start=1):
        if should_stop is not None and should_stop():
            raise ArchiveCancelled
        frames = []
        for exchange in EXCHANGES:
            path = partition_path(store, exchange, symbol, day)
            if path.is_file():
                frames.append(pd.read_parquet(path))
        if frames:
            day_trades = pd.concat(frames, ignore_index=True)
            day_trades = day_trades.loc[
                (day_trades["timestamp"] >= since_ts) & (day_trades["timestamp"] < until_ts)
            ]
            if not day_trades.empty:
                parts.append(aggregate_footprint(day_trades, timeframe, price_step))
        emit(event="footprint", label=day, done=position, total=len(days))
    if not parts:
        return pd.DataFrame()
    footprint = pd.concat(parts).sort_index()
    # A tick can land in a bar that two day-partitions both cover (the range
    # filter is per day), and the OKX archive day is UTC+8 -- re-aggregating by
    # key removes any duplicated cell instead of summing it twice.
    levels = [name for name in footprint.index.names]
    grouped = footprint.groupby(level=levels, sort=True)
    summed = grouped.sum(min_count=1)
    coverage = [column for column in footprint.columns if column.endswith(("_covered", "_has_data"))]
    if coverage:
        flags = grouped[coverage].any()
        for column in coverage:
            summed[column] = flags[column].astype(bool)
    else:
        summed = grouped.first()
    return summed.loc[:, footprint.columns]


def download_dataset(
    exchanges: Sequence[str],
    since: str,
    until: str,
    *,
    symbol: str = SYMBOL,
    instrument: str | None = None,
    dataset: str = DEFAULT_DATASET,
    market: str = "um",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    contract_size: float | None = None,
    workers: int = 4,
    chunk_rows: int = 1_000_000,
    retries: int = 4,
    timeout: float = 120.0,
    checksum: bool = True,
    force: bool = False,
    dry_run: bool = False,
    price_step: float | None = None,
    timeframe: str = "1min",
    candles_interval: str | None = None,
    progress: Callable[[ArchiveProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> DatasetResult:
    """Ticks (+ optional footprint and candles) in one call -- what the GUI uses.

    ``price_step`` builds a footprint Parquet from the ingested ticks;
    ``candles_interval`` fetches Binance candles as a CSV the backtester can
    load. Both are optional: without them this is :func:`download_archives` plus
    a tidy result object. Progress events come from all three stages, so a
    progress bar can span the whole job.
    """
    outcome = DatasetResult(store=str(Path(data_dir)), dry_run=dry_run)
    out_dir = Path(data_dir)
    interval = candles_interval if candles_interval else DEFAULT_INTERVAL

    def emit(**kwargs: Any) -> None:
        if progress is not None:
            progress(ArchiveProgress(**kwargs))

    archives_result = download_archives(
        exchanges, since, until, symbol=symbol, instrument=instrument, dataset=dataset,
        market=market, data_dir=data_dir, contract_size=contract_size, workers=workers,
        chunk_rows=chunk_rows, retries=retries, timeout=timeout, checksum=checksum,
        force=force, dry_run=dry_run, progress=progress, should_stop=should_stop,
    )
    outcome.rows = archives_result.rows
    outcome.files = archives_result.files
    outcome.skipped = archives_result.skipped
    outcome.missing = archives_result.missing
    outcome.failures = archives_result.failures
    outcome.planned = archives_result.planned
    outcome.cancelled = archives_result.cancelled
    if dry_run:
        return outcome
    if archives_result.cancelled:
        # A cancelled run still has whatever completed on disk, but a partial
        # footprint would silently become "the whole range" downstream.
        return outcome

    if price_step:
        emit(event="stage", message=f"footprint: {timeframe} x {price_step:g} накопление")
        try:
            footprint = footprint_from_store(
                TradeStore(data_dir), symbol=symbol, since=since, until=until,
                timeframe=timeframe, price_step=price_step, progress=progress,
                should_stop=should_stop,
            )
        except ArchiveCancelled:
            outcome.cancelled = True
            return outcome
        except (ValueError, OSError) as exc:
            outcome.failures.append(f"footprint: {exc}")
        else:
            if footprint.empty:
                outcome.failures.append(
                    "footprint: нет тиков за выбранный период в хранилище"
                )
            else:
                folder = out_dir / "footprint"
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / (
                    f"{_encode_symbol(symbol)}_{timeframe}_step{price_step:g}"
                    f"_{parse_day(since)}_{parse_day(until)}.parquet"
                )
                footprint.to_parquet(target)
                outcome.footprint = footprint
                outcome.footprint_path = str(target)

    if candles_interval:
        emit(event="stage", message=f"candles: binance {interval}")
        try:
            candles = download_candles(
                since, until, symbol=symbol, interval=interval, market=market,
                instrument=instrument, workers=workers, retries=retries, timeout=timeout,
                checksum=checksum, progress=progress, should_stop=should_stop,
            )
        except ArchiveCancelled:
            outcome.cancelled = True
            return outcome
        except (ValueError, RuntimeError, OSError) as exc:
            outcome.failures.append(f"candles: {exc}")
        else:
            if candles.empty:
                outcome.failures.append("candles: биржа не отдала ни одного бара")
            else:
                folder = out_dir / "candles"
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / (
                    f"{native_instrument('binance', symbol, override=instrument)}"
                    f"-{interval}_{parse_day(since)}_{parse_day(until)}.csv"
                )
                candles.to_csv(target)
                outcome.candles = candles
                outcome.candles_path = str(target)
    return outcome


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def parse_day(value: str) -> str:
    """Validate a ``YYYY-MM-DD`` UTC day and return it normalized."""
    text = str(value).strip()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(f"Dates must be YYYY-MM-DD (UTC), got {value!r}") from exc


def day_range(since: str, until: str) -> list[str]:
    """UTC days in ``[since, until)``."""
    start = datetime.strptime(parse_day(since), "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(parse_day(until), "%Y-%m-%d").replace(tzinfo=UTC)
    if start >= end:
        raise ValueError("since must be strictly before until")
    days: list[str] = []
    cursor = start
    while cursor < end:
        days.append(cursor.date().isoformat())
        cursor += timedelta(days=1)
    return days


# --------------------------------------------------------------------------- #
# CLI plumbing (shared by `algo-trades archive` and the fetch_ticks.py shim)
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-e", "--exchange", nargs="+", choices=EXCHANGES,
                        default=["binance"], help="venues to fetch (default: binance)")
    parser.add_argument("--symbol", default=SYMBOL,
                        help=f"CCXT-style instrument (default: {SYMBOL})")
    parser.add_argument("--instrument", default=None,
                        help="venue-native id override, e.g. BTCUSDT or BTC-USDT-SWAP")
    parser.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET,
                        help="binance only: aggTrades (default, smaller) or trades")
    parser.add_argument("--market", choices=("um", "cm"), default="um",
                        help="binance futures market: um (USDT-margined) or cm (coin)")
    parser.add_argument("--since", required=True, help="start day, YYYY-MM-DD (UTC), inclusive")
    parser.add_argument("--until", required=True, help="end day, YYYY-MM-DD (UTC), exclusive")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"trade store root (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--contract-size", type=float, default=None,
                        help="base units per contract (OKX only; auto-detected by default)")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    parser.add_argument("--chunk-rows", type=int, default=1_000_000,
                        help="rows parsed per chunk while streaming")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request seconds")
    parser.add_argument("--no-checksum", action="store_true",
                        help="skip sha256 verification of Binance archives")
    parser.add_argument("--force", action="store_true",
                        help="re-download days already recorded as complete")
    parser.add_argument("--candles", metavar="INTERVAL", nargs="?", const=DEFAULT_INTERVAL,
                        default=None,
                        help="also fetch Binance candles, e.g. --candles 1m "
                             f"(choices: {', '.join(KLINE_INTERVALS)})")
    parser.add_argument("--footprint", action="store_true",
                        help="aggregate the ingested ticks into a footprint Parquet")
    parser.add_argument("--price-step", type=float, default=10.0,
                        help="footprint price bucket in quote units (with --footprint)")
    parser.add_argument("--timeframe", default="1min",
                        help="footprint bar frequency, pandas offset (with --footprint)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the file plan and exit")


def format_event(event: ArchiveProgress) -> str | None:
    """One human-readable line for a progress event; None when not worth showing.

    Shared by the console command and the GUI log pane so both say the same
    thing about the same job.
    """
    if event.event == "plan":
        pending = max(event.total - event.done, 0)
        text = f"{event.exchange or 'candles'}: {pending} file(s) to fetch, {event.done} present"
        return text + (f" — {event.message}" if event.message else "")
    if event.event == "file":
        return f"  {event.exchange} {event.label}: {event.rows:,} rows"
    if event.event == "footprint":
        # Per-day lines would bury everything else on a multi-month range.
        if event.done == event.total or event.done % 25 == 0 or event.done == 1:
            return f"  футпринт: {event.done}/{event.total} дней"
        return None
    if event.event == "stage":
        return f"{event.message}..."
    if event.event == "missing":
        return f"  {event.exchange} {event.label}: ещё не опубликован, пропуск"
    if event.event == "error":
        return f"  ! {event.exchange} {event.label}: {event.message}"
    if event.event == "exchange_done":
        return f"{event.exchange}: {event.rows:,} rows in {event.done} file(s)"
    return None


def run_from_args(args: argparse.Namespace) -> int:
    """Console entry point; returns a process exit code."""

    def report(event: ArchiveProgress) -> None:
        line = format_event(event)
        if line is None:
            return
        print(line, file=sys.stderr if event.event == "error" else sys.stdout)

    try:
        result = download_dataset(
            args.exchange, args.since, args.until,
            symbol=args.symbol, instrument=args.instrument, dataset=args.dataset,
            market=args.market, data_dir=args.data_dir, contract_size=args.contract_size,
            workers=args.workers, chunk_rows=args.chunk_rows, retries=args.retries,
            timeout=args.timeout, checksum=not args.no_checksum, force=args.force,
            dry_run=args.dry_run, price_step=args.price_step if args.footprint else None,
            timeframe=args.timeframe, candles_interval=args.candles, progress=report,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for line in result.planned:
            print(f"  {line}")
        print("dry run: nothing downloaded")
        return 1 if result.failures else 0

    print("-" * 60)
    for exchange in args.exchange:
        print(f"{exchange:>8}: {result.rows.get(exchange, 0):,} rows written")
    if result.skipped:
        print(f"skipped {result.skipped} file(s) already in the store (--force redoes them)")
    if result.footprint_path:
        bars = result.footprint.index.get_level_values("bar_start")
        print(
            f"footprint: {len(result.footprint):,} cells, {bars.nunique():,} bars, "
            f"{bars.min()} .. {bars.max()}\n           -> {result.footprint_path}"
        )
    if result.candles_path:
        candles = result.candles
        print(
            f"candles:   {len(candles):,} bars "
            f"({candles.index[0]:%Y-%m-%d %H:%M} .. {candles.index[-1]:%Y-%m-%d %H:%M} UTC)"
            f"\n           -> {result.candles_path}"
        )
    for failure in result.failures:
        print(f"! {failure}", file=sys.stderr)
    if result.cancelled:
        print("cancelled: completed days are recorded, re-run to resume")
    print(f"store: {result.store}")

    if not result.footprint_path and not args.dry_run:
        print(
            "\nNext steps:\n"
            f"  algo-trades archive --since {args.since} --until {args.until} --footprint "
            f"--candles 1m --data-dir {args.data_dir}\n"
            f"  algo-trades footprint --price-step 10 --timeframe 1min "
            f"--data-dir {args.data_dir} --out footprint.parquet"
        )
    return 1 if (result.failures or result.cancelled) else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="algo-trades archive",
        description=(
            "Download bulk exchange archives (Binance Vision / Bybit / OKX) into the "
            "canonical trade store for footprint and order-flow research."
        ),
    )
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


__all__ = [
    "ArchiveCancelled", "ArchiveProgress", "ArchiveReport", "ArchiveResult",
    "DatasetRequest", "DatasetResult", "Unit", "add_arguments", "archive_days",
    "build_kline_units", "build_units", "day_range", "download_archives",
    "download_candles", "download_dataset", "fetch_unit", "footprint_from_store",
    "format_event", "ingest_file", "ingest_kline_file", "main", "native_instrument",
    "okx_contract_size", "pandas_to_binance_interval", "parse_day", "parse_symbol",
    "partition_path", "read_binance_klines", "read_binance_trades",
    "read_bybit_trades", "read_okx_trades", "run_from_args",
]
