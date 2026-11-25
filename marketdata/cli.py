"""Command-line ingestion: download trades, build footprints, export features.

Entry point is the ``algo-trades`` console script::

    algo-trades archive -e binance --since 2024-01-01 --until 2024-02-01 --candles 1m
    algo-trades download -e binance okx bybit --since 2024-01-01 --until 2024-02-01
    algo-trades footprint --price-step 10 --out footprint.parquet
    algo-trades features --window 20 --out features.csv
    algo-trades status

``archive`` reads the exchanges' bulk archives (fast, complete, free) and is the
recommended path for anything longer than a few days; ``download`` pages public
trades through CCXT and is kept for recent windows and venues without archives.

Times are epoch milliseconds, offset-aware ISO-8601, or a bare ``YYYY-MM-DD``
(optionally with ``HH:MM``), which this CLI reads as UTC -- a date on a command
line carries no other offset, so no silent guessing happens anywhere below.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any, Callable, Sequence

import pandas as pd

from . import archives
from .archives import DEFAULT_DATA_DIR
from .core import EXCHANGES, QUANTITY_SOURCES, SYMBOL, aggregate_footprint
from .features import bar_features_from_footprint
from .sources import DownloadReport, download_trades
from .store import TradeStore

DATE_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$")


def parse_time(value: str) -> int | str | pd.Timestamp:
    """CLI time value: epoch ms, offset-aware ISO, or a bare date read as UTC."""
    text = value.strip()
    if DATE_LIKE.match(text):
        return pd.Timestamp(text.replace(" ", "T"), tz="UTC")
    return text


def _write(frame: pd.DataFrame, path: str, *, index: bool) -> None:
    if path.endswith(".parquet"):
        frame.to_parquet(path)
    elif path.endswith(".csv"):
        frame.to_csv(path, index=index)
    else:
        raise ValueError("Output must end with .parquet or .csv")


def _select(args: argparse.Namespace) -> pd.DataFrame:
    """Read trades from the store and apply the CLI's optional filters."""
    # read() takes (exchange, symbol): passing the symbol positionally filtered by
    # exchange and returned nothing.
    trades = TradeStore(args.data_dir).read(symbol=args.symbol, sort=True)
    if trades.empty:
        return trades
    if args.exchange:
        trades = trades.loc[trades["exchange"].isin(args.exchange)]
    if getattr(args, "since", None):
        trades = trades.loc[trades["timestamp"] >= parse_time(args.since)]
    if getattr(args, "until", None):
        trades = trades.loc[trades["timestamp"] < parse_time(args.until)]
    return trades


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def run_download(
    args: argparse.Namespace,
    client_factory: Callable[[str], Any] | None = None,
) -> int:
    store = TradeStore(args.data_dir)
    since, until = parse_time(args.since), parse_time(args.until)
    failed = 0
    for exchange in args.exchange:
        report = DownloadReport()
        client = client_factory(exchange) if client_factory else None
        try:
            count = download_trades(
                exchange, since, until, store,
                symbol=args.symbol,
                limit=args.limit,
                exchange_client=client,
                contract_size=args.contract_size,
                quantity_source=args.quantity_source,
                report=report,
                strict_pagination=args.strict_pagination,
            )
        except Exception as exc:  # noqa: BLE001 - one venue must not stop the rest
            failed += 1
            print(f"{exchange}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        size = "-" if report.contract_size is None else f"{report.contract_size:g}"
        print(
            f"{exchange}: {count} rows written, {report.pages} pages, "
            f"{report.malformed_rows} malformed, contract_size={size}"
        )
        for issue in report.issues:
            print(f"  ! {issue}")
        if report.incomplete_buckets:
            print(f"  ! incomplete millisecond buckets: {report.incomplete_buckets}")
    return 1 if failed else 0


def run_footprint(args: argparse.Namespace) -> int:
    trades = _select(args)
    if trades.empty:
        print("No trades in the store for this selection.", file=sys.stderr)
        return 1
    footprint = aggregate_footprint(
        trades, timeframe=args.timeframe, price_step=args.price_step
    )
    bars = footprint.index.get_level_values("bar_start")
    print(
        f"{len(footprint)} cells, {bars.nunique()} bars, "
        f"{bars.min()} .. {bars.max()}"
    )
    if args.out:
        _write(footprint, args.out, index=True)
        print(f"written: {args.out}")
    return 0


def run_features(args: argparse.Namespace) -> int:
    trades = _select(args)
    if trades.empty:
        print("No trades in the store for this selection.", file=sys.stderr)
        return 1
    footprint = aggregate_footprint(
        trades, timeframe=args.timeframe, price_step=args.price_step
    )
    features = bar_features_from_footprint(footprint, window=args.window)
    print(f"{len(features)} bars, columns: {', '.join(features.columns)}")
    if args.out:
        _write(features, args.out, index=True)
        print(f"written: {args.out}")
    return 0


def run_status(args: argparse.Namespace) -> int:
    store = TradeStore(args.data_dir)
    manifest = store.manifest_data()
    partitions = sorted(store.canonical.glob("exchange=*/symbol=*/date=*/trades.parquet"))
    print(f"store: {store.root}")
    print(f"partitions: {len(partitions)}")
    checkpoints = store.checkpoints()
    if checkpoints:
        print("checkpoints:")
        for name, cursor in sorted(checkpoints.items()):
            print(f"  {name}: {pd.Timestamp(cursor, unit='ms', tz='UTC')}")
    for entry in manifest.get("downloads", [])[-5:]:
        size = entry.get("contract_size")
        print(
            f"download {entry['at']}: {entry['exchange']} {entry['symbol']} "
            f"{entry['rows_written']} rows, contract_size="
            f"{'-' if size is None else format(size, 'g')}"
        )
    for issue in manifest.get("issues", [])[-5:]:
        print(f"issue {issue['at']} [{issue['exchange']}]: {issue['message']}")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _add_store_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"trade store root (default: {DEFAULT_DATA_DIR})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algo-trades",
        description="Download crypto-futures trades and build footprint features.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    archive = commands.add_parser(
        "archive",
        help="fetch bulk exchange tick archives (binance/bybit/okx), optionally candles",
        description=(
            "Download the exchanges' own bulk archives into the canonical trade store. "
            "Far faster and more complete than paging CCXT, and free."
        ),
    )
    archives.add_arguments(archive)
    archive.set_defaults(handler=archives.run_from_args)

    download = commands.add_parser("download", help="fetch public trades via CCXT")
    download.add_argument("-e", "--exchange", nargs="+", choices=EXCHANGES,
                          default=list(EXCHANGES))
    download.add_argument("--symbol", default=SYMBOL)
    download.add_argument("--since", required=True, help="epoch ms, ISO-8601, or YYYY-MM-DD (UTC)")
    download.add_argument("--until", required=True, help="exclusive upper bound")
    download.add_argument("--limit", type=int, default=1000, help="trades per page")
    download.add_argument("--contract-size", type=float, default=None,
                          help="base units per contract; detected from the venue by default")
    download.add_argument("--quantity-source", choices=QUANTITY_SOURCES, default="auto")
    download.add_argument("--strict-pagination", action="store_true",
                          help="fail instead of warning when a millisecond cannot be paged")
    _add_store_options(download)
    download.set_defaults(handler=run_download)

    for name, handler, help_text in (
        ("footprint", run_footprint, "aggregate stored trades into a footprint"),
        ("features", run_features, "export bar-aligned order-flow features"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--symbol", default=None, help="filter by symbol")
        command.add_argument("--exchange", nargs="+", choices=EXCHANGES, default=None)
        command.add_argument("--since", default=None)
        command.add_argument("--until", default=None)
        command.add_argument("--timeframe", default="1min", help="pandas bar frequency")
        # No default: a symbol-agnostic step would quietly produce a useless
        # grid (BTC at step 1 has one level per tick), so make it explicit.
        command.add_argument("--price-step", type=float, required=True,
                             help="price bucket size in quote units")
        command.add_argument("--out", default=None, help=".parquet or .csv output path")
        _add_store_options(command)
        command.set_defaults(handler=handler)
        if name == "features":
            command.add_argument("--window", type=int, default=20,
                                 help="rolling window in bars")

    status = commands.add_parser("status", help="show store, checkpoint and issue summary")
    _add_store_options(status)
    status.set_defaults(handler=run_status)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        return run_download(args, client_factory)
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
