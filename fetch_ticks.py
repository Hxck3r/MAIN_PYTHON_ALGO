#!/usr/bin/env python3
"""Standalone launcher for the bulk tick-archive downloader.

The implementation now lives in the package as :mod:`marketdata.archives`, so this
script, the ``algo-trades archive`` command and the GUI all share one code path.
Kept as a convenience entry point for a fresh checkout::

    python fetch_ticks.py --since 2024-01-01 --until 2024-02-01
    python fetch_ticks.py -e binance okx --since 2024-01-01 --until 2024-02-01 \\
        --candles 1m --data-dir marketdata_store

Equivalent, and preferred once the package is installed::

    algo-trades archive --since 2024-01-01 --until 2024-02-01

:mod:`marketdata.archives` documents the venue semantics (verified against the
live archives), the OKX UTC+8 day caveat, and the programmatic API the GUI uses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _bootstrap() -> None:
    """Make the repository root importable when run from a checkout."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "marketdata" / "core.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise SystemExit(
        "Could not find the project root (a directory containing marketdata/core.py) "
        f"starting from {here}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    _bootstrap()
    from marketdata import archives

    parser = argparse.ArgumentParser(
        prog="fetch_ticks",
        description=(
            "Download bulk exchange tick archives (Binance Vision / Bybit / OKX) into "
            "the canonical trade store for footprint and order-flow research. "
            "Binance candles (--candles) are available too."
        ),
    )
    archives.add_arguments(parser)
    return archives.run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
