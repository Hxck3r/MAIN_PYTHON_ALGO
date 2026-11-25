"""Causal order-flow features for bar-aligned footprint data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: per-cell value columns produced by aggregate_footprint/build_footprint
FOOTPRINT_VALUE_COLUMNS = ("buy_qty", "sell_qty", "delta", "total_qty", "trade_count")
#: suffixes of boolean coverage columns that must never be summed as volume
COVERAGE_SUFFIXES = ("_has_data", "_covered", "_covered_flag")


def _resolve_column(
    frame: pd.DataFrame, preferred: str, suffix: str
) -> str | None:
    """Pick the requested column, or the single variant of it."""
    if preferred in frame.columns:
        return preferred
    plain = suffix.lstrip("_")
    if plain in frame.columns:
        return plain
    candidates = [
        column for column in frame.columns
        if column.endswith(suffix) and not column.startswith("combined_")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise ValueError(
        f"Ambiguous column {preferred!r}: several venue variants exist "
        f"({sorted(candidates)}); pass the column name explicitly"
    )


def _is_coverage_column(column: str) -> bool:
    return column.endswith(COVERAGE_SUFFIXES) or column in ("has_data", "covered")


def add_order_flow_features(
    frame: pd.DataFrame,
    *,
    delta_column: str = "combined_delta",
    buy_column: str = "combined_buy_qty",
    sell_column: str = "combined_sell_qty",
    volume_column: str = "combined_total_qty",
    trade_count_column: str | None = None,
    window: int = 20,
) -> pd.DataFrame:
    """Return a copy with causal order-flow features.

    Every rolling operation includes only the current and previous rows. Missing
    exchange coverage remains missing; it is not silently converted into data.
    When only a single venue is present its ``<venue>_delta`` style columns are
    used automatically.
    """
    if window < 1:
        raise ValueError("window must be positive")
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex) and not out.index.is_monotonic_increasing:
        out = out.sort_index(kind="stable")

    delta_name = _resolve_column(out, delta_column, "_delta")
    buy_name = _resolve_column(out, buy_column, "_buy_qty")
    sell_name = _resolve_column(out, sell_column, "_sell_qty")
    missing = [
        name for name, resolved in (
            (delta_column, delta_name), (buy_column, buy_name), (sell_column, sell_name),
        ) if resolved is None
    ]
    if missing:
        raise ValueError(f"Missing footprint columns: {missing}")

    delta = pd.to_numeric(out[delta_name], errors="coerce")
    buy = pd.to_numeric(out[buy_name], errors="coerce")
    sell = pd.to_numeric(out[sell_name], errors="coerce")
    volume_name = _resolve_column(out, volume_column, "_total_qty")
    volume = (
        pd.to_numeric(out[volume_name], errors="coerce")
        if volume_name is not None else buy + sell
    )

    out["delta"] = delta
    out["cumulative_delta"] = delta.fillna(0).cumsum().where(delta.notna())
    out["buy_sell_ratio"] = buy / sell.replace(0, np.nan)
    out["volume_imbalance"] = (buy - sell) / (buy + sell).replace(0, np.nan)
    if trade_count_column and trade_count_column in out:
        out["trade_count"] = pd.to_numeric(out[trade_count_column], errors="coerce")
    elif "trade_count" not in out:
        count_columns = [c for c in out if c.endswith("_trade_count")]
        out["trade_count"] = (
            out[count_columns].sum(axis=1, min_count=1) if count_columns else np.nan
        )
    out["rolling_delta"] = delta.rolling(window, min_periods=1).sum().where(delta.notna())
    out["rolling_volume"] = volume.rolling(window, min_periods=1).sum().where(volume.notna())
    mean = delta.rolling(window, min_periods=2).mean()
    std = delta.rolling(window, min_periods=2).std(ddof=0)
    out["delta_zscore"] = (delta - mean) / std.replace(0, np.nan)
    return out


def bar_features_from_footprint(
    footprint: pd.DataFrame, *, window: int = 20, include_cells: bool = True
) -> pd.DataFrame:
    """Collapse price cells into bar-aligned order-flow features.

    Value columns are summed cell by cell; ``combined_*`` and per-venue volumes
    are summed independently (never one into the other, which would double
    count). Coverage flags are OR-ed, so a venue without data stays ``False``
    instead of being converted into zero observed volume.

    ``include_cells`` adds the within-bar structure of the footprint
    (:func:`cell_order_flow_features`): POC, value area, diagonal imbalance,
    absorption and delta divergence. It needs ``price_level`` to mean something,
    so it is skipped automatically when the index has no such level (a per-venue
    ``(bar_start, venue)`` frame, say). A frame that is not indexed by
    ``bar_start`` at all is rejected.
    """
    if footprint is None or footprint.empty:
        return pd.DataFrame()
    index = footprint.index
    if not isinstance(index, pd.MultiIndex) or "bar_start" not in index.names:
        raise ValueError(
            "Footprint must be indexed by (bar_start, price_level)"
        )
    grouped = footprint.groupby(level="bar_start")
    value_suffixes = tuple(f"_{name}" for name in FOOTPRINT_VALUE_COLUMNS)
    value_columns = [
        column for column in footprint.columns
        if column.endswith(value_suffixes) or column in FOOTPRINT_VALUE_COLUMNS
    ]
    if not value_columns:
        raise ValueError("Footprint has no volume/delta columns to aggregate")
    bars = grouped[value_columns].sum(min_count=1)
    coverage_columns = [
        column for column in footprint.columns if _is_coverage_column(column)
    ]
    if coverage_columns:
        flags = grouped[coverage_columns].any()
        for column in coverage_columns:
            bars[column] = flags[column].astype(bool)
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars.index.name = "bar_start"
    bars = add_order_flow_features(bars, window=window)
    if include_cells and "price_level" in index.names:
        cells = cell_order_flow_features(footprint)
        if not cells.empty:
            # Cell statistics are per bar, so they join 1:1 onto the bar frame.
            bars = bars.join(cells, how="left", suffixes=("", "_cell"))
    return bars


# --------------------------------------------------------------------------- #
# Within-bar (price cell) structure
# --------------------------------------------------------------------------- #
#: columns produced by cell_order_flow_features
CELL_FEATURE_COLUMNS = (
    "bar_delta", "cell_volume", "level_count", "poc_price", "poc_volume", "poc_share",
    "poc_migration", "value_area_high", "value_area_low", "value_area_share",
    "bar_high_cell", "bar_low_cell", "bar_vwap", "vwap_minus_poc", "bar_range_ticks",
    "buy_imbalance_levels", "sell_imbalance_levels", "imbalance_stack", "imbalance_skew",
    "absorption_price", "absorption_volume", "absorption_signal",
    "delta_divergence",
)


def _infer_price_step(levels: np.ndarray) -> float:
    """Smallest positive gap between distinct price levels, or 0.0 if unknown."""
    unique = np.unique(levels[np.isfinite(levels)])
    if unique.size < 2:
        return 0.0
    gaps = np.diff(unique)
    gaps = gaps[gaps > 0]
    return float(gaps.min()) if gaps.size else 0.0


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest consecutive ``True`` run."""
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _value_area(volume: np.ndarray, poc_index: int, target: float) -> tuple[int, int, float]:
    """Grow a value area out from the POC two levels at a time.

    The Market Profile convention: at each step compare the next two levels up
    against the next two down and take whichever side carries more volume. The
    area therefore always contains the POC and is contiguous.
    """
    count = len(volume)
    low = high = poc_index
    accumulated = float(volume[poc_index])
    while accumulated < target and (low > 0 or high < count - 1):
        up = float(volume[high + 1:high + 3].sum()) if high < count - 1 else -1.0
        down = float(volume[max(0, low - 2):low].sum()) if low > 0 else -1.0
        if (up >= down or low == 0) and high < count - 1:
            high += 1
            accumulated += float(volume[high])
        elif low > 0:
            low -= 1
            accumulated += float(volume[low])
        else:
            break
    return low, high, accumulated


def _describe_bar(
    cells: pd.DataFrame,
    *,
    step: float,
    imbalance_ratio: float,
    value_area_share: float,
    absorption_quantile: float,
    absorption_delta_share: float,
) -> dict[str, float]:
    """Per-bar summary of one bar's price cells (cells sorted by level)."""
    levels = cells["level"].to_numpy(dtype="float64")
    volume = cells["volume"].to_numpy(dtype="float64")
    delta = np.nan_to_num(cells["delta"].to_numpy(dtype="float64"), nan=0.0)
    buy = np.nan_to_num(cells["buy"].to_numpy(dtype="float64"), nan=0.0)
    sell = np.nan_to_num(cells["sell"].to_numpy(dtype="float64"), nan=0.0)
    count = len(volume)
    total = float(volume.sum())
    if total <= 0:
        return {}

    # Ties on volume resolve to the lowest level: cells are sorted ascending and
    # argmax returns the first maximum.
    poc_index = int(np.argmax(volume))
    poc_price = float(levels[poc_index])
    half = step / 2.0
    vwap = float(((levels + half) * volume).sum() / total)

    low_index, high_index, area_volume = _value_area(
        volume, poc_index, value_area_share * total
    )

    # Diagonal imbalance: a cell's aggressive volume against the resting volume
    # one tick away on the other side. The edge cells are compared with
    # themselves, which is the usual reading when a bar has no neighbour there.
    buy_imbalance = np.zeros(count, dtype=bool)
    sell_imbalance = np.zeros(count, dtype=bool)
    for position in range(count):
        reference = sell[position - 1] if position > 0 else sell[position]
        if buy[position] > 0 and buy[position] >= imbalance_ratio * reference:
            buy_imbalance[position] = True
        reference = buy[position + 1] if position < count - 1 else buy[position]
        if sell[position] > 0 and sell[position] >= imbalance_ratio * reference:
            sell_imbalance[position] = True
    buy_levels = int(buy_imbalance.sum())
    sell_levels = int(sell_imbalance.sum())
    total_levels = buy_levels + sell_levels
    buy_stack = _longest_run(buy_imbalance)
    sell_stack = _longest_run(sell_imbalance)
    # Signed so a single column carries "who owns the stack"; positive = buyers.
    stack = buy_stack if buy_stack >= sell_stack else -sell_stack

    # Absorption: a heavy cell whose delta is near zero means aggressive flow was
    # met by resting size. Negative delta -> sellers absorbed by bids (bullish),
    # positive delta -> buyers absorbed by offers (bearish).
    threshold = float(np.quantile(volume, absorption_quantile))
    heavy = volume >= threshold
    flat = np.abs(delta) <= absorption_delta_share * volume
    absorbing = np.flatnonzero(heavy & flat)
    if absorbing.size:
        chosen = int(absorbing[np.argmax(volume[absorbing])])
        absorption_price = float(levels[chosen])
        absorption_volume = float(volume[chosen])
        absorption_signal = 1 if delta[chosen] < 0 else (-1 if delta[chosen] > 0 else 0)
    else:
        absorption_price = np.nan
        absorption_volume = np.nan
        absorption_signal = 0

    return {
        "bar_delta": float(delta.sum()),
        "cell_volume": total,
        "level_count": float(count),
        "poc_price": poc_price,
        "poc_volume": float(volume[poc_index]),
        "poc_share": float(volume[poc_index]) / total,
        "value_area_high": float(levels[high_index]) + step,
        "value_area_low": float(levels[low_index]),
        "value_area_share": area_volume / total,
        # A price level covers [level, level + step), so the bar's high is the top
        # of its highest traded bucket, not the bucket's floor.
        "bar_high_cell": float(levels[-1]) + step,
        "bar_low_cell": float(levels[0]),
        "bar_vwap": vwap,
        "vwap_minus_poc": vwap - poc_price,
        "bar_range_ticks": (float(levels[-1]) - float(levels[0])) / step + 1.0 if step else float(count),
        "buy_imbalance_levels": float(buy_levels),
        "sell_imbalance_levels": float(sell_levels),
        "imbalance_stack": float(stack),
        "imbalance_skew": (buy_levels - sell_levels) / total_levels if total_levels else 0.0,
        "absorption_price": absorption_price,
        "absorption_volume": absorption_volume,
        "absorption_signal": float(absorption_signal),
    }


def cell_order_flow_features(
    footprint: pd.DataFrame,
    *,
    value_column: str = "combined_total_qty",
    delta_column: str = "combined_delta",
    buy_column: str = "combined_buy_qty",
    sell_column: str = "combined_sell_qty",
    imbalance_ratio: float = 3.0,
    value_area_share: float = 0.7,
    absorption_quantile: float = 0.9,
    absorption_delta_share: float = 0.15,
    price_step: float | None = None,
) -> pd.DataFrame:
    """Order-flow structure inside each bar, from its footprint price cells.

    Returns one row per bar (indexed by ``bar_start``, UTC) with the classic
    footprint readings:

    ``poc_price``/``poc_volume``/``poc_share``
        the price level that traded the most and how dominant it was;
    ``poc_migration``
        change of the POC against the previous bar -- value drifting up or down;
    ``value_area_high``/``value_area_low``/``value_area_share``
        the contiguous area around the POC holding ~``value_area_share`` of the
        volume (Market Profile expansion, two levels at a time);
    ``buy_imbalance_levels``/``sell_imbalance_levels``/``imbalance_stack``/``imbalance_skew``
        cells whose aggressive volume is at least ``imbalance_ratio`` times the
        resting volume one tick away (diagonal imbalance). A zero reference
        counts as a full imbalance, which is how footprint charts read it. The
        stack is the longest consecutive run, signed by side: positive = buyers;
    ``absorption_price``/``absorption_volume``/``absorption_signal``
        the heaviest cell (top ``1 - absorption_quantile`` of the bar's cells)
        whose delta stays within ``absorption_delta_share`` of its volume. Signal
        +1 = sellers absorbed by bids (bullish), -1 = buyers absorbed by offers;
    ``delta_divergence``
        +1 when the bar made a lower low than the previous bar yet net delta was
        positive, -1 when it made a higher high with negative delta. A new low
        and a new high at once (an outside bar) resolves as +1;
    ``bar_high_cell``/``bar_low_cell``/``bar_vwap``/``bar_range_ticks``
        the bar's traded range and volume-weighted price, rebuilt from the cells.

    ``price_step`` is inferred from the smallest gap between distinct levels when
    not given; it only shifts the reconstructed extremes and the VWAP half-tick
    centring, never the rankings.

    Lookahead: a bar's cells are only complete when the bar closes. Every value
    here describes a *finished* bar, so a strategy may act on it at the close of
    that bar (``Bar.index``) and must not use it earlier.
    """
    if footprint is None or footprint.empty:
        return pd.DataFrame()
    index = footprint.index
    if not isinstance(index, pd.MultiIndex) or set(index.names) < {"bar_start", "price_level"}:
        raise ValueError("Footprint must be indexed by (bar_start, price_level)")
    if imbalance_ratio <= 1:
        raise ValueError("imbalance_ratio must be greater than 1")
    if not 0 < value_area_share <= 1:
        raise ValueError("value_area_share must be in (0, 1]")
    if not 0 < absorption_quantile <= 1:
        raise ValueError("absorption_quantile must be in (0, 1]")
    if not 0 <= absorption_delta_share <= 1:
        raise ValueError("absorption_delta_share must be in [0, 1]")

    resolved = {}
    for name, preferred, suffix in (
        ("volume", value_column, "_total_qty"),
        ("delta", delta_column, "_delta"),
        ("buy", buy_column, "_buy_qty"),
        ("sell", sell_column, "_sell_qty"),
    ):
        column = _resolve_column(footprint, preferred, suffix)
        if column is None:
            raise ValueError(f"Footprint has no {name} column (looked for {preferred!r})")
        resolved[name] = footprint[column]

    levels = index.get_level_values("price_level").to_numpy(dtype="float64")
    step = float(price_step) if price_step else _infer_price_step(levels)

    cells = pd.DataFrame({
        "bar": index.get_level_values("bar_start"),
        "level": levels,
        "volume": pd.to_numeric(resolved["volume"], errors="coerce").to_numpy(dtype="float64"),
        "delta": pd.to_numeric(resolved["delta"], errors="coerce").to_numpy(dtype="float64"),
        "buy": pd.to_numeric(resolved["buy"], errors="coerce").to_numpy(dtype="float64"),
        "sell": pd.to_numeric(resolved["sell"], errors="coerce").to_numpy(dtype="float64"),
    })
    # A cell with no observed volume on any venue carries no information; a zero
    # row would silently become a "level with no buying", which is not the same
    # thing as a level nobody traded.
    cells = cells.loc[np.isfinite(cells["volume"]) & (cells["volume"] > 0)
                      & np.isfinite(cells["level"])]
    if cells.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="bar_start"))

    rows: dict[Any, dict[str, float]] = {}
    for bar, group in cells.groupby("bar", sort=True):
        summary = _describe_bar(
            group.sort_values("level", kind="stable"),
            step=step, imbalance_ratio=imbalance_ratio,
            value_area_share=value_area_share,
            absorption_quantile=absorption_quantile,
            absorption_delta_share=absorption_delta_share,
        )
        if summary:
            rows[bar] = summary
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="bar_start"))

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True), name="bar_start")
    out = out.sort_index()

    # Cross-bar derivations stay causal: they only look backwards.
    previous_poc = out["poc_price"].shift(1)
    out["poc_migration"] = out["poc_price"] - previous_poc
    previous_high = out["bar_high_cell"].shift(1)
    previous_low = out["bar_low_cell"].shift(1)
    lower_low = (out["bar_low_cell"] < previous_low).fillna(False)
    higher_high = (out["bar_high_cell"] > previous_high).fillna(False)
    bullish = lower_low & (out["bar_delta"] > 0)
    bearish = higher_high & (out["bar_delta"] < 0)
    out["delta_divergence"] = np.select(
        [bullish.to_numpy(), bearish.to_numpy()], [1.0, -1.0], default=0.0
    )
    return out.loc[:, list(CELL_FEATURE_COLUMNS)]


__all__ = [
    "CELL_FEATURE_COLUMNS", "add_order_flow_features", "bar_features_from_footprint",
    "cell_order_flow_features",
]