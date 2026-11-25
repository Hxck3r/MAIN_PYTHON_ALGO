"""Order-flow strategy trading the *inside* of the footprint.

``FootprintDeltaStrategy`` reads only bar-level totals (rolling delta). This one
reads the structure of each bar's price cells -- diagonal imbalance, absorption,
POC migration and delta divergence -- which is where footprint charts actually
carry information: totals say how much net aggression there was, cells say where
it happened and whether it was met by resting size.

The idea, in one line: buy when aggressive buyers are stacked at the offer, the
bar's net delta is unusually positive, and any heavy flat-delta cell (absorption)
is on the bid side -- value is being accepted higher. Mirror for shorts.

Every input is a *finished* bar's footprint, so signals are evaluated at that
bar's close and the broker fills on the next bar's open. Nothing here reads a
future cell, a future bar's POC, or the current bar's unfinished volume.

Validation so far, honestly: on one day of BTC-USDT perp (2024-01-02, 1-min
bars, binance+bybit+okx) the conditions fire often but the edge is not there yet
-- buy signals showed a mean forward 5-bar return of about -1.1 bp and sell
signals about +1.7 bp, i.e. on that sample "follow the aggression" mildly
*reverts*, and the P&L was dominated by commission. One day proves nothing
either way, and the side conventions were cross-checked against the venues'
own files, so the reading is left as designed: this is a starting point to be
optimised and walk-forward-validated over months of archives, not a verdict.
"""
from __future__ import annotations

import numpy as np

from backtester import Bar, BaseStrategy


class FootprintImbalanceStrategy(BaseStrategy):
    """Diagonal imbalance + absorption + delta z-score, with ATR stops."""

    params = {
        # rolling window for the delta z-score and relative volume
        "window": 20,
        # a side needs this many imbalanced price levels inside the bar
        "imbalance_levels": 2,
        # minimum delta z-score in the trade's direction
        "delta_z": 1.0,
        # 1 = require an absorption signal on the right side (stricter, fewer trades)
        "require_absorption": 0,
        # 1 = require delta divergence (new low with net buying / new high with selling)
        "require_divergence": 0,
        # bar volume must be at least this multiple of its rolling mean (0 = off)
        "min_relative_volume": 0.0,
        # exit when delta z-score falls below this (a negative number)
        "exit_z": -0.5,
        # ATR risk management
        "atr_window": 14,
        "atr_stop": 2.0,
        "atr_take": 3.0,
    }

    #: cell features this strategy cannot run without
    required_features = (
        "delta", "delta_zscore", "buy_imbalance_levels", "sell_imbalance_levels",
        "imbalance_stack", "absorption_signal", "delta_divergence", "poc_price",
    )

    def init(self) -> None:
        missing = [name for name in self.required_features
                   if name not in self.full_features.columns]
        if missing:
            raise ValueError(
                "FootprintImbalanceStrategy needs footprint features (missing: "
                + ", ".join(missing)
                + "). Load a footprint Parquet in the data panel, or pass "
                "market_features=bar_features_from_footprint(footprint) to the engine."
            )
        features = self.full_features
        self._delta_z = features["delta_zscore"].to_numpy(dtype=float)
        self._delta = features["delta"].to_numpy(dtype=float)
        # Bars with no footprint row join as NaN. For counts and signals that
        # means "nothing observed", which is a real 0 -- but a missing delta
        # z-score must stay NaN so `next` refuses to trade on it.
        self._buy_levels = np.nan_to_num(features["buy_imbalance_levels"].to_numpy(dtype=float))
        self._sell_levels = np.nan_to_num(features["sell_imbalance_levels"].to_numpy(dtype=float))
        self._stack = np.nan_to_num(features["imbalance_stack"].to_numpy(dtype=float))
        self._absorption = np.nan_to_num(features["absorption_signal"].to_numpy(dtype=float))
        self._divergence = np.nan_to_num(features["delta_divergence"].to_numpy(dtype=float))
        self._poc = features["poc_price"].to_numpy(dtype=float)

        volume = features.get("combined_total_qty")
        if volume is None:
            volume = features.get("cell_volume")
        if volume is None:
            volume = np.full(len(features), np.nan)
        volume = np.asarray(volume, dtype=float)
        window = max(2, int(self.p["window"]))
        mean = np.full(len(volume), np.nan)
        for i in range(window - 1, len(volume)):
            window_slice = volume[i - window + 1:i + 1]
            if np.isfinite(window_slice).any():
                mean[i] = np.nanmean(window_slice)
        with np.errstate(invalid="ignore", divide="ignore"):
            self._relative_volume = volume / mean

        self._atr = self._average_true_range(int(self.p["atr_window"]))

        # The POC is a price, so it belongs on the price chart next to the candles.
        self.plot_line("POC (footprint)", self._poc, color="#f5b041", overlay=True)

    def _average_true_range(self, window: int) -> np.ndarray:
        """Wilder-free simple ATR; only past bars are ever read."""
        high = self.full_high
        low = self.full_low
        close = self.full_close
        count = len(high)
        true_range = np.full(count, np.nan)
        if count == 0:
            return true_range
        true_range[0] = high[0] - low[0]
        if count > 1:
            previous_close = close[:-1]
            true_range[1:] = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - previous_close),
                           np.abs(low[1:] - previous_close)),
            )
        atr = np.full(count, np.nan)
        for i in range(count):
            start = max(0, i - window + 1)
            chunk = true_range[start:i + 1]
            if np.isfinite(chunk).any():
                atr[i] = np.nanmean(chunk)
        return atr

    def next(self, bar: Bar) -> None:
        i = bar.index
        delta_z = self._delta_z[i]
        if not np.isfinite(delta_z):
            return

        required_levels = float(self.p["imbalance_levels"])
        delta_z_threshold = float(self.p["delta_z"])
        relative_volume = self._relative_volume[i]
        min_relative_volume = float(self.p["min_relative_volume"])
        if min_relative_volume > 0 and (
            not np.isfinite(relative_volume) or relative_volume < min_relative_volume
        ):
            return

        absorption = self._absorption[i]
        divergence = self._divergence[i]
        # A cell that absorbed the other side must never be traded against: with
        # the flag off it is only a veto, with it on it becomes a requirement.
        if self.p["require_absorption"]:
            long_side_ok = absorption > 0
            short_side_ok = absorption < 0
        else:
            long_side_ok = absorption >= 0
            short_side_ok = absorption <= 0
        if self.p["require_divergence"]:
            long_side_ok = long_side_ok and divergence > 0
            short_side_ok = short_side_ok and divergence < 0

        # Stacked imbalance alone is enough only when it is deep (>= 2x the
        # per-level requirement); otherwise the two counts must both agree.
        buy_levels = self._buy_levels[i]
        sell_levels = self._sell_levels[i]
        stack = self._stack[i]
        buyers_own_the_bar = (
            (buy_levels >= required_levels and buy_levels > sell_levels)
            or stack >= 2 * required_levels
        )
        sellers_own_the_bar = (
            (sell_levels >= required_levels and sell_levels > buy_levels)
            or stack <= -2 * required_levels
        )

        position = self.position
        if position is None:
            if buyers_own_the_bar and delta_z >= delta_z_threshold and long_side_ok:
                self._enter("buy", bar, "footprint_imbalance_buy")
            elif sellers_own_the_bar and delta_z <= -delta_z_threshold and short_side_ok:
                self._enter("sell", bar, "footprint_imbalance_sell")
            return

        exit_z = float(self.p["exit_z"])
        if position.is_long and delta_z <= exit_z:
            self.close_position(tag="footprint_delta_flip")
        elif not position.is_long and delta_z >= -exit_z:
            self.close_position(tag="footprint_delta_flip")

    def _enter(self, side: str, bar: Bar, tag: str) -> None:
        """Enter with ATR-based protective levels, sized by risk."""
        atr = self._atr[bar.index]
        stop_loss = take_profit = None
        if np.isfinite(atr) and atr > 0:
            stop_distance = float(self.p["atr_stop"]) * atr
            take_distance = float(self.p["atr_take"]) * atr
            if side == "buy":
                stop_loss = bar.close - stop_distance
                take_profit = bar.close + take_distance
            else:
                stop_loss = bar.close + stop_distance
                take_profit = bar.close - take_distance
        order = self.buy if side == "buy" else self.sell
        order(stop_loss=stop_loss, take_profit=take_profit, tag=tag)


__all__ = ["FootprintImbalanceStrategy"]
