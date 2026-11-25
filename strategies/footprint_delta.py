"""Example event-driven strategy consuming causal footprint features."""
from __future__ import annotations

import numpy as np

from backtester import Bar, BaseStrategy


class FootprintDeltaStrategy(BaseStrategy):
    """Trade when rolling combined delta crosses a configurable threshold."""

    params = {"threshold": 0.0}

    def init(self) -> None:
        required = ("rolling_delta", "volume_imbalance")
        missing = [name for name in required if name not in self.full_features]
        if missing:
            raise ValueError(
                "FootprintDeltaStrategy requires feature columns: " + ", ".join(missing)
            )
        self.rolling_delta = self.full_features["rolling_delta"].to_numpy(dtype=float)
        self.imbalance = self.full_features["volume_imbalance"].to_numpy(dtype=float)
        self.plot_line("Rolling delta", self.rolling_delta, color="#48c9b0", overlay=False)

    def next(self, bar: Bar) -> None:
        delta = self.rolling_delta[bar.index]
        imbalance = self.imbalance[bar.index]
        if not np.isfinite(delta) or not np.isfinite(imbalance):
            return
        threshold = float(self.p["threshold"])
        if delta > threshold and imbalance > 0 and (self.position is None or not self.position.is_long):
            self.buy(tag="footprint_delta_buy")
        elif delta < -threshold and imbalance < 0 and (self.position is None or self.position.is_long):
            self.sell(tag="footprint_delta_sell")
