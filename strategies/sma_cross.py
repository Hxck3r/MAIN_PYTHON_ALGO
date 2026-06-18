"""
Демонстрационная стратегия: пересечение двух скользящих средних.

Логика (аналог типового советника MT5):
    * быстрая SMA пересекает медленную снизу вверх  -> покупка;
    * быстрая SMA пересекает медленную сверху вниз  -> продажа (разворот);
    * каждая позиция защищена Stop-Loss и Take-Profit в % от цены входа.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import Bar, BaseStrategy


class SmaCrossStrategy(BaseStrategy):
    """Классический кроссовер SMA с защитными уровнями."""

    params = {
        "fast_period": 10,
        "slow_period": 30,
        "sl_pct": 0.02,   # Stop-Loss: 2% от цены входа
        "tp_pct": 0.04,   # Take-Profit: 4% от цены входа
    }

    def init(self) -> None:
        fast = int(self.p["fast_period"])
        slow = int(self.p["slow_period"])
        if fast >= slow:
            raise ValueError("fast_period должен быть меньше slow_period")

        close = pd.Series(self.full_close)
        # rolling(i) использует только бары <= i -- заглядывания в будущее нет
        self.fast_sma: np.ndarray = close.rolling(fast).mean().to_numpy()
        self.slow_sma: np.ndarray = close.rolling(slow).mean().to_numpy()

        # Линии скользящих на графике (пересечения видны там, где входит бот)
        self.plot_line(f"SMA {fast}", self.fast_sma, color="#f5b041")
        self.plot_line(f"SMA {slow}", self.slow_sma, color="#5dade2")

    def next(self, bar: Bar) -> None:
        i = bar.index
        if i < 1 or np.isnan(self.slow_sma[i - 1]):
            return  # индикаторы ещё не прогрелись

        crossed_up = (
            self.fast_sma[i - 1] <= self.slow_sma[i - 1]
            and self.fast_sma[i] > self.slow_sma[i]
        )
        crossed_down = (
            self.fast_sma[i - 1] >= self.slow_sma[i - 1]
            and self.fast_sma[i] < self.slow_sma[i]
        )

        price = bar.close
        if crossed_up and (self.position is None or not self.position.is_long):
            self.buy(
                stop_loss=price * (1.0 - self.p["sl_pct"]),
                take_profit=price * (1.0 + self.p["tp_pct"]),
                tag="sma_cross_up",
            )
        elif crossed_down and (self.position is None or self.position.is_long):
            self.sell(
                stop_loss=price * (1.0 + self.p["sl_pct"]),
                take_profit=price * (1.0 - self.p["tp_pct"]),
                tag="sma_cross_down",
            )
