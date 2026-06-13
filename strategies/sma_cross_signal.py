"""
SMA-кроссовер в виде сигнальной стратегии (для быстрого numba-движка).

Логика идентична SmaCrossStrategy, но выражена массивами сигналов:
поэтому класс исполняется как событийным BacktestEngine (через
универсальный SignalStrategy.next), так и быстрым run_fast_backtest --
результаты совпадают.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.fast_engine import SignalStrategy


class SmaCrossSignal(SignalStrategy):
    """Классический кроссовер SMA с защитными уровнями (сигнальная форма)."""

    params = {
        "fast_period": 10,
        "slow_period": 30,
        "sl_pct": 0.02,
        "tp_pct": 0.04,
    }

    def compute_signals(self) -> None:
        fast = int(self.p["fast_period"])
        slow = int(self.p["slow_period"])
        if fast >= slow:
            raise ValueError("fast_period должен быть меньше slow_period")

        sl_pct = float(self.p["sl_pct"])
        tp_pct = float(self.p["tp_pct"])
        prices = self.full_close
        close = pd.Series(prices)
        fast_sma = close.rolling(fast).mean().to_numpy()
        slow_sma = close.rolling(slow).mean().to_numpy()

        # Векторное пересечение: prev <= / current > (и наоборот). Сдвиг массивов
        # вместо Python-цикла -- индикаторы NaN на прогреве дают False в сравнении.
        prev_f, prev_s = fast_sma[:-1], slow_sma[:-1]
        cur_f, cur_s = fast_sma[1:], slow_sma[1:]
        valid = ~np.isnan(prev_s)
        crossed_up = valid & (prev_f <= prev_s) & (cur_f > cur_s)
        crossed_down = valid & (prev_f >= prev_s) & (cur_f < cur_s)

        up = np.flatnonzero(crossed_up) + 1      # +1: сигнал на текущем баре i
        down = np.flatnonzero(crossed_down) + 1
        self._sig[up] = 1
        self._sl[up] = prices[up] * (1.0 - sl_pct)
        self._tp[up] = prices[up] * (1.0 + tp_pct)
        self._sig[down] = -1
        self._sl[down] = prices[down] * (1.0 + sl_pct)
        self._tp[down] = prices[down] * (1.0 - tp_pct)
