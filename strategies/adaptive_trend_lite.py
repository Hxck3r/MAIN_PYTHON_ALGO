"""Стратегия на основе индикатора TA Adaptive Trend Lite (пересечение EMA)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.fast_engine import SignalStrategy


class TAAdaptiveTrendLite(SignalStrategy):
    """
    Стратегия TA Adaptive Trend Lite.
    
    Вход в лонг при пересечении быстрой EMA снизу вверх медленной EMA.
    Вход в шорт при пересечении быстрой EMA сверху вниз медленной EMA.
    Параметры по умолчанию соответствуют оригиналу из Pine Script (21 и 55).
    """

    params = {
        "fast_len": 21,
        "slow_len": 55,
    }

    def compute_signals(self) -> None:
        fast_len = int(self.p["fast_len"])
        slow_len = int(self.p["slow_len"])
        
        if fast_len >= slow_len:
            raise ValueError("fast_len должен быть меньше slow_len")

        prices = self.full_close
        close_series = pd.Series(prices)
        
        # Расчёт экспоненциальных скользящих средних (EMA)
        # В Pandas ewm(span=...) использует ту же формулу, что и ta.ema в Pine Script
        fast_ema = close_series.ewm(span=fast_len, adjust=False).mean().to_numpy()
        slow_ema = close_series.ewm(span=slow_len, adjust=False).mean().to_numpy()

        # Выделяем сдвиги для выявления пересечений (причинно, без lookahead)
        prev_fast, prev_slow = fast_ema[:-1], slow_ema[:-1]
        cur_fast, cur_slow = fast_ema[1:], slow_ema[1:]
        
        # Проверяем прогрев (медленная EMA не должна быть NaN)
        valid = ~np.isnan(prev_slow)
        
        # Поиск пересечений (ta.crossover / ta.crossunder)
        crossed_up = valid & (prev_fast <= prev_slow) & (cur_fast > cur_slow)
        crossed_down = valid & (prev_fast >= prev_slow) & (cur_fast < cur_slow)

        # Индексы сигналов (+1, так как срезы срезали первый элемент ряда)
        up_idx = np.flatnonzero(crossed_up) + 1
        down_idx = np.flatnonzero(crossed_down) + 1
        
        # Заполнение сигнального массива: +1 лонг, -1 шорт
        # Встречный сигнал автоматически закроет текущую позицию (неттинг)
        self._sig[up_idx] = 1
        self._sig[down_idx] = -1