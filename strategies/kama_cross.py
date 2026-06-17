"""
Стратегия: пересечение двух адаптивных скользящих Кауфмана (KAMA).

KAMA (Kaufman Adaptive Moving Average) подстраивает скорость сглаживания под
«эффективность» движения цены: на трендовом участке она быстрая (близка к цене),
в боковике -- медленная (отсекает шум). Это даёт меньше ложных пересечений,
чем у обычных SMA/EMA.

Логика:
    * быстрая KAMA пересекает медленную снизу вверх  -> вход в лонг;
    * быстрая KAMA пересекает медленную сверху вниз  -> вход в шорт
      (если шорты не отключены параметром ``allow_short``);
    * стоп-лосс выставляется на расстоянии ``atr_mult`` * ATR от цены входа.

Заглядывания в будущее нет: и KAMA, и ATR в точке i используют только бары <= i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import Bar, BaseStrategy


def _kama(close: np.ndarray, er_period: int, fast: int, slow: int) -> np.ndarray:
    """Адаптивная скользящая Кауфмана.

    er_period -- окно расчёта Efficiency Ratio;
    fast/slow -- периоды быстрой и медленной EMA, задающие границы скорости
    сглаживания.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    if n <= er_period:
        return kama

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    # change = |close[i] - close[i-er_period]| -- общее смещение за окно;
    # volatility = сумма |dclose| за окно -- суммарный путь цены.
    diff = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(diff).rolling(er_period).sum().to_numpy()
    change = np.abs(close - np.roll(close, er_period))

    # Инициализируем KAMA ценой на первом «прогретом» баре.
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        vol = volatility[i]
        er = change[i] / vol if vol != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    return kama


class KamaCrossStrategy(BaseStrategy):
    """Кроссовер двух KAMA со стопом по ATR и опциональным отключением шортов."""

    params = {
        "fast_er": 10,       # окно Efficiency Ratio быстрой KAMA
        "slow_er": 30,       # окно Efficiency Ratio медленной KAMA
        "fast_sc": 2,        # период быстрой EMA (граница скорости)
        "slow_sc": 30,       # период медленной EMA (граница скорости)
        "atr_period": 14,    # окно ATR для стопа
        "atr_mult": 3.0,     # стоп = atr_mult * ATR от цены входа
        "allow_short": True, # False -> входы в шорт отключены (только лонг)
    }

    def init(self) -> None:
        fast_er = int(self.p["fast_er"])
        slow_er = int(self.p["slow_er"])
        if fast_er >= slow_er:
            raise ValueError("fast_er должен быть меньше slow_er")

        close = self.full_close
        self.fast_kama: np.ndarray = _kama(
            close, fast_er, int(self.p["fast_sc"]), int(self.p["slow_sc"])
        )
        self.slow_kama: np.ndarray = _kama(
            close, slow_er, int(self.p["fast_sc"]), int(self.p["slow_sc"])
        )

        # ATR Уайлдера: EMA истинного диапазона (alpha = 1/period).
        high = pd.Series(self.full_high)
        low = pd.Series(self.full_low)
        prev_close = pd.Series(close).shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        period = int(self.p["atr_period"])
        self.atr: np.ndarray = tr.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()

    def next(self, bar: Bar) -> None:
        i = bar.index
        if i < 1 or np.isnan(self.slow_kama[i - 1]) or np.isnan(self.atr[i]):
            return  # индикаторы ещё не прогрелись

        crossed_up = (
            self.fast_kama[i - 1] <= self.slow_kama[i - 1]
            and self.fast_kama[i] > self.slow_kama[i]
        )
        crossed_down = (
            self.fast_kama[i - 1] >= self.slow_kama[i - 1]
            and self.fast_kama[i] < self.slow_kama[i]
        )

        price = bar.close
        stop_dist = self.p["atr_mult"] * self.atr[i]

        if crossed_up and (self.position is None or not self.position.is_long):
            self.buy(stop_loss=price - stop_dist, tag="kama_cross_up")
        elif crossed_down:
            if not self.p["allow_short"]:
                # Шорты отключены: пересечение вниз лишь закрывает лонг.
                if self.position is not None and self.position.is_long:
                    self.close_position(tag="kama_exit_long")
            elif self.position is None or self.position.is_long:
                self.sell(stop_loss=price + stop_dist, tag="kama_cross_down")
