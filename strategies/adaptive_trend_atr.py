"""Событийная стратегия на основе TA Adaptive Trend Lite с ATR стопами и частичным закрытием."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import Bar, BaseStrategy


class TAAdaptiveTrendATR(BaseStrategy):
    """
    Стратегия TA Adaptive Trend Lite с интеграцией ATR.
    
    Поддерживает:
    - Стоп-лосс и Тейк-профит на основе ATR.
    - Опциональное частичное закрытие позиции (2 тейка по 25% объема).
    """

    params = {
        "fast_len": 21,
        "slow_len": 55,
        "atr_len": 14,
        "sl_atr_mult": 1.5,      # Множитель ATR для Стоп-Лосса
        "tp1_atr_mult": 1.5,     # Множитель ATR для Тейк-Профита 1 (25% объема)
        "tp2_atr_mult": 3.0,     # Множитель ATR для Тейк-Профита 2 (25% объема)
        "use_partial_close": True,  # Включение/отключение частичных закрытий
    }

    def init(self) -> None:
        fast_len = int(self.p["fast_len"])
        slow_len = int(self.p["slow_len"])
        atr_len = int(self.p["atr_len"])
        
        if fast_len >= slow_len:
            raise ValueError("fast_len должен быть меньше slow_len")

        # Предрасчет индикаторов по полным массивам (без lookahead bias)
        close_series = pd.Series(self.full_close)
        high_series = pd.Series(self.full_high)
        low_series = pd.Series(self.full_low)

        # 1. Рассчитываем EMA
        self.fast_ema = close_series.ewm(span=fast_len, adjust=False).mean().to_numpy()
        self.slow_ema = close_series.ewm(span=slow_len, adjust=False).mean().to_numpy()

        # 2. Рассчитываем True Range и ATR
        prev_close = close_series.shift(1)
        tr1 = high_series - low_series
        tr2 = (high_series - prev_close).abs()
        tr3 = (low_series - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Классический ATR через Wilder's MMA (или RMA), аналог ta.atr в Pine Script
        self.atr = tr.ewm(alpha=1.0 / atr_len, adjust=False).mean().to_numpy()

        # Линии быстрой/медленной EMA на графике (их пересечения = сигналы входа)
        self.plot_line(f"EMA {fast_len}", self.fast_ema, color="#f5b041")
        self.plot_line(f"EMA {slow_len}", self.slow_ema, color="#5dade2")

        # Переменные состояния для отслеживания частичных закрытий внутри одной позиции
        self.tp1_hit = False
        self.tp2_hit = False

    def next(self, bar: Bar) -> None:
        i = bar.index
        
        # Проверяем прогрев индикаторов
        if i < 1 or np.isnan(self.slow_ema[i]) or np.isnan(self.atr[i]):
            return

        # Логика пересечений на основе предыдущего и текущего баров
        crossed_up = (self.fast_ema[i - 1] <= self.slow_ema[i - 1] and self.fast_ema[i] > self.slow_ema[i])
        crossed_down = (self.fast_ema[i - 1] >= self.slow_ema[i - 1] and self.fast_ema[i] < self.slow_ema[i])

        price = bar.close
        current_atr = self.atr[i]

        # Если позиции нет, сбрасываем флаги частичных закрытий
        if self.position is None:
            self.tp1_hit = False
            self.tp2_hit = False

        # --- ЛОГИКА ДЛЯ ОТКРЫТОЙ ПОЗИЦИИ ---
        if self.position is not None:
            # Смена тренда — безусловный выход всем остатком (разворот или закрытие)
            if (self.position.is_long and crossed_down) or (not self.position.is_long and crossed_up):
                self.close_position(tag="trend_flip")
                # Флаги сбросятся в следующем баре, так как position станет None

            # Если включено частичное закрытие, проверяем достижение целей по ATR внутри бара
            elif self.p["use_partial_close"]:
                entry_price = self.position.entry_price
                current_size = self.position.size

                if self.position.is_long:
                    # Уровни для лонга
                    tp1_level = entry_price + current_atr * self.p["tp1_atr_mult"]
                    tp2_level = entry_price + current_atr * self.p["tp2_atr_mult"]

                    # Проверяем Тейк 1 (25% от первоначального объема)
                    if not self.tp1_hit and bar.high >= tp1_level:
                        # Так как мы не знаем первоначальный объем (он мог быть изменен автосайзингом),
                        # если мы закрываем первый кусок, то 25% от текущего полного объема — это 25%.
                        # Если позиция уменьшится, для TP2 расчет будет от остатка.
                        # Чтобы закрыть ровно 25% от старта: на первом шаге берем 25% от текущего объема.
                        close_size = current_size * 0.25
                        self.sell(size=close_size, tag="partial_tp1")
                        self.tp1_hit = True

                    # Проверяем Тейк 2 (еще 25% от первоначального объема, что равно ~33.3% от остатка 75%)
                    if self.tp1_hit and not self.tp2_hit and bar.high >= tp2_level:
                        # 25% от 100% первоначальных — это 1/3 от оставшихся 75%.
                        close_size = current_size * 0.3333
                        self.sell(size=close_size, tag="partial_tp2")
                        self.tp2_hit = True

                else:
                    # Уровни для шорта
                    tp1_level = entry_price - current_atr * self.p["tp1_atr_mult"]
                    tp2_level = entry_price - current_atr * self.p["tp2_atr_mult"]

                    if not self.tp1_hit and bar.low <= tp1_level:
                        close_size = current_size * 0.25
                        self.buy(size=close_size, tag="partial_tp1")
                        self.tp1_hit = True

                    if self.tp1_hit and not self.tp2_hit and bar.low <= tp2_level:
                        close_size = current_size * 0.3333
                        self.buy(size=close_size, tag="partial_tp2")
                        self.tp2_hit = True

        # --- ЛОГИКА ВХОДА В ПОЗИЦИЮ ---
        # Входим, только если позиции сейчас нет (чтобы не плодить ордера на каждом баре)
        if self.position is None:
            if crossed_up:
                # Рассчитываем жесткий Стоп-Лосс по ATR
                sl_level = price - current_atr * self.p["sl_atr_mult"]
                self.buy(
                    stop_loss=sl_level,
                    tag="entry_long"
                )
            elif crossed_down:
                sl_level = price + current_atr * self.p["sl_atr_mult"]
                self.sell(
                    stop_loss=sl_level,
                    tag="entry_short"
                )