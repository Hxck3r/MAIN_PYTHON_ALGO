"""
KAMA Strategy — порт советника KAMA_Strategy_Bot_v2.mq5 на движок бэктестера.

Логика (точный перенос MQL5-версии, RSI-фильтр исключён по требованию):
    * Сигнал входа по дельте KAMA, превысившей порог EntryFilter * StdDev(дельт);
    * выходы: разворот KAMA, ATR Trailing Stop (программный), PSAR Stop;
    * аварийный физический Stop-Loss = EmergencySL * ATR от цены входа;
    * фильтры: сессионный (часы), волатильный (ATR/avgATR), HTF KAMA (старший ТФ);
    * частичное закрытие в 2 уровня по доле от начального объёма (цели в ATR);
    * двусторонняя торговля либо только LONG.

Соответствие индексов: в MQL5 стратегия работает на открытии нового бара и
читает kama[1] (последний ЗАКРЫТЫЙ бар), kama[2] и т.д. В этом движке next(bar i)
вызывается на закрытии бара i, а ордер исполняется на открытии i+1 — поэтому
аналог MQL5 kama[1] == kama[i], kama[2] == kama[i-1] (заглядывания в будущее нет).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import Bar, BaseStrategy


# ====================================================================== #
#  Индикаторы (повторяют поведение iAMA / iATR / iSAR из MetaTrader 5)
# ====================================================================== #
def kama(price: np.ndarray, period: int, fast: int, slow: int) -> np.ndarray:
    """Kaufman Adaptive Moving Average (аналог iAMA, PRICE_CLOSE).

    ER = |p[i]-p[i-period]| / Σ|p[k]-p[k-1]|; SC = (ER*(fastSC-slowSC)+slowSC)^2;
    AMA[i] = AMA[i-1] + SC * (p[i] - AMA[i-1]). Первое значение — seed = цена.
    """
    n = len(price)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    # Кумулятивная сумма |p[k]-p[k-1]|: волатильность окна за O(1) вместо
    # пересуммирования окна на каждом баре (O(n*period) -> O(n))
    cum_abs_diff = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(price)))))
    out[period - 1] = price[period - 1]  # затравочное значение
    for i in range(period, n):
        change = abs(price[i] - price[i - period])
        volatility = cum_abs_diff[i] - cum_abs_diff[i - period]
        er = change / volatility if volatility != 0.0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (price[i] - out[i - 1])
    return out


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int) -> np.ndarray:
    """Average True Range со сглаживанием Уайлдера (аналог iATR)."""
    n = len(high)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])
    out[period] = np.mean(tr[1 : period + 1])  # первое значение — SMA TR
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def parabolic_sar(high: np.ndarray, low: np.ndarray,
                  step: float, max_step: float) -> np.ndarray:
    """Parabolic SAR (аналог iSAR)."""
    n = len(high)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    bull = True
    af = step
    ep = high[0]
    sar = low[0]
    out[0] = sar
    for i in range(1, n):
        prior = sar
        if bull:
            sar = prior + af * (ep - prior)
            sar = min(sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar:
                bull = False
                sar = ep
                ep = low[i]
                af = step
        else:
            sar = prior + af * (ep - prior)
            sar = max(sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar:
                bull = True
                sar = ep
                ep = high[i]
                af = step
        if bull and high[i] > ep:
            ep = high[i]
            af = min(af + step, max_step)
        elif not bull and low[i] < ep:
            ep = low[i]
            af = min(af + step, max_step)
        out[i] = sar
    return out


# ====================================================================== #
#  Стратегия
# ====================================================================== #
class KamaStrategy(BaseStrategy):
    """Порт KAMA_Strategy_Bot_v2 (без RSI-фильтра)."""

    params = {
        # --- Основные настройки ---
        "position_size": 0.1,     # объём позиции в единицах актива (аналог лота)
        "long_only": True,        # True = только LONG; False = двусторонняя торговля

        # --- Выходы из позиции ---
        "use_kama_exit": True,    # выход по развороту KAMA
        "use_atr_trailing": False,  # ATR Trailing Stop (программный)
        "use_psar_exit": False,   # PSAR Stop Loss
        "emergency_sl_atr": 5.0,  # аварийный физический SL (множитель ATR, 0 = выкл)

        # --- Настройки KAMA ---
        "kama_length": 14,
        "kama_fast": 2,
        "kama_slow": 20,

        # --- KAMA фильтр входа/выхода ---
        "entry_filter": 1.5,      # множитель StdDev для входа
        "exit_filter": 0.5,       # множитель StdDev для выхода

        # --- ATR Trailing Stop ---
        "atr_period": 14,
        "atr_multiplier": 3.0,
        "atr_trigger": "wick",    # "wick" (по тени) или "close" (по закрытию)

        # --- PSAR Stop ---
        "psar_start": 0.02,
        "psar_max": 0.2,

        # --- HTF фильтр (старший ТФ) ---
        "use_htf_filter": True,
        "htf_rule": "1h",         # правило ресемпла pandas (аналог PERIOD_H1)

        # --- Волатильный фильтр ---
        "use_vol_filter": False,
        "vol_atr_avg_period": 20,
        "vol_min_mult": 0.5,
        "vol_max_mult": 3.0,

        # --- Сессионный фильтр ---
        "use_time_filter": True,
        "session_start": 8,       # час начала торговли (по времени баров)
        "session_end": 20,        # час конца торговли

        # --- Частичное закрытие (2 уровня) ---
        "use_partial_close": False,
        "partial_close_ratio": 0.25,  # доля от начального объёма на каждый уровень
        "partial_mult1": 1.0,         # ATR-множитель цели 1-го уровня
        "partial_mult2": 2.0,         # ATR-множитель цели 2-го уровня
    }

    # ------------------------------------------------------------------ #
    #  Инициализация: предрасчёт всех индикаторов и фильтров
    # ------------------------------------------------------------------ #
    def init(self) -> None:
        length = int(self.p["kama_length"])
        fast = int(self.p["kama_fast"])
        slow = int(self.p["kama_slow"])

        close = self.full_close
        high = self.full_high
        low = self.full_low

        # --- Базовые индикаторы ---
        self.kama = kama(close, length, fast, slow)
        self.atr = atr_wilder(high, low, close, int(self.p["atr_period"]))
        self.psar = parabolic_sar(
            high, low, float(self.p["psar_start"]), float(self.p["psar_max"])
        )

        # --- Дельта KAMA и пороги входа/выхода (StdDev дельт за length баров) ---
        kdelta = pd.Series(self.kama).diff()          # kama[i] - kama[i-1]
        std = kdelta.rolling(length).std(ddof=0)      # популяционный StdDev
        self.kdelta = kdelta.to_numpy()
        self.entry_maf = (float(self.p["entry_filter"]) * std).to_numpy()
        self.exit_maf = (float(self.p["exit_filter"]) * std).to_numpy()

        # --- Среднее ATR для волатильного фильтра ---
        vol_period = int(self.p["vol_atr_avg_period"])
        self.atr_avg = pd.Series(self.atr).rolling(vol_period).mean().to_numpy()

        # --- HTF KAMA bias (1 = бычий, -1 = медвежий, 0 = нейтр./нет данных) ---
        self.htf_bias = self._compute_htf_bias(length, fast, slow)

        # --- Состояние управления позицией ---
        self._reset_state()

    def _reset_state(self) -> None:
        """Сброс трейлинга, флагов частичного закрытия и начального объёма."""
        self.trail_long = 0.0
        self.trail_short = 0.0
        self.partial1 = False     # 1-й уровень частичного закрытия пройден
        self.partial2 = False     # 2-й уровень частичного закрытия пройден
        self.init_vol = 0.0       # начальный объём позиции (для долей)

    def _compute_htf_bias(self, length: int, fast: int, slow: int) -> np.ndarray:
        """Маппинг направления KAMA старшего ТФ на бары базового ТФ.

        HTF-бар становится доступен только после своего закрытия (avail = старт +
        период), поэтому merge_asof(backward) исключает заглядывание в будущее.
        Соответствует MQL5: khtf[1] (последний закрытый HTF-бар) vs khtf[2].
        """
        n = len(self.full_close)
        if not self.p["use_htf_filter"]:
            return np.zeros(n)

        rule = str(self.p["htf_rule"])
        df = self.data
        htf_close = (
            df["close"].resample(rule, label="left", closed="left").last().dropna()
        )
        if len(htf_close) <= length:
            return np.zeros(n)

        khtf = kama(htf_close.to_numpy(), length, fast, slow)
        htf_delta = np.diff(khtf, prepend=np.nan)     # khtf[j] - khtf[j-1]
        offset = pd.tseries.frequencies.to_offset(rule)
        avail = htf_close.index + offset              # момент закрытия HTF-бара

        htf_df = (
            pd.DataFrame({"avail": avail, "bias": np.sign(htf_delta)})
            .dropna()
            .sort_values("avail")
        )
        base = pd.DataFrame({"ts": df.index})
        merged = pd.merge_asof(
            base, htf_df, left_on="ts", right_on="avail", direction="backward"
        )
        # NaN (нет закрытого HTF-бара) трактуем как 0 — фильтр не ограничивает
        return np.nan_to_num(merged["bias"].to_numpy(), nan=0.0)

    # ------------------------------------------------------------------ #
    #  Фильтры
    # ------------------------------------------------------------------ #
    def _in_session(self, bar: Bar) -> bool:
        if not self.p["use_time_filter"]:
            return True
        hour = bar.datetime.hour
        return self.p["session_start"] <= hour < self.p["session_end"]

    def _volatility_ok(self, i: int) -> bool:
        if not self.p["use_vol_filter"]:
            return True
        avg = self.atr_avg[i]
        if not np.isfinite(avg) or avg <= 0:
            return False
        ratio = self.atr[i] / avg
        return self.p["vol_min_mult"] <= ratio <= self.p["vol_max_mult"]

    # ------------------------------------------------------------------ #
    #  Торговая логика
    # ------------------------------------------------------------------ #
    def next(self, bar: Bar) -> None:
        i = bar.index
        if i < 1:
            return
        # Прогрев индикаторов: нужны дельта KAMA, порог и ATR
        if (
            not np.isfinite(self.kdelta[i])
            or not np.isfinite(self.entry_maf[i])
            or not np.isfinite(self.atr[i])
        ):
            return

        delta = self.kdelta[i]            # аналог kama1delta = kama[1]-kama[2]
        entry_maf = self.entry_maf[i]
        exit_maf = self.exit_maf[i]
        atr = self.atr[i]
        psar = self.psar[i]
        close = self._close[i]
        high = self._high[i]
        low = self._low[i]

        # Программный ATR trailing: уровни от текущего закрытого бара
        atr_long = low - atr * self.p["atr_multiplier"]
        atr_short = high + atr * self.p["atr_multiplier"]

        pos = self.position

        # Позиции нет — сбрасываем состояние и переходим к сигналам входа
        if pos is None:
            self._reset_state()
            self._try_entry(bar, delta, entry_maf, atr, close, atr_long, atr_short)
            return

        # Восстановление начального объёма (например, после рестарта движка)
        if self.init_vol <= 0.0:
            self.init_vol = pos.size

        if pos.is_long:
            self._manage_long(bar, pos, delta, exit_maf, atr, psar,
                              close, low, atr_long)
        else:
            self._manage_short(bar, pos, delta, exit_maf, atr, psar,
                               close, high, atr_short)

    # ------------------------------------------------------------------ #
    #  Управление LONG
    # ------------------------------------------------------------------ #
    def _manage_long(self, bar, pos, delta, exit_maf, atr, psar,
                     close, low, atr_long) -> None:
        # Обновляем трейлинг (двигается только вверх)
        if self.p["use_atr_trailing"]:
            if self.trail_long == 0.0 or atr_long > self.trail_long:
                self.trail_long = atr_long

        # --- Сигналы закрытия ---
        close_long = False
        if self.p["use_psar_exit"] and close < psar:
            close_long = True
        if self.p["use_kama_exit"] and delta < 0 and abs(delta) > exit_maf:
            close_long = True
        if self.p["use_atr_trailing"] and self.trail_long > 0.0:
            check = close if self.p["atr_trigger"] == "close" else low
            if check <= self.trail_long:
                close_long = True

        if close_long:
            self.close_position(tag="kama_exit_long")
            return

        # --- Частичное закрытие (2 уровня по доле от начального объёма) ---
        if self.p["use_partial_close"] and self.init_vol > 0.0:
            part = min(self.init_vol * self.p["partial_close_ratio"], pos.size)
            if not self.partial1 and part > 0 and \
                    close >= pos.entry_price + self.p["partial_mult1"] * atr:
                self.sell(size=part, tag="partial_long_1")
                self.partial1 = True
            elif self.partial1 and not self.partial2 and part > 0 and \
                    close >= pos.entry_price + self.p["partial_mult2"] * atr:
                part = min(part, pos.size)
                if part > 0:
                    self.sell(size=part, tag="partial_long_2")
                    self.partial2 = True

    # ------------------------------------------------------------------ #
    #  Управление SHORT
    # ------------------------------------------------------------------ #
    def _manage_short(self, bar, pos, delta, exit_maf, atr, psar,
                      close, high, atr_short) -> None:
        if self.p["use_atr_trailing"]:
            if self.trail_short == 0.0 or atr_short < self.trail_short:
                self.trail_short = atr_short

        close_short = False
        if self.p["use_psar_exit"] and close > psar:
            close_short = True
        if self.p["use_kama_exit"] and delta > 0 and delta > exit_maf:
            close_short = True
        if self.p["use_atr_trailing"] and self.trail_short > 0.0:
            check = close if self.p["atr_trigger"] == "close" else high
            if check >= self.trail_short:
                close_short = True

        if close_short:
            self.close_position(tag="kama_exit_short")
            return

        if self.p["use_partial_close"] and self.init_vol > 0.0:
            part = min(self.init_vol * self.p["partial_close_ratio"], pos.size)
            if not self.partial1 and part > 0 and \
                    close <= pos.entry_price - self.p["partial_mult1"] * atr:
                self.buy(size=part, tag="partial_short_1")
                self.partial1 = True
            elif self.partial1 and not self.partial2 and part > 0 and \
                    close <= pos.entry_price - self.p["partial_mult2"] * atr:
                part = min(part, pos.size)
                if part > 0:
                    self.buy(size=part, tag="partial_short_2")
                    self.partial2 = True

    # ------------------------------------------------------------------ #
    #  Сигналы входа
    # ------------------------------------------------------------------ #
    def _try_entry(self, bar, delta, entry_maf, atr, close,
                   atr_long, atr_short) -> None:
        i = bar.index
        if not self._in_session(bar) or not self._volatility_ok(i):
            return

        htf = self.htf_bias[i]
        size = float(self.p["position_size"])
        emer = float(self.p["emergency_sl_atr"])

        # --- LONG ---
        long_signal = delta > 0 and delta > entry_maf
        if self.p["use_htf_filter"] and htf == -1:
            long_signal = False
        if long_signal:
            sl = close - emer * atr if emer > 0 else None
            self.buy(size=size, stop_loss=sl, tag="kama_long")
            self.trail_long = atr_long if self.p["use_atr_trailing"] else 0.0
            self.init_vol = size
            self.partial1 = self.partial2 = False
            return

        # --- SHORT (только при двусторонней торговле) ---
        if self.p["long_only"]:
            return
        short_signal = delta < 0 and abs(delta) > entry_maf
        if self.p["use_htf_filter"] and htf == 1:
            short_signal = False
        if short_signal:
            sl = close + emer * atr if emer > 0 else None
            self.sell(size=size, stop_loss=sl, tag="kama_short")
            self.trail_short = atr_short if self.p["use_atr_trailing"] else 0.0
            self.init_vol = size
            self.partial1 = self.partial2 = False
