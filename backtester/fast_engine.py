"""
Быстрый бэктестер сигнальных стратегий на Numba JIT.

Событийное ядро (backtest_engine.BacktestEngine) универсально, но платит
за гибкость: на каждом баре создаётся Python-объект Bar, вызывается
strategy.next(), ордера проходят через объектную модель брокера. Для самого
частого класса стратегий -- «сигнальных» (одна неттинг-позиция, рыночный
вход на открытии следующего бара, защитные SL/TP) -- весь прогон можно
выразить как цикл по numpy-массивам и скомпилировать его Numba. Это даёт
десятки крат ускорения, что критично для Grid Search / WFA на слабом CPU.

Стратегия описывает себя массивами длины n (заполняются в compute_signals):
    sig[i]   -- решение на ЗАКРЫТИИ бара i: +1 встать в лонг, -1 в шорт, 0 -- ничего
                (исполняется рыночно на ОТКРЫТИИ бара i+1);
    sl[i]/tp[i] -- абсолютные уровни Stop-Loss / Take-Profit ордера (NaN -- нет);
    size[i]  -- явный размер (NaN -- авто);
    risk[i]  -- сайзинг от риска, доля эквити под SL (NaN -- не используется).

Семантика исполнения ПОБИТОВО совпадает с Broker из backtest_engine:
тот же расчёт цены входа/выхода (спред/слиппедж), приоритет SL над TP
внутри бара, гэпы через уровень, неттинг с разворотом, сайзинг от риска,
кап по плечу, принудительное закрытие в конце данных. Один и тот же класс
SignalStrategy исполняется и событийным движком (через next()), и быстрым
ядром -- результаты идентичны (см. tests/test_fast_engine.py).
"""
from __future__ import annotations

from typing import Any, Optional, Type

import numpy as np
import pandas as pd
from numba import njit

from .backtest_engine import (
    BacktestEngine,
    BacktestResult,
    ExitReason,
    OrderSide,
    Trade,
)
from .base_strategy import Bar, BaseStrategy

# Числовые коды причины выхода для kernel (Python-перечисление -- дорого в njit)
_REASON = {
    1: ExitReason.SIGNAL,
    2: ExitReason.STOP_LOSS,
    3: ExitReason.TAKE_PROFIT,
    4: ExitReason.END_OF_DATA,
}


# ---------------------------------------------------------------------- #
# Базовый класс сигнальной стратегии
# ---------------------------------------------------------------------- #
class SignalStrategy(BaseStrategy):
    """Стратегия, выражаемая массивами сигналов (для быстрого numba-движка).

    Наследник реализует compute_signals(), заполняя self._sig / _sl / _tp /
    _size / _risk (массивы длины n, уже созданы к моменту вызова). Класс
    одновременно совместим с событийным BacktestEngine -- next() здесь
    универсальный и читает те же массивы, поэтому оба движка дают один
    результат.
    """

    def init(self) -> None:
        n = len(self.full_close)
        self._sig = np.zeros(n, dtype=np.int64)
        self._sl = np.full(n, np.nan, dtype=np.float64)
        self._tp = np.full(n, np.nan, dtype=np.float64)
        self._size = np.full(n, np.nan, dtype=np.float64)
        self._risk = np.full(n, np.nan, dtype=np.float64)
        self.compute_signals()

    def compute_signals(self) -> None:
        """Заполняет массивы сигналов. Реализуется наследником."""
        raise NotImplementedError

    def next(self, bar: Bar) -> None:
        i = bar.index
        s = int(self._sig[i])
        if s == 0:
            return
        pos = self.position
        kw: dict[str, Any] = {}
        sl, tp = self._sl[i], self._tp[i]
        size, risk = self._size[i], self._risk[i]
        if not np.isnan(sl):
            kw["stop_loss"] = float(sl)
        if not np.isnan(tp):
            kw["take_profit"] = float(tp)
        if not np.isnan(risk):
            kw["risk_pct"] = float(risk)
        elif not np.isnan(size):
            kw["size"] = float(size)

        if s == 1 and (pos is None or not pos.is_long):
            self.buy(**kw)
        elif s == -1 and (pos is None or pos.is_long):
            self.sell(**kw)


# ---------------------------------------------------------------------- #
# Валидация сигналов и проверка причинности (анти-lookahead)
# ---------------------------------------------------------------------- #
def _validate_signal_arrays(strategy: "SignalStrategy") -> None:
    """Проверки сайзинга/SL для баров с сигналом -- зеркало Broker._submit.

    В быстром ядре NaN означает «не задано», поэтому ошибка в стратегии
    (risk_pct без stop_loss, размер <= 0, size вместе с risk_pct) иначе
    превратилась бы в молчаливо пропущенный ордер. Здесь она всплывает явно,
    как исключение в событийном движке.
    """
    where = strategy._sig != 0
    if not where.any():
        return
    sl = strategy._sl[where]
    size = strategy._size[where]
    risk = strategy._risk[where]
    has_size = ~np.isnan(size)
    has_risk = ~np.isnan(risk)

    if np.any(has_size & (size <= 0)):
        raise ValueError("Размер ордера должен быть положительным")
    if np.any(has_size & has_risk):
        raise ValueError("size и risk_pct взаимоисключающие")
    if np.any(has_risk & (risk <= 0)):
        raise ValueError("risk_pct должен быть положительным")
    if np.any(has_risk & np.isnan(sl)):
        raise ValueError(
            "Сайзинг от риска требует stop_loss: размер позиции "
            "вычисляется из дистанции до стопа"
        )


def _compute_signal_arrays(
    strategy_cls: Type["SignalStrategy"],
    data: pd.DataFrame,
    params: Optional[dict[str, Any]],
) -> dict[str, np.ndarray]:
    strategy = strategy_cls(**(params or {}))
    strategy._bind(None, data)
    strategy.init()
    return {
        "sig": strategy._sig, "sl": strategy._sl, "tp": strategy._tp,
        "size": strategy._size, "risk": strategy._risk,
    }


def assert_no_lookahead(
    strategy_cls: Type["SignalStrategy"],
    data: pd.DataFrame,
    params: Optional[dict[str, Any]] = None,
    prefixes: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> None:
    """Проверка причинности сигналов сигнальной стратегии.

    Сигнал на баре j обязан зависеть только от данных <= j: значит, он не
    должен меняться от того, есть ли в данных бары ПОСЛЕ j. Сигналы
    пересчитываются на префиксах data[:k] и сравниваются с тем же срезом
    сигналов полного прогона. Любое расхождение -- заглядывание в будущее
    (center=True, shift(-k), bfill, нормировка по всему ряду и т.п.).

    Это ловит то, что тест на эквивалентность движков поймать НЕ может:
    оба движка читают одни и те же предрассчитанные сигналы, поэтому их
    совпадение не доказывает причинность самих сигналов.

    prefixes: доли (0, 1] или абсолютные длины (>= 2) проверяемых префиксов.
    Бросает AssertionError с номером первого «протёкшего» бара.
    """
    if not issubclass(strategy_cls, SignalStrategy):
        raise TypeError("assert_no_lookahead применим только к SignalStrategy")
    full = _compute_signal_arrays(strategy_cls, data, params)
    n = len(data)
    for pref in prefixes:
        k = int(round(pref * n)) if 0 < pref <= 1 else int(pref)
        if k < 2 or k >= n:
            continue
        part = _compute_signal_arrays(strategy_cls, data.iloc[:k], params)
        for name, arr in part.items():
            ref = full[name][:k]
            if name == "sig":
                mism = arr != ref
            else:  # float-массивы: NaN на одинаковых местах считаем совпадением
                mism = ~((arr == ref) | (np.isnan(arr) & np.isnan(ref)))
            if mism.any():
                j = int(np.flatnonzero(mism)[0])
                raise AssertionError(
                    f"Заглядывание в будущее в {strategy_cls.__name__}: сигнал "
                    f"'{name}' на баре {j} меняется при добавлении баров после "
                    f"{k} (полный={full[name][j]!r}, префикс[:{k}]={arr[j]!r})"
                )


# ---------------------------------------------------------------------- #
# Numba-ядро: реплика Broker над массивами
# ---------------------------------------------------------------------- #
@njit(cache=True)
def _run_kernel(
    open_, high, low, close,
    sig, sl_arr, tp_arr, size_arr, risk_arr,
    cash0, commission_pct, commission_fixed, slippage_pct, spread_pct,
    max_leverage, has_leverage, default_size_pct, warmup,
):  # pragma: no cover -- JIT-компилируется, профилируется отдельно
    n = open_.shape[0]
    equity = np.empty(n, dtype=np.float64)

    t_entry_bar = np.empty(n, dtype=np.int64)
    t_exit_bar = np.empty(n, dtype=np.int64)
    t_side = np.empty(n, dtype=np.int64)
    t_size = np.empty(n, dtype=np.float64)
    t_entry_price = np.empty(n, dtype=np.float64)
    t_exit_price = np.empty(n, dtype=np.float64)
    t_commission = np.empty(n, dtype=np.float64)
    t_pnl = np.empty(n, dtype=np.float64)
    t_pnl_pct = np.empty(n, dtype=np.float64)
    t_reason = np.empty(n, dtype=np.int64)
    nt = 0

    cash = cash0
    pos_side = 0          # 0 нет / 1 лонг / -1 шорт
    pos_size = 0.0
    pos_entry = 0.0
    pos_sl = np.nan
    pos_tp = np.nan
    pos_entry_bar = 0
    pos_entry_comm = 0.0
    last_close = np.nan

    pend = 0
    pend_sl = np.nan
    pend_tp = np.nan
    pend_size = np.nan
    pend_risk = np.nan

    for j in range(n):
        o = open_[j]
        h = high[j]
        l = low[j]  # noqa: E741 — OHLC-конвенция o/h/l/c в numba-ядре
        c = close[j]

        # --- 1) исполнение отложенного рыночного ордера на открытии бара --- #
        if pend != 0:
            side = pend
            if side == 1:
                price = o * (1.0 + slippage_pct) * (1.0 + spread_pct)
            else:
                price = o * (1.0 - slippage_pct)

            if pos_side != 0 and not np.isnan(last_close):
                eq = cash + pos_side * (last_close - pos_entry) * pos_size
            else:
                eq = cash
            eq_pos = eq if eq > 0.0 else 0.0

            if not np.isnan(pend_size):
                size = pend_size
            elif not np.isnan(pend_risk):
                dist = side * (price - pend_sl)
                size = eq_pos * pend_risk / dist if dist > 0.0 else 0.0
            else:
                size = eq_pos * default_size_pct / price

            if has_leverage and price > 0.0:
                offset = 0.0
                if pos_side != 0:
                    offset = pos_size if pos_side != side else -pos_size
                cap = eq_pos * max_leverage / price + offset
                if cap < 0.0:
                    cap = 0.0
                if size > cap:
                    size = cap

            if size <= 0.0:
                pend = 0
            else:
                remaining = size
                # неттинг: встречный ордер сначала закрывает позицию
                if pos_side != 0 and pos_side != side:
                    close_size = pos_size if pos_size < remaining else remaining
                    exit_comm = abs(price * close_size) * commission_pct + commission_fixed
                    entry_comm_part = pos_entry_comm * (close_size / pos_size)
                    gross = pos_side * (price - pos_entry) * close_size
                    cash += gross - exit_comm
                    t_entry_bar[nt] = pos_entry_bar
                    t_exit_bar[nt] = j
                    t_side[nt] = pos_side
                    t_size[nt] = close_size
                    t_entry_price[nt] = pos_entry
                    t_exit_price[nt] = price
                    t_commission[nt] = exit_comm + entry_comm_part
                    t_pnl[nt] = gross - exit_comm - entry_comm_part
                    t_pnl_pct[nt] = pos_side * (price / pos_entry - 1.0) * 100.0
                    t_reason[nt] = 1
                    nt += 1
                    pos_size -= close_size
                    pos_entry_comm -= entry_comm_part
                    remaining -= close_size
                    if pos_size <= 1e-12:
                        pos_side = 0
                        pos_size = 0.0
                        pos_entry_comm = 0.0
                        pos_sl = np.nan
                        pos_tp = np.nan

                if remaining > 1e-12:
                    comm = abs(price * remaining) * commission_pct + commission_fixed
                    cash -= comm
                    if pos_side == 0:
                        pos_side = side
                        pos_size = remaining
                        pos_entry = price
                        pos_entry_bar = j
                        pos_entry_comm = comm
                        pos_sl = pend_sl
                        pos_tp = pend_tp
                    else:
                        total = pos_size + remaining
                        pos_entry = (pos_entry * pos_size + price * remaining) / total
                        pos_size = total
                        pos_entry_comm += comm
                        if not np.isnan(pend_sl):
                            pos_sl = pend_sl
                        if not np.isnan(pend_tp):
                            pos_tp = pend_tp
                pend = 0

        # --- 2) проверка SL/TP внутри бара (приоритет SL) ------------------ #
        if pos_side == 1:
            sl_hit = (not np.isnan(pos_sl)) and l <= pos_sl
            tp_hit = (not np.isnan(pos_tp)) and h >= pos_tp
            if sl_hit:
                raw = o if o < pos_sl else pos_sl
                price = raw * (1.0 - slippage_pct)
                reason = 2
            elif tp_hit:
                raw = o if o > pos_tp else pos_tp
                price = raw
                reason = 3
            else:
                reason = 0
        elif pos_side == -1:
            sl_hit = (not np.isnan(pos_sl)) and h >= pos_sl
            tp_hit = (not np.isnan(pos_tp)) and l <= pos_tp
            if sl_hit:
                raw = o if o > pos_sl else pos_sl
                price = raw * (1.0 + slippage_pct) * (1.0 + spread_pct)
                reason = 2
            elif tp_hit:
                raw = o if o < pos_tp else pos_tp
                price = raw * (1.0 + spread_pct)
                reason = 3
            else:
                reason = 0
        else:
            reason = 0

        if reason != 0:
            exit_comm = abs(price * pos_size) * commission_pct + commission_fixed
            gross = pos_side * (price - pos_entry) * pos_size
            cash += gross - exit_comm
            t_entry_bar[nt] = pos_entry_bar
            t_exit_bar[nt] = j
            t_side[nt] = pos_side
            t_size[nt] = pos_size
            t_entry_price[nt] = pos_entry
            t_exit_price[nt] = price
            t_commission[nt] = exit_comm + pos_entry_comm
            t_pnl[nt] = gross - exit_comm - pos_entry_comm
            t_pnl_pct[nt] = pos_side * (price / pos_entry - 1.0) * 100.0
            t_reason[nt] = reason
            nt += 1
            pos_side = 0
            pos_size = 0.0
            pos_entry_comm = 0.0
            pos_sl = np.nan
            pos_tp = np.nan

        # --- 3) последний бар: принудительное закрытие по close ------------ #
        if j == n - 1:
            if pos_side != 0:
                if pos_side == 1:
                    price = c
                else:
                    price = c * (1.0 + spread_pct)
                exit_comm = abs(price * pos_size) * commission_pct + commission_fixed
                gross = pos_side * (price - pos_entry) * pos_size
                cash += gross - exit_comm
                t_entry_bar[nt] = pos_entry_bar
                t_exit_bar[nt] = j
                t_side[nt] = pos_side
                t_size[nt] = pos_size
                t_entry_price[nt] = pos_entry
                t_exit_price[nt] = price
                t_commission[nt] = exit_comm + pos_entry_comm
                t_pnl[nt] = gross - exit_comm - pos_entry_comm
                t_pnl_pct[nt] = pos_side * (price / pos_entry - 1.0) * 100.0
                t_reason[nt] = 4
                nt += 1
                pos_side = 0
                pos_size = 0.0
                pos_entry_comm = 0.0
        # --- 4) решение стратегии на закрытии бара (исполнится на j+1) ----- #
        elif j >= warmup:
            s = sig[j]
            if s == 1 and pos_side != 1:
                pend = 1
                pend_sl = sl_arr[j]
                pend_tp = tp_arr[j]
                pend_size = size_arr[j]
                pend_risk = risk_arr[j]
            elif s == -1 and pos_side != -1:
                pend = -1
                pend_sl = sl_arr[j]
                pend_tp = tp_arr[j]
                pend_size = size_arr[j]
                pend_risk = risk_arr[j]
            else:
                pend = 0

        # --- 5) переоценка счёта по закрытию бара -------------------------- #
        if pos_side != 0:
            equity[j] = cash + pos_side * (c - pos_entry) * pos_size
        else:
            equity[j] = cash
        last_close = c

    return (
        equity,
        t_entry_bar[:nt], t_exit_bar[:nt], t_side[:nt], t_size[:nt],
        t_entry_price[:nt], t_exit_price[:nt], t_commission[:nt],
        t_pnl[:nt], t_pnl_pct[:nt], t_reason[:nt],
    )


# ---------------------------------------------------------------------- #
# Обёртка: запуск и сборка BacktestResult
# ---------------------------------------------------------------------- #
def run_fast_backtest(
    data: pd.DataFrame,
    strategy_cls: Type[SignalStrategy],
    strategy_params: Optional[dict[str, Any]] = None,
    cash: float = 10_000.0,
    commission_pct: float = 0.0,
    commission_fixed: float = 0.0,
    slippage_pct: float = 0.0,
    spread_pct: float = 0.0,
    max_leverage: Optional[float] = None,
    default_size_pct: float = 0.95,
    warmup_bars: int = 0,
) -> BacktestResult:
    """Прогон сигнальной стратегии через numba-ядро.

    Сигнатура и семантика совпадают с BacktestEngine; принимает только
    наследников SignalStrategy. Возвращает обычный BacktestResult, поэтому
    metrics / Monte-Carlo / визуализация работают без изменений.
    """
    if not issubclass(strategy_cls, SignalStrategy):
        raise TypeError(
            "Быстрый движок принимает только SignalStrategy "
            f"(получено {strategy_cls.__name__})"
        )
    # Те же проверки данных, что и у событийного движка (колонки OHLCV,
    # DatetimeIndex, сортировка): иначе ядро молча отработало бы на мусоре.
    BacktestEngine._validate_data(data)
    if warmup_bars < 0 or warmup_bars >= len(data):
        raise ValueError("warmup_bars должен быть в диапазоне [0, len(data))")

    strategy = strategy_cls(**(strategy_params or {}))
    strategy._bind(None, data)  # broker не нужен: next() в быстром пути не зовётся
    strategy.init()
    _validate_signal_arrays(strategy)

    index = data.index
    out = _run_kernel(
        data["open"].to_numpy(np.float64),
        data["high"].to_numpy(np.float64),
        data["low"].to_numpy(np.float64),
        data["close"].to_numpy(np.float64),
        strategy._sig, strategy._sl, strategy._tp, strategy._size, strategy._risk,
        float(cash), float(commission_pct), float(commission_fixed),
        float(slippage_pct), float(spread_pct),
        float(max_leverage) if max_leverage is not None else 0.0,
        max_leverage is not None,
        float(default_size_pct), int(warmup_bars),
    )
    (equity, entry_bar, exit_bar, side, size, entry_price, exit_price,
     commission, pnl, pnl_pct, reason) = out

    # Время сделок боксится пачкой через index.take(...).tolist(): поэлементный
    # index[i] боксит Timestamp десятками микросекунд (узкое место на 3.14),
    # bulk-вариант -- в ~20 раз быстрее при идентичном результате.
    entry_times = index.take(entry_bar).tolist()
    exit_times = index.take(exit_bar).tolist()
    trades: list[Trade] = []
    for k in range(len(entry_bar)):
        trades.append(Trade(
            id=k + 1,
            side=OrderSide.BUY if side[k] == 1 else OrderSide.SELL,
            size=float(size[k]),
            entry_price=float(entry_price[k]),
            exit_price=float(exit_price[k]),
            entry_time=entry_times[k],
            exit_time=exit_times[k],
            entry_bar=int(entry_bar[k]),
            exit_bar=int(exit_bar[k]),
            commission=float(commission[k]),
            pnl=float(pnl[k]),
            pnl_pct=float(pnl_pct[k]),
            exit_reason=_REASON[int(reason[k])],
        ))

    return BacktestResult(
        equity_curve=pd.Series(equity, index=index, name="equity"),
        trades=trades,
        orders=[],
        data=data,
        initial_cash=float(cash),
        strategy_name=strategy_cls.__name__,
        strategy_params=dict(strategy.p),
    )
