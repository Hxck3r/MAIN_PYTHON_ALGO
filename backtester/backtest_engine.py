"""
Ядро бэктестера: событийная (event-driven) симуляция «бар за баром».

Выбран событийный обход, а не векторизация: он позволяет писать произвольно
сложные стратегии (отложенные ордера, трейлинг, частичные закрытия в будущем)
и без изменений ядра наращивать Walk-Forward Analysis и Монте-Карло поверх
него -- они лишь многократно вызывают BacktestEngine.run() на разных срезах
данных / перестановках сделок.

Модель исполнения (без заглядывания в будущее):
    * Ордер, выставленный в strategy.next(bar i), исполняется не раньше
      бара i+1 (рыночный -- по цене открытия i+1 с проскальзыванием).
    * Лимитный ордер исполняется, когда цена бара пересекает limit_price.
    * Stop-Loss / Take-Profit проверяются внутри бара по high/low;
      если в одном баре задеты оба уровня -- консервативно считается SL.
    * Учёт позиции -- неттинг (одна агрегированная позиция, как в MT5):
      встречный ордер сначала закрывает текущую позицию, остаток
      открывает противоположную.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Type

import numpy as np
import pandas as pd

from .base_strategy import Bar, BaseStrategy


# ---------------------------------------------------------------------- #
# Типы и сущности
# ---------------------------------------------------------------------- #
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> OrderSide:
        return OrderSide.SELL if self is OrderSide.BUY else OrderSide.BUY


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ExitReason(Enum):
    SIGNAL = "signal"            # закрытие по сигналу стратегии / развороту
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    END_OF_DATA = "end_of_data"  # принудительное закрытие в конце теста


@dataclass
class Order:
    """Торговый ордер (Market или Limit) с опциональными SL/TP."""

    id: int
    side: OrderSide
    type: OrderType
    size: Optional[float]               # None -> автосайзинг при исполнении
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None  # стоп-ордер на вход (buy-stop/sell-stop)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_pct: Optional[float] = None    # сайзинг от риска: доля эквити под SL
    tag: str = ""
    status: OrderStatus = OrderStatus.PENDING
    created_bar: int = -1
    filled_bar: Optional[int] = None
    filled_price: Optional[float] = None

    @property
    def is_pending(self) -> bool:
        return self.status is OrderStatus.PENDING


@dataclass
class Position:
    """Открытая агрегированная позиция (неттинг, как в MT5)."""

    side: OrderSide
    size: float
    entry_price: float            # средневзвешенная цена входа
    entry_bar: int
    entry_time: pd.Timestamp
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_commission: float = 0.0
    tag: str = ""                 # тег ордера, открывшего позицию

    @property
    def is_long(self) -> bool:
        return self.side is OrderSide.BUY

    def unrealized_pnl(self, price: float) -> float:
        return self.side.value * (price - self.entry_price) * self.size


@dataclass
class Trade:
    """Завершённая сделка (раунд-трип): вход и выход."""

    id: int
    side: OrderSide
    size: float
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_bar: int
    exit_bar: int
    commission: float             # суммарная комиссия (вход + выход)
    pnl: float                    # чистая прибыль с учётом комиссий
    pnl_pct: float                # доходность сделки в % к цене входа
    exit_reason: ExitReason
    tag: str = ""

    @property
    def duration_bars(self) -> int:
        return self.exit_bar - self.entry_bar


# ---------------------------------------------------------------------- #
# Брокер
# ---------------------------------------------------------------------- #
class Broker:
    """Симулятор брокера: исполнение ордеров, комиссии, проскальзывание, SL/TP.

    Параметры
    ---------
    cash:             начальный капитал.
    commission_pct:   комиссия в долях от оборота за каждую сторону
                      (0.0005 = 0.05%).
    commission_fixed: фиксированная комиссия за исполнение ордера.
    slippage_pct:     проскальзывание в долях цены, всегда против трейдера
                      (применяется к рыночным и стоповым исполнениям).
    spread_pct:       спред bid/ask в долях цены. Данные считаются bid-ценами
                      (как графики MT5): любая ПОКУПКА (вход в лонг, выход из
                      шорта, включая SL/TP шорта) исполняется по ask =
                      price * (1 + spread_pct); продажи -- по цене графика.
    max_leverage:     максимальное плечо: ноционал позиции не может превышать
                      equity * max_leverage (None -- без ограничения,
                      как маржинальный счёт с бесконечным плечом).
                      Размер ордера, превышающий лимит, урезается; ордер,
                      на который маржи нет вовсе, отклоняется (REJECTED).
    default_size_pct: доля эквити для автосайзинга, когда size не задан.
    """

    def __init__(
        self,
        cash: float = 10_000.0,
        commission_pct: float = 0.0,
        commission_fixed: float = 0.0,
        slippage_pct: float = 0.0,
        spread_pct: float = 0.0,
        max_leverage: Optional[float] = None,
        default_size_pct: float = 0.95,
    ) -> None:
        if cash <= 0:
            raise ValueError("Начальный капитал должен быть положительным")
        if spread_pct < 0:
            raise ValueError("Спред не может быть отрицательным")
        if max_leverage is not None and max_leverage <= 0:
            raise ValueError("max_leverage должен быть положительным или None")
        self.initial_cash = cash
        self.cash = cash
        self.commission_pct = commission_pct
        self.commission_fixed = commission_fixed
        self.slippage_pct = slippage_pct
        self.spread_pct = spread_pct
        self.max_leverage = max_leverage
        self.default_size_pct = default_size_pct

        self.position: Optional[Position] = None
        self.orders: list[Order] = []          # история всех ордеров
        self.pending_orders: list[Order] = []
        self.trades: list[Trade] = []

        self._order_ids = itertools.count(1)
        self._trade_ids = itertools.count(1)
        self._current_bar: Optional[Bar] = None
        self._last_price: float = math.nan

    # ------------------------------------------------------------------ #
    # Состояние счёта
    # ------------------------------------------------------------------ #
    @property
    def equity(self) -> float:
        """Баланс + плавающая прибыль открытой позиции."""
        if self.position is None or math.isnan(self._last_price):
            return self.cash
        return self.cash + self.position.unrealized_pnl(self._last_price)

    # ------------------------------------------------------------------ #
    # Публичный API для стратегии
    # ------------------------------------------------------------------ #
    def buy(self, **kwargs: Any) -> Order:
        return self._submit(OrderSide.BUY, **kwargs)

    def sell(self, **kwargs: Any) -> Order:
        return self._submit(OrderSide.SELL, **kwargs)

    def close_position(self, tag: str = "") -> Optional[Order]:
        """Рыночный ордер на полное закрытие текущей позиции."""
        if self.position is None:
            return None
        return self._submit(
            self.position.side.opposite, size=self.position.size, tag=tag or "close"
        )

    def cancel_all_orders(self) -> None:
        for order in self.pending_orders:
            order.status = OrderStatus.CANCELLED
        self.pending_orders.clear()

    def cancel_order(self, order: Order) -> None:
        if order in self.pending_orders:
            order.status = OrderStatus.CANCELLED
            self.pending_orders.remove(order)

    # ------------------------------------------------------------------ #
    # Обработка бара (вызывается движком)
    # ------------------------------------------------------------------ #
    def process_bar(self, bar: Bar) -> None:
        """Исполняет отложенные ордера и проверяет SL/TP на баре."""
        self._current_bar = bar

        for order in list(self.pending_orders):
            fill_price = self._fill_price(order, bar)
            if fill_price is not None:
                self._execute(order, fill_price, bar)

        self._check_sl_tp(bar)
        self._last_price = bar.close

    def finalize(self, bar: Bar) -> None:
        """Принудительно закрывает позицию в конце данных по цене закрытия."""
        self.cancel_all_orders()
        if self.position is not None:
            exit_side = self.position.side.opposite
            self._close_position_part(
                size=self.position.size,
                price=self._apply_spread(bar.close, exit_side),
                bar=bar,
                reason=ExitReason.END_OF_DATA,
            )
        self._last_price = bar.close

    # ------------------------------------------------------------------ #
    # Внутренняя кухня
    # ------------------------------------------------------------------ #
    def _submit(
        self,
        side: OrderSide,
        size: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_pct: Optional[float] = None,
        tag: str = "",
    ) -> Order:
        if size is not None and size <= 0:
            raise ValueError("Размер ордера должен быть положительным")
        if limit_price is not None and stop_price is not None:
            raise ValueError("limit_price и stop_price взаимоисключающие")
        if risk_pct is not None:
            if size is not None:
                raise ValueError("size и risk_pct взаимоисключающие")
            if not risk_pct > 0:
                raise ValueError("risk_pct должен быть положительным")
            if stop_loss is None:
                raise ValueError(
                    "Сайзинг от риска требует stop_loss: размер позиции "
                    "вычисляется из дистанции до стопа"
                )
        if stop_price is not None:
            order_type = OrderType.STOP
        elif limit_price is not None:
            order_type = OrderType.LIMIT
        else:
            order_type = OrderType.MARKET
        order = Order(
            id=next(self._order_ids),
            side=side,
            type=order_type,
            size=size,
            limit_price=limit_price,
            stop_price=stop_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_pct=risk_pct,
            tag=tag,
            created_bar=self._current_bar.index if self._current_bar else -1,
        )
        self.orders.append(order)
        self.pending_orders.append(order)
        return order

    def _fill_price(self, order: Order, bar: Bar) -> Optional[float]:
        """Возвращает цену исполнения ордера на данном баре или None."""
        # Ордер не может исполниться на баре своего создания --
        # стратегия видела этот бар уже закрытым.
        if order.created_bar >= bar.index:
            return None

        if order.type is OrderType.MARKET:
            return self._apply_spread(
                self._apply_slippage(bar.open, order.side), order.side
            )

        if order.type is OrderType.STOP:
            assert order.stop_price is not None
            stop = order.stop_price
            if order.side is OrderSide.BUY:
                if bar.open >= stop:
                    raw = bar.open       # гэп через уровень -- по открытию
                elif bar.high >= stop:
                    raw = stop
                else:
                    return None
            elif bar.open <= stop:
                raw = bar.open
            elif bar.low <= stop:
                raw = stop
            else:
                return None
            return self._apply_spread(
                self._apply_slippage(raw, order.side), order.side
            )

        assert order.limit_price is not None
        limit = order.limit_price
        if order.side is OrderSide.BUY:
            if bar.open <= limit:
                raw = bar.open           # открылись ниже лимита -- цена лучше
            elif bar.low <= limit:
                raw = limit
            else:
                return None
        elif bar.open >= limit:
            raw = bar.open
        elif bar.high >= limit:
            raw = limit
        else:
            return None
        return self._apply_spread(raw, order.side)

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Проскальзывание всегда ухудшает цену для трейдера."""
        return price * (1.0 + side.value * self.slippage_pct)

    def _apply_spread(self, price: float, side: OrderSide) -> float:
        """Покупка исполняется по ask = bid + спред (данные -- bid-цены)."""
        if side is OrderSide.BUY:
            return price * (1.0 + self.spread_pct)
        return price

    def _commission(self, price: float, size: float) -> float:
        return abs(price * size) * self.commission_pct + self.commission_fixed

    def _auto_size(self, price: float) -> float:
        return max(self.equity * self.default_size_pct, 0.0) / price

    def _risk_size(self, order: Order, price: float) -> float:
        """Размер позиции от риска: equity * risk_pct / дистанция до стопа.

        Потеря при выходе ровно по SL равна risk_pct от эквити на момент
        входа (без учёта комиссий, слиппеджа и гэпов через уровень --
        они могут увеличить фактический убыток).
        """
        assert order.stop_loss is not None and order.risk_pct is not None
        distance = order.side.value * (price - order.stop_loss)
        if distance <= 0:  # стоп по неправильную сторону от цены входа
            return 0.0
        return max(self.equity, 0.0) * order.risk_pct / distance

    def _cap_leverage(self, order: Order, size: float, price: float) -> float:
        """Урезает размер так, чтобы ноционал не превышал equity * max_leverage.

        Особенно важно при risk_pct-сайзинге: близкий стоп даёт огромный
        размер (equity * risk / distance) -- без капа это бесконечное плечо.
        Объём, закрывающий встречную позицию, маржи не требует и в лимит
        не входит.
        """
        if self.max_leverage is None or price <= 0:
            return size
        offset = 0.0
        pos = self.position
        if pos is not None:
            if pos.side is not order.side:
                offset = pos.size       # неттинг: закрытие освобождает маржу
            else:
                offset = -pos.size      # доливка: маржа уже занята позицией
        cap = max(self.equity, 0.0) * self.max_leverage / price + offset
        return min(size, max(cap, 0.0))

    def _execute(self, order: Order, price: float, bar: Bar) -> None:
        """Исполняет ордер по цене price с учётом неттинга позиции."""
        if order.size is not None:
            size = order.size
        elif order.risk_pct is not None:
            size = self._risk_size(order, price)
        else:
            size = self._auto_size(price)
        size = self._cap_leverage(order, size, price)
        if size <= 0:
            order.status = OrderStatus.REJECTED
            self.pending_orders.remove(order)
            return

        order.status = OrderStatus.FILLED
        order.filled_bar = bar.index
        order.filled_price = price
        self.pending_orders.remove(order)

        remaining = size
        pos = self.position
        if pos is not None and pos.side is not order.side:
            close_size = min(pos.size, remaining)
            self._close_position_part(
                size=close_size, price=price, bar=bar,
                reason=ExitReason.SIGNAL, tag=order.tag,
            )
            remaining -= close_size

        if remaining > 1e-12:
            self._open_or_increase(order, remaining, price, bar)

    def _open_or_increase(
        self, order: Order, size: float, price: float, bar: Bar
    ) -> None:
        commission = self._commission(price, size)
        self.cash -= commission

        pos = self.position
        if pos is None:
            self.position = Position(
                side=order.side,
                size=size,
                entry_price=price,
                entry_bar=bar.index,
                entry_time=bar.datetime,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                entry_commission=commission,
                tag=order.tag,
            )
        else:
            # Доливка: пересчёт средневзвешенной цены входа
            total = pos.size + size
            pos.entry_price = (pos.entry_price * pos.size + price * size) / total
            pos.size = total
            pos.entry_commission += commission
            if order.stop_loss is not None:
                pos.stop_loss = order.stop_loss
            if order.take_profit is not None:
                pos.take_profit = order.take_profit

    def _close_position_part(
        self,
        size: float,
        price: float,
        bar: Bar,
        reason: ExitReason,
        tag: str = "",
    ) -> None:
        pos = self.position
        assert pos is not None
        size = min(size, pos.size)

        exit_commission = self._commission(price, size)
        entry_commission_part = pos.entry_commission * (size / pos.size)
        gross = pos.side.value * (price - pos.entry_price) * size
        pnl = gross - exit_commission - entry_commission_part

        self.cash += gross - exit_commission
        # Комиссия входа уже списана с cash при открытии -- в pnl сделки
        # включаем её долю только для корректной статистики по сделке.

        self.trades.append(
            Trade(
                id=next(self._trade_ids),
                side=pos.side,
                size=size,
                entry_price=pos.entry_price,
                exit_price=price,
                entry_time=pos.entry_time,
                exit_time=bar.datetime,
                entry_bar=pos.entry_bar,
                exit_bar=bar.index,
                commission=exit_commission + entry_commission_part,
                pnl=pnl,
                pnl_pct=pos.side.value * (price / pos.entry_price - 1.0) * 100.0,
                exit_reason=reason,
                tag=tag or pos.tag,  # тег закрытия, иначе тег входа (SL/TP)
            )
        )

        pos.size -= size
        pos.entry_commission -= entry_commission_part
        if pos.size <= 1e-12:
            self.position = None

    def _check_sl_tp(self, bar: Bar) -> None:
        """Проверка Stop-Loss / Take-Profit внутри бара (по high/low).

        Консервативное допущение: если за один бар задеты и SL, и TP,
        первым считается срабатывание Stop-Loss.
        """
        pos = self.position
        if pos is None:
            return

        sl, tp = pos.stop_loss, pos.take_profit
        if pos.is_long:
            sl_hit = sl is not None and bar.low <= sl
            tp_hit = tp is not None and bar.high >= tp
        else:
            sl_hit = sl is not None and bar.high >= sl
            tp_hit = tp is not None and bar.low <= tp

        exit_side = pos.side.opposite
        if sl_hit:
            assert sl is not None
            # Гэп через уровень: исполняемся по цене открытия, она хуже SL
            raw = min(bar.open, sl) if pos.is_long else max(bar.open, sl)
            price = self._apply_spread(
                self._apply_slippage(raw, exit_side), exit_side
            )
            self._close_position_part(
                size=pos.size, price=price, bar=bar, reason=ExitReason.STOP_LOSS
            )
        elif tp_hit:
            assert tp is not None
            raw = max(bar.open, tp) if pos.is_long else min(bar.open, tp)
            price = self._apply_spread(raw, exit_side)
            self._close_position_part(
                size=pos.size, price=price, bar=bar, reason=ExitReason.TAKE_PROFIT
            )


# ---------------------------------------------------------------------- #
# Результат прогона
# ---------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    """Результат одного прогона бэктеста."""

    equity_curve: pd.Series           # эквити по закрытию каждого бара
    trades: list[Trade]
    orders: list[Order]
    data: pd.DataFrame
    initial_cash: float
    strategy_name: str
    strategy_params: dict[str, Any]
    #: индикаторы для графика (см. BaseStrategy.plot_line); каждый элемент --
    #: dict(name, values, color, overlay, width). Пусто, если стратегия их не
    #: объявляет. default_factory -- чтобы старый код, строящий BacktestResult
    #: без этого поля, не падал.
    indicators: list[dict[str, Any]] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def net_profit(self) -> float:
        return self.final_equity - self.initial_cash

    def trades_df(self) -> pd.DataFrame:
        """Сделки в виде DataFrame (удобно для анализа и визуализации)."""
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "id", "side", "size", "entry_price", "exit_price",
                    "entry_time", "exit_time", "entry_bar", "exit_bar",
                    "commission", "pnl", "pnl_pct", "exit_reason",
                    "duration_bars", "tag",
                ]
            )
        return pd.DataFrame(
            {
                "id": t.id,
                "side": t.side.name,
                "size": t.size,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_bar": t.entry_bar,
                "exit_bar": t.exit_bar,
                "commission": t.commission,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason.value,
                "duration_bars": t.duration_bars,
                "tag": t.tag,
            }
            for t in self.trades
        )

    def stats(self, periods_per_year: Optional[int] = None) -> Any:
        """Расчёт метрик (см. metrics.compute_metrics)."""
        from .metrics import compute_metrics

        return compute_metrics(
            self.equity_curve, self.trades_df(), periods_per_year=periods_per_year
        )


# ---------------------------------------------------------------------- #
# Движок
# ---------------------------------------------------------------------- #
class BacktestEngine:
    """Исторический симулятор: прогоняет стратегию по данным бар за баром.

    Пример::

        engine = BacktestEngine(
            data=df,
            strategy_cls=SmaCross,
            strategy_params={"fast": 10, "slow": 30},
            cash=10_000,
            commission_pct=0.0005,
            slippage_pct=0.0001,
        )
        result = engine.run()
        print(result.stats())
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_cls: Type[BaseStrategy],
        strategy_params: Optional[dict[str, Any]] = None,
        cash: float = 10_000.0,
        commission_pct: float = 0.0,
        commission_fixed: float = 0.0,
        slippage_pct: float = 0.0,
        spread_pct: float = 0.0,
        max_leverage: Optional[float] = None,
        default_size_pct: float = 0.95,
        warmup_bars: int = 0,
    ) -> None:
        """warmup_bars: первые N баров стратегия не вызывается (торговли нет),
        но индикаторы в init() считаются по всему срезу -- к началу торговли
        они уже прогреты. Используется WFA, чтобы прогрев не съедал OOS-окно.
        """
        self._validate_data(data)
        if warmup_bars < 0 or warmup_bars >= len(data):
            raise ValueError("warmup_bars должен быть в диапазоне [0, len(data))")
        self.data = data
        self.strategy_cls = strategy_cls
        self.strategy_params = dict(strategy_params or {})
        self.warmup_bars = int(warmup_bars)
        self.broker_kwargs: dict[str, Any] = dict(
            cash=cash,
            commission_pct=commission_pct,
            commission_fixed=commission_fixed,
            slippage_pct=slippage_pct,
            spread_pct=spread_pct,
            max_leverage=max_leverage,
            default_size_pct=default_size_pct,
        )

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"В данных не хватает колонок: {sorted(missing)}")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Ожидается DataFrame с DatetimeIndex")
        if not data.index.is_monotonic_increasing:
            raise ValueError("Индекс данных должен быть отсортирован по возрастанию")
        if len(data) < 2:
            raise ValueError("Слишком мало данных для симуляции")

    def run(self) -> BacktestResult:
        broker = Broker(**self.broker_kwargs)
        strategy = self.strategy_cls(**self.strategy_params)
        strategy._bind(broker, self.data)
        strategy.init()

        n = len(self.data)
        index = self.data.index
        opens = self.data["open"].to_numpy()
        highs = self.data["high"].to_numpy()
        lows = self.data["low"].to_numpy()
        closes = self.data["close"].to_numpy()
        volumes = self.data["volume"].to_numpy()
        equity = np.empty(n, dtype=np.float64)

        for i in range(n):
            bar = Bar(
                i,
                float(opens[i]),
                float(highs[i]),
                float(lows[i]),
                float(closes[i]),
                float(volumes[i]),
                index,
            )
            # 1) исполнение отложенного: ордера прошлых баров, SL/TP
            broker.process_bar(bar)
            # 2) последний бар: принудительное закрытие, стратегию не вызываем
            if i == n - 1:
                broker.finalize(bar)
            elif i >= self.warmup_bars:
                # 3) торговая логика стратегии на закрытии бара
                strategy._set_index(i)
                strategy.next(bar)
            # 4) переоценка счёта по закрытию бара
            equity[i] = broker.equity

        return BacktestResult(
            equity_curve=pd.Series(equity, index=index, name="equity"),
            trades=broker.trades,
            orders=broker.orders,
            data=self.data,
            initial_cash=broker.initial_cash,
            strategy_name=self.strategy_cls.__name__,
            strategy_params=dict(strategy.p),
            indicators=list(getattr(strategy, "_plots", [])),
        )


# ---------------------------------------------------------------------- #
# Диспетчер движка
# ---------------------------------------------------------------------- #
def run_backtest(
    data: pd.DataFrame,
    strategy_cls: Type[BaseStrategy],
    strategy_params: Optional[dict[str, Any]] = None,
    *,
    prefer_fast: bool = True,
    **engine_kwargs: Any,
) -> BacktestResult:
    """Прогон стратегии с автоматическим выбором движка.

    Для наследников SignalStrategy (при установленной numba) используется
    быстрое numba-ядро run_fast_backtest -- десятки крат быстрее на сетках и
    в WFA; для остальных стратегий и при отсутствии numba -- событийный
    BacktestEngine. Оба пути дают идентичный BacktestResult (эквивалентность
    зафиксирована в tests/test_fast_engine.py), поэтому метрики / WFA /
    Монте-Карло / визуализация работают одинаково.

    prefer_fast=False принудительно использует событийный движок (нужно,
    например, в тестах эквивалентности).
    """
    if prefer_fast:
        try:
            from .fast_engine import SignalStrategy, run_fast_backtest
        except ImportError:
            pass  # numba не установлена -- тихо откатываемся на событийный движок
        else:
            if isinstance(strategy_cls, type) and issubclass(
                strategy_cls, SignalStrategy
            ):
                return run_fast_backtest(
                    data=data,
                    strategy_cls=strategy_cls,
                    strategy_params=strategy_params,
                    **engine_kwargs,
                )
    return BacktestEngine(
        data=data,
        strategy_cls=strategy_cls,
        strategy_params=strategy_params,
        **engine_kwargs,
    ).run()
