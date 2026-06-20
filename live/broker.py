"""LiveBroker -- брокер стратегии поверх биржевых фьючерсов.

Повторяет публичный интерфейс backtester.Broker (buy / sell / close_position /
cancel_all_orders + свойства position / equity / cash), но вместо симуляции
ставит реальные рыночные ордера через ExchangeClient. Благодаря этому
существующие стратегии запускаются «вживую» без единой правки.

Семантика повторяет неттинг бэктестера (счёт в режиме one-way):
    * ордер противоположной стороны сначала закрывает текущую позицию,
      остаток открывает противоположную (биржа в one-way режиме делает это
      автоматически при рыночном ордере нужного объёма);
    * частичное закрытие -- ордер той же логики с size меньше позиции;
    * SL/TP, переданные при входе, выставляются БИРЖЕВЫМИ reduce-only
      стоп/тейк-ордерами (срабатывают между барами -- безопаснее интрабар-
      проверки бэктеста) и переразмериваются под позицию на каждом баре.

Размеры (size) и позиции выражены в КОНТРАКТАХ ccxt (как поле position.contracts):
strategy.position.size -- в контрактах, поэтому explicit size из стратегии
(например, частичное закрытие 0.25*size) тоже в контрактах. Автосайзинг и
сайзинг от риска переводят ноционал/базовый объём в контракты через contractSize.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

from backtester.backtest_engine import OrderSide, Position

from .exchange import ExchangeClient, LivePosition

EmitFn = Callable[[str, str, Optional[dict]], None]


class ProtectiveOrderError(RuntimeError):
    """Не удалось защитить живую позицию стоп-лоссом.

    Поднимается из sync_protective_orders, когда у ОТКРЫТОЙ позиции есть
    желаемый стоп-лосс, но биржа его не приняла (или уровень на неверной
    стороне). Движок ловит это отдельно от ExchangeError и аварийно закрывает
    позицию: торговать без стопа на реале (особенно на DEX, где биржевые
    стоп-ордера поддерживаются ограниченно) опаснее, чем выйти в кэш.
    Отключается флагом ExecutionConfig.require_protective=False.
    """


@dataclass
class ExecutionConfig:
    """Как исполнять входы и защитные выходы (для экономии на комиссии).

    entry_limit -- открывать позицию лимитным (мейкерским) ордером вместо
        рыночного. Лимитный ордер может не исполниться сразу: если к
        следующему бару он не залился, движок снимает его как устаревший и
        локальная модель позиции НЕ меняется (истина читается с биржи).
    post_only   -- лимитные входы только как мейкер (биржа отклонит тейкер-филл).
    limit_offset_bps -- на сколько базисных пунктов отступить лимит-ценой в свою
        пользу от close, если стратегия не задала limit_price (buy ниже, sell выше).
    tp_maker    -- тейк-профит выставлять reduce-only ЛИМИТНЫМ ордером (мейкер)
        вместо тейк-маркета. Стоп-лосс всегда остаётся стоп-маркетом (гарантия
        выхода важнее комиссии).
    max_slippage_pct -- максимальное проскальзывание для РЫНОЧНОГО входа в %.
        Если к моменту исполнения цена ушла ПРОТИВ входа дальше этого процента
        от цены закрытия бара, ордер не выставляется (вход пропускается).
        0 -- проверка выключена. Закрытия и защитные ордера не блокируются.
    require_protective -- если стратегия задала стоп-лосс, но биржа его не
        приняла, аварийно закрыть позицию и остановить цикл (см.
        ProtectiveOrderError). True по умолчанию -- безопасно для реала и DEX,
        где биржевые стопы поддерживаются ограниченно. False оставляет позицию
        под управлением стратегии (выход по сигналу на закрытии бара) -- ставьте
        осознанно, только если стратегия не полагается на биржевой стоп.
    """

    entry_limit: bool = False
    post_only: bool = True
    limit_offset_bps: float = 1.0
    tp_maker: bool = False
    max_slippage_pct: float = 0.0
    require_protective: bool = True


@dataclass
class RiskLimits:
    """Аппаратные предохранители («рубильник»). 0 -- предохранитель выключен.

    max_notional_per_order -- потолок ноционала одного входного ордера в USDT
        (защита от «толстого пальца»/бага сайзинга); ордер сверх лимита
        отклоняется (НЕ урезается, чтобы не искажать намерение стратегии).
    max_orders_per_hour    -- максимум ВХОДНЫХ ордеров за скользящий час
        (анти-зацикливание при ошибке стратегии).
    max_daily_loss_pct     -- стоп торговли на сутки (UTC) при просадке эквити от
        стартового баланса дня на N%: позиция закрывается, цикл останавливается.

    Закрытия и защитные ордера НИКОГДА не блокируются лимитами -- действие,
    снижающее риск, должно проходить всегда.
    """

    max_notional_per_order: float = 0.0
    max_orders_per_hour: int = 0
    max_daily_loss_pct: float = 0.0


class SizingConfig:
    """Как выбирать размер позиции, когда стратегия не задаёт size явно.

    mode:
        "notional" -- фиксированный ноционал в USDT на вход (notional_usdt);
        "balance_pct" -- процент от общего баланса (balance_pct, 0..100).
    leverage -- информативно (плечо выставляется на бирже отдельно); ноционал
    уже подразумевает использование плеча.
    """

    def __init__(
        self,
        mode: str = "notional",
        notional_usdt: float = 100.0,
        balance_pct: float = 5.0,
    ) -> None:
        self.mode = mode
        self.notional_usdt = float(notional_usdt)
        self.balance_pct = float(balance_pct)

    def target_notional(self, equity: float) -> float:
        if self.mode == "balance_pct":
            return max(equity, 0.0) * self.balance_pct / 100.0
        return self.notional_usdt


class LiveBroker:
    """Брокер стратегии, исполняющий ордера на бирже (USDT-перпетуалы)."""

    def __init__(
        self,
        client: ExchangeClient,
        symbol: str,
        sizing: SizingConfig,
        emit: EmitFn,
        dry_run: bool = False,
        execution: Optional[ExecutionConfig] = None,
        risk: Optional[RiskLimits] = None,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.sizing = sizing
        self.emit = emit
        self.dry_run = dry_run
        self.execution = execution or ExecutionConfig()
        self.risk = risk or RiskLimits()

        try:
            self._contract_size = client.contract_size(symbol)
        except Exception:  # noqa: BLE001 - без рынков считаем 1 контракт = 1 базовый
            self._contract_size = 1.0

        # Состояние, переживающее пересборку стратегии (она создаётся каждый бар):
        self._position: Optional[LivePosition] = None
        self._entry_time: Optional[pd.Timestamp] = None
        self._desired_sl: Optional[float] = None
        self._desired_tp: Optional[float] = None
        self._free_usdt: float = 0.0
        self._total_usdt: float = 0.0
        self._last_price: float = math.nan
        self._window_index: Optional[pd.DatetimeIndex] = None
        self._now: Optional[pd.Timestamp] = None
        # последняя синхронизированная защитная конфигурация (для дедупликации)
        self._synced_protect: tuple = ()
        # незалитый лимитный вход (id ордера + намерение по стороне/SL/TP)
        self._pending_entry: Optional[dict] = None
        # времена (monotonic) ВХОДНЫХ ордеров за скользящий час -- для рейт-лимита
        self._order_times: deque[float] = deque()

    # ------------------------------------------------------------------ #
    # Обновление состояния на каждом новом баре (вызывается движком)
    # ------------------------------------------------------------------ #
    def begin_bar(
        self,
        window_index: pd.DatetimeIndex,
        last_price: float,
        free_usdt: float,
        total_usdt: float,
        now: pd.Timestamp,
    ) -> None:
        self._window_index = window_index
        self._last_price = float(last_price)
        self._free_usdt = float(free_usdt)
        self._total_usdt = float(total_usdt)
        self._now = now
        if not self.dry_run:
            # Истина о позиции -- с биржи (могла измениться по SL/TP между барами)
            self._position = self.client.fetch_position(self.symbol)
        self._reconcile_pending_entry()
        if self._position is None:
            self._entry_time = None
            self._desired_sl = None
            self._desired_tp = None

    def _reconcile_pending_entry(self) -> None:
        """Свести незалитый лимитный вход с реальной позицией на новом баре.

        Если позиция появилась -- лимит исполнился: перенимаем его SL/TP/время
        входа. Если позиции нет -- лимит не залился: снимаем устаревший ордер
        (цена ушла). Локальная модель при этом НЕ домысливала фантомный филл.
        """
        pend = self._pending_entry
        if pend is None:
            return
        if self._position is not None and self._position.size > 0:
            self._entry_time = self._now
            if pend.get("sl") is not None:
                self._desired_sl = pend["sl"]
            if pend.get("tp") is not None:
                self._desired_tp = pend["tp"]
            self.emit("info", "Лимитный вход исполнен", None)
        else:
            oid = pend.get("id")
            if oid is not None:
                self.client.cancel_order(str(oid), self.symbol)
            self.emit("info", "Лимитный вход не залился — снят как устаревший", None)
        self._pending_entry = None

    # ------------------------------------------------------------------ #
    # Свойства, ожидаемые стратегией
    # ------------------------------------------------------------------ #
    @property
    def position(self) -> Optional[Position]:
        """Позиция в виде backtester.Position (для совместимости со стратегиями)."""
        lp = self._position
        if lp is None or lp.size <= 0:
            return None
        entry_bar = self._entry_bar()
        entry_time = self._entry_time or (
            self._window_index[entry_bar] if self._window_index is not None
            else pd.Timestamp.utcnow()
        )
        return Position(
            side=OrderSide.BUY if lp.is_long else OrderSide.SELL,
            size=lp.size,
            entry_price=lp.entry_price,
            entry_bar=entry_bar,
            entry_time=entry_time,
            stop_loss=self._desired_sl,
            take_profit=self._desired_tp,
        )

    def _entry_bar(self) -> int:
        """Индекс бара входа в координатах текущего окна (для min_hold)."""
        if self._window_index is None:
            return 0
        if self._entry_time is None:
            return 0  # вход неизвестен/давно -- min_hold не блокирует
        pos = int(self._window_index.searchsorted(self._entry_time))
        return max(0, min(pos, len(self._window_index) - 1))

    @property
    def equity(self) -> float:
        return self._total_usdt

    @property
    def cash(self) -> float:
        return self._free_usdt

    # ------------------------------------------------------------------ #
    # Торговый API стратегии
    # ------------------------------------------------------------------ #
    def buy(self, **kwargs: Any) -> None:
        self._submit(OrderSide.BUY, **kwargs)

    def sell(self, **kwargs: Any) -> None:
        self._submit(OrderSide.SELL, **kwargs)

    def close_position(self, tag: str = "") -> None:
        lp = self._position
        if lp is None or lp.size <= 0:
            return
        side = "sell" if lp.is_long else "buy"
        # Закрытие -- рыночное (гарантированный выход) и не подпадает под риск-лимиты.
        order = self._market(side, lp.size, reduce_only=True, tag=tag or "close")
        if order is None:
            return  # биржа отклонила -- позиция как была, фантомного филла не делаем
        fill = self._fill_price(order)
        closed_amt = self._order_filled(order, lp.size)
        self._apply_fill(OrderSide.SELL if lp.is_long else OrderSide.BUY, closed_amt,
                         sl=None, tp=None, reduce=True, fill_price=fill)

    def cancel_all_orders(self) -> None:
        if self.dry_run:
            return
        try:
            self.client.cancel_all_orders(self.symbol)
        except Exception as exc:  # noqa: BLE001
            self.emit("error", f"Не удалось отменить ордера: {exc}", None)

    # ------------------------------------------------------------------ #
    # Реализация
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
    ) -> None:
        contracts = self._resolve_size(side, size, risk_pct, stop_loss)
        if contracts <= 0:
            self.emit("info", f"Сигнал {side.name}: нулевой размер — пропуск", None)
            return
        # Предохранители: входной ордер (reduce_only=False) проходит риск-проверки.
        if not self._risk_allows_entry(contracts):
            return
        ccxt_side = "buy" if side is OrderSide.BUY else "sell"

        # Лимитный (мейкерский) вход, если включён в ExecutionConfig либо стратегия
        # сама задала limit_price. Иначе -- рыночное исполнение на закрытии бара.
        use_limit = self.execution.entry_limit or (limit_price is not None)
        if use_limit:
            self._submit_limit_entry(
                side, ccxt_side, contracts, limit_price,
                stop_loss, take_profit, tag,
            )
            return

        order = self._market(ccxt_side, contracts, reduce_only=False, tag=tag)
        if order is None:
            return  # биржа отклонила -- фантомного филла не делаем (фикс №2)
        self._register_entry_order()
        self._apply_fill(
            side, self._order_filled(order, contracts),
            sl=stop_loss, tp=take_profit, reduce=False,
            fill_price=self._fill_price(order),
        )

    def _submit_limit_entry(
        self,
        side: OrderSide,
        ccxt_side: str,
        contracts: float,
        limit_price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        tag: str,
    ) -> None:
        """Поставить лимитный (мейкерский) вход без оптимистичного филла.

        Локальная модель позиции НЕ меняется: исполнение неизвестно до
        следующего бара (см. _reconcile_pending_entry). В сухом прогоне нет
        реального стакана -- считаем лимит исполненным по лимит-цене.
        """
        px = self._entry_limit_price(side, limit_price)
        notional = contracts * (self._contract_size or 1.0) * px
        label = "сухой прогон" if self.dry_run else "ЛИМИТ"
        msg = (
            f"[{label}] {ccxt_side.upper()} {contracts:g} конт. {self.symbol} "
            f"@ {px:g} (~{notional:.2f} USDT, мейкер)" + (f" • {tag}" if tag else "")
        )
        if self.dry_run:
            self.emit("order", msg, {"side": ccxt_side, "contracts": contracts,
                                     "price": px, "dry": True})
            self._apply_fill(side, contracts, sl=stop_loss, tp=take_profit,
                             reduce=False, fill_price=px)
            return
        try:
            order = self.client.create_limit_order(
                self.symbol, ccxt_side, contracts, px,
                reduce_only=False, post_only=self.execution.post_only,
            )
        except Exception as exc:  # noqa: BLE001
            self.emit("error", f"Ошибка лимитного входа: {exc}", None)
            return
        self._register_entry_order()
        self.emit("order", msg, {"id": order.get("id"), "side": ccxt_side,
                                 "contracts": contracts, "price": px})
        self._pending_entry = {
            "id": order.get("id"), "side": side,
            "sl": stop_loss, "tp": take_profit,
        }

    def _entry_limit_price(self, side: OrderSide, limit_price: Optional[float]) -> float:
        """Цена лимитного входа: явная из стратегии или close со смещением мейкера."""
        if limit_price is not None:
            return float(limit_price)
        off = max(self.execution.limit_offset_bps, 0.0) / 10000.0
        px = self._last_price
        return px * (1.0 - off) if side is OrderSide.BUY else px * (1.0 + off)

    # ------------------------------------------------------------------ #
    # Предохранители (kill switch / circuit breaker)
    # ------------------------------------------------------------------ #
    def _risk_allows_entry(self, contracts: float) -> bool:
        """Проверки перед ВХОДНЫМ ордером. False -> ордер не отправляется."""
        cap = self.risk.max_notional_per_order
        if cap > 0:
            notional = contracts * (self._contract_size or 1.0) * self._last_price
            if notional > cap:
                self.emit(
                    "error",
                    f"⛔ Ордер отклонён: ноционал ~{notional:.2f} USDT превышает "
                    f"лимит {cap:.2f} USDT (предохранитель «толстого пальца»)",
                    {"risk": "max_notional"},
                )
                return False
        limit = self.risk.max_orders_per_hour
        if limit > 0:
            self._prune_order_times()
            if len(self._order_times) >= limit:
                self.emit(
                    "error",
                    f"⛔ Ордер отклонён: достигнут лимит {limit} входов/час "
                    f"(анти-зацикливание)",
                    {"risk": "max_orders_per_hour"},
                )
                return False
        return True

    def _register_entry_order(self) -> None:
        """Зафиксировать факт входного ордера для рейт-лимита."""
        if self.risk.max_orders_per_hour > 0:
            self._order_times.append(time.monotonic())

    def _prune_order_times(self) -> None:
        cutoff = time.monotonic() - 3600.0
        while self._order_times and self._order_times[0] < cutoff:
            self._order_times.popleft()

    def _resolve_size(
        self,
        side: OrderSide,
        size: Optional[float],
        risk_pct: Optional[float],
        stop_loss: Optional[float],
    ) -> float:
        """Размер ордера в контрактах (повторяет логику Broker бэктестера)."""
        price = self._last_price
        if not math.isfinite(price) or price <= 0:
            return 0.0
        cs = self._contract_size or 1.0
        if size is not None:                       # явный размер в контрактах
            contracts = float(size)
        elif risk_pct is not None and stop_loss is not None:
            distance = side.value * (price - stop_loss)
            if distance <= 0:
                return 0.0
            base_units = max(self._total_usdt, 0.0) * float(risk_pct) / distance
            contracts = base_units / cs
        else:                                      # автосайзинг по ноционалу
            notional = self.sizing.target_notional(self._total_usdt)
            base_units = notional / price
            contracts = base_units / cs
        if self.dry_run:
            return max(contracts, 0.0)
        contracts = self.client.amount_to_precision(self.symbol, contracts)
        min_amt = self.client.min_amount(self.symbol)
        if min_amt and 0 < contracts < min_amt:
            self.emit(
                "info",
                f"Размер {contracts:g} < минимума биржи {min_amt:g} — округляю до минимума",
                None,
            )
            contracts = min_amt
        return max(contracts, 0.0)

    def _slippage_blocks_entry(self, side: str) -> bool:
        """True -> рыночный вход отменён: цена ушла против входа дальше лимита.

        Сравнивает текущую цену биржи с ценой закрытия бара, по которой стратегия
        приняла решение. Если НЕБЛАГОПРИЯТНОЕ движение (для buy цена выросла, для
        sell упала) превышает max_slippage_pct, вход не выставляется. Только для
        входов: закрытия и защитные ордера (reduce_only) никогда не блокируются.
        """
        max_slip = self.execution.max_slippage_pct
        ref = self._last_price
        if max_slip <= 0 or self.dry_run or not math.isfinite(ref) or ref <= 0:
            return False
        try:
            current = self.client.fetch_ticker_last(self.symbol)
        except Exception:  # noqa: BLE001 - цену не узнали -> не блокируем вход
            return False
        move_pct = (current - ref) / ref * 100.0
        adverse = move_pct if side == "buy" else -move_pct
        if adverse > max_slip:
            self.emit(
                "info",
                f"⏭ Вход {side.upper()} пропущен: цена ушла на {adverse:+.2f}% "
                f"(> лимита проскальзывания {max_slip:.2f}%): {ref:g} → {current:g}",
                {"slippage_pct": adverse, "limit_pct": max_slip},
            )
            return True
        return False

    def _market(
        self, side: str, contracts: float, reduce_only: bool, tag: str
    ) -> Optional[dict]:
        """Рыночный ордер. Возвращает ордер при успехе, None при отказе биржи.

        Возврат None -- сигнал вызывающему НЕ обновлять локальную модель позиции
        (иначе она разойдётся с биржей -- фантомный филл, фикс №2).
        """
        if not reduce_only and self._slippage_blocks_entry(side):
            return None
        notional = contracts * (self._contract_size or 1.0) * self._last_price
        label = "сухой прогон" if self.dry_run else "ОРДЕР"
        msg = (
            f"[{label}] {side.upper()} {contracts:g} конт. {self.symbol} "
            f"(~{notional:.2f} USDT)" + (" reduceOnly" if reduce_only else "")
            + (f" • {tag}" if tag else "")
        )
        if self.dry_run:
            self.emit("order", msg, {"side": side, "contracts": contracts, "dry": True})
            return {"dry": True}
        try:
            order = self.client.create_market_order(
                self.symbol, side, contracts, reduce_only=reduce_only
            )
        except Exception as exc:  # noqa: BLE001
            self.emit("error", f"Ошибка ордера: {exc}", None)
            return None
        self.emit("order", msg, {"id": order.get("id"), "side": side,
                                 "contracts": contracts})
        return order

    @staticmethod
    def _order_filled(order: Optional[dict], requested: float) -> float:
        """Фактически залитый объём из ответа биржи (фолбэк -- запрошенный).

        При частичном филле рыночного ордера биржа возвращает `filled` < запроса;
        учитываем именно его, чтобы локальная модель позиции не разошлась с биржей
        (на следующем баре истина всё равно перечитывается, но в пределах бара
        стратегия может довыставлять ордера). Если поля нет/пустое -- считаем
        ордер залитым полностью (как и было).
        """
        if order:
            val = order.get("filled")
            if val is not None:
                try:
                    f = float(val)
                    if f > 0:
                        return f
                except (TypeError, ValueError):
                    pass
        return requested

    def _fill_price(self, order: Optional[dict]) -> float:
        """Фактическая средняя цена филла из ответа биржи (фолбэк -- last_price)."""
        if order:
            for key in ("average", "price"):
                val = order.get(key)
                if val:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return self._last_price

    def _apply_fill(
        self,
        side: OrderSide,
        contracts: float,
        sl: Optional[float],
        tp: Optional[float],
        reduce: bool,
        fill_price: Optional[float] = None,
    ) -> None:
        """Обновляет локальную модель позиции по ФАКТУ исполнения (неттинг).

        Вызывается только после подтверждённого биржей ордера (фикс №2). На
        следующем баре истина перечитывается с биржи (begin_bar), но в пределах
        бара стратегия может выставить несколько ордеров (частичные закрытия +
        разворот), поэтому модель должна корректно неттиться сразу. ``fill_price``
        -- фактическая средняя цена филла; фолбэк -- цена закрытия бара.
        """
        price = fill_price if (fill_price and math.isfinite(fill_price)) else self._last_price
        pos = self._position
        signed = (contracts if side is OrderSide.BUY else -contracts)
        prev_signed = pos.signed_size if pos else 0.0
        new_signed = prev_signed + signed

        opened_new_side = (
            (prev_signed <= 0 < new_signed) or (prev_signed >= 0 > new_signed)
        )
        if abs(new_signed) < 1e-12:
            self._position = None
            self._entry_time = None
            self._desired_sl = self._desired_tp = None
            return

        new_side = "long" if new_signed > 0 else "short"
        if pos is None or opened_new_side:
            entry_price = price
            self._entry_time = self._now
            self._desired_sl, self._desired_tp = sl, tp
        else:
            # доливка/частичное закрытие той же стороны -- средняя цена
            if abs(new_signed) > abs(prev_signed) and pos is not None:
                total = abs(new_signed)
                entry_price = (
                    pos.entry_price * abs(prev_signed) + price * contracts
                ) / total
            else:
                entry_price = pos.entry_price if pos else price
            if sl is not None:
                self._desired_sl = sl
            if tp is not None:
                self._desired_tp = tp
        self._position = LivePosition(
            side=new_side, size=abs(new_signed), entry_price=entry_price,
        )

    # ------------------------------------------------------------------ #
    # Защитные ордера (SL/TP) -- синхронизируются движком после next()
    # ------------------------------------------------------------------ #
    def _protective_ref(self, lp: LivePosition) -> Optional[float]:
        """Опорная цена для проверки стороны SL/TP (вход позиции; фолбэк -- last)."""
        ref = lp.entry_price if lp.entry_price and lp.entry_price > 0 else self._last_price
        return ref if math.isfinite(ref) and ref > 0 else None

    def _level_on_correct_side(self, lp: LivePosition, level: float, is_sl: bool) -> bool:
        """True, если защитный уровень на правильной стороне от цены.

        Для лонга: стоп ниже, тейк выше; для шорта -- наоборот. Если опорной
        цены нет (не знаем вход) -- не блокируем (возвращаем True).
        """
        ref = self._protective_ref(lp)
        if ref is None:
            return True
        if lp.is_long:
            return level < ref if is_sl else level > ref
        return level > ref if is_sl else level < ref

    def sync_protective_orders(self) -> None:
        """Переставляет биржевые SL/TP под текущую позицию (если изменились).

        Стоп-лосс критичен: если он желателен, но не встал (отказ биржи или
        неверная сторона уровня), поднимаем ProtectiveOrderError -- движок
        закроет позицию (не оставляем её без защиты). Тейк-профит некритичен:
        при сбое только предупреждаем.
        """
        if self.dry_run:
            return
        # Незалитый лимитный вход ждёт исполнения и не защищён позицией -- его
        # ордер нельзя трогать cancel_all'ом, пока он висит в стакане.
        if self._pending_entry is not None:
            return
        lp = self._position
        if lp is None or lp.size <= 0:
            key = ()
            if key != self._synced_protect:
                self.cancel_all_orders()
                self._synced_protect = key
            return

        tp_maker = self.execution.tp_maker
        key = (lp.side, round(lp.size, 8), self._desired_sl, self._desired_tp, tp_maker)
        if key == self._synced_protect:
            return
        self.cancel_all_orders()
        exit_side = "sell" if lp.is_long else "buy"

        # --- Стоп-лосс (критично) -------------------------------------- #
        sl_failed = False
        if self._desired_sl is not None:
            if not self._level_on_correct_side(lp, self._desired_sl, is_sl=True):
                self.emit(
                    "error",
                    f"⛔ Стоп-лосс {self._desired_sl:g} на НЕВЕРНОЙ стороне для "
                    f"{lp.side} (вход ~{lp.entry_price:g}) — не выставляю",
                    {"protective": "sl_invalid"},
                )
                sl_failed = True
            else:
                try:
                    # Стоп-лосс -- всегда стоп-маркет: гарантия выхода важнее комиссии.
                    self.client.create_stop_order(
                        self.symbol, exit_side, lp.size, self._desired_sl, kind="stop"
                    )
                except Exception as exc:  # noqa: BLE001
                    self.emit("error", f"⛔ Не удалось выставить СТОП-ЛОСС: {exc}",
                              {"protective": "sl"})
                    sl_failed = True

        # --- Тейк-профит (некритично) ---------------------------------- #
        if self._desired_tp is not None and self._level_on_correct_side(
            lp, self._desired_tp, is_sl=False
        ):
            try:
                if tp_maker:
                    # Тейк-профит мейкером: reduce-only лимит по цели (экономит комиссию).
                    self.client.create_limit_order(
                        self.symbol, exit_side, lp.size, self._desired_tp,
                        reduce_only=True, post_only=True,
                    )
                else:
                    self.client.create_stop_order(
                        self.symbol, exit_side, lp.size, self._desired_tp,
                        kind="take_profit",
                    )
            except Exception as exc:  # noqa: BLE001
                self.emit("error", f"Тейк-профит не выставлен (не критично): {exc}",
                          {"protective": "tp"})
        elif self._desired_tp is not None:
            self.emit(
                "info",
                f"Тейк-профит {self._desired_tp:g} на неверной стороне для "
                f"{lp.side} — пропуск",
                None,
            )

        # --- Позиция осталась без стопа? ------------------------------- #
        if sl_failed:
            self._synced_protect = ()   # форсим повтор/обработку на след. баре
            if self.execution.require_protective:
                raise ProtectiveOrderError(
                    "стоп-лосс не выставлен — позиция осталась бы без защиты"
                )
            self.emit(
                "error",
                "⚠️ ВНИМАНИЕ: позиция БЕЗ стоп-лосса (require_protective=False)",
                {"protective": "naked"},
            )
            return

        self._synced_protect = key
        if self._desired_sl is not None or self._desired_tp is not None:
            self.emit(
                "info",
                f"Защитные ордера: SL={self._desired_sl} TP={self._desired_tp} "
                f"на {lp.size:g} конт.",
                None,
            )

    # ------------------------------------------------------------------ #
    # Сериализация состояния (для персистентности между перезапусками)
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        """Намерения, которые нельзя перечитать с биржи (SL/TP, лимит-вход)."""
        pe = None
        if self._pending_entry is not None:
            pe = {
                "id": self._pending_entry.get("id"),
                "side": self._pending_entry["side"].name,
                "sl": self._pending_entry.get("sl"),
                "tp": self._pending_entry.get("tp"),
            }
        return {
            "entry_time": str(self._entry_time) if self._entry_time is not None else None,
            "desired_sl": self._desired_sl,
            "desired_tp": self._desired_tp,
            "pending_entry": pe,
        }

    def restore(self, data: dict) -> None:
        """Восстановить намерения из snapshot() (мягко к битым полям).

        Главное -- вернуть desired_sl/desired_tp, чтобы первый sync после
        рестарта НЕ снял защитные ордера у живой позиции, а переставил их
        идентично (см. комментарий в state.py).
        """
        if not data:
            return
        et = data.get("entry_time")
        try:
            self._entry_time = pd.Timestamp(et) if et else None
        except (ValueError, TypeError):
            self._entry_time = None
        self._desired_sl = data.get("desired_sl")
        self._desired_tp = data.get("desired_tp")
        pe = data.get("pending_entry")
        if pe and pe.get("side") in (OrderSide.BUY.name, OrderSide.SELL.name):
            self._pending_entry = {
                "id": pe.get("id"),
                "side": OrderSide[pe["side"]],
                "sl": pe.get("sl"),
                "tp": pe.get("tp"),
            }
