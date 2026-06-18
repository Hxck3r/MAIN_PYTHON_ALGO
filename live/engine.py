"""LiveTradingEngine -- цикл реальной торговли по закрытым барам.

Чистый Python без Qt (тестируемо и переиспользуемо). Раз в poll-интервал
опрашивает биржу; при появлении НОВОГО ЗАКРЫТОГО бара выполняет один шаг
стратегии и даёт брокеру выставить ордера. О каждом значимом действии шлёт
EngineEvent через колбэк on_event (GUI оборачивает его в Qt-сигнал).

Стратегия пересобирается на каждом баре над свежим окном истории -- индикаторы
в init() считаются заново (как в бэктесте), заглядывания вперёд нет, состояние
позиции переживает пересборку в самом брокере.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Type

import pandas as pd

from backtester.base_strategy import Bar, BaseStrategy

from .broker import ExecutionConfig, LiveBroker, RiskLimits, SizingConfig
from .exchange import ExchangeClient, ExchangeError, drop_forming_candle


@dataclass
class EngineEvent:
    """Событие торгового цикла для UI/журнала."""

    kind: str                 # status|info|error|order|signal|position|balance|candles
    message: str = ""
    data: Optional[dict] = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EngineConfig:
    """Параметры запуска торгового цикла."""

    symbol: str
    timeframe: str = "1h"
    strategy_cls: Optional[Type[BaseStrategy]] = None
    strategy_params: dict[str, Any] = field(default_factory=dict)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    leverage: int = 5
    margin_mode: str = "isolated"
    window_bars: int = 800
    poll_seconds: float = 10.0
    dry_run: bool = False
    persist_state: bool = True   # сохранять состояние сессии на диск (для VPS-рестартов)


class LiveTradingEngine:
    """Событийный цикл live-торговли поверх ExchangeClient + LiveBroker."""

    def __init__(
        self,
        client: ExchangeClient,
        config: EngineConfig,
        on_event: Callable[[EngineEvent], None],
    ) -> None:
        self.client = client
        self.cfg = config
        self._on_event = on_event
        self._stop = threading.Event()
        self._last_processed_ts: Optional[pd.Timestamp] = None
        # Дневной предохранитель убытка: дата (UTC) и стартовый эквити суток.
        self._risk_day: Optional[Any] = None
        self._day_start_equity: float = 0.0
        self._store = self._make_store()

    def _make_store(self):
        """StateStore для текущей сессии или None (в сухом прогоне/при отказе)."""
        if not self.cfg.persist_state or self.cfg.dry_run:
            return None
        try:
            from .state import StateStore
            return StateStore.for_session(
                self.client.exchange_id, self.client.mode,
                self.cfg.symbol, self.cfg.timeframe,
            )
        except Exception:  # noqa: BLE001 - персистентность не критична для торговли
            return None

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        self._stop.set()

    @property
    def is_running(self) -> bool:
        return not self._stop.is_set()

    def _emit(self, kind: str, message: str = "", data: Optional[dict] = None) -> None:
        try:
            self._on_event(EngineEvent(kind=kind, message=message, data=data))
        except Exception:  # noqa: BLE001 - получатель не должен ронять цикл
            pass

    def _sleep(self, seconds: float) -> None:
        """Прерываемый сон: просыпается на stop() быстрее интервала."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self._stop.is_set():
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        cfg = self.cfg
        if cfg.strategy_cls is None:
            self._emit("error", "Стратегия не задана")
            return

        mode = "ДЕМО" if self.client.is_demo else "РЕАЛ"
        dry = " (сухой прогон)" if cfg.dry_run else ""
        self._emit(
            "status",
            f"Старт: {self.client.meta.label} [{mode}]{dry} • {cfg.symbol} • "
            f"{cfg.timeframe} • {cfg.strategy_cls.__name__}",
            {"running": True},
        )

        # Плечо/режим маржи -- best-effort (ошибки не критичны)
        if not cfg.dry_run:
            self.client.set_leverage(cfg.leverage, cfg.symbol, cfg.margin_mode)

        broker = LiveBroker(
            client=self.client,
            symbol=cfg.symbol,
            sizing=cfg.sizing,
            emit=lambda kind, msg, data: self._emit(kind, msg, data),
            dry_run=cfg.dry_run,
            execution=cfg.execution,
            risk=cfg.risk,
        )

        # Восстановление состояния прошлой сессии (SL/TP, лимит-вход, базис убытка):
        # ДО первого бара, чтобы sync не снял защитные ордера у живой позиции.
        self._restore_state(broker)

        # Стартовый снимок: история для графика + баланс + позиция
        try:
            df0 = self.client.fetch_ohlcv(cfg.symbol, cfg.timeframe, cfg.window_bars)
            self._emit("candles", "История загружена", {"df": df0})
            self._emit_balance()
            self._emit_position(broker)
            closed0 = drop_forming_candle(df0, cfg.timeframe)
            if not closed0.empty:
                self._last_processed_ts = closed0.index[-1]
                self._emit_plots_for(closed0, broker)  # индикаторы на график сразу
            self._persist_state(broker)
        except ExchangeError as exc:
            self._emit("error", f"Старт не удался: {exc}", {"fatal": True})
            self._emit("status", "Остановлено", {"running": False})
            return

        self._emit("info", f"Ждём закрытия нового бара ({cfg.timeframe})…")

        # Основной цикл
        while not self._stop.is_set():
            self._sleep(cfg.poll_seconds)
            if self._stop.is_set():
                break
            try:
                df = self.client.fetch_ohlcv(cfg.symbol, cfg.timeframe, cfg.window_bars)
            except ExchangeError as exc:
                self._emit("error", f"Сеть/биржа: {exc} — повтор позже")
                self._sleep(min(30.0, cfg.poll_seconds * 3))
                continue

            self._emit("candles", "", {"df": df})  # живое обновление графика
            closed = drop_forming_candle(df, cfg.timeframe)
            if closed.empty:
                continue
            new_ts = closed.index[-1]
            if self._last_processed_ts is not None and new_ts <= self._last_processed_ts:
                continue  # нового закрытого бара ещё нет

            # --- Новый закрытый бар: шаг стратегии ---
            last_price = float(closed["close"].iloc[-1])
            self._emit(
                "signal",
                f"Новый бар {new_ts:%Y-%m-%d %H:%M} • close={last_price:g}",
                {"ts": str(new_ts), "close": last_price},
            )
            try:
                bal = self._emit_balance()
                broker.begin_bar(
                    closed.index, last_price,
                    free_usdt=bal["free"], total_usdt=bal["total"],
                    now=pd.Timestamp(new_ts),
                )
                # Предохранитель дневного убытка ПЕРЕД новыми сделками
                if self._daily_loss_tripped(bal["total"], pd.Timestamp(new_ts)):
                    self._trip_kill_switch(broker)
                    break
                self._run_strategy_step(broker, closed)
                broker.sync_protective_orders()
                self._emit_position(broker)
            except ExchangeError as exc:
                self._emit("error", f"Биржа на баре: {exc} — продолжаю")
            except Exception as exc:  # noqa: BLE001 - ошибка стратегии: фатально
                self._emit(
                    "error",
                    f"Ошибка стратегии: {exc}\n{traceback.format_exc()}",
                    {"fatal": True},
                )
                break

            self._last_processed_ts = new_ts
            self._persist_state(broker)

        self._emit("status", "Остановлено", {"running": False})

    # ------------------------------------------------------------------ #
    # Предохранитель дневного убытка (kill switch)
    # ------------------------------------------------------------------ #
    def _daily_loss_tripped(self, equity: float, now: pd.Timestamp) -> bool:
        """True, если просадка эквити от старта суток (UTC) достигла лимита."""
        limit = self.cfg.risk.max_daily_loss_pct
        if limit <= 0:
            return False
        day = now.tz_convert("UTC").date() if now.tzinfo else now.date()
        if self._risk_day != day:                 # новые сутки -> новый базис
            self._risk_day = day
            self._day_start_equity = equity
            return False
        base = self._day_start_equity
        if base <= 0:
            self._day_start_equity = equity
            return False
        loss_pct = (base - equity) / base * 100.0
        if loss_pct >= limit:
            self._emit(
                "error",
                f"⛔ РУБИЛЬНИК: дневной убыток {loss_pct:.2f}% ≥ лимита {limit:.2f}% "
                f"(старт суток {base:.2f} → {equity:.2f} USDT). Торговля остановлена.",
                {"risk": "max_daily_loss", "fatal": True},
            )
            return True
        return False

    def _trip_kill_switch(self, broker: LiveBroker) -> None:
        """Закрыть позицию, снять ордера и остановить цикл."""
        try:
            broker.close_position(tag="kill-switch")
            broker.cancel_all_orders()
            self._emit_position(broker)
        except ExchangeError as exc:
            self._emit("error", f"Рубильник: не удалось закрыться чисто: {exc}", None)
        self._persist_state(broker)
        self._stop.set()

    # ------------------------------------------------------------------ #
    # Персистентность состояния сессии (для перезапусков на VPS)
    # ------------------------------------------------------------------ #
    def _restore_state(self, broker: LiveBroker) -> None:
        if self._store is None:
            return
        st = self._store.load()
        if not st:
            return
        rd = st.get("risk_day")
        if rd:
            try:
                from datetime import date
                self._risk_day = date.fromisoformat(str(rd))
                self._day_start_equity = float(st.get("day_start_equity") or 0.0)
            except (ValueError, TypeError):
                self._risk_day = None
                self._day_start_equity = 0.0
        broker.restore(st.get("broker") or {})
        if (broker._desired_sl is not None or broker._desired_tp is not None
                or broker._pending_entry is not None):
            self._emit(
                "info",
                f"Восстановлено состояние сессии: SL={broker._desired_sl} "
                f"TP={broker._desired_tp}"
                + (" • ждёт лимит-вход" if broker._pending_entry else ""),
                None,
            )

    def _persist_state(self, broker: LiveBroker) -> None:
        if self._store is None:
            return
        st = {
            "risk_day": self._risk_day.isoformat() if self._risk_day else None,
            "day_start_equity": self._day_start_equity,
            "last_processed_ts": (
                str(self._last_processed_ts)
                if self._last_processed_ts is not None else None
            ),
            "broker": broker.snapshot(),
        }
        self._store.save(st)

    # ------------------------------------------------------------------ #
    def _run_strategy_step(self, broker: LiveBroker, window: pd.DataFrame) -> None:
        cfg = self.cfg
        strat = cfg.strategy_cls(**cfg.strategy_params)  # type: ignore[misc]
        strat._bind(broker, window)
        strat.init()
        i = len(window) - 1
        strat._set_index(i)
        row = window.iloc[i]
        bar = Bar(
            i, float(row["open"]), float(row["high"]), float(row["low"]),
            float(row["close"]), float(row["volume"]), window.index,
        )
        strat.next(bar)
        # индикаторы стратегии -> на график (пересчитаны над этим же окном)
        plots = getattr(strat, "_plots", None)
        if plots:
            self._emit("indicators", "", {"plots": plots, "index": window.index})

    def _emit_plots_for(self, window: pd.DataFrame, broker: LiveBroker) -> None:
        """Снять индикаторы стратегии над окном (без торговли) -- для графика."""
        cfg = self.cfg
        try:
            strat = cfg.strategy_cls(**cfg.strategy_params)  # type: ignore[misc]
            strat._bind(broker, window)
            strat.init()
        except Exception:  # noqa: BLE001 - индикаторы для графика не критичны
            return
        plots = getattr(strat, "_plots", None)
        if plots:
            self._emit("indicators", "", {"plots": plots, "index": window.index})

    def _emit_balance(self) -> dict[str, Any]:
        bal = self.client.fetch_quote_balance()
        cur = bal.get("currency", "USDT")
        self._emit(
            "balance",
            f"Баланс: {bal['total']:.2f} {cur} (свободно {bal['free']:.2f})",
            bal,
        )
        return bal

    def _emit_position(self, broker: LiveBroker) -> None:
        if broker.dry_run:
            lp = broker._position  # в сухом прогоне -- локальная модель
        else:
            try:
                lp = self.client.fetch_position(self.cfg.symbol)
                broker._position = lp
            except ExchangeError:
                lp = broker._position
        if lp is None or lp.size <= 0:
            self._emit("position", "Позиция: нет (плоско)", {"flat": True})
        else:
            self._emit(
                "position",
                f"Позиция: {lp.side.upper()} {lp.size:g} @ {lp.entry_price:g} "
                f"• uPnL {lp.unrealized_pnl:+.2f}",
                {
                    "side": lp.side, "size": lp.size,
                    "entry_price": lp.entry_price, "upnl": lp.unrealized_pnl,
                },
            )
