"""Юнит-тесты LiveBroker против фейковой биржи (без сети).

Проверяют, что брокер корректно транслирует намерения стратегии
(buy/sell/close + неттинг + сайзинг + SL/TP) в биржевые ордера.
"""
from __future__ import annotations

import pandas as pd
import pytest

from live.broker import (
    ExecutionConfig,
    LiveBroker,
    ProtectiveOrderError,
    RiskLimits,
    SizingConfig,
)
from live.exchange import LivePosition, drop_forming_candle


class FakeExchange:
    """Минимальная имитация ExchangeClient: ведёт нетто-позицию в контрактах."""

    def __init__(self) -> None:
        self.orders: list[tuple] = []          # (side, amount, reduce_only)
        self.limit_orders: list[tuple] = []    # (side, amount, price, reduce_only, post_only)
        self.stops: list[tuple] = []           # (side, amount, price, kind)
        self.canceled_ids: list = []
        self.cancels = 0
        self.net = 0.0                         # signed contracts
        self.reject = False                    # рыночный ордер -> отказ биржи
        self.ticker = 100.0                    # текущая цена (для проверки проскальзывания)
        self.fill_ratio = 1.0                  # доля заливки рыночного ордера (частичный филл)
        self.fail_stop_kinds: set[str] = set() # виды стоп-ордеров, которые «отклоняет» биржа

    def contract_size(self, symbol: str) -> float:
        return 1.0

    def fetch_ticker_last(self, symbol: str) -> float:
        return self.ticker

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return round(amount, 6)

    def min_amount(self, symbol: str) -> float:
        return 0.0

    def create_market_order(self, symbol, side, amount, reduce_only=False, params=None):
        if self.reject:
            raise RuntimeError("биржа отклонила ордер")
        self.orders.append((side, amount, reduce_only))
        filled = amount * self.fill_ratio
        self.net += filled if side == "buy" else -filled
        return {"id": len(self.orders), "average": 100.0, "filled": filled}

    def create_limit_order(self, symbol, side, amount, price,
                           reduce_only=False, post_only=False, params=None):
        self.limit_orders.append((side, amount, price, reduce_only, post_only))
        # Лимит «висит» в стакане: нетто не меняем (фиксируется в тесте вручную).
        return {"id": f"lim{len(self.limit_orders)}"}

    def create_stop_order(self, symbol, side, amount, stop_price, kind="stop"):
        if kind in self.fail_stop_kinds:
            raise RuntimeError(f"биржа отклонила {kind}")
        self.stops.append((side, amount, stop_price, kind))
        return {"id": "stop"}

    def cancel_order(self, order_id, symbol):
        self.canceled_ids.append(order_id)

    def cancel_all_orders(self, symbol):
        self.cancels += 1
        self.stops.clear()

    def fetch_position(self, symbol):
        if abs(self.net) < 1e-12:
            return None
        side = "long" if self.net > 0 else "short"
        return LivePosition(side=side, size=abs(self.net), entry_price=100.0)


def _events():
    out = []
    return out, (lambda kind, msg, data: out.append((kind, msg)))


def _index(n=10):
    return pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")


def _new_broker(sizing=None, dry_run=False, execution=None, risk=None):
    ex = FakeExchange()
    events, emit = _events()
    broker = LiveBroker(
        client=ex, symbol="BTC/USDT:USDT",
        sizing=sizing or SizingConfig(mode="notional", notional_usdt=100.0),
        emit=emit, dry_run=dry_run, execution=execution, risk=risk,
    )
    return ex, broker, events


def _begin(broker, price=100.0, total=1000.0, free=1000.0):
    idx = _index()
    broker.begin_bar(idx, price, free_usdt=free, total_usdt=total, now=idx[-1])


# --------------------------------------------------------------------------- #
def test_auto_size_market_buy_from_flat():
    ex, broker, _ = _new_broker()
    _begin(broker, price=100.0)
    broker.buy()                                  # ноционал 100 / цена 100 = 1 контракт
    assert ex.orders == [("buy", 1.0, False)]
    assert broker.position is not None
    assert broker.position.is_long
    assert broker.position.size == pytest.approx(1.0)


def test_explicit_size_partial_close_reduces_long():
    ex, broker, _ = _new_broker()
    _begin(broker, price=100.0)
    broker.buy()                                  # +1 long
    _begin(broker, price=100.0)                   # биржа подтверждает long 1.0
    broker.sell(size=0.25)                         # частичное закрытие 25%
    assert ex.orders[-1] == ("sell", 0.25, False)
    assert broker.position.size == pytest.approx(0.75)
    assert broker.position.is_long                # всё ещё long


def test_close_position_is_reduce_only():
    ex, broker, _ = _new_broker()
    _begin(broker, price=100.0)
    broker.buy()
    _begin(broker, price=100.0)
    broker.close_position()
    side, amount, reduce_only = ex.orders[-1]
    assert side == "sell" and reduce_only is True
    assert amount == pytest.approx(1.0)
    assert broker.position is None


def test_flip_short_to_long_nets_through_zero():
    ex, broker, _ = _new_broker()
    _begin(broker, price=100.0)
    broker.sell()                                  # открыть short 1.0
    assert ex.net == pytest.approx(-1.0)
    _begin(broker, price=100.0)                    # биржа: short 1.0
    # авто-ордер buy 1.0: в one-way режиме закрывает short и оставляет ~флэт
    broker.buy()
    assert ex.orders[-1] == ("buy", 1.0, False)
    assert ex.net == pytest.approx(0.0)


def test_risk_sizing_uses_stop_distance():
    # risk 1% от 1000 USDT = 10 USDT; дистанция до стопа = 100-95 = 5 -> 2 контракта
    ex, broker, _ = _new_broker()
    _begin(broker, price=100.0, total=1000.0)
    broker.buy(risk_pct=0.01, stop_loss=95.0)
    assert ex.orders[-1][0] == "buy"
    assert ex.orders[-1][1] == pytest.approx(2.0)


def test_protective_orders_placed_and_resized():
    ex, broker, _ = _new_broker()
    _begin(broker, price=100.0)
    broker.buy(stop_loss=95.0, take_profit=110.0)
    broker.sync_protective_orders()
    kinds = {(s[0], s[3]) for s in ex.stops}       # (side, kind)
    assert ("sell", "stop") in kinds               # SL для лонга — продажа
    assert ("sell", "take_profit") in kinds        # TP для лонга — продажа
    assert all(s[1] == pytest.approx(1.0) for s in ex.stops)


def test_dry_run_places_no_real_orders():
    ex, broker, events = _new_broker(dry_run=True)
    _begin(broker, price=100.0)
    broker.buy()
    broker.sync_protective_orders()
    assert ex.orders == []                          # на биржу ничего не ушло
    assert ex.stops == []
    assert broker.position is not None              # но локальная модель ведётся
    assert any(kind == "order" for kind, _ in events)


# --------------------------------------------------------------------------- #
# Фикс №2: отклонённый ордер не порождает фантомную позицию
# --------------------------------------------------------------------------- #
def test_rejected_order_does_not_update_local_model():
    ex, broker, events = _new_broker()
    ex.reject = True
    _begin(broker, price=100.0)
    broker.buy()
    assert ex.orders == []                          # на бирже ничего не открылось
    assert broker.position is None                  # и локальная модель не выдумала вход
    assert any(kind == "error" for kind, _ in events)


def test_rejected_close_keeps_position():
    ex, broker, _ = _new_broker()
    _begin(broker, price=100.0)
    broker.buy()
    _begin(broker, price=100.0)                      # биржа: long 1.0
    ex.reject = True
    broker.close_position()
    assert broker.position is not None              # отказ закрытия -> позиция как была
    assert broker.position.size == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Фикс №5: предохранители (kill switch / circuit breaker)
# --------------------------------------------------------------------------- #
def test_max_notional_rejects_oversized_order():
    # ноционал 100 USDT при лимите 50 -> вход отклоняется (защита «толстого пальца»)
    ex, broker, events = _new_broker(risk=RiskLimits(max_notional_per_order=50.0))
    _begin(broker, price=100.0)
    broker.buy()
    assert ex.orders == []
    assert broker.position is None
    assert any(kind == "error" for kind, _ in events)


def test_max_notional_allows_within_cap():
    ex, broker, _ = _new_broker(risk=RiskLimits(max_notional_per_order=150.0))
    _begin(broker, price=100.0)
    broker.buy()                                     # ноционал 100 < 150 -> проходит
    assert ex.orders == [("buy", 1.0, False)]


def test_max_orders_per_hour_blocks_runaway():
    ex, broker, _ = _new_broker(risk=RiskLimits(max_orders_per_hour=1))
    _begin(broker, price=100.0)
    broker.buy()                                     # 1-й вход — ок
    broker.buy()                                     # 2-й за тот же час — заблокирован
    assert len(ex.orders) == 1


def test_close_bypasses_rate_limit():
    # закрытие не должно блокироваться лимитом частоты входов
    ex, broker, _ = _new_broker(risk=RiskLimits(max_orders_per_hour=1))
    _begin(broker, price=100.0)
    broker.buy()
    _begin(broker, price=100.0)
    broker.close_position()                          # reduce-only закрытие проходит
    assert ex.orders[-1][2] is True                  # последний ордер reduceOnly


# --------------------------------------------------------------------------- #
# Максимальное проскальзывание: рыночный вход отменяется при сильном движении
# --------------------------------------------------------------------------- #
def test_slippage_blocks_entry_when_price_moved_against():
    # лимит 0.5%, цена ушла вверх на 1% -> рыночный buy не выставляется
    ex, broker, events = _new_broker(execution=ExecutionConfig(max_slippage_pct=0.5))
    _begin(broker, price=100.0)
    ex.ticker = 101.0
    broker.buy()
    assert ex.orders == []
    assert broker.position is None
    assert any(kind == "info" for kind, _ in events)


def test_slippage_allows_entry_within_limit():
    ex, broker, _ = _new_broker(execution=ExecutionConfig(max_slippage_pct=2.0))
    _begin(broker, price=100.0)
    ex.ticker = 101.0                               # +1% < лимита 2% -> проходит
    broker.buy()
    assert ex.orders == [("buy", 1.0, False)]


def test_slippage_ignores_favorable_move():
    # для buy падение цены — В ПОЛЬЗУ входа, не блокируется даже при узком лимите
    ex, broker, _ = _new_broker(execution=ExecutionConfig(max_slippage_pct=0.5))
    _begin(broker, price=100.0)
    ex.ticker = 99.0
    broker.buy()
    assert ex.orders == [("buy", 1.0, False)]


def test_slippage_never_blocks_close():
    ex, broker, _ = _new_broker(execution=ExecutionConfig(max_slippage_pct=0.1))
    _begin(broker, price=100.0)
    broker.buy()                                    # вход без движения (ticker=100)
    _begin(broker, price=100.0)
    ex.ticker = 200.0                               # огромное движение
    broker.close_position()                         # закрытие обязано пройти
    assert ex.orders[-1][2] is True                 # reduceOnly закрытие исполнено


def test_slippage_disabled_by_default_does_not_query_ticker():
    # при выключенной проверке (default 0) тикер не запрашивается даже при движении
    ex, broker, _ = _new_broker()
    _begin(broker, price=100.0)
    ex.ticker = 150.0
    broker.buy()
    assert ex.orders == [("buy", 1.0, False)]


# --------------------------------------------------------------------------- #
# Фикс №7: лимитные (мейкерские) входы и мейкерский тейк-профит
# --------------------------------------------------------------------------- #
def test_limit_entry_places_post_only_without_optimistic_fill():
    ex, broker, _ = _new_broker(execution=ExecutionConfig(entry_limit=True))
    _begin(broker, price=100.0)
    broker.buy(stop_loss=95.0)
    assert ex.orders == []                            # рыночного ордера нет
    assert len(ex.limit_orders) == 1
    side, amount, price, reduce_only, post_only = ex.limit_orders[0]
    assert side == "buy" and post_only is True and reduce_only is False
    assert broker.position is None                    # фантомного филла нет (ждём биржу)


def test_limit_entry_adopts_sl_when_filled():
    ex, broker, _ = _new_broker(execution=ExecutionConfig(entry_limit=True))
    _begin(broker, price=100.0)
    broker.buy(stop_loss=95.0, take_profit=110.0)
    ex.net = 1.0                                      # симулируем заливку лимита
    _begin(broker, price=100.0)                       # новый бар -> сверка
    assert broker.position is not None
    assert broker.position.size == pytest.approx(1.0)
    assert broker._desired_sl == pytest.approx(95.0)
    assert broker._desired_tp == pytest.approx(110.0)
    assert broker._pending_entry is None


def test_limit_entry_cancelled_when_not_filled():
    ex, broker, _ = _new_broker(execution=ExecutionConfig(entry_limit=True))
    _begin(broker, price=100.0)
    broker.buy(stop_loss=95.0)
    # лимит не залился: net остаётся 0
    _begin(broker, price=100.0)
    assert broker.position is None
    assert broker._pending_entry is None
    assert ex.canceled_ids == ["lim1"]                # устаревший лимит снят


def test_tp_maker_uses_reduce_only_limit():
    ex, broker, _ = _new_broker(execution=ExecutionConfig(tp_maker=True))
    _begin(broker, price=100.0)
    broker.buy(stop_loss=95.0, take_profit=110.0)     # рыночный вход (entry_limit=False)
    broker.sync_protective_orders()
    # SL остаётся стоп-маркетом
    assert ("sell", "stop") in {(s[0], s[3]) for s in ex.stops}
    # TP -- reduce-only постонли лимит на 110
    assert len(ex.limit_orders) == 1
    side, amount, price, reduce_only, post_only = ex.limit_orders[0]
    assert side == "sell" and reduce_only is True and post_only is True
    assert price == pytest.approx(110.0)


# --------------------------------------------------------------------------- #
# Персистентность состояния (устойчивость к перезапускам на VPS)
# --------------------------------------------------------------------------- #
def test_state_store_round_trip(tmp_path):
    from live.state import StateStore

    store = StateStore(tmp_path / "session.json")
    assert store.load() == {}                          # нет файла -> пусто
    store.save({"day_start_equity": 1000.0, "broker": {"desired_sl": 95.0}})
    loaded = store.load()
    assert loaded["broker"]["desired_sl"] == 95.0
    assert loaded["day_start_equity"] == 1000.0
    store.clear()
    assert store.load() == {}                          # удалён


def test_state_store_tolerates_corrupt_file(tmp_path):
    from live.state import StateStore

    path = tmp_path / "bad.json"
    path.write_text("{ это не json", encoding="utf-8")
    assert StateStore(path).load() == {}               # битый файл -> пусто, без падения


def test_snapshot_restore_keeps_protective_orders_after_restart():
    # после рестарта первый sync ПЕРЕСТАВЛЯЕТ SL/TP, а не снимает их у живой позиции
    ex, broker, _ = _new_broker()
    _begin(broker, price=100.0)
    broker.buy(stop_loss=95.0, take_profit=110.0)
    snap = broker.snapshot()
    assert snap["desired_sl"] == 95.0 and snap["desired_tp"] == 110.0

    # «Рестарт»: свежий брокер, биржа уже держит позицию long 1.0
    ex2 = FakeExchange()
    ex2.net = 1.0
    _events_out, emit = _events()
    fresh = LiveBroker(
        client=ex2, symbol="BTC/USDT:USDT",
        sizing=SizingConfig(mode="notional", notional_usdt=100.0), emit=emit,
    )
    fresh.restore(snap)
    assert fresh._desired_sl == 95.0 and fresh._desired_tp == 110.0
    _begin(fresh, price=100.0)                          # читает позицию с биржи
    fresh.sync_protective_orders()
    kinds = {s[3] for s in ex2.stops}
    assert "stop" in kinds and "take_profit" in kinds   # защита переставлена, не снята


def test_snapshot_restore_pending_limit_entry():
    ex, broker, _ = _new_broker(execution=ExecutionConfig(entry_limit=True))
    _begin(broker, price=100.0)
    broker.buy(stop_loss=95.0)
    snap = broker.snapshot()
    assert snap["pending_entry"]["side"] == "BUY"
    assert snap["pending_entry"]["sl"] == 95.0

    fresh = _new_broker()[1]
    fresh.restore(snap)
    assert fresh._pending_entry is not None
    assert fresh._pending_entry["side"].name == "BUY"
    assert fresh._pending_entry["sl"] == 95.0


# --------------------------------------------------------------------------- #
# Защита позиции: без стоп-лосса не торгуем (особенно важно для DEX)
# --------------------------------------------------------------------------- #
def test_protective_sl_failure_raises_and_blocks_naked():
    # биржа отклонила стоп -> позиция осталась бы голой -> критическая ошибка
    ex, broker, _ = _new_broker()
    ex.fail_stop_kinds = {"stop"}
    _begin(broker, price=100.0)
    broker.buy(stop_loss=95.0)
    with pytest.raises(ProtectiveOrderError):
        broker.sync_protective_orders()


def test_protective_naked_allowed_when_flag_off():
    # require_protective=False -> не бросаем, но громко предупреждаем
    ex, broker, events = _new_broker(
        execution=ExecutionConfig(require_protective=False)
    )
    ex.fail_stop_kinds = {"stop"}
    _begin(broker, price=100.0)
    broker.buy(stop_loss=95.0)
    broker.sync_protective_orders()                  # не бросает
    assert any(kind == "error" for kind, _ in events)


def test_protective_invalid_sl_side_is_critical():
    # лонг со стопом ВЫШЕ входа -- неверная сторона -> критично (не выставляем мусор)
    ex, broker, _ = _new_broker()
    _begin(broker, price=100.0)
    broker.buy(stop_loss=105.0)                       # вход ~100, стоп выше -- неверно
    with pytest.raises(ProtectiveOrderError):
        broker.sync_protective_orders()
    assert ex.stops == []                             # битый стоп на биржу не ушёл


def test_protective_tp_failure_is_non_critical():
    # стоп встал, тейк отклонён -> торговля продолжается (TP не критичен)
    ex, broker, events = _new_broker()
    ex.fail_stop_kinds = {"take_profit"}
    _begin(broker, price=100.0)
    broker.buy(stop_loss=95.0, take_profit=110.0)
    broker.sync_protective_orders()                  # не бросает
    assert ("sell", 95.0, "stop") in [(s[0], s[2], s[3]) for s in ex.stops]
    assert any(kind == "error" and "ейк" in msg for kind, msg in events)


def test_partial_fill_updates_model_by_filled_amount():
    # рыночный ордер залился наполовину -> локальная модель = фактический объём
    ex, broker, _ = _new_broker()
    ex.fill_ratio = 0.5
    _begin(broker, price=100.0)
    broker.buy(size=1.0)                              # запрос 1.0, залилось 0.5
    assert ex.orders[-1] == ("buy", 1.0, False)       # на биржу ушёл запрошенный объём
    assert broker.position is not None
    assert broker.position.size == pytest.approx(0.5)  # но в модели -- фактический филл


# --------------------------------------------------------------------------- #
def test_drop_forming_candle_removes_open_bar():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": range(5)}, index=idx)
    # «сейчас» = время последней свечи + 10 минут -> последний бар ещё открыт
    now_ms = int(idx[-1].timestamp() * 1000) + 10 * 60 * 1000
    out = drop_forming_candle(df, "1h", now_ms=now_ms)
    assert len(out) == 4
    # «сейчас» сильно позже -> последний бар уже закрыт, ничего не режем
    now_ms2 = int(idx[-1].timestamp() * 1000) + 2 * 3600 * 1000
    assert len(drop_forming_candle(df, "1h", now_ms=now_ms2)) == 5
