"""Тесты ядра: модель исполнения, инварианты учёта, отсутствие lookahead."""
from __future__ import annotations

import numpy as np
import pytest

from backtester import (
    BacktestEngine,
    Bar,
    BaseStrategy,
    ExitReason,
    OrderStatus,
    OrderType,
)
from strategies import SmaCrossStrategy

from conftest import make_bars


class BuyOnce(BaseStrategy):
    """Один рыночный ордер на заданном баре."""

    params = {"at": 1, "size": 1.0, "sl": 0.0, "tp": 0.0}

    def init(self) -> None:
        self.done = False

    def next(self, bar: Bar) -> None:
        if not self.done and bar.index == self.p["at"]:
            self.buy(
                size=self.p["size"],
                stop_loss=self.p["sl"] or None,
                take_profit=self.p["tp"] or None,
            )
            self.done = True


class SellOnce(BaseStrategy):
    params = {"at": 1, "size": 1.0}

    def init(self) -> None:
        self.done = False

    def next(self, bar: Bar) -> None:
        if not self.done and bar.index == self.p["at"]:
            self.sell(size=self.p["size"])
            self.done = True


class BuyStopOnce(BaseStrategy):
    """Buy-stop на заданном уровне, выставляется на первом баре."""

    params = {"stop": 0.0, "size": 1.0}

    def init(self) -> None:
        self.done = False

    def next(self, bar: Bar) -> None:
        if not self.done:
            self.buy(size=self.p["size"], stop_price=self.p["stop"])
            self.done = True


class RiskEntry(BaseStrategy):
    """Вход с сайзингом от риска и очень близким стопом."""

    params = {"risk_pct": 0.1, "sl_dist": 0.1}

    def init(self) -> None:
        self.done = False

    def next(self, bar: Bar) -> None:
        if not self.done:
            self.buy(risk_pct=self.p["risk_pct"], stop_loss=bar.close - self.p["sl_dist"])
            self.done = True


FLAT = [(100, 101, 99, 100)] * 10


def test_market_fills_next_open():
    data = make_bars([(100, 101, 99, 100), (100, 101, 99, 100),
                      (105, 106, 104, 105), (105, 106, 104, 105)])
    result = BacktestEngine(data, BuyOnce, {"at": 1}).run()
    order = result.orders[0]
    assert order.status is OrderStatus.FILLED
    assert order.created_bar == 1
    assert order.filled_bar == 2
    assert order.filled_price == pytest.approx(105.0)  # open бара 2


def test_buy_pays_spread_sell_does_not():
    data = make_bars(FLAT)
    res = BacktestEngine(data, BuyOnce, {"at": 1}, spread_pct=0.01).run()
    assert res.orders[0].filled_price == pytest.approx(100.0 * 1.01)
    # закрытие лонга (продажа) в finalize -- по цене графика, без спреда
    assert res.trades[-1].exit_price == pytest.approx(100.0)

    res_short = BacktestEngine(data, SellOnce, {"at": 1}, spread_pct=0.01).run()
    assert res_short.orders[0].filled_price == pytest.approx(100.0)
    # выкуп шорта (покупка) платит спред
    assert res_short.trades[-1].exit_price == pytest.approx(100.0 * 1.01)


def test_slippage_against_trader():
    data = make_bars(FLAT)
    res = BacktestEngine(data, BuyOnce, {"at": 1}, slippage_pct=0.002).run()
    assert res.orders[0].filled_price == pytest.approx(100.0 * 1.002)


def test_buy_stop_fills_on_breakout():
    data = make_bars([
        (100, 101, 99, 100),   # бар 0: выставлен buy-stop на 103
        (100, 102, 99, 101),   # бар 1: high 102 < 103 -- не исполняется
        (101, 104, 100, 103),  # бар 2: high 104 >= 103 -- fill по 103
        (103, 104, 102, 103),
    ])
    res = BacktestEngine(data, BuyStopOnce, {"stop": 103.0}).run()
    order = res.orders[0]
    assert order.type is OrderType.STOP
    assert order.filled_bar == 2
    assert order.filled_price == pytest.approx(103.0)


def test_buy_stop_gap_fills_at_open():
    data = make_bars([
        (100, 101, 99, 100),
        (110, 112, 109, 111),  # гэп выше уровня 103 -- fill по open 110
        (111, 112, 110, 111),
    ])
    res = BacktestEngine(data, BuyStopOnce, {"stop": 103.0}).run()
    assert res.orders[0].filled_price == pytest.approx(110.0)


def test_max_leverage_caps_risk_sizing():
    data = make_bars(FLAT)
    # без капа: size = 10000 * 0.1 / 0.1 = 10000 единиц (ноционал 100x)
    res = BacktestEngine(data, RiskEntry, cash=10_000.0).run()
    uncapped = res.trades[0].size
    assert uncapped == pytest.approx(10_000.0, rel=0.01)

    res_capped = BacktestEngine(data, RiskEntry, cash=10_000.0, max_leverage=5.0).run()
    capped = res_capped.trades[0].size
    # ноционал <= equity * 5: 10000 * 5 / 100 = 500 единиц
    assert capped == pytest.approx(500.0, rel=0.01)


def test_sl_priority_when_sl_and_tp_in_same_bar():
    data = make_bars([
        (100, 101, 99, 100),
        (100, 106, 94, 100),   # задеты и SL (95) и TP (105) -> консервативно SL
        (100, 101, 99, 100),
    ])
    res = BacktestEngine(data, BuyOnce, {"at": 0, "sl": 95.0, "tp": 105.0}).run()
    trade = res.trades[0]
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_price == pytest.approx(95.0)


def test_limit_fills_at_better_open():
    class BuyLimit(BaseStrategy):
        params = {"limit": 100.0}

        def init(self) -> None:
            self.done = False

        def next(self, bar: Bar) -> None:
            if not self.done:
                self.buy(size=1.0, limit_price=self.p["limit"])
                self.done = True

    data = make_bars([
        (100, 101, 99, 100),
        (98, 99, 97, 98),      # открылись ниже лимита 100 -- fill по 98
        (98, 99, 97, 98),
    ])
    res = BacktestEngine(data, BuyLimit).run()
    assert res.orders[0].filled_price == pytest.approx(98.0)


def test_cash_conservation(synth_1h):
    """Финальная эквити == старт + сумма PnL сделок (комиссии внутри PnL)."""
    res = BacktestEngine(
        synth_1h, SmaCrossStrategy,
        cash=10_000.0, commission_pct=0.0005, slippage_pct=0.0002,
        spread_pct=0.0001,
    ).run()
    assert len(res.trades) > 5
    pnl_sum = sum(t.pnl for t in res.trades)
    assert res.final_equity == pytest.approx(10_000.0 + pnl_sum, abs=1e-6)


def test_no_lookahead_truncation(synth_1h):
    """Будущие бары не влияют на прошлые сделки.

    Прогон на полных данных и на префиксе должен дать идентичные сделки,
    завершившиеся до конца префикса (END_OF_DATA не сравнивается).
    """
    cut = 1000
    full = BacktestEngine(synth_1h, SmaCrossStrategy, cash=10_000.0).run()
    part = BacktestEngine(synth_1h.iloc[:cut], SmaCrossStrategy, cash=10_000.0).run()

    full_done = [t for t in full.trades if t.exit_bar < cut - 1]
    part_done = [t for t in part.trades if t.exit_reason is not ExitReason.END_OF_DATA]
    assert len(part_done) == len(full_done)
    for a, b in zip(part_done, full_done, strict=True):
        assert a.entry_time == b.entry_time
        assert a.exit_time == b.exit_time
        assert a.entry_price == pytest.approx(b.entry_price)
        assert a.exit_price == pytest.approx(b.exit_price)
        assert a.size == pytest.approx(b.size)


def test_warmup_blocks_trading(synth_1h):
    warmup = 200
    res = BacktestEngine(
        synth_1h.iloc[:600], SmaCrossStrategy, warmup_bars=warmup
    ).run()
    assert all(o.created_bar >= warmup for o in res.orders)
    # эквити до конца прогрева плоская
    head = res.equity_curve.iloc[: warmup + 1].to_numpy()
    assert np.allclose(head, head[0])


def test_unknown_strategy_param_raises():
    with pytest.raises(ValueError, match="Неизвестные параметры"):
        SmaCrossStrategy(nonexistent=1)


def test_stop_and_limit_mutually_exclusive():
    data = make_bars(FLAT)

    class BadOrder(BaseStrategy):
        def init(self) -> None:
            pass

        def next(self, bar: Bar) -> None:
            self.buy(size=1.0, limit_price=99.0, stop_price=101.0)

    with pytest.raises(ValueError, match="взаимоисключающие"):
        BacktestEngine(data, BadOrder).run()


def test_strategy_indicators_captured_for_chart():
    """Линии, объявленные через plot_line() в init(), попадают в результат.

    Это контракт оверлеев на графике: GUI рисует именно BacktestResult.indicators.
    """
    data = make_bars(FLAT * 5)  # 50 баров -- хватает на прогрев SMA(10)
    result = BacktestEngine(
        data, SmaCrossStrategy, {"fast_period": 5, "slow_period": 10}
    ).run()
    names = [ind["name"] for ind in result.indicators]
    assert names == ["SMA 5", "SMA 10"]
    fast = result.indicators[0]
    assert fast["overlay"] is True
    assert len(fast["values"]) == len(data)
