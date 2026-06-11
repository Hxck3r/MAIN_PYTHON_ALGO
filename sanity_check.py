"""
Проверки честности движка бэктестинга (sanity-тесты).

Запуск:
    python sanity_check.py

Каждая проверка печатает OK или FAIL с пояснением. Если все OK --
ядру можно доверять. Запускайте этот скрипт после любых правок
в backtester/, чтобы убедиться, что ничего не сломалось.
"""
from __future__ import annotations

import sys

import numpy as np

from backtester import (
    BacktestEngine,
    Bar,
    BaseStrategy,
    ExitReason,
    OrderSide,
    generate_synthetic_data,
)
from strategies import SmaCrossStrategy

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PASSED: list[str] = []
FAILED: list[str] = []


def report(name: str, ok: bool, details: str = "") -> None:
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {name}" + (f" -- {details}" if details else ""))
    (PASSED if ok else FAILED).append(name)


# ---------------------------------------------------------------------- #
# Вспомогательные стратегии для тестов
# ---------------------------------------------------------------------- #
class DoNothingStrategy(BaseStrategy):
    """Никогда не торгует."""

    def init(self) -> None:
        pass

    def next(self, bar: Bar) -> None:
        pass


class AlwaysInWithLevelsStrategy(BaseStrategy):
    """Входит в лонг каждый раз, когда позиции нет; SL/TP в % от цены.

    Уровни SL/TP записываются в tag ордера, чтобы после прогона можно
    было сверить фактическую цену выхода с заявленным уровнем.
    """

    params = {"sl_pct": 0.01, "tp_pct": 0.02}

    def init(self) -> None:
        pass

    def next(self, bar: Bar) -> None:
        if self.position is None:
            sl = bar.close * (1.0 - self.p["sl_pct"])
            tp = bar.close * (1.0 + self.p["tp_pct"])
            self.buy(stop_loss=sl, take_profit=tp, tag=f"{sl:.10f}|{tp:.10f}")


class RiskSizedStrategy(BaseStrategy):
    """Лонг с сайзингом от риска: убыток по SL должен равняться risk_pct эквити."""

    params = {"risk_pct": 0.02, "sl_pct": 0.01}

    def init(self) -> None:
        pass

    def next(self, bar: Bar) -> None:
        if self.position is None:
            self.buy(
                risk_pct=self.p["risk_pct"],
                stop_loss=bar.close * (1.0 - self.p["sl_pct"]),
            )


# ---------------------------------------------------------------------- #
# Проверки
# ---------------------------------------------------------------------- #
def check_1_balance_consistency() -> None:
    """Сумма pnl всех сделок == изменение счёта (деньги не теряются)."""
    data = generate_synthetic_data(n_bars=1500, seed=11)
    result = BacktestEngine(
        data, SmaCrossStrategy,
        strategy_params={"fast_period": 8, "slow_period": 40},
        cash=10_000, commission_pct=0.001, slippage_pct=0.0002,
    ).run()

    pnl_sum = sum(t.pnl for t in result.trades)
    account_change = result.final_equity - result.initial_cash
    diff = abs(pnl_sum - account_change)
    report(
        "Сходимость баланса: сумма pnl сделок == изменение счёта",
        diff < 1e-6,
        f"сделок {len(result.trades)}, расхождение {diff:.2e}",
    )


def check_2_do_nothing() -> None:
    """Пустая стратегия: ноль сделок, эквити не меняется."""
    data = generate_synthetic_data(n_bars=500, seed=22)
    result = BacktestEngine(
        data, DoNothingStrategy, cash=10_000, commission_pct=0.001
    ).run()

    flat = bool(np.all(result.equity_curve.to_numpy() == 10_000.0))
    report(
        "Пустая стратегия: 0 сделок и ровная эквити",
        len(result.trades) == 0 and flat,
        f"сделок {len(result.trades)}, эквити постоянна: {flat}",
    )


def check_3_costs_hurt() -> None:
    """Издержки (комиссия + слиппедж) уменьшают прибыль на тех же данных."""
    data = generate_synthetic_data(n_bars=1500, seed=33)
    params = {"fast_period": 8, "slow_period": 40}

    free = BacktestEngine(
        data, SmaCrossStrategy, strategy_params=params,
        commission_pct=0.0, slippage_pct=0.0,
    ).run()
    costly = BacktestEngine(
        data, SmaCrossStrategy, strategy_params=params,
        commission_pct=0.002, slippage_pct=0.0005,
    ).run()

    report(
        "Издержки уменьшают прибыль (комиссия 0.2% + слиппедж 0.05%)",
        costly.net_profit < free.net_profit and len(free.trades) > 0,
        f"без издержек {free.net_profit:,.2f}, с издержками {costly.net_profit:,.2f}",
    )


def check_4_sl_tp_levels() -> None:
    """Цена выхода по SL не лучше уровня стопа, по TP -- не хуже тейка."""
    data = generate_synthetic_data(n_bars=1500, volatility=0.008, seed=44)
    result = BacktestEngine(
        data, AlwaysInWithLevelsStrategy,
        commission_pct=0.0, slippage_pct=0.0,  # без шума, проверяем сами уровни
    ).run()

    sl_trades = [t for t in result.trades if t.exit_reason is ExitReason.STOP_LOSS]
    tp_trades = [t for t in result.trades if t.exit_reason is ExitReason.TAKE_PROFIT]
    eps = 1e-9

    sl_ok = all(
        t.exit_price <= float(t.tag.split("|")[0]) + eps for t in sl_trades
    )
    tp_ok = all(
        t.exit_price >= float(t.tag.split("|")[1]) - eps for t in tp_trades
    )
    long_only = all(t.side is OrderSide.BUY for t in result.trades)

    report(
        "SL/TP исполняются по своим уровням (или хуже при гэпе, не лучше)",
        sl_ok and tp_ok and long_only and (sl_trades or tp_trades),
        f"выходов по SL: {len(sl_trades)}, по TP: {len(tp_trades)}",
    )


def check_5_reproducibility() -> None:
    """Два одинаковых запуска дают идентичный результат."""
    data = generate_synthetic_data(n_bars=1000, seed=55)
    kwargs = dict(
        strategy_params={"fast_period": 10, "slow_period": 30},
        commission_pct=0.0005, slippage_pct=0.0001,
    )
    a = BacktestEngine(data, SmaCrossStrategy, **kwargs).run()
    b = BacktestEngine(data, SmaCrossStrategy, **kwargs).run()

    same_equity = a.equity_curve.equals(b.equity_curve)
    same_trades = len(a.trades) == len(b.trades) and all(
        x.pnl == y.pnl for x, y in zip(a.trades, b.trades)
    )
    report(
        "Воспроизводимость: повторный запуск идентичен",
        same_equity and same_trades,
        f"финальная эквити {a.final_equity:,.2f} == {b.final_equity:,.2f}",
    )


def check_6_risk_sizing() -> None:
    """Сайзинг от риска: убыток по SL == risk_pct от эквити на момент входа.

    Без комиссий и слиппеджа выход ровно по уровню SL теряет точно
    risk_pct; гэп через уровень может потерять больше, но никогда меньше.
    """
    data = generate_synthetic_data(n_bars=1500, volatility=0.008, seed=66)
    result = BacktestEngine(
        data, RiskSizedStrategy, cash=10_000,
        commission_pct=0.0, slippage_pct=0.0,
    ).run()
    risk_pct = RiskSizedStrategy.params["risk_pct"]

    sl_trades = [t for t in result.trades if t.exit_reason is ExitReason.STOP_LOSS]
    never_less = True   # убыток не меньше заявленного риска
    exact_hits = 0      # выходы ровно по уровню (без гэпа) -- точное равенство
    for t in sl_trades:
        # Эквити на момент входа = эквити закрытия предыдущего бара
        # (позиции не было, исполнение на открытии entry_bar)
        equity_before = float(result.equity_curve.iloc[t.entry_bar - 1])
        expected_loss = equity_before * risk_pct
        tol = equity_before * 1e-9
        if -t.pnl < expected_loss - tol:
            never_less = False
        if abs(-t.pnl - expected_loss) <= tol:
            exact_hits += 1

    report(
        "Сайзинг от риска: убыток по SL == risk_pct эквити (или больше при гэпе)",
        bool(sl_trades) and never_less and exact_hits > 0,
        f"выходов по SL: {len(sl_trades)}, из них ровно по уровню: {exact_hits}",
    )


def main() -> int:
    print("=" * 70)
    print("ПРОВЕРКИ ЧЕСТНОСТИ ДВИЖКА")
    print("=" * 70)
    for check in (
        check_1_balance_consistency,
        check_2_do_nothing,
        check_3_costs_hurt,
        check_4_sl_tp_levels,
        check_5_reproducibility,
        check_6_risk_sizing,
    ):
        check()

    print("-" * 70)
    print(f"Итого: пройдено {len(PASSED)}, провалено {len(FAILED)}")
    if FAILED:
        print("Провалены:", ", ".join(FAILED))
        return 1
    print("Все проверки пройдены -- ядру можно доверять.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
