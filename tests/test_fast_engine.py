"""Быстрый numba-движок должен побитово совпадать с событийным ядром."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import BacktestEngine, generate_synthetic_data
from backtester.fast_engine import (
    SignalStrategy,
    assert_no_lookahead,
    run_fast_backtest,
)
from strategies.sma_cross_signal import SmaCrossSignal


class _LeakyStrategy(SignalStrategy):
    """Намеренное заглядывание: лонг, если ЗАКРЫТИЕ следующего бара выше."""

    params = {"sl_pct": 0.02, "tp_pct": 0.04}

    def compute_signals(self) -> None:
        c = self.full_close
        future = np.empty_like(c)
        future[:-1] = c[1:]
        future[-1] = np.nan
        idx = np.flatnonzero(future > c)
        self._sig[idx] = 1
        self._sl[idx] = c[idx] * (1.0 - self.p["sl_pct"])
        self._tp[idx] = c[idx] * (1.0 + self.p["tp_pct"])


def _max_equity_diff(a, b) -> float:
    return float(np.abs(a.equity_curve.to_numpy() - b.equity_curve.to_numpy()).max())


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return generate_synthetic_data(n_bars=4000, freq="15min", seed=11)


@pytest.mark.parametrize(
    "params",
    [
        {"fast_period": 10, "slow_period": 30},
        {"fast_period": 5, "slow_period": 50, "sl_pct": 0.03, "tp_pct": 0.05},
        {"fast_period": 20, "slow_period": 80, "sl_pct": 0.01, "tp_pct": 0.10},
    ],
)
@pytest.mark.parametrize(
    "engine_kwargs",
    [
        {},
        {"commission_pct": 0.0005, "slippage_pct": 0.0001},
        {"commission_pct": 0.0005, "slippage_pct": 0.0002, "spread_pct": 0.0003},
        {"commission_pct": 0.0005, "slippage_pct": 0.0001, "max_leverage": 2.0},
    ],
)
def test_fast_matches_event_engine(data, params, engine_kwargs):
    ev = BacktestEngine(
        data=data, strategy_cls=SmaCrossSignal, strategy_params=params, **engine_kwargs
    ).run()
    fa = run_fast_backtest(data, SmaCrossSignal, params, **engine_kwargs)

    assert _max_equity_diff(ev, fa) == 0.0
    assert len(ev.trades) == len(fa.trades)
    et, ft = ev.trades_df(), fa.trades_df()
    np.testing.assert_array_equal(et["pnl"].to_numpy(), ft["pnl"].to_numpy())
    np.testing.assert_array_equal(et["entry_bar"].to_numpy(), ft["entry_bar"].to_numpy())
    assert (et["exit_reason"].to_numpy() == ft["exit_reason"].to_numpy()).all()


def test_warmup_equivalence(data):
    ev = BacktestEngine(
        data=data, strategy_cls=SmaCrossSignal,
        strategy_params={"fast_period": 10, "slow_period": 30}, warmup_bars=120,
    ).run()
    fa = run_fast_backtest(
        data, SmaCrossSignal, {"fast_period": 10, "slow_period": 30}, warmup_bars=120
    )
    assert _max_equity_diff(ev, fa) == 0.0


def test_risk_sizing_equivalence(data):
    class SmaRisk(SmaCrossSignal):
        params = dict(SmaCrossSignal.params, risk=0.01)

        def compute_signals(self) -> None:
            super().compute_signals()
            self._risk[self._sig != 0] = float(self.p["risk"])

    ev = BacktestEngine(
        data=data, strategy_cls=SmaRisk, strategy_params={},
        commission_pct=0.0005, max_leverage=5.0,
    ).run()
    fa = run_fast_backtest(
        data, SmaRisk, {}, commission_pct=0.0005, max_leverage=5.0
    )
    assert _max_equity_diff(ev, fa) == 0.0
    assert len(ev.trades) == len(fa.trades) > 0


def test_rejects_non_signal_strategy(data):
    with pytest.raises(TypeError):
        run_fast_backtest(data, BacktestEngine, {})  # не SignalStrategy


# --- Правка 1: валидация данных, как в событийном движке ------------------- #
def test_validates_unsorted_index(data):
    shuffled = data.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="отсортирован"):
        run_fast_backtest(shuffled, SmaCrossSignal, {})


def test_validates_missing_column(data):
    no_volume = data.drop(columns=["volume"])
    with pytest.raises(ValueError, match="не хватает колонок"):
        run_fast_backtest(no_volume, SmaCrossSignal, {})


# --- Правка 3: misuse сайзинга всплывает явно (как в Broker._submit) -------- #
def _sig_variant(mutate):
    class Variant(SmaCrossSignal):
        def compute_signals(self) -> None:
            super().compute_signals()
            mutate(self)
    return Variant


def test_risk_without_sl_raises(data):
    def mutate(s):
        s._sl[:] = np.nan
        s._risk[s._sig != 0] = 0.01
    with pytest.raises(ValueError, match="stop_loss"):
        run_fast_backtest(data, _sig_variant(mutate), {})


def test_size_and_risk_mutually_exclusive(data):
    def mutate(s):
        m = s._sig != 0
        s._size[m] = 1.0
        s._risk[m] = 0.01
    with pytest.raises(ValueError, match="взаимоисключающие"):
        run_fast_backtest(data, _sig_variant(mutate), {})


def test_negative_size_raises(data):
    def mutate(s):
        s._size[s._sig != 0] = -1.0
    with pytest.raises(ValueError, match="положительным"):
        run_fast_backtest(data, _sig_variant(mutate), {})


# --- Правка 2: проверка причинности (анти-lookahead) ----------------------- #
def test_assert_no_lookahead_passes_for_causal_strategy(data):
    # rolling-средние причинны -> исключения быть не должно
    assert_no_lookahead(SmaCrossSignal, data, {"fast_period": 10, "slow_period": 30})


def test_assert_no_lookahead_detects_leak(data):
    with pytest.raises(AssertionError, match="Заглядывание в будущее"):
        assert_no_lookahead(_LeakyStrategy, data)


def test_metrics_consistent(data):
    ev = BacktestEngine(
        data=data, strategy_cls=SmaCrossSignal,
        strategy_params={"fast_period": 10, "slow_period": 30},
        commission_pct=0.0005,
    ).run()
    fa = run_fast_backtest(
        data, SmaCrossSignal, {"fast_period": 10, "slow_period": 30},
        commission_pct=0.0005,
    )
    assert ev.stats().sharpe_ratio == pytest.approx(fa.stats().sharpe_ratio, abs=1e-12)
    assert ev.stats().net_profit == pytest.approx(fa.stats().net_profit, abs=1e-9)
