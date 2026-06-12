"""Тесты Монте-Карло: детерминизм, инварианты методов, формы массивов."""
from __future__ import annotations

import numpy as np
import pytest

from backtester import BacktestEngine, MonteCarloConfig, run_monte_carlo
from strategies import SmaCrossStrategy


@pytest.fixture(scope="module")
def backtest(synth_1h):
    result = BacktestEngine(synth_1h, SmaCrossStrategy, cash=10_000.0).run()
    assert len(result.trades) >= 5
    return result


def test_shapes_and_determinism(backtest):
    cfg = MonteCarloConfig(n_sims=200, seed=123)
    a = run_monte_carlo(backtest, cfg)
    b = run_monte_carlo(backtest, MonteCarloConfig(n_sims=200, seed=123))
    assert a.n_sims == 200
    assert len(a.total_return_pct) == 200
    assert len(a.max_dd_pct) == 200
    np.testing.assert_allclose(a.final_equity, b.final_equity)


def test_permute_fixed_sizing_total_invariant(backtest):
    """Перестановка без возврата при фикс. сайзинге не меняет итоговый PnL."""
    cfg = MonteCarloConfig(n_sims=100, method="permute", sizing="fixed", seed=1)
    res = run_monte_carlo(backtest, cfg)
    assert res.final_equity.max() - res.final_equity.min() < 1e-6
    assert res.final_equity[0] == pytest.approx(res.original_curve[-1])


def test_probabilities_in_range(backtest):
    res = run_monte_carlo(backtest, MonteCarloConfig(n_sims=300, seed=5))
    for value in (
        res.prob_profit_pct,
        res.prob_target_pct,
        res.prob_dd_exceeded_pct,
        res.risk_of_ruin_pct,
        res.original_return_rank_pct,
        res.p_loss_pct,
    ):
        assert 0.0 <= value <= 100.0
