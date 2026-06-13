"""Тесты метрик: просадка, Sharpe, экспозиция, новые поля."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.backtest_engine import BacktestEngine
from backtester.metrics import (
    CRITERIA,
    compute_metrics,
    compute_optimization_metric,
    exposure_pct,
    max_drawdown_info,
    metric_needs_trades,
    sharpe_ratio,
)
from strategies.sma_cross import SmaCrossStrategy


def _equity(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1D")
    return pd.Series(values, index=index, dtype=np.float64, name="equity")


def test_max_drawdown_known_values():
    dd_abs, dd_pct, duration = max_drawdown_info(_equity([100, 120, 90, 130]))
    assert dd_abs == pytest.approx(30.0)
    assert dd_pct == pytest.approx(25.0)
    assert duration == 1


def test_sharpe_zero_volatility_is_zero():
    returns = np.full(100, 0.01)
    assert sharpe_ratio(returns, periods_per_year=252) == 0.0


def test_exposure_merges_overlapping_trades():
    trades = pd.DataFrame({"entry_bar": [1, 3, 10], "exit_bar": [5, 7, 12]})
    # объединение [1,7) и [10,12) = 6 + 2 = 8 баров из 20
    assert exposure_pct(trades, n_bars=20) == pytest.approx(40.0)


def test_exposure_empty_trades():
    assert exposure_pct(pd.DataFrame(), n_bars=100) == 0.0


def test_compute_metrics_new_fields():
    equity = _equity([100, 101, 99, 102, 104, 103, 106])
    trades = pd.DataFrame(
        {
            "pnl": [1.0, -2.0, 3.0],
            "duration_bars": [1, 1, 2],
            "commission": [0.1, 0.1, 0.1],
            "entry_bar": [0, 2, 4],
            "exit_bar": [1, 3, 6],
        }
    )
    m = compute_metrics(equity, trades, periods_per_year=252)
    d = m.to_dict()
    assert "calmar_ratio" in d and "annualized_volatility_pct" in d
    assert m.annualized_volatility_pct > 0
    assert m.exposure_pct == pytest.approx(4 / 7 * 100.0)
    assert m.calmar_ratio != 0.0  # есть и просадка, и рост


def test_compute_metrics_empty_equity():
    m = compute_metrics(pd.Series(dtype=np.float64), pd.DataFrame())
    assert m.total_trades == 0
    assert m.exposure_pct == 0.0


@pytest.mark.parametrize("metric", sorted(set(CRITERIA.values())))
def test_optimization_metric_matches_full_metrics(synth_1h, metric):
    """Лёгкий путь оптимизатора численно совпадает с полным compute_metrics."""
    result = BacktestEngine(
        data=synth_1h,
        strategy_cls=SmaCrossStrategy,
        strategy_params={"fast_period": 10, "slow_period": 40},
        commission_pct=0.0005,
    ).run()
    trades_df = result.trades_df()
    full = getattr(result.stats(), metric)
    fast = compute_optimization_metric(
        result.equity_curve,
        trades_df if metric_needs_trades(metric) else None,
        metric,
    )
    assert fast == pytest.approx(full, rel=1e-9, abs=1e-9)


def test_optimization_metric_empty_equity_matches():
    empty = pd.Series(dtype=np.float64)
    for metric in set(CRITERIA.values()):
        assert compute_optimization_metric(empty, pd.DataFrame(), metric) == 0.0
