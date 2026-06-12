"""Тесты метрик: просадка, Sharpe, экспозиция, новые поля."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.metrics import (
    compute_metrics,
    exposure_pct,
    max_drawdown_info,
    sharpe_ratio,
)


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
