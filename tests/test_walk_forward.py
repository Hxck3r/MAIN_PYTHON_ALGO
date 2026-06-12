"""Тесты Walk-Forward: разбиение на окна, warm-up OOS, сшивка."""
from __future__ import annotations

import pandas as pd
import pytest

import math

import numpy as np

from backtester import WalkForwardConfig, run_walk_forward, window_bounds
from backtester.walk_forward import _spearman, _window_plateau_pct
from strategies import SmaCrossStrategy

GRID = {"fast_period": [5, 10], "slow_period": [20]}


def _config(days_total: int = 50, is_days: int = 20, oos_days: int = 10,
            **kwargs) -> WalkForwardConfig:
    start = pd.Timestamp("2022-01-01")
    return WalkForwardConfig(
        start=start,
        end=start + pd.Timedelta(days=days_total),
        in_sample_days=is_days,
        out_sample_days=oos_days,
        **kwargs,
    )


def test_window_bounds_rolling_count():
    bounds = window_bounds(_config())
    # IS 20д + OOS 10д, шаг 10д, период 50д: окна [0,30), [10,40), [20,50)
    assert len(bounds) == 3
    for is_start, is_end, oos_start, oos_end in bounds:
        assert is_end == oos_start
        assert (is_end - is_start).days == 20
        assert (oos_end - oos_start).days == 10


def test_window_bounds_overlap_rejected():
    with pytest.raises(ValueError, match="перекрылись"):
        window_bounds(_config(step_days=5))


def test_run_walk_forward_basic(synth_1h):
    result = run_walk_forward(
        data=synth_1h, strategy_cls=SmaCrossStrategy,
        config=_config(), grid=GRID, metric="net_profit", min_trades=0,
    )
    assert result.n_windows >= 1
    assert not result.stitched_equity.empty
    assert result.stitched_equity.index.is_monotonic_increasing
    # сшитая кривая лежит внутри OOS-периода
    first_oos = result.windows[0].oos_start
    assert result.stitched_equity.index[0] >= first_oos


def test_run_walk_forward_warmup_trims_equity(synth_1h):
    warm = run_walk_forward(
        data=synth_1h, strategy_cls=SmaCrossStrategy,
        config=_config(), grid=GRID, metric="net_profit", min_trades=0,
        warmup_bars=50,
    )
    # эквити каждого окна начинается строго с начала OOS, прогрев отрезан
    for w in warm.windows:
        if w.valid:
            assert w.oos_equity.index[0] >= w.oos_start
            assert w.oos_equity.index[-1] < w.oos_end
    # все сделки -- внутри OOS-границ своих окон
    for w in warm.windows:
        if w.valid and not w.oos_trades.empty:
            assert (w.oos_trades["entry_time"] >= w.oos_start).all()


def test_warmup_zero_matches_old_behavior(synth_1h):
    a = run_walk_forward(
        data=synth_1h, strategy_cls=SmaCrossStrategy,
        config=_config(), grid=GRID, metric="net_profit", min_trades=0,
    )
    b = run_walk_forward(
        data=synth_1h, strategy_cls=SmaCrossStrategy,
        config=_config(), grid=GRID, metric="net_profit", min_trades=0,
        warmup_bars=0,
    )
    pd.testing.assert_series_equal(a.stitched_equity, b.stitched_equity)


# --- Spearman -------------------------------------------------------------- #
def test_spearman_monotonic_and_ties():
    # строго возрастающая связь -> +1, обратная -> -1
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _spearman(x, x * 2 + 1) == pytest.approx(1.0)
    assert _spearman(x, -x) == pytest.approx(-1.0)
    # совпадения значений (средние ранги) не ломают расчёт
    assert _spearman(np.array([1.0, 1.0, 2.0, 3.0]),
                     np.array([1.0, 2.0, 2.0, 3.0])) == pytest.approx(0.9, abs=0.2)


def test_spearman_needs_three_points():
    assert math.isnan(_spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
    # нет разброса -> неопределено
    assert math.isnan(_spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))


# --- Plateau --------------------------------------------------------------- #
def test_plateau_flat_vs_spike():
    grid = {"a": [1, 2, 3]}
    # плато: соседи оптимума почти так же хороши -> 100%
    flat = {(("a", 2),): 1.0, (("a", 1),): 0.95, (("a", 3),): 0.9}
    assert _window_plateau_pct(flat, {"a": 2}, grid) == pytest.approx(100.0)
    # изолированный пик: соседи рушатся -> 0%
    spike = {(("a", 2),): 1.0, (("a", 1),): 0.1, (("a", 3),): 0.1}
    assert _window_plateau_pct(spike, {"a": 2}, grid) == pytest.approx(0.0)


def test_plateau_nan_without_neighbors():
    # сетку никто не покрыл соседями -> NaN
    grid = {"a": [1, 2, 3]}
    assert math.isnan(_window_plateau_pct({(("a", 2),): 1.0}, {"a": 2}, grid))


def test_wfa_result_has_robustness_metrics(synth_1h):
    result = run_walk_forward(
        data=synth_1h, strategy_cls=SmaCrossStrategy,
        config=_config(), grid={"fast_period": [5, 10, 15], "slow_period": [20, 30]},
        metric="sharpe_ratio", min_trades=0,
    )
    # поля присутствуют и имеют корректный тип/диапазон
    assert isinstance(result.is_oos_rank_corr, float)
    if math.isfinite(result.is_oos_rank_corr):
        assert -1.0 <= result.is_oos_rank_corr <= 1.0
    if math.isfinite(result.avg_plateau_pct):
        assert 0.0 <= result.avg_plateau_pct <= 100.0
    # текстовый отчёт по параметрам не падает
    assert "Spearman" in result.param_report()
