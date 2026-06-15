"""Тесты PBO/CSCV: переподгонка ~0.5 на шуме, ~0 при переносимом отборе."""
from __future__ import annotations

import numpy as np
import pytest

from backtester import (
    build_trial_returns,
    pbo_from_optimization,
    probability_of_backtest_overfitting,
)
from backtester.pbo import PBOResult, unique_configs
from backtester import GridSearchOptimizer, make_backtest_objective
from strategies import SmaCrossStrategy


def test_pbo_high_when_selecting_among_noise():
    """Отбор лучшего среди конфигов без преимущества -- переподгонка: PBO высок.

    IS-лучший «поймал» удачные блоки, на комплементарных OOS-блоках он
    систематически проседает (CSCV) -> относительный OOS-ранг ниже медианы.
    """
    rng = np.random.default_rng(0)
    matrix = rng.normal(0.0, 1.0, size=(480, 20))  # T наблюдений × N конфигов
    res = probability_of_backtest_overfitting(matrix, n_splits=16)
    assert isinstance(res, PBOResult)
    assert res.n_combinations == 12870  # C(16, 8)
    assert res.pbo > 0.55              # явный сигнал переподгонки


def test_pbo_low_when_one_config_dominates():
    """Если один конфиг устойчиво лучший и на IS, и на OOS -> PBO ~ 0."""
    rng = np.random.default_rng(1)
    n = 20
    mus = np.linspace(0.0, 2.85, n)              # хорошо разделённые средние
    matrix = rng.normal(0.0, 1.0, size=(480, n)) + mus  # столбец j лучше j-1
    res = probability_of_backtest_overfitting(matrix, n_splits=16)
    assert res.pbo < 0.05
    assert res.prob_oos_loss < 0.05             # отобранный почти всегда в плюсе


def test_pbo_rejects_too_few_configs():
    with pytest.raises(ValueError, match="2 конфигурации"):
        probability_of_backtest_overfitting(np.zeros((100, 1)), n_splits=4)


def test_pbo_rejects_odd_splits():
    with pytest.raises(ValueError, match="чётным"):
        probability_of_backtest_overfitting(np.zeros((100, 5)), n_splits=7)


def test_build_trial_returns_shape(synth_1h):
    configs = [
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 10, "slow_period": 30},
        {"fast_period": 15, "slow_period": 40},
    ]
    matrix = build_trial_returns(synth_1h, SmaCrossStrategy, configs)
    assert matrix.shape == (len(synth_1h), 3)
    assert (matrix.index == synth_1h.index).all()
    assert not matrix.isna().any().any()  # первый бар -> 0, не NaN


def test_unique_configs_dedup():
    history = [
        ({"a": 1, "b": 2}, 0.5),
        ({"a": 1, "b": 2}, 0.5),  # дубль
        ({"a": 3, "b": 4}, 0.7),
    ]
    assert unique_configs(history) == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_build_trial_returns_skips_failing_configs(synth_1h):
    """Конфиг с недопустимыми параметрами (fast>=slow) пропускается, не валит."""
    configs = [
        {"fast_period": 5, "slow_period": 20},    # ок
        {"fast_period": 30, "slow_period": 20},   # fast>=slow -> стратегия падает
        {"fast_period": 10, "slow_period": 40},   # ок
    ]
    matrix = build_trial_returns(synth_1h, SmaCrossStrategy, configs)
    assert matrix.shape == (len(synth_1h), 2)     # упавший конфиг отброшен
    assert list(matrix.columns) == [0, 1]         # колонки перенумерованы


def test_pbo_from_optimization_handles_invalid_combos(synth_1h):
    """Сетка с пересечением fast/slow: невалидные комбинации не ломают PBO."""
    grid = {"fast_period": [10, 30, 60], "slow_period": [20, 40]}
    obj = make_backtest_objective(
        synth_1h, SmaCrossStrategy, metric="sharpe_ratio", min_trades=0,
    )
    opt = GridSearchOptimizer(obj, param_grid=grid).optimize()
    # часть комбинаций (fast>=slow) получили -inf в истории
    assert any(not np.isfinite(s) for _, s in opt.history)
    res = pbo_from_optimization(synth_1h, SmaCrossStrategy, opt.history, n_splits=4)
    assert isinstance(res, PBOResult)
    assert res.n_trials >= 2                       # остались только валидные


def test_pbo_from_optimization_end_to_end(synth_1h):
    grid = {"fast_period": [5, 10, 15, 20], "slow_period": [25, 40]}
    objective = make_backtest_objective(
        data=synth_1h, strategy_cls=SmaCrossStrategy, metric="sharpe_ratio",
        min_trades=0,
    )
    opt = GridSearchOptimizer(objective, param_grid=grid).optimize()
    res = pbo_from_optimization(
        synth_1h, SmaCrossStrategy, opt.history, n_splits=8,
    )
    assert isinstance(res, PBOResult)
    assert 0.0 <= res.pbo <= 1.0
    assert res.n_trials == 8  # 4 × 2 различных конфигурации
    assert "Probability of Backtest Overfitting" in str(res)
