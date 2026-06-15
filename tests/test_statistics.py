"""Тесты статистики против переоптимизации: PSR/DSR, плато, permutation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from backtester import (
    DeflatedSharpeResult,
    GridSearchOptimizer,
    deflated_sharpe_from_optimization,
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    haircut_sharpe_ratio,
    make_backtest_objective,
    min_track_record_length,
    permutation_test,
    permute_bars,
    probabilistic_sharpe_ratio,
    select_plateau_params,
)
from backtester.statistics import _norm_ppf
from strategies import SmaCrossStrategy


def test_norm_ppf_known_quantiles():
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _norm_ppf(0.025) == pytest.approx(-1.959964, abs=1e-5)


def test_psr_strong_edge_close_to_one():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.01, 0.01, size=300)  # SR за период ~1
    assert probabilistic_sharpe_ratio(returns) > 0.99


def test_psr_zero_edge_is_half():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, size=300)
    returns = returns - returns.mean()  # точно нулевой Sharpe
    assert probabilistic_sharpe_ratio(returns) == pytest.approx(0.5, abs=1e-9)


def test_expected_max_sharpe_grows_with_trials():
    var = 0.01
    e10 = expected_max_sharpe(10, var)
    e1000 = expected_max_sharpe(1000, var)
    assert 0 < e10 < e1000


def test_dsr_below_psr():
    """Поправка на перебор всегда снижает уверенность."""
    rng = np.random.default_rng(2)
    returns = rng.normal(0.002, 0.01, size=500)
    res = deflated_sharpe_ratio(returns, trial_scores=np.zeros(1000))
    assert res.n_trials == 1000
    assert res.dsr < res.psr_zero
    assert res.expected_max_sr > 0


def test_plateau_prefers_stable_region():
    """Изолированный пик проигрывает плато с хорошими соседями."""
    history = []
    scores = {0: 0, 1: 0, 2: 8, 3: 9, 4: 8, 5: 0, 6: 0, 7: 0, 8: 0, 9: 12, 10: 0}
    for a, s in scores.items():
        history.append(({"a": a}, float(s)))
    params, plateau_score = select_plateau_params(history, radius=1)
    # argmax выбрал бы a=9 (12), плато-выбор -- центр области 8-9-8
    assert params == {"a": 3}
    assert plateau_score == pytest.approx((8 + 9 + 8) / 3)


def test_permute_bars_preserves_distribution(synth_1h):
    data = synth_1h.iloc[:300]
    permuted = permute_bars(data, np.random.default_rng(0))
    assert len(permuted) == len(data)
    assert (permuted.index == data.index).all()
    # геометрия баров сохранена
    assert (permuted["high"] >= permuted[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (permuted["low"] <= permuted[["open", "close"]].min(axis=1) + 1e-9).all()
    # мультимножество бар-доходностей close/prev_close то же
    orig = (data["close"] / data["close"].shift(1)).dropna()
    perm = (permuted["close"] / permuted["close"].shift(1)).dropna()
    assert np.allclose(np.sort(orig.to_numpy()), np.sort(perm.to_numpy()), rtol=1e-9)
    # а порядок -- другой
    assert not np.allclose(orig.to_numpy(), perm.to_numpy())


def test_deflated_sharpe_from_optimization(synth_1h):
    """Хелпер прогоняет лучшие параметры и дефлирует Sharpe на число испытаний."""
    grid = {"fast_period": [5, 10, 15], "slow_period": [20, 30]}
    objective = make_backtest_objective(
        data=synth_1h, strategy_cls=SmaCrossStrategy, metric="sharpe_ratio",
        min_trades=0,
    )
    opt = GridSearchOptimizer(objective, param_grid=grid).optimize()

    dsr = deflated_sharpe_from_optimization(
        synth_1h, SmaCrossStrategy, opt.best_params, n_trials=len(opt.history),
    )
    assert isinstance(dsr, DeflatedSharpeResult)
    assert dsr.n_trials == len(opt.history) == 6
    assert 0.0 <= dsr.dsr <= 1.0
    # поправка на перебор не делает DSR выше «сырого» PSR
    assert dsr.dsr <= dsr.psr_zero + 1e-12
    # вердикт печатается в отчёте
    assert "Deflated Sharpe" in str(dsr)


# --- Канонический DSR (дисперсия по испытаниям + эффективное N) ----------- #
def test_effective_trials_collapses_duplicates():
    """Дубликаты конфигов не считаются независимыми испытаниями."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.01, size=(400, 5))   # 5 независимых сигналов
    matrix = np.hstack([base, base])               # 10 колонок = 5 дублей пар
    assert effective_number_of_trials(matrix, corr_threshold=0.95) == 5


def test_dsr_canonical_uses_trial_variance():
    """Канонический путь: больше разброс Sharpe по испытаниям -> ниже DSR."""
    rng = np.random.default_rng(4)
    returns = rng.normal(0.003, 0.01, size=500)
    small = np.linspace(0.04, 0.06, 50)            # узкий разброс trial-Sharpe
    large = np.linspace(-0.10, 0.20, 50)           # широкий разброс
    res_small = deflated_sharpe_ratio(returns, np.empty(0), trial_sharpes=small)
    res_large = deflated_sharpe_ratio(returns, np.empty(0), trial_sharpes=large)
    assert res_small.canonical and res_large.canonical
    assert res_small.sr_variance == pytest.approx(np.var(small, ddof=1))
    # шире разброс испытаний -> выше порог шума -> ниже уверенность
    assert res_large.expected_max_sr > res_small.expected_max_sr
    assert res_large.dsr <= res_small.dsr


# --- MinTRL ---------------------------------------------------------------- #
def test_mintrl_sufficient_for_strong_long_track():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.01, 0.01, size=1000)    # SR за период ~1
    res = min_track_record_length(returns, benchmark_sr=0.0, confidence=0.95)
    assert res.sufficient
    assert res.min_length < res.n_observations


def test_mintrl_infinite_without_edge():
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0, 0.01, size=200)
    returns = returns - returns.mean()             # ровно нулевой Sharpe
    res = min_track_record_length(returns, benchmark_sr=0.0)
    assert not res.sufficient
    assert math.isinf(res.min_length)


# --- Haircut Sharpe -------------------------------------------------------- #
def test_haircut_zero_for_single_trial():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.01, 0.01, size=500)
    res = haircut_sharpe_ratio(returns, n_trials=1)
    assert res.haircut_pct == pytest.approx(0.0, abs=1e-6)


def test_haircut_grows_with_trials():
    rng = np.random.default_rng(8)
    returns = rng.normal(0.005, 0.01, size=500)
    h1 = haircut_sharpe_ratio(returns, n_trials=1)
    h100 = haircut_sharpe_ratio(returns, n_trials=200)
    assert h100.haircut_pct >= h1.haircut_pct
    assert h100.adjusted_sr_annual <= h1.adjusted_sr_annual
    assert "Haircut Sharpe" in str(h100)


def test_permutation_test_runs(synth_1h):
    res = permutation_test(
        data=synth_1h.iloc[:400],
        strategy_cls=SmaCrossStrategy,
        metric="net_profit",
        n_permutations=5,
        seed=0,
    )
    assert 0.0 < res.p_value <= 1.0
    assert len(res.permuted_scores) == 5
