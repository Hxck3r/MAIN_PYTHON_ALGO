"""Тесты статистики против переоптимизации: PSR/DSR, плато, permutation."""
from __future__ import annotations

import numpy as np
import pytest

from backtester import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
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
