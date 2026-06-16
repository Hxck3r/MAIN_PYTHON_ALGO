"""Мост ML → движок MAIN: ``MLStrategy`` проигрывает OOS-вероятности модели
через штатный бэктест и совместима с аппаратом робастности (Монте-Карло).

Ключевая проверка причинности: стратегия не открывает сделок раньше первого
out-of-fold предсказания (т.е. не торгует на in-sample участке).
"""
from __future__ import annotations

import numpy as np
import pytest

from backtester import MonteCarloConfig, generate_synthetic_data, run_monte_carlo
from backtester.backtest_engine import run_backtest
from ml.experiment import build_bundle, save_bundle
from ml.features.engineering import FeatureConfig, build_feature_set
from ml.trainer import TrainConfig, train
from strategies import STRATEGIES
from strategies.ml_strategy import MLStrategy


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    """Обучить модель на синтетике и сохранить артефакт прогона на диск."""
    df = generate_synthetic_data(n_bars=1500, freq="1h", seed=7)
    fs = build_feature_set(df, FeatureConfig())
    mr = train(fs, TrainConfig(model_name="random_forest", validation="walk_forward"))
    bundle = build_bundle(mr, fs, df, source_path="synthetic.csv")
    path = save_bundle(bundle, tmp_path_factory.mktemp("models") / "run.joblib")
    return df, path, bundle


def test_strategy_is_registered():
    assert "ML модель (классификатор)" in STRATEGIES
    assert STRATEGIES["ML модель (классификатор)"] is MLStrategy


def test_ml_strategy_runs_through_engine(bundle_path):
    df, path, _ = bundle_path
    result = run_backtest(
        df, MLStrategy, {"bundle_path": str(path)}, prefer_fast=False,
    )
    assert len(result.equity_curve) == len(df)
    stats = result.stats()
    assert stats.total_trades >= 0
    assert np.isfinite(result.final_equity)


def test_no_trades_before_first_oos_signal(bundle_path):
    """Причинность: ни одна сделка не входит раньше первого OOS-предсказания."""
    df, path, bundle = bundle_path
    # Позиция первого бара с известной вероятностью (граница in/out-of-sample).
    aligned = bundle.oos_proba.reindex(df.index).to_numpy(dtype=float)
    first_oos = int(np.argmax(~np.isnan(aligned)))
    assert first_oos > 0  # warmup + первый train-фолд не торгуются

    result = run_backtest(
        df, MLStrategy, {"bundle_path": str(path)}, prefer_fast=False,
    )
    for tr in result.trades:
        assert tr.entry_bar >= first_oos


def test_short_mode_runs(bundle_path):
    df, path, _ = bundle_path
    result = run_backtest(
        df, MLStrategy,
        {"bundle_path": str(path), "allow_short": True, "min_hold": 5},
        prefer_fast=False,
    )
    assert len(result.equity_curve) == len(df)


def test_monte_carlo_on_ml_strategy(bundle_path):
    """Аппарат робастности MAIN применяется к ML-стратегии без изменений."""
    df, path, _ = bundle_path
    result = run_backtest(
        df, MLStrategy,
        {"bundle_path": str(path), "sl_pct": 0.02, "tp_pct": 0.03},
        prefer_fast=False,
    )
    if result.stats().total_trades < 5:
        pytest.skip("Недостаточно сделок для осмысленного Монте-Карло")
    mc = run_monte_carlo(result, MonteCarloConfig(n_sims=200, method="resample"))
    assert mc.n_trades == result.stats().total_trades
    assert len(mc.total_return_pct) == 200
