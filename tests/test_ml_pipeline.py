"""Headless-проверки ML-ядра (пакет ``ml``): признаки, purged walk-forward,
обучение, артефакт прогона. Без Qt и без GUI.

Запуск: ``pytest tests/test_ml_pipeline.py -q``.
"""
from __future__ import annotations

import pytest

from backtester import generate_synthetic_data
from ml.experiment import build_bundle, load_bundle, save_bundle
from ml.features.engineering import FeatureConfig, build_feature_set
from ml.trainer import TrainConfig, train
from ml.validation import PurgedWalkForwardSplit


@pytest.fixture(scope="module")
def ohlcv():
    """Достаточно длинный синтетический ряд для 5-фолдового walk-forward."""
    return generate_synthetic_data(n_bars=1500, freq="1h", seed=7)


def test_purged_split_has_no_leakage(ohlcv):
    """Train строго в прошлом и отделён от test зазором ≥ horizon+embargo."""
    horizon, embargo = 3, 2
    sp = PurgedWalkForwardSplit(n_splits=5, horizon=horizon, embargo=embargo)
    folds = list(sp.split(ohlcv))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()                       # прошлое < будущее
        assert test_idx.min() - train_idx.max() >= horizon + embargo  # зазор соблюдён
        assert len(set(train_idx) & set(test_idx)) == 0               # без пересечений


def test_feature_set_has_features_and_target(ohlcv):
    fs = build_feature_set(ohlcv, FeatureConfig())
    assert len(fs.feature_columns) > 5
    assert fs.target_column in fs.frame.columns
    assert set(fs.frame[fs.target_column].unique()) <= {0, 1}


def test_triple_barrier_target_is_binary(ohlcv):
    fs = build_feature_set(
        ohlcv, FeatureConfig(target_mode="triple_barrier", target_horizon=10)
    )
    assert fs.horizon == 10
    assert set(fs.frame[fs.target_column].unique()) <= {0, 1}
    assert len(fs.feature_columns) > 5


def test_walk_forward_trains_and_oos(ohlcv):
    fs = build_feature_set(ohlcv, FeatureConfig())
    mr = train(fs, TrainConfig(model_name="random_forest", validation="walk_forward"))
    assert mr.validation == "walk_forward"
    assert len(mr.fold_metrics) >= 1
    assert "cv_auc_mean" in mr.metrics
    assert 0.05 <= mr.threshold <= 0.95
    # OOS-предсказания выровнены и непустые — их «проигрывает» MLStrategy.
    assert len(mr.y_true) == len(mr.y_proba) == len(mr.test_index) > 0
    assert mr.importances  # важности извлечены


def test_holdout_mode_runs(ohlcv):
    fs = build_feature_set(ohlcv, FeatureConfig())
    mr = train(fs, TrainConfig(model_name="random_forest", validation="holdout"))
    assert mr.validation == "holdout"
    assert 0.0 <= mr.metrics["accuracy"] <= 1.0


def test_progress_callback_can_cancel(ohlcv):
    """Возврат False из колбэка кооперативно прерывает обучение."""
    from ml.trainer import TrainingCancelled

    fs = build_feature_set(ohlcv, FeatureConfig())

    def cb(pct, msg):
        return False  # сразу просим остановиться

    with pytest.raises(TrainingCancelled):
        train(fs, TrainConfig(model_name="random_forest"), progress=cb)


def test_bundle_roundtrip_preserves_oos_proba(ohlcv, tmp_path):
    fs = build_feature_set(ohlcv, FeatureConfig())
    mr = train(fs, TrainConfig(model_name="random_forest"))
    bundle = build_bundle(mr, fs, ohlcv, source_path="synthetic.csv")
    assert len(bundle.oos_proba) == len(mr.y_proba)

    path = save_bundle(bundle, tmp_path / "run.joblib")
    assert path.exists()
    assert path.with_suffix(".json").exists()        # сводка-сайдкар

    loaded = load_bundle(path)
    assert loaded.feature_columns == mr.feature_columns
    assert loaded.threshold == pytest.approx(mr.threshold)
    assert loaded.oos_proba.equals(bundle.oos_proba)


def test_auc_ci_significance(ohlcv):
    """DeLong CI отличает реальный сигнал от случайного."""
    import numpy as np

    from ml.significance import auc_ci

    rng = np.random.default_rng(0)
    y = np.array([0] * 200 + [1] * 200)
    sep = np.concatenate([rng.normal(0, 1, 200), rng.normal(2, 1, 200)])
    auc, lo, hi, p = auc_ci(y, sep)
    assert auc > 0.8 and lo > 0.5 and p < 0.01            # значимый край

    noise = rng.normal(0, 1, 400)
    _, lo2, _, p2 = auc_ci(y, noise)
    assert lo2 <= 0.5 and p2 > 0.05                       # незначимо


def test_information_coefficient():
    import numpy as np

    from ml.significance import information_coefficient

    rng = np.random.default_rng(1)
    proba = np.linspace(0.0, 1.0, 200)
    fwd = (proba - 0.5) + rng.normal(0, 0.1, 200)         # положительная связь
    ic, p = information_coefficient(proba, fwd)
    assert ic > 0.5 and p < 0.01


def test_train_attaches_significance_and_diagnostics(ohlcv):
    fs = build_feature_set(ohlcv, FeatureConfig())
    mr = train(fs, TrainConfig(model_name="random_forest"))
    for key in ("auc_ci_low", "auc_ci_high", "auc_pvalue", "ic", "ic_pvalue",
                "baseline_acc", "pos_rate", "variance_ratio", "n_samples"):
        assert key in mr.metrics


def test_calendar_and_regime_features(ohlcv):
    """Часовой ряд → есть календарные признаки; режимные — всегда."""
    fs = build_feature_set(ohlcv, FeatureConfig(use_calendar=True, use_regime=True))
    assert any(c.startswith("cal_") for c in fs.feature_columns)
    assert "vol_regime" in fs.feature_columns and "mom_50" in fs.feature_columns
    # отключение убирает их
    fs2 = build_feature_set(ohlcv, FeatureConfig(use_calendar=False, use_regime=False))
    assert not any(c.startswith("cal_") for c in fs2.feature_columns)
    assert "vol_regime" not in fs2.feature_columns


def test_hyperparameter_tuning_runs(ohlcv):
    from ml.tuning import tune

    fs = build_feature_set(ohlcv, FeatureConfig())
    best, score, history = tune(
        fs, TrainConfig(model_name="random_forest", n_splits=3), n_trials=3,
    )
    assert isinstance(best, dict) and best
    assert len(history) >= 1


def test_feature_cache_hit(ohlcv, tmp_path, monkeypatch):
    """Повторный вызов берёт признаки из кеша (один и тот же FeatureSet)."""
    from ml import cache, config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    cfg = FeatureConfig()
    fs1 = cache.cached_feature_set(ohlcv, cfg)
    assert list(tmp_path.glob("fs_*.joblib")), "кеш признаков не записан"
    fs2 = cache.cached_feature_set(ohlcv, cfg)            # должен прийти из кеша
    assert fs1.frame.equals(fs2.frame)
    assert fs1.feature_columns == fs2.feature_columns


@pytest.mark.skipif(
    "lightgbm" not in __import__("ml.models", fromlist=["MODEL_SPECS"]).MODEL_SPECS,
    reason="LightGBM не установлен в окружении",
)
def test_lightgbm_pipeline(ohlcv):
    fs = build_feature_set(ohlcv, FeatureConfig())
    mr = train(fs, TrainConfig(model_name="lightgbm", validation="walk_forward"))
    assert 0.0 <= mr.metrics["accuracy"] <= 1.0
    assert len(mr.y_proba) == len(mr.test_index)


# --- A1: веса уникальности и горизонт меток -------------------------------- #
def test_concurrency_weights_normalized_and_penalize_overlap():
    """Веса нормированы к среднему 1; перекрытые (середина ряда) весят меньше краёв."""
    import numpy as np

    from ml.sample_weights import concurrency_weights, time_decay

    n = 200
    event_end = np.minimum(np.arange(n) + 5, n - 1)   # каждая метка живёт 6 баров
    w = concurrency_weights(event_end, n)
    assert w.shape == (n,)
    assert np.all(w > 0) and abs(w.mean() - 1.0) < 1e-9
    mid = n // 2
    # Первые бары мало с кем перекрываются (низкая параллельность) → весят заметно
    # больше, чем плотно перекрытая середина ряда.
    assert w[0] > w[mid]

    decayed = time_decay(np.ones(n), last_w=0.5)
    assert decayed[0] < decayed[-1]                    # старые наблюдения весят меньше
    assert abs(decayed.mean() - 1.0) < 1e-9


def test_triple_barrier_events_t1_within_horizon(ohlcv):
    from ml.features.labeling import triple_barrier_events

    h = 10
    ev = triple_barrier_events(ohlcv["high"], ohlcv["low"], ohlcv["close"], horizon=h)
    valid = ev.dropna()
    assert set(valid["label"].unique()) <= {0.0, 1.0}
    assert valid["t1"].min() >= 1 and valid["t1"].max() <= h


def test_feature_set_carries_event_end(ohlcv):
    fs = build_feature_set(ohlcv, FeatureConfig(target_mode="triple_barrier", target_horizon=8))
    assert fs.event_end is not None
    assert len(fs.event_end) == len(fs.frame)
    pos = __import__("numpy").arange(len(fs.frame))
    assert (fs.event_end >= pos).all()                 # конец метки не раньше её начала


def test_sample_weights_change_training_not_break(ohlcv):
    """Обучение с весами и без — оба проходят и дают валидные OOS-предсказания."""
    fs = build_feature_set(ohlcv, FeatureConfig(target_mode="triple_barrier", target_horizon=10))
    for use_w in (True, False):
        mr = train(fs, TrainConfig(model_name="random_forest", n_splits=4,
                                   use_sample_weights=use_w))
        assert len(mr.y_proba) == len(mr.test_index) > 0


# --- A3/A4: OOS-доходность, Sharpe по фолдам, дефляция ---------------------- #
def test_oos_returns_and_fold_sharpe(ohlcv):
    fs = build_feature_set(ohlcv, FeatureConfig())
    mr = train(fs, TrainConfig(model_name="random_forest", n_splits=4))
    assert len(mr.oos_returns) == len(mr.test_index) > 0
    assert len(mr.oos_fwd) == len(mr.test_index)
    assert "oos_sharpe" in mr.metrics
    assert all("sharpe" in fm for fm in mr.fold_metrics)


def test_deflation_penalizes_multiple_trials(ohlcv):
    """DSR с поправкой на перебор не выше PSR без поправки (порог шума ≥ 0)."""
    fs = build_feature_set(ohlcv, FeatureConfig())
    mr = train(fs, TrainConfig(model_name="random_forest", n_splits=4,
                               tune_trial_sharpes=[0.4, 0.1, -0.2, 0.6, 0.05]))
    assert "psr" in mr.metrics
    if "dsr" in mr.metrics:                             # появляется при n_trials>1
        assert mr.metrics["n_trials"] == 5
        assert mr.metrics["dsr"] <= mr.metrics["psr"] + 1e-9


# --- A2/C: MDA-важность и float32 ------------------------------------------ #
def test_mda_importance_optional(ohlcv):
    fs = build_feature_set(ohlcv, FeatureConfig())
    mr = train(fs, TrainConfig(model_name="random_forest", n_splits=3,
                               compute_mda=True, mda_repeats=2))
    assert mr.importances_mda                           # посчитана
    assert set(mr.importances_mda) <= set(fs.feature_columns)


def test_float32_and_feature_selection(ohlcv):
    fs = build_feature_set(ohlcv, FeatureConfig(use_float32=True, feature_select=True))
    assert all(str(fs.frame[c].dtype) == "float32" for c in fs.feature_columns)
    # отбор не оставляет дублирующих по сильной корреляции (число признаков уменьшилось
    # или осталось, но обучение всё равно проходит)
    mr = train(fs, TrainConfig(model_name="random_forest", n_splits=3))
    assert 0.0 <= mr.metrics["accuracy"] <= 1.0
