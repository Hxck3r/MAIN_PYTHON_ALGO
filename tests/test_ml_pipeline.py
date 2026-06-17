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


# --- Фракционное дифференцирование (AFML гл. 5) ---------------------------- #
def test_fracdiff_weights_and_transform(ohlcv):
    import numpy as np

    from ml.features.fracdiff import ffd_weights, frac_diff_ffd

    w = ffd_weights(0.4, threshold=1e-3)
    assert w[-1] == 1.0                                # самый новый коэффициент = 1
    assert len(w) > 5 and abs(w).min() >= 1e-3 * 0.99  # обрезка по порогу

    fd = frac_diff_ffd(np.log(ohlcv["close"]), 0.4, threshold=1e-3)
    width = len(w)
    assert fd.iloc[:width - 1].isna().all()            # разогрев — NaN
    assert fd.iloc[width - 1:].notna().all()
    # d=0 возвращает исходный ряд (память не тронута)
    same = frac_diff_ffd(ohlcv["close"], 0.0)
    assert np.allclose(same.to_numpy(), ohlcv["close"].to_numpy())


def test_fracdiff_feature_added(ohlcv):
    fs = build_feature_set(ohlcv, FeatureConfig(use_fracdiff=True))
    assert "fracdiff" in fs.feature_columns


# --- Bet sizing (AFML гл. 10) ---------------------------------------------- #
def test_bet_size_monotone_and_bounded():
    from ml.bet_sizing import bet_size_from_proba

    assert bet_size_from_proba(0.5) == 0.0
    sizes = [bet_size_from_proba(p) for p in (0.5, 0.6, 0.7, 0.8, 0.9)]
    assert all(0.0 <= s <= 1.0 for s in sizes)
    assert sizes == sorted(sizes)                      # растёт с уверенностью
    # симметрия относительно ½
    assert abs(bet_size_from_proba(0.8) - bet_size_from_proba(0.2)) < 1e-9


# --- Издержки в OOS-оценке (честный net-Sharpe) ---------------------------- #
def test_signal_returns_costs_penalize_turnover():
    """Издержки вычитаются при каждой смене позиции (обороте); cost=0 — валовая."""
    import numpy as np
    import pandas as pd

    from ml.trainer import _signal_returns

    idx = pd.RangeIndex(6)
    fwd = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)   # нулевой доход
    proba = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1])             # позиция дёргается каждый бар
    gross = _signal_returns(fwd, idx, proba, 0.5, cost=0.0)
    net = _signal_returns(fwd, idx, proba, 0.5, cost=0.001)
    assert gross.sum() == 0.0                                    # без издержек — ноль
    assert net.sum() < 0.0                                       # частый оборот съедает капитал


def test_train_cost_bps_reduces_oos_sharpe(ohlcv):
    """С издержками OOS-Sharpe не выше валового и cost_bps пишется в метрики."""
    fs = build_feature_set(ohlcv, FeatureConfig(target_mode="triple_barrier", target_horizon=10))
    gross = train(fs, TrainConfig(model_name="random_forest", n_splits=4, cost_bps=0.0))
    net = train(fs, TrainConfig(model_name="random_forest", n_splits=4, cost_bps=10.0))
    assert gross.metrics["cost_bps"] == 0.0 and net.metrics["cost_bps"] == 10.0
    assert net.metrics["oos_sharpe"] <= gross.metrics["oos_sharpe"] + 1e-9


# --- Temporal-safe калибровка (без утечки во времени) ---------------------- #
def test_calibrated_training_extracts_importances(ohlcv):
    """Калибровка через FrozenEstimator: модель обучается, важности достаются из базы."""
    fs = build_feature_set(ohlcv, FeatureConfig(target_mode="triple_barrier", target_horizon=10))
    mr = train(fs, TrainConfig(model_name="random_forest", n_splits=4, calibrate=True))
    assert mr.calibrated
    assert mr.importances                      # важности извлечены из базовой модели
    assert len(mr.y_proba) == len(mr.test_index) > 0


# --- CPCV (AFML гл. 12) ---------------------------------------------------- #
def test_cpcv_splitter_paths_and_no_leakage():
    """Число сочетаний/путей по формуле; train без пересечений и с purge-зазором."""
    from math import comb

    import numpy as np
    import pandas as pd

    from ml.cpcv import CombinatorialPurgedCV, make_groups

    n, N, k, h, emb = 600, 6, 2, 5, 2
    X = pd.DataFrame({"a": np.arange(n)})
    cv = CombinatorialPurgedCV(n_groups=N, test_groups=k, horizon=h, embargo=emb)
    assert cv.n_combos() == comb(N, k)
    assert cv.n_paths() == comb(N - 1, k - 1)
    groups = make_groups(n, N)
    combos = list(cv.split(X))
    assert len(combos) == comb(N, k)
    gap = h + emb
    for tr, te, combo in combos:
        assert len(set(tr) & set(te)) == 0                      # train/test не пересекаются
        tr_set = set(tr.tolist())
        for g in combo:                                         # purge-зазор вокруг тест-блоков
            s, e = groups[g]
            assert not (tr_set & set(range(max(0, s - gap), min(n, e + gap))))


def test_cpcv_assemble_paths_shapes():
    """Сборка путей: P путей полной длины из предсказаний по группам."""
    import numpy as np

    from ml.cpcv import assemble_paths

    preds = {0: [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
             1: [np.array([0.5]), np.array([0.6])],
             2: [np.array([0.7, 0.8, 0.9]), np.array([0.1, 0.2, 0.3])]}
    paths = assemble_paths(preds, n_groups=3)
    assert len(paths) == 2
    assert all(p.shape == (6,) for p in paths)                  # 2+1+3 баров на путь


def test_cpcv_training_distribution(ohlcv):
    """CPCV-обучение даёт распределение Sharpe по путям и единый OOS-ряд."""
    import numpy as np

    fs = build_feature_set(ohlcv, FeatureConfig(target_mode="triple_barrier", target_horizon=10))
    mr = train(fs, TrainConfig(model_name="random_forest", validation="cpcv",
                               n_splits=6, cpcv_k=2, cost_bps=2.0))
    assert mr.validation == "cpcv"
    n_paths = int(mr.metrics["cpcv_n_paths"])
    assert n_paths > 0 and len(mr.cpcv_sharpes) == n_paths
    for key in ("cpcv_sharpe_mean", "cpcv_sharpe_std", "cpcv_sharpe_min", "cpcv_prob_positive"):
        assert key in mr.metrics
    assert 0.0 <= mr.metrics["cpcv_prob_positive"] <= 1.0
    assert len(mr.y_proba) == len(mr.test_index) > 0            # единый OOS-ряд для моста
    assert np.isfinite(mr.metrics.get("oos_sharpe", float("nan")))


def test_cpcv_bundle_roundtrip_preserves_sharpes(ohlcv, tmp_path):
    fs = build_feature_set(ohlcv, FeatureConfig(target_mode="triple_barrier", target_horizon=10))
    mr = train(fs, TrainConfig(model_name="random_forest", validation="cpcv",
                               n_splits=6, cpcv_k=2))
    bundle = build_bundle(mr, fs, ohlcv, source_path="synthetic.csv")
    assert bundle.cpcv_sharpes == mr.cpcv_sharpes
    path = save_bundle(bundle, tmp_path / "cpcv.joblib")
    loaded = load_bundle(path)
    assert loaded.cpcv_sharpes == mr.cpcv_sharpes


# --- Реестр прогонов (гигиена селекционного смещения) ---------------------- #
def test_registry_scans_and_groups_runs(ohlcv, tmp_path):
    """scan_runs читает сводки, сортирует по Sharpe; группировка считает прогоны на данных."""
    from ml.registry import scan_runs, selection_bias_by_data

    fs = build_feature_set(ohlcv, FeatureConfig(target_mode="triple_barrier", target_horizon=10))
    # Три прогона на ОДНИХ данных — модельный случай селекционного перебора.
    for i in range(3):
        mr = train(fs, TrainConfig(model_name="random_forest", n_splits=4))
        bundle = build_bundle(mr, fs, ohlcv, source_path="synthetic.csv")
        save_bundle(bundle, tmp_path / f"run_{i}.joblib")

    rows = scan_runs(tmp_path)
    assert len(rows) == 3
    sharpes = [r.oos_sharpe for r in rows if r.oos_sharpe == r.oos_sharpe]
    assert sharpes == sorted(sharpes, reverse=True)        # отсортировано по убыванию

    bias = selection_bias_by_data(rows)
    assert len(bias) == 1                                  # одни данные → одна группа
    grp = next(iter(bias.values()))
    assert grp.n_runs == 3                                 # все три прогона на этих данных


def test_registry_ignores_broken_json(tmp_path):
    """Битый/чужой json и отсутствие .joblib тихо пропускаются."""
    from ml.registry import scan_runs

    (tmp_path / "broken.json").write_text("{not valid", encoding="utf-8")
    (tmp_path / "orphan.json").write_text('{"metrics": {}}', encoding="utf-8")  # нет .joblib
    assert scan_runs(tmp_path) == []


# --- Мета-разметка (AFML гл. 3) -------------------------------------------- #
def test_meta_labeling_pipeline(ohlcv):
    import numpy as np

    cfg = FeatureConfig(target_mode="meta", target_horizon=10,
                        primary_kind="ma_cross", primary_fast=15, primary_slow=40)
    fs = build_feature_set(ohlcv, cfg)
    assert fs.side_column == "prim_side"
    assert "prim_side" in fs.feature_columns
    assert set(fs.frame[fs.target_column].unique()) <= {0, 1}

    mr = train(fs, TrainConfig(model_name="random_forest", n_splits=4))
    finite = mr.oos_side[np.isfinite(mr.oos_side)]
    assert len(mr.oos_side) == len(mr.test_index) > 0
    assert set(np.unique(finite)) <= {-1.0, 1.0}        # сторона ∈ {-1,+1}
