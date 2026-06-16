"""Подбор гиперпараметров случайным поиском на purged walk-forward CV.

Опциональный шаг (включается галочкой в панели). Перебирает N случайных
конфигураций модели и выбирает лучшую по среднему OOS-AUC по фолдам — на той же
честной кросс-валидации, что и финальное обучение (без заглядывания в будущее).

Лёгкий внутренний поиск без внешних зависимостей (без Optuna), чтобы работать на
слабом ПК. Число проб ограничено и задаётся пользователем — каждый trial = полный
walk-forward, поэтому стоимость линейна по числу проб.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np

from ml.domain import FeatureSet
from ml.trainer import ProgressCb, TrainConfig, _emit, _train_walk_forward

log = logging.getLogger(__name__)


def _sample_params(model_name: str, rng: np.random.Generator) -> dict[str, Any]:
    """Случайная конфигурация гиперпараметров для заданной модели."""
    if model_name == "lightgbm":
        return {
            "num_leaves": int(rng.integers(15, 64)),
            "learning_rate": float(10 ** rng.uniform(-2.0, -1.0)),
            "n_estimators": int(rng.integers(150, 500)),
            "min_child_samples": int(rng.integers(10, 60)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "reg_lambda": float(rng.uniform(0.0, 5.0)),
        }
    if model_name == "random_forest":
        max_features = ["sqrt", "log2", 0.5][int(rng.integers(0, 3))]
        return {
            "n_estimators": int(rng.integers(150, 500)),
            "max_depth": int(rng.integers(3, 13)),
            "min_samples_leaf": int(rng.integers(1, 20)),
            "max_features": max_features,
        }
    if model_name == "gradient_boosting":
        return {
            "n_estimators": int(rng.integers(100, 400)),
            "max_depth": int(rng.integers(2, 5)),
            "learning_rate": float(10 ** rng.uniform(-1.7, -1.0)),
        }
    if model_name == "logistic_regression":
        return {"C": float(10 ** rng.uniform(-2.0, 1.0))}
    return {}


def _score(mr) -> float:
    s = mr.metrics.get("cv_auc_mean", float("nan"))
    if not np.isfinite(s):
        s = mr.metrics.get("roc_auc", float("nan"))
    return s


def tune(
    fs: FeatureSet,
    cfg: TrainConfig,
    n_trials: int = 15,
    progress: ProgressCb = None,
    seed: int = 42,
) -> tuple[dict[str, Any], float, list[float]]:
    """Подобрать гиперпараметры на purged walk-forward CV.

    Возвращает ``(лучшие_параметры, лучший_AUC, sharpe_проб)`` — список Sharpe
    OOS-сигнала по каждой удачной пробе используется для дефляции результата (DSR):
    он отражает разброс «лучшего из N» и штрафует множественное тестирование.
    """
    rng = np.random.default_rng(seed)
    best_params: dict[str, Any] = {}
    best_score = -np.inf
    trial_sharpes: list[float] = []

    for i in range(max(1, n_trials)):
        params = _sample_params(cfg.model_name, rng)
        _emit(progress, int(100 * i / max(1, n_trials)),
              f"Подбор гиперпараметров {i + 1}/{n_trials}…")
        # Калибровку/подбор порога на этапе поиска отключаем — нужен только AUC.
        trial_cfg = replace(cfg, params=params, calibrate=False, optimize_threshold=False)
        try:
            mr = _train_walk_forward(fs, trial_cfg, None)
        except Exception as exc:  # noqa: BLE001 — пропускаем вырожденные конфиги
            log.warning("Trial %d пропущен: %s", i + 1, exc)
            continue
        score = _score(mr)
        trial_sharpes.append(float(mr.metrics.get("oos_sharpe", float("nan"))))
        if np.isfinite(score) and score > best_score:
            best_score, best_params = score, params

    log.info("Подбор завершён: лучший OOS-AUC %.3f при %s", best_score, best_params)
    return best_params, float(best_score), trial_sharpes
