"""Фабрика ML-моделей (scikit-learn).

Каждая модель описана спецификацией: человекочитаемое имя, конструктор, параметры по
умолчанию и признак необходимости масштабирования. Линейные модели заворачиваются в
``Pipeline`` со ``StandardScaler``; деревьям масштабирование не нужно.

Добавить свою модель = добавить запись в :data:`MODEL_SPECS`.

Импорт sklearn/lightgbm — тяжёлый, поэтому этот модуль грузится ЛЕНИВО: основное
приложение его не импортирует, пока пользователь не откроет вкладку машинного обучения.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# LightGBM — основная модель под слабый CPU (гистограммный бустинг: быстро и точно).
# Импорт необязательный: если пакет не установлен, модель просто не появится в списке.
try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:  # pragma: no cover - зависит от окружения
    _HAS_LGBM = False


def safe_n_jobs() -> int:
    """Число потоков под слабый ПК: все ядра минус одно (≥1) — GUI не «замерзает».

    ``n_jobs=-1`` забирал бы все ядра, и интерфейс на маломощной машине подвисал бы во
    время обучения. Оставляем одно ядро системе/интерфейсу.
    """
    return max(1, (os.cpu_count() or 2) - 1)


@dataclass
class ModelSpec:
    label: str
    builder: Callable[..., BaseEstimator]
    defaults: dict[str, Any] = field(default_factory=dict)
    needs_scaling: bool = False
    supports_class_weight: bool = False   # принимает class_weight="balanced"


MODEL_SPECS: dict[str, ModelSpec] = {}

if _HAS_LGBM:
    MODEL_SPECS["lightgbm"] = ModelSpec(
        label="LightGBM",
        builder=LGBMClassifier,
        # Параметры подобраны под слабый ПК: умеренное число деревьев, ограниченные
        # листья, сабсэмплинг — быстрая тренировка без существенной потери качества.
        defaults={
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": -1,
        },
        supports_class_weight=True,
    )

MODEL_SPECS.update({
    "random_forest": ModelSpec(
        label="Random Forest",
        builder=RandomForestClassifier,
        defaults={"n_estimators": 300, "max_depth": 6, "n_jobs": -1, "random_state": 42},
        supports_class_weight=True,
    ),
    "gradient_boosting": ModelSpec(
        label="Gradient Boosting",
        builder=GradientBoostingClassifier,
        defaults={"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "random_state": 42},
    ),
    "logistic_regression": ModelSpec(
        label="Logistic Regression",
        builder=LogisticRegression,
        defaults={"C": 1.0, "max_iter": 1000},
        needs_scaling=True,
        supports_class_weight=True,
    ),
})


def available_models() -> list[tuple[str, str]]:
    """Список ``(ключ, человекочитаемое_имя)`` для выпадающего списка в GUI."""
    return [(key, spec.label) for key, spec in MODEL_SPECS.items()]


def build_model(
    name: str,
    params: dict[str, Any] | None = None,
    *,
    balance_classes: bool = False,
) -> BaseEstimator:
    """Собрать модель по ключу. Линейные модели оборачиваются в Pipeline со скейлером.

    ``balance_classes`` добавляет ``class_weight="balanced"`` тем моделям, что его
    поддерживают (полезно при перекосе классов в target).
    """
    if name not in MODEL_SPECS:
        raise KeyError(f"Неизвестная модель: {name}")
    spec = MODEL_SPECS[name]
    merged = {**spec.defaults, **(params or {})}
    if balance_classes and spec.supports_class_weight:
        merged.setdefault("class_weight", "balanced")
    # Под слабый ПК: -1 (все ядра) → все ядра минус одно, чтобы интерфейс не подвисал.
    if merged.get("n_jobs") in (-1, None) and "n_jobs" in merged:
        merged["n_jobs"] = safe_n_jobs()
    estimator = spec.builder(**merged)
    if spec.needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    return estimator


def default_model_name() -> str:
    """Ключ модели по умолчанию: LightGBM, если доступен, иначе Random Forest."""
    return "lightgbm" if "lightgbm" in MODEL_SPECS else "random_forest"


def final_estimator(model: BaseEstimator) -> BaseEstimator:
    """Достать итоговый estimator: разворачивает Pipeline и калибровочную обёртку."""
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]
    # CalibratedClassifierCV прячет базовую модель внутри calibrated_classifiers_.
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        cc = model.calibrated_classifiers_[0]
        base = getattr(cc, "estimator", None)
        if base is not None:
            return final_estimator(base)
    return model
