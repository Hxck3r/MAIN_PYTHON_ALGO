"""Самодостаточное ML-ядро (перенесено из проекта ML_trading).

Слои:

* :mod:`ml.features` — индикаторы, разметка target, сборка ``FeatureSet``;
* :mod:`ml.validation` — purged walk-forward CV (защита от переобучения);
* :mod:`ml.models` — фабрика классических моделей (LightGBM / RF / GB / Logistic);
* :mod:`ml.trainer` — обучение + честная OOS-оценка + калибровка + подбор порога;
* :mod:`ml.experiment` — артефакт прогона ``RunBundle`` (с OOS-вероятностями).

ML-предсказания подаются в штатный движок MAIN через
``strategies.ml_strategy.MLStrategy`` — поэтому Monte-Carlo / PBO / Walk-Forward /
Deflated Sharpe применяются к ML-стратегии без изменений.

Импорт пакета намеренно «лёгкий»: тяжёлые sklearn/lightgbm подтягиваются лениво,
только при первом обращении к ``ml.trainer`` / ``ml.models`` — это держит старт
приложения быстрым на слабых ПК.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # подсказки типов без рантайм-импорта тяжёлых модулей
    # Реэкспорт для статических анализаторов: в рантайме имена резолвятся лениво
    # через __getattr__ (см. _LAZY ниже), поэтому ruff не видит их использования.
    from ml.domain import FeatureSet, ModelResult  # noqa: F401
    from ml.experiment import RunBundle  # noqa: F401

# Имя -> (подмодуль, атрибут). Резолвится лениво в __getattr__.
_LAZY: dict[str, tuple[str, str]] = {
    "FeatureConfig": ("ml.features.engineering", "FeatureConfig"),
    "build_feature_set": ("ml.features.engineering", "build_feature_set"),
    "FeatureSet": ("ml.domain", "FeatureSet"),
    "ModelResult": ("ml.domain", "ModelResult"),
    "PurgedWalkForwardSplit": ("ml.validation", "PurgedWalkForwardSplit"),
    "TrainConfig": ("ml.trainer", "TrainConfig"),
    "train": ("ml.trainer", "train"),
    "available_models": ("ml.models", "available_models"),
    "default_model_name": ("ml.models", "default_model_name"),
    "MODEL_SPECS": ("ml.models", "MODEL_SPECS"),
    "RunBundle": ("ml.experiment", "RunBundle"),
    "build_bundle": ("ml.experiment", "build_bundle"),
    "save_bundle": ("ml.experiment", "save_bundle"),
    "load_bundle": ("ml.experiment", "load_bundle"),
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:  # PEP 562 — ленивая загрузка символов пакета
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'ml' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])
