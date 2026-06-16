"""Кеш матрицы признаков на диске — оптимизация под слабый ПК.

Расчёт признаков (индикаторы по всему ряду) — заметная доля времени обучения.
При повторных прогонах с теми же данными и настройками признаков (например, перебор
модели/числа фолдов/порога) пересчитывать их незачем. Кеш ключуется отпечатком данных
(``data_fingerprint``) и хэшем :class:`FeatureConfig`, поэтому при смене данных или
настроек он автоматически инвалидируется.

Хранится в ``.cache/ml_features`` в корне проекта (в .gitignore). Кеш — необязательная
оптимизация: любая ошибка чтения/записи просто означает обычный пересчёт.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict

import pandas as pd

from ml import config
from ml.experiment import data_fingerprint

log = logging.getLogger(__name__)


def _config_hash(cfg) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def cached_feature_set(df: pd.DataFrame, cfg, *, enabled: bool = True):
    """Вернуть ``FeatureSet`` из кеша или посчитать и закешировать.

    ``enabled=False`` полностью отключает кеш (всегда честный пересчёт).
    """
    from ml.features.engineering import build_feature_set

    if not enabled:
        return build_feature_set(df, cfg)

    key = f"{data_fingerprint(df, None)['sha1']}_{_config_hash(cfg)}"
    path = config.CACHE_DIR / f"fs_{key}.joblib"

    if path.exists():
        try:
            from joblib import load

            return load(path)
        except Exception as exc:  # noqa: BLE001 — кеш не критичен
            log.warning("Кеш признаков повреждён (%s) — пересчитываю.", exc)

    fs = build_feature_set(df, cfg)
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from joblib import dump

        dump(fs, path)
    except Exception as exc:  # noqa: BLE001 — запись кеша не критична
        log.warning("Не удалось записать кеш признаков: %s", exc)
    return fs
