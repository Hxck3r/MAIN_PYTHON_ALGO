"""Воспроизводимый артефакт прогона («Run»).

Сохраняется не голая модель, а весь контекст результата: модель + список признаков +
порог решения + конфиг признаков + метрики + **отпечаток данных** (хэш, период, число
баров) + **out-of-fold вероятности** (их «проигрывает» MLStrategy в бэктесте без
заглядывания в будущее). Это даёт воспроизводимость и честный бэктест на OOS-сегменте.

Модуль не зависит от Qt. Рядом с ``.joblib`` (полный артефакт с моделью) пишется
``.json`` — человекочитаемая сводка метрик для быстрого просмотра без загрузки модели.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ml import config
from ml.domain import FeatureSet, ModelResult

log = logging.getLogger(__name__)

ARTIFACT_VERSION = 3          # v3: + oos_true/oos_fwd/importances_mda (интеракт. визуализация)


def data_fingerprint(df: pd.DataFrame, source_path: Optional[str]) -> dict[str, Any]:
    """Короткий отпечаток данных: хэш содержимого + период + число баров."""
    try:
        digest = hashlib.sha1(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()[:16]
    except Exception:  # noqa: BLE001 - хэш не критичен
        digest = "n/a"
    idx = df.index
    return {
        "source": Path(source_path).name if source_path else None,
        "n_bars": int(len(df)),
        "start": str(idx[0]) if len(idx) else None,
        "end": str(idx[-1]) if len(idx) else None,
        "sha1": digest,
    }


@dataclass
class RunBundle:
    """Полный артефакт обученного прогона."""

    model: Any
    model_name: str
    feature_columns: list[str]
    threshold: float
    validation: str
    calibrated: bool
    metrics: dict[str, float]
    # Out-of-fold вероятности класса «1» по времени — основа честного бэктеста.
    oos_proba: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Истинные метки и «сырая» доходность следующего бара на OOS — для интерактивной
    # визуализации (калибровка/ROC/гистограмма/эквити) после загрузки артефакта.
    oos_true: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    oos_fwd: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Сторона первичного сигнала на OOS-барах (мета-разметка) — её играет MetaLabelStrategy.
    oos_side: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fold_metrics: list[dict] = field(default_factory=list)
    importances: dict[str, float] = field(default_factory=dict)
    importances_mda: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    feature_config: dict = field(default_factory=dict)
    data_fingerprint: dict = field(default_factory=dict)
    app_version: str = config.APP_VERSION
    artifact_version: int = ARTIFACT_VERSION
    created_at: str = ""

    def summary(self) -> dict[str, Any]:
        """Метаданные без модели и без рядов по барам — для JSON-сводки."""
        d = asdict(self)
        for heavy in ("model", "oos_proba", "oos_true", "oos_fwd", "oos_side"):
            d.pop(heavy, None)
        return d


def build_bundle(
    mr: ModelResult,
    fs: FeatureSet,
    raw_df: pd.DataFrame,
    source_path: Optional[str],
) -> RunBundle:
    """Собрать :class:`RunBundle` из результатов пайплайна."""
    return RunBundle(
        model=mr.model,
        model_name=mr.model_name,
        feature_columns=list(mr.feature_columns),
        threshold=float(mr.threshold),
        validation=mr.validation,
        calibrated=mr.calibrated,
        metrics=dict(mr.metrics),
        oos_proba=pd.Series(mr.y_proba.values, index=mr.y_proba.index, name="oos_proba"),
        oos_true=pd.Series(mr.y_true.values, index=mr.y_true.index, name="oos_true"),
        oos_fwd=pd.Series(mr.oos_fwd.values, index=mr.oos_fwd.index, name="oos_fwd")
        if len(mr.oos_fwd) else pd.Series(dtype=float),
        oos_side=pd.Series(mr.oos_side.values, index=mr.oos_side.index, name="oos_side")
        if len(mr.oos_side) else pd.Series(dtype=float),
        fold_metrics=list(mr.fold_metrics),
        importances=dict(mr.importances),
        importances_mda=dict(mr.importances_mda),
        confusion=list(mr.confusion),
        feature_config=dict(fs.config_meta),
        data_fingerprint=data_fingerprint(raw_df, source_path),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def save_bundle(bundle: RunBundle, path: str | Path) -> Path:
    """Сохранить полный артефакт (.joblib) и сводку (.json рядом)."""
    from joblib import dump

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, path)
    sidecar = path.with_suffix(".json")
    try:
        sidecar.write_text(
            json.dumps(bundle.summary(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - сводка не критична
        log.warning("Не удалось записать JSON-сводку: %s", exc)
    log.info("Прогон сохранён: %s", path)
    return path


def load_bundle(path: str | Path) -> RunBundle:
    """Загрузить артефакт прогона."""
    from joblib import load

    obj = load(path)
    if not isinstance(obj, RunBundle):
        raise ValueError("Файл не является артефактом прогона ML.")
    return obj
