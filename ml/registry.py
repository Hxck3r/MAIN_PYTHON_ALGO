"""Реестр прогонов: гигиена селекционного смещения по сохранённым моделям.

Зачем это нужно
---------------
Главный способ обмануть себя в квант-исследовании — обучить много моделей на одних и
тех же данных и оставить «лучшую по Sharpe». Дефляция (DSR) внутри одного прогона
учитывает только перебор гиперпараметров ВНУТРИ него, но не сам факт «я запустил 17
прогонов». Этот модуль собирает все сохранённые прогоны (по лёгким ``.json``-сводкам,
без загрузки моделей), группирует их по **отпечатку данных** и считает разброс
OOS-Sharpe по группе — чтобы пользователь видел: лучший результат завышен отбором, а
доверять стоит DSR/CPCV, а не максимуму.

Модуль на чистом Python (json/pathlib/statistics), без Qt/ML — тестируется headless.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Any

log = logging.getLogger(__name__)


def _f(d: dict, key: str, default: float = float("nan")) -> float:
    """Достать число из словаря метрик (NaN/дефолт, если нет/не число)."""
    try:
        v = d.get(key, default)
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


@dataclass
class RunRow:
    """Одна строка реестра — сводка одного сохранённого прогона."""

    path: Path                  # путь к артефакту .joblib (модель + OOS)
    created_at: str
    model_name: str
    validation: str
    auc: float                  # OOS ROC-AUC (cv_auc_mean или roc_auc)
    oos_sharpe: float           # net издержек, за бар
    psr: float
    dsr: float
    cpcv_prob_positive: float   # доля путей CPCV с краем > 0 (NaN, если не CPCV)
    cost_bps: float
    data_sha1: str
    data_period: str
    n_bars: int


def _row_from_summary(json_path: Path, summary: dict[str, Any]) -> RunRow:
    """Собрать :class:`RunRow` из распарсенной ``.json``-сводки прогона."""
    m = summary.get("metrics", {}) or {}
    fp = summary.get("data_fingerprint", {}) or {}
    auc = _f(m, "cv_auc_mean")
    if auc != auc:                       # NaN → откат на holdout-AUC
        auc = _f(m, "roc_auc")
    start, end = fp.get("start"), fp.get("end")
    period = f"{str(start)[:10]} … {str(end)[:10]}" if start and end else "—"
    return RunRow(
        path=json_path.with_suffix(".joblib"),
        created_at=str(summary.get("created_at", ""))[:19].replace("T", " "),
        model_name=str(summary.get("model_name", "—")),
        validation=str(summary.get("validation", "—")),
        auc=auc,
        oos_sharpe=_f(m, "oos_sharpe"),
        psr=_f(m, "psr"),
        dsr=_f(m, "dsr"),
        cpcv_prob_positive=_f(m, "cpcv_prob_positive"),
        cost_bps=_f(m, "cost_bps", 0.0),
        data_sha1=str(fp.get("sha1", "n/a")),
        data_period=period,
        n_bars=int(fp.get("n_bars", 0) or 0),
    )


def scan_runs(models_dir: str | Path) -> list[RunRow]:
    """Прочитать все ``.json``-сводки в каталоге моделей (без загрузки .joblib).

    Возвращает строки, отсортированные по net OOS-Sharpe убыванию (NaN — в конец).
    Битые/чужие json тихо пропускаются.
    """
    models_dir = Path(models_dir)
    rows: list[RunRow] = []
    if not models_dir.exists():
        return rows
    for jp in sorted(models_dir.glob("*.json")):
        try:
            summary = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — чужой/битый json
            log.debug("Пропущен %s: %s", jp.name, exc)
            continue
        if "metrics" not in summary or not jp.with_suffix(".joblib").exists():
            continue
        rows.append(_row_from_summary(jp, summary))

    def _key(r: RunRow) -> float:
        return r.oos_sharpe if r.oos_sharpe == r.oos_sharpe else float("-inf")

    rows.sort(key=_key, reverse=True)
    return rows


@dataclass
class GroupBias:
    """Сводка селекционного смещения по группе прогонов на одних данных."""

    data_sha1: str
    data_period: str
    n_runs: int
    best_sharpe: float
    sharpe_std: float           # разброс OOS-Sharpe по прогонам (шум отбора)
    best_dsr: float             # DSR лучшего прогона (если считался) — честнее максимума


def selection_bias_by_data(rows: list[RunRow]) -> dict[str, GroupBias]:
    """Сгруппировать прогоны по отпечатку данных и оценить смещение отбора.

    Чем больше прогонов на ОДНИХ данных и чем шире разброс их OOS-Sharpe, тем сильнее
    «лучший» завышен отбором — реальный край ближе к DSR/CPCV, а не к максимуму.
    """
    groups: dict[str, list[RunRow]] = {}
    for r in rows:
        groups.setdefault(r.data_sha1, []).append(r)

    out: dict[str, GroupBias] = {}
    for sha1, grp in groups.items():
        sharpes = [r.oos_sharpe for r in grp if r.oos_sharpe == r.oos_sharpe]
        best = max(sharpes) if sharpes else float("nan")
        std = pstdev(sharpes) if len(sharpes) > 1 else 0.0
        best_run = max(grp, key=lambda r: (r.oos_sharpe if r.oos_sharpe == r.oos_sharpe else float("-inf")))
        out[sha1] = GroupBias(
            data_sha1=sha1,
            data_period=grp[0].data_period,
            n_runs=len(grp),
            best_sharpe=best,
            sharpe_std=std,
            best_dsr=best_run.dsr,
        )
    return out
