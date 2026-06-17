"""Диагностика данных и целевой переменной — понять, почему модель «не учится».

Часто причина не в модели, а в данных/таргете: близкий к случайному блужданию ряд,
сильный перекос классов или слишком трудная цель (направление на 1 бар). Эти метрики
выводятся во вкладке, чтобы пользователь видел контекст, а не только AUC.

Без Qt — тестируется headless.
"""
from __future__ import annotations

import numpy as np

from ml.domain import FeatureSet


def variance_ratio(returns: np.ndarray, q: int = 5) -> float:
    """Тест отношения дисперсий Ло–Маккинли.

    VR ≈ 1 — случайное блуждание (направление непредсказуемо по построению);
    VR > 1 — тренд (моментум), VR < 1 — возврат к среднему. Оба отклонения от 1
    означают, что в ряду ЕСТЬ структура, которую модель может ловить.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < q * 5:
        return float("nan")
    var_1 = r.var(ddof=1)
    if var_1 <= 0:
        return float("nan")
    rolled = np.convolve(r, np.ones(q), mode="valid")     # суммы по q баров
    var_q = rolled.var(ddof=1)
    return float(var_q / (q * var_1))


def vr_label(vr: float) -> str:
    """Человекочитаемая трактовка variance ratio."""
    if not np.isfinite(vr):
        return "—"
    if vr > 1.15:
        return "тренд (есть моментум)"
    if vr < 0.85:
        return "возврат к среднему"
    return "близко к случайному блужданию"


def data_target_diagnostics(fs: FeatureSet) -> dict[str, float]:
    """Числовые диагностики FeatureSet (для вывода в панели)."""
    frame = fs.frame
    y = frame[fs.target_column].astype(int)
    n = len(frame)
    pos_rate = float(y.mean()) if n else float("nan")
    baseline_acc = max(pos_rate, 1.0 - pos_rate) if n else float("nan")
    rets = frame["close"].pct_change().to_numpy()
    vr = variance_ratio(rets, q=5)
    return {
        "n_samples": float(n),
        "pos_rate": pos_rate,
        "baseline_acc": baseline_acc,
        "variance_ratio": vr,
    }
