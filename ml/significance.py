"""Статистическая значимость качества модели — честная оценка «есть ли край».

Высокий AUC сам по себе ничего не гарантирует: на коротком ряду он легко
оказывается случайным. Поэтому считаем:

* **Доверительный интервал AUC** методом DeLong (Sun & Xu, 2014) — замкнутая форма
  без пересэмплинга. Если нижняя граница > 0.5, преимущество над случайностью
  статистически значимо.
* **Information Coefficient (IC)** — ранговая корреляция (Спирмен) между предсказанной
  вероятностью роста и фактической будущей доходностью. Прямой «экономический» сигнал:
  IC > 0 со значимым p — модель ранжирует сделки в нужную сторону.

Модуль на numpy/scipy, без Qt — тестируется headless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    """Средние ранги (с учётом совпадений) — ядро быстрого DeLong."""
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def auc_ci(
    y_true, y_score, alpha: float = 0.95
) -> tuple[float, float, float, float]:
    """AUC + доверительный интервал (DeLong) + p-value против AUC=0.5.

    Возвращает ``(auc, ci_low, ci_high, p_value)``. ``p_value`` — односторонняя
    вероятность получить такой AUC случайно (меньше — лучше). При вырожденных
    входах (один класс) возвращает NaN.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    tz = _midrank(np.concatenate([pos, neg]))
    tx = _midrank(pos)
    ty = _midrank(neg)
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)

    v10 = (tz[:m] - tx) / n                   # компоненты по положительным
    v01 = 1.0 - (tz[m:] - ty) / m             # компоненты по отрицательным
    s10 = v10.var(ddof=1) / m if m > 1 else 0.0
    s01 = v01.var(ddof=1) / n if n > 1 else 0.0
    var = s10 + s01
    if not np.isfinite(var) or var <= 0:
        return float(auc), float("nan"), float("nan"), float("nan")

    se = float(np.sqrt(var))
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    lo = max(0.0, auc - z * se)
    hi = min(1.0, auc + z * se)
    p = float(stats.norm.sf((auc - 0.5) / se))   # H0: AUC = 0.5, односторонний
    return float(auc), float(lo), float(hi), p


def information_coefficient(proba, forward_returns) -> tuple[float, float]:
    """IC = ранговая корреляция (Спирмен) вероятности и будущей доходности.

    Возвращает ``(ic, p_value)``. Выравнивает ряды по общему индексу и отбрасывает
    NaN. При недостатке данных возвращает ``(nan, nan)``.
    """
    proba = pd.Series(proba)
    fwd = pd.Series(forward_returns)
    df = pd.concat([proba.rename("p"), fwd.rename("r")], axis=1).dropna()
    if len(df) < 10 or df["p"].nunique() < 2 or df["r"].nunique() < 2:
        return float("nan"), float("nan")
    res = stats.spearmanr(df["p"].to_numpy(), df["r"].to_numpy())
    return float(res.statistic), float(res.pvalue)
