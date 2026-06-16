"""Фракционное дифференцирование (López de Prado, AFML гл. 5).

Зачем это нужно
---------------
Доходности (``pct_change``) делают ряд стационарным, но **стирают почти всю память**:
для ML это потеря предиктивной информации. Фракционное дифференцирование берёт
дробный порядок ``d`` (обычно 0.2–0.6), при котором ряд уже стационарен (проходит
ADF/VR-тесты), но **память сохранена** — это часто лучшие признаки, чем чистые
доходности.

Здесь реализован метод фиксированного окна (FFD, snippet 5.3 AFML): веса обрезаются
по порогу ``threshold``, поэтому окно конечной длины — расчёт линеен по ряду (дёшево
на слабом ПК), а результат не «течёт» по длине истории.

Чистый numpy/pandas, без Qt/ML — тестируется headless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ffd_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Веса FFD для порядка ``d`` (от старых к новым), обрезанные по ``threshold``.

    Рекуррента биномиального ряда: ``w_k = -w_{k-1} * (d - k + 1) / k``, ``w_0 = 1``.
    Члены меньше ``threshold`` по модулю отбрасываются → конечное окно.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
        if k > 10_000:                      # страховка от вырожденного d
            break
    return np.array(w[::-1], dtype=float)   # порядок: старые → новые


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Фракционно продифференцировать ряд методом фиксированного окна.

    Первые ``len(weights)-1`` значений — ``NaN`` (нет полного окна), что согласуется с
    разогревом прочих индикаторов и отбрасывается общим ``dropna`` пайплайна.
    """
    if d <= 0:                              # d=0 — исходный ряд (память не тронута)
        return series.astype(float).copy()
    x = np.asarray(series, dtype=float)
    w = ffd_weights(d, threshold)
    width = len(w)
    out = np.full(x.size, np.nan)
    if x.size >= width:
        # Кросс-корреляция окна (старые→новые) с весами (старые→новые) = FFD-значение.
        out[width - 1:] = np.correlate(x, w, mode="valid")
    return pd.Series(out, index=series.index, name=series.name)
