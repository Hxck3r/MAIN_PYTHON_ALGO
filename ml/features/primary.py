"""Первичный сигнал для мета-разметки (López de Prado, AFML гл. 3).

В схеме мета-лейблинга **сторону** сделки (long/short) задаёт простое первичное правило,
а ML-модель учится лишь решать, **стоит ли** эту сделку брать. Здесь — пара дешёвых,
прозрачных первичных правил, выдающих сторону ``side ∈ {-1, +1}`` (0 — на разогреве).

Сторона строго причинна (считается по прошлым барам), поэтому её можно и обучать как
признак, и переигрывать в стратегии без заглядывания в будущее.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import indicators as ind

PRIMARY_KINDS = ("ma_cross", "momentum")


def primary_signal(
    close: pd.Series, kind: str = "ma_cross", fast: int = 20, slow: int = 50
) -> pd.Series:
    """Сторона первичного сигнала ``{-1, +1}`` (NaN на разогреве).

    * ``ma_cross`` — знак разности быстрой и медленной SMA (тренд-следование);
    * ``momentum`` — знак доходности за ``fast`` баров (импульс).
    """
    if kind == "momentum":
        mom = close - close.shift(max(1, fast))
        side = np.sign(mom)
    else:  # ma_cross
        sma_fast = ind.sma(close, max(1, fast))
        sma_slow = ind.sma(close, max(fast + 1, slow))
        side = np.sign(sma_fast - sma_slow)
    side = side.replace(0.0, np.nan)          # 0 (равенство) трактуем как «нет стороны»
    return side.rename("prim_side")
