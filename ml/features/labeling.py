"""Разметка целевой переменной для классификации.

Два подхода:

* **direction** — направление через ``horizon`` баров (знак будущей доходности с порогом).
  Прост, но экономически «плоский»: одинаково трактует +0.1% и +10%.
* **triple_barrier** — метод тройного барьера (López de Prado). Для каждого бара ставятся
  три барьера на горизонте ``horizon``:

  - верхний (take-profit) = ``close * (1 + pt * vol)``;
  - нижний (stop-loss)    = ``close * (1 - sl * vol)``;
  - вертикальный          = конец окна ``horizon``.

  Метка = какой барьер сработал первым: ``1`` (вверх) если первым задет верхний,
  ``0`` (вниз) если нижний; при достижении лишь вертикального — по знаку доходности.
  Барьеры масштабируются волатильностью, поэтому таргет учитывает риск/доходность сделки
  и не вырождается на спокойных/бурных участках.

Модуль на чистом numpy/pandas, без Qt/ML — тестируется headless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewm_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """Оценка удельной волатильности бара = EWMA-СКО доходностей (единица барьеров)."""
    returns = close.pct_change()
    return returns.ewm(span=span, min_periods=span // 2 + 1).std()


def direction_labels(
    close: pd.Series, horizon: int, threshold: float
) -> pd.Series:
    """Метка направления: 1, если доходность через ``horizon`` баров > порога."""
    fwd_ret = close.shift(-horizon) / close - 1.0
    out = (fwd_ret > threshold).astype("float")
    out.iloc[-horizon:] = np.nan          # у последних баров нет будущего
    return out


def triple_barrier_events(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    horizon: int,
    pt: float = 1.5,
    sl: float = 1.5,
    vol_span: int = 20,
) -> pd.DataFrame:
    """Тройной барьер: метка ``{0,1,NaN}`` + смещение до касания ``t1`` (в барах).

    Возвращает кадр с колонками:

    * ``label`` — какой барьер сработал первым (1 — вверх, 0 — вниз/стоп);
    * ``t1`` — на сколько баров вперёд от ``i`` сработал барьер (``1..horizon``); нужно
      для весов уникальности (длина «жизни» метки). Для строк без метки — ``NaN``.

    Барьеры проверяются по внутрибаровым ``high``/``low`` (реалистичнее, чем по close).
    Бары с неполным окном впереди (``i + horizon > n-1``) остаются ``NaN``.
    """
    vol = ewm_volatility(close, vol_span).to_numpy()
    c = close.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    n = len(c)
    labels = np.full(n, np.nan)
    t1 = np.full(n, np.nan)

    for i in range(n):
        if i + horizon > n - 1:           # неполное окно — пропуск (нет look-ahead)
            break
        vi = vol[i]
        if not np.isfinite(vi) or vi <= 0:
            continue
        upper = c[i] * (1.0 + pt * vi)
        lower = c[i] * (1.0 - sl * vi)
        label = None
        touch = horizon                    # по умолчанию — вертикальный барьер
        for j in range(i + 1, i + horizon + 1):
            hit_up = h[j] >= upper
            hit_dn = lo[j] <= lower
            if hit_up and hit_dn:
                label, touch = 0.0, j - i  # оба за один бар — консервативно «стоп первым»
                break
            if hit_up:
                label, touch = 1.0, j - i
                break
            if hit_dn:
                label, touch = 0.0, j - i
                break
        if label is None:                  # сработал лишь вертикальный барьер
            label = 1.0 if c[i + horizon] > c[i] else 0.0
        labels[i] = label
        t1[i] = touch

    return pd.DataFrame({"label": labels, "t1": t1}, index=close.index)


def triple_barrier_labels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    horizon: int,
    pt: float = 1.5,
    sl: float = 1.5,
    vol_span: int = 20,
) -> pd.Series:
    """Метки тройного барьера ({0.0, 1.0, NaN}) — тонкая обёртка над событиями."""
    events = triple_barrier_events(high, low, close, horizon, pt, sl, vol_span)
    return events["label"].rename(None)
