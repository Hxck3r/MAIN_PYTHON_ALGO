"""Веса наблюдений для обучения на перекрывающихся метках (López de Prado, AFML гл. 4).

Зачем это нужно
---------------
Метка бара ``i`` смотрит вперёд на ``horizon`` баров (направление) или до касания
барьера (тройной барьер). Поэтому метки соседних баров **перекрываются во времени** и
несут почти одну и ту же информацию — выборка не-IID. Без поправки модель переоценивает
«густо засеянные» периоды (мало уникальной информации, но много почти одинаковых строк),
что завышает оптимизм оценки и провоцирует переобучение.

Решение AFML — взвесить наблюдения по **уникальности**: чем больше меток одновременно
«живёт» на барах, которые покрывает метка ``i``, тем меньше её уникальный вклад и тем
меньше её вес. Дополнительно поддержан линейный **временной декей** (более старые
наблюдения весят меньше — рынок нестационарен).

Модуль на чистом numpy (O(n) через разностный массив и префиксные суммы — дёшево на
слабом ПК), без Qt/sklearn — тестируется headless.
"""
from __future__ import annotations

import numpy as np


def concurrency_weights(event_end: np.ndarray, n: int) -> np.ndarray:
    """Веса уникальности для ``n`` наблюдений по их горизонтам.

    Параметры
    ---------
    event_end : позиционный индекс последнего бара, который покрывает метка наблюдения
        ``i`` (метка живёт на ``[i, event_end[i]]``). Должен быть ``>= i``.
    n : число наблюдений (= длине ряда).

    Возвращает массив весов длины ``n``, нормированный к среднему ``1.0`` (чтобы не менять
    эффективный масштаб регуляризации модели). Сумма весов ≈ ``n``.

    Алгоритм (AFML ``mpNumCoEvents`` + ``mpSampleW``):
    1. concurrency ``c_t`` — сколько меток одновременно активны на баре ``t`` (разностный
       массив + cumsum, O(n));
    2. вес ``w_i`` = среднее по ``[i, event_end_i]`` от ``1/c_t`` (префиксная сумма, O(n)).
    """
    starts = np.arange(n)
    ends = np.clip(np.asarray(event_end, dtype=np.int64), starts, n - 1)

    # --- 1. Параллельность меток на каждом баре (difference array) ---
    diff = np.zeros(n + 1, dtype=np.float64)
    np.add.at(diff, starts, 1.0)
    np.add.at(diff, ends + 1, -1.0)
    concurrency = np.cumsum(diff[:n])
    concurrency[concurrency < 1.0] = 1.0           # защита от деления на ноль

    # --- 2. Средняя по горизонту обратная параллельность (prefix sums) ---
    inv = 1.0 / concurrency
    prefix = np.concatenate(([0.0], np.cumsum(inv)))
    span = (ends - starts + 1).astype(np.float64)
    weights = (prefix[ends + 1] - prefix[starts]) / span

    mean = weights.mean()
    if mean > 0:
        weights = weights / mean
    return weights


def time_decay(weights: np.ndarray, last_w: float = 1.0) -> np.ndarray:
    """Линейный временной декей весов (AFML 4.5.2, упрощённый, по позиции).

    ``weights`` упорядочены по времени (старые → новые). ``last_w ∈ (0, 1]`` — множитель
    для самого СТАРОГО наблюдения; самое новое умножается на ``1.0``. ``last_w = 1.0`` —
    декей выключен. Результат пере-нормируется к среднему ``1.0``.
    """
    n = len(weights)
    if n == 0 or last_w >= 1.0:
        return weights
    last_w = max(0.0, float(last_w))
    factor = np.linspace(last_w, 1.0, n)           # старые → новые
    out = np.asarray(weights, dtype=np.float64) * factor
    mean = out.mean()
    if mean > 0:
        out = out / mean
    return out
