"""Комбинаторно-очищенная кросс-валидация (CPCV, López de Prado, AFML гл. 12).

Зачем это нужно
---------------
Walk-forward даёт ОДИН out-of-sample путь — одну точку Sharpe. На коротком финансовом
ряду эта точка крайне шумна: «повезло с одним разбиением» неотличимо от реального края.
CPCV строит МНОГО out-of-sample путей из одних и тех же данных и даёт **распределение**
OOS-доходности — это честный ответ на вопрос «край устойчив или это артефакт разбиения».

Идея
----
1. Ряд делится на ``N`` смежных групп.
2. Перебираются все ``C(N, k)`` сочетаний из ``k`` тест-групп; на каждом модель учится
   на остальных группах (с purge+embargo на стыках), предсказывает на тест-группах.
3. Из предсказаний собираются ``P = C(N-1, k-1)`` полноразмерных backtest-путей: в каждом
   пути каждая группа встречается ровно один раз (взята из своего сочетания). Распределение
   Sharpe по этим путям и есть результат.

Каждая группа тестируется в ``C(N-1, k-1)`` сочетаниях — отсюда число путей. Стоимость =
``C(N, k)`` обучений модели (на слабом ПК число групп/``k`` задаёт пользователь).

Модуль на чистом numpy/itertools, без Qt/ML — тестируется headless.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterator

import numpy as np


def make_groups(n: int, n_groups: int) -> list[tuple[int, int]]:
    """Границы ``n_groups`` смежных групп: список ``(start, end)`` (end эксклюзивен).

    Группы покрывают весь ряд ``[0, n)`` без пропусков и пересечений.
    """
    n_groups = max(2, min(int(n_groups), n))
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


@dataclass
class CombinatorialPurgedCV:
    """Генератор сочетаний (train_idx, test_idx, test_groups) с purge + embargo.

    Параметры
    ---------
    n_groups : число смежных групп ``N`` (≥ 3).
    test_groups : размер тест-комбинации ``k`` (1 ≤ k < N). Число путей = ``C(N-1, k-1)``.
    horizon : горизонт метки в барах (для purge на стыках тест-блоков).
    embargo : доп. зазор в барах вокруг тест-блоков (сверх горизонта).
    """

    n_groups: int = 6
    test_groups: int = 2
    horizon: int = 1
    embargo: int = 0

    def _k(self) -> int:
        return max(1, min(int(self.test_groups), max(1, int(self.n_groups) - 1)))

    def n_combos(self) -> int:
        """Сколько обучений модели потребуется (= ``C(N, k)``)."""
        return comb(max(2, int(self.n_groups)), self._k())

    def n_paths(self) -> int:
        """Сколько out-of-sample backtest-путей даст схема (= ``C(N-1, k-1)``)."""
        return comb(max(1, int(self.n_groups) - 1), self._k() - 1)

    def split(self, X) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
        """Выдать ``(train_idx, test_idx, test_groups)`` для каждого сочетания.

        ``train_idx`` очищен: вокруг каждого тест-блока вырезаны ``horizon+embargo``
        баров с обеих сторон (метки соседних баров пересекаются по времени).
        """
        n = len(X)
        groups = make_groups(n, self.n_groups)
        N = len(groups)
        k = max(1, min(self._k(), N - 1))
        gap = max(0, int(self.horizon)) + max(0, int(self.embargo))
        all_idx = np.arange(n)
        for combo in combinations(range(N), k):
            test_mask = np.zeros(n, dtype=bool)
            for g in combo:
                s, e = groups[g]
                test_mask[s:e] = True
            train_mask = ~test_mask
            # purge + embargo: вырезаем зазор по обе стороны каждого тест-блока.
            for g in combo:
                s, e = groups[g]
                train_mask[max(0, s - gap):min(n, e + gap)] = False
            train_idx = all_idx[train_mask]
            test_idx = all_idx[test_mask]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx, combo


def assemble_paths(
    preds_by_group: dict[int, list[np.ndarray]], n_groups: int
) -> list[np.ndarray]:
    """Собрать backtest-пути из предсказаний по группам.

    ``preds_by_group[g]`` — список массивов вероятностей для баров группы ``g`` (по одному
    на каждое сочетание, где ``g`` была тестовой), в порядке перебора сочетаний. Путь ``j``
    конкатенирует ``j``-е предсказание каждой группы по порядку групп — получается полный
    OOS-ряд. Число путей = минимум по группам числа их предсказаний.
    """
    if not preds_by_group:
        return []
    n_paths = min(len(preds_by_group.get(g, [])) for g in range(n_groups))
    paths: list[np.ndarray] = []
    for j in range(n_paths):
        parts = [preds_by_group[g][j] for g in range(n_groups)]
        paths.append(np.concatenate(parts))
    return paths
