"""Временная кросс-валидация для финансовых рядов: purged walk-forward.

Зачем это нужно
---------------
Обычный train/test сплит и тем более случайный ``KFold`` дают **завышенную** оценку
на временных рядах: модель «подсматривает» будущее. Здесь реализован честный подход
(López de Prado, *Advances in Financial Machine Learning*):

* **Walk-forward** — несколько последовательных out-of-sample фолдов; обучение всегда
  на прошлом, проверка на будущем (расширяющееся или скользящее окно).
* **Purging** — из train выбрасываются бары, чья метка (она смотрит на ``horizon``
  баров вперёд) пересекается с тестовым окном; иначе на стыке возникает утечка.
* **Embargo** — дополнительный зазор после/около теста: соседние во времени бары
  скоррелированы, поэтому ещё несколько баров исключаются.

Сплиттер работает с **позиционными** индексами (как ``TimeSeriesSplit``), не зависит
от Qt/ML-библиотек и тестируется headless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedWalkForwardSplit:
    """Генератор разбиений (train_idx, test_idx) с purge + embargo.

    Параметры
    ---------
    n_splits : число out-of-sample фолдов.
    horizon : горизонт метки в барах (на сколько вперёд смотрит target). Последние
        ``horizon`` баров train у границы теста удаляются (purge).
    embargo : доп. зазор в барах между train и test (сверх ``horizon``).
    mode : ``"expanding"`` — train растёт от начала; ``"sliding"`` — окно фиксированной длины.
    max_train_size : длина train-окна для ``sliding`` (по умолчанию = размеру первого блока).
    test_size : размер одного тестового фолда в барах (по умолчанию ``n // (n_splits + 1)``).
    """

    n_splits: int = 5
    horizon: int = 1
    embargo: int = 0
    mode: str = "expanding"
    max_train_size: int | None = None
    test_size: int | None = None

    def get_n_splits(self) -> int:
        return self.n_splits

    @property
    def _gap(self) -> int:
        # Метка бара i охватывает [i, i+horizon]; чтобы не пересечь test_start,
        # удаляем последние (horizon + embargo) баров train у границы.
        return max(0, self.horizon) + max(0, self.embargo)

    def split(self, X) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Выдать пары (train_idx, test_idx) позиционных индексов."""
        n = len(X)
        if self.n_splits < 1:
            raise ValueError("n_splits должно быть ≥ 1.")
        test_size = self.test_size or max(1, n // (self.n_splits + 1))
        gap = self._gap

        # Тестовые окна примыкают к концу ряда, идут последовательно.
        first_test_start = n - self.n_splits * test_size
        min_train = gap + 1
        if first_test_start - gap < min_train:
            raise ValueError(
                "Недостаточно данных для walk-forward с такими параметрами "
                f"(n={n}, n_splits={self.n_splits}, test_size={test_size}, gap={gap})."
            )

        indices = np.arange(n)
        for k in range(self.n_splits):
            test_start = first_test_start + k * test_size
            test_end = test_start + test_size if k < self.n_splits - 1 else n
            train_end = test_start - gap
            if train_end <= 0:
                continue
            if self.mode == "sliding":
                window = self.max_train_size or first_test_start - gap
                train_start = max(0, train_end - window)
            else:  # expanding
                train_start = 0
            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx
