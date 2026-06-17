"""Прогон ручного sanity-набора (sanity_check.main) внутри pytest.

sanity_check.py исторически запускался руками и проверяет базовые инварианты
движка (сходимость баланса, отсутствие сделок у пустой стратегии, влияние
издержек, уровни SL/TP, воспроизводимость, риск-сайзинг). Здесь он заведён в
автопрогон/CI: main() возвращает 0 при успехе и 1 при провале любого инварианта.
"""
from __future__ import annotations

import sanity_check


def test_sanity_invariants_pass() -> None:
    assert sanity_check.main() == 0
