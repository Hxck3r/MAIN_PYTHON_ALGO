"""Offscreen GUI smoke-тест (gui_smoke_test.main) внутри pytest.

Поднимает реальное окно в режиме QT_QPA_PLATFORM=offscreen и прогоняет
бэктест -> оптимизацию -> WFA -> матрицу -> Монте-Карло -> PBO -> ML,
проверяя, что панели заполняются. Пропускается, если PySide6 недоступен
(например, в окружении без Qt); в CI офскрин-плагин ставится отдельным шагом.
"""
from __future__ import annotations

import os

import pytest

# Офскрин до любого импорта Qt (на случай запуска вне CI, где переменная не задана).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def test_gui_smoke() -> None:
    import gui_smoke_test

    gui_smoke_test.main()
