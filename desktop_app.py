"""
Десктопный «Тестер стратегий» (PySide6 + finplot/pyqtgraph).

Запуск из папки проекта:
    python desktop_app.py
"""
from __future__ import annotations

import os
import sys

# Жёстко выбираем PySide6 как Qt-биндинг для pyqtgraph/finplot,
# чтобы не было конфликта, если в системе установлен ещё и PyQt6.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AlgoTester")
    window = MainWindow()
    window.show()           
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
