"""Диалог «Реестр прогонов»: сравнение сохранённых ML-моделей и гигиена отбора.

Показывает все сохранённые прогоны (из лёгких ``.json``-сводок, без загрузки моделей),
отсортированные по net OOS-Sharpe, и предупреждает о селекционном смещении: сколько
прогонов сделано на ОДНИХ данных и насколько широк разброс их Sharpe (чем больше — тем
сильнее «лучший» завышен отбором; доверять стоит DSR/CPCV, а не максимуму). Двойной клик
по строке загружает соответствующий артефакт в панель.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ml.registry import RunRow, scan_runs, selection_bias_by_data

COLOR_GOOD = "#16c784"
COLOR_BAD = "#ff4d4d"
COLOR_WARN = "#ff9f43"

_COLS = ["Дата", "Модель", "Валидация", "AUC", "OOS-Sharpe (net)", "PSR", "DSR",
         "CPCV % путей>0", "Издержки", "Период данных", "Прогонов на данных"]


def _num(v: float, fmt: str = "{:.3f}", dash: str = "—") -> str:
    return fmt.format(v) if isinstance(v, (int, float)) and math.isfinite(v) else dash


class RunRegistryDialog(QDialog):
    """Таблица сохранённых прогонов с предупреждением о смещении отбора."""

    #: путь к выбранному .joblib для загрузки в панель
    run_chosen = Signal(str)

    def __init__(self, models_dir: str, parent=None) -> None:
        super().__init__(parent)
        self._models_dir = models_dir
        self.setWindowTitle("📊 Реестр прогонов ML")
        self.resize(1040, 560)

        root = QVBoxLayout(self)
        self.lbl_info = QLabel()
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("padding: 4px 6px;")
        root.addWidget(self.lbl_info)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        root.addWidget(self.table, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 Обновить")
        self.btn_refresh.clicked.connect(self.reload)
        self.btn_load = QPushButton("📂 Загрузить выбранную")
        self.btn_load.clicked.connect(self._load_selected)
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_load)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

        self._rows: list[RunRow] = []
        self.reload()

    # ------------------------------------------------------------------ #
    def reload(self) -> None:
        """Перечитать сводки прогонов с диска и заполнить таблицу."""
        self._rows = scan_runs(self._models_dir)
        bias = selection_bias_by_data(self._rows)
        self._fill_info(bias)

        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._rows))
        for i, r in enumerate(self._rows):
            n_runs = bias[r.data_sha1].n_runs if r.data_sha1 in bias else 1
            cells = [
                r.created_at, r.model_name, r.validation,
                _num(r.auc), _num(r.oos_sharpe), _num(r.psr), _num(r.dsr),
                _num(r.cpcv_prob_positive * 100.0, "{:.0f}%"),
                f"{r.cost_bps:.1f} б.п." if math.isfinite(r.cost_bps) and r.cost_bps else "—",
                r.data_period, str(n_runs),
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem()
                # Числовые колонки сортируем как числа (а не как строки).
                if c in (3, 4, 5, 6):
                    val = (r.auc, r.oos_sharpe, r.psr, r.dsr)[c - 3]
                    item.setData(Qt.DisplayRole, round(val, 4) if math.isfinite(val) else -9.99)
                else:
                    item.setData(Qt.DisplayRole, text)
                color = self._cell_color(c, r)
                if color:
                    item.setForeground(QColor(color))
                self.table.setItem(i, c, item)
            # Путь к артефакту прячем в первой ячейке для загрузки по клику.
            self.table.item(i, 0).setData(Qt.UserRole, str(r.path))
        self.table.setSortingEnabled(True)
        self.btn_load.setEnabled(bool(self._rows))

    @staticmethod
    def _cell_color(col: int, r: RunRow) -> str | None:
        if col == 4 and math.isfinite(r.oos_sharpe):          # OOS-Sharpe
            return COLOR_GOOD if r.oos_sharpe > 0 else COLOR_BAD
        if col == 6 and math.isfinite(r.dsr):                 # DSR
            return COLOR_GOOD if r.dsr >= 0.95 else COLOR_WARN
        if col == 7 and math.isfinite(r.cpcv_prob_positive):  # CPCV
            return COLOR_GOOD if r.cpcv_prob_positive >= 0.95 else (
                COLOR_WARN if r.cpcv_prob_positive >= 0.6 else COLOR_BAD)
        return None

    def _fill_info(self, bias: dict) -> None:
        if not self._rows:
            self.lbl_info.setText(
                "Сохранённых прогонов нет. Обучите модель и нажмите «Сохранить».")
            self.lbl_info.setStyleSheet("padding: 4px 6px; color: gray;")
            return
        # Берём самую «перебранную» группу данных как главное предупреждение.
        worst = max(bias.values(), key=lambda g: g.n_runs)
        if worst.n_runs > 1:
            self.lbl_info.setText(
                f"Всего прогонов: {len(self._rows)}. На одних данных ({worst.data_period}) "
                f"уже {worst.n_runs} прогонов, разброс OOS-Sharpe σ={worst.sharpe_std:.3f}. "
                f"⚠ «Лучший» (Sharpe {worst.best_sharpe:.3f}) завышен отбором по {worst.n_runs} "
                "попыткам — доверяйте DSR/CPCV и подтверждению на новых данных, а не максимуму."
            )
            self.lbl_info.setStyleSheet(f"padding: 4px 6px; color: {COLOR_WARN}; font-weight: bold;")
        else:
            self.lbl_info.setText(
                f"Всего прогонов: {len(self._rows)}. Двойной клик — загрузить прогон в панель.")
            self.lbl_info.setStyleSheet("padding: 4px 6px; color: #c8c8c8;")

    # ------------------------------------------------------------------ #
    def _on_double_click(self, row: int, _col: int) -> None:
        self._emit_row(row)

    def _load_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self._emit_row(rows[0].row())

    def _emit_row(self, row: int) -> None:
        item = self.table.item(row, 0)
        path = item.data(Qt.UserRole) if item else None
        if path:
            self.run_chosen.emit(str(path))
            self.accept()
