"""Дашборд PBO (вероятность переподгодки) методом CSCV.

PBO отвечает на вопрос отбора, на который НЕ отвечает Deflated Sharpe:
«систематически ли конфигурация, лучшая на in-sample, проваливается на
out-of-sample?». Метод CSCV режет историю всех испытаний оптимизации на
блоки и перебирает все способы разбить их пополам на IS/OOS; PBO -- доля
разбиений, где IS-лучший оказался ниже OOS-медианы.

Слева -- настройки и сводка; справа:
    * гистограмма логитов относительного OOS-ранга (левее 0 -- переподгонка);
    * IS vs OOS Sharpe отобранной конфигурации по всем разбиениям.

Трактовка: PBO ~ 0 -- отбор переносится на новые данные; PBO >= 0.5 --
выбор лучшего по IS равносилен подбрасыванию монеты на OOS (переподгонка).
Запускается по последней оптимизации (вкладка «Оптимизация»).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from backtester.pbo import PBOResult

from .chart_widget import BG, COLOR_START_CASH, make_grid_item

COLOR_GOOD = "#16c784"
COLOR_BAD = "#ff4d4d"
COLOR_WARN = "#ff9f43"
HIST_BRUSH = (79, 156, 247, 160)
SCATTER_BRUSH = (79, 156, 247, 120)


def _pbo_color(pbo: float) -> str:
    """Цвет PBO: меньше -- лучше (0 робастно, >=0.5 переподгонка)."""
    if pbo < 0.3:
        return COLOR_GOOD
    if pbo < 0.5:
        return COLOR_WARN
    return COLOR_BAD


class PboSummaryTable(QTableWidget):
    """Двухколоночная сводка CSCV/PBO."""

    def __init__(self, parent=None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Метрика", "Значение"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

    def show_result(self, r: PBOResult) -> None:
        verdict = (
            "переподгонка: отбор по IS не переносится на OOS"
            if r.pbo >= 0.5
            else "отбор по IS переносится на OOS (приемлемо)"
        )
        median_logit = float(np.median(r.logits)) if len(r.logits) else float("nan")
        rows: list[tuple[str, str, Optional[str]]] = [
            ("PBO (вероятность переподгонки)", f"{r.pbo:.3f}", _pbo_color(r.pbo)),
            ("Вердикт", verdict, _pbo_color(r.pbo)),
            ("— Параметры CSCV —", "", None),
            ("Конфигураций (N)", str(r.n_trials), None),
            ("Блоков (S) / комбинаций",
             f"{r.n_splits} / {r.n_combinations:,}", None),
            ("— Диагностика —", "", None),
            ("P(убыток отобранного на OOS)", f"{r.prob_oos_loss:.3f}",
             COLOR_GOOD if r.prob_oos_loss < 0.3 else
             (COLOR_WARN if r.prob_oos_loss < 0.6 else COLOR_BAD)),
            ("Медиана логита (>0 — перенос)", f"{median_logit:+.3f}",
             COLOR_GOOD if median_logit > 0 else COLOR_BAD),
            ("Наклон деградации OOS~IS", f"{r.degradation_slope:+.3f}", None),
        ]
        self.setRowCount(len(rows))
        for row, (name, value, color) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if not value:  # строка-разделитель
                name_item.setForeground(QColor("#888888"))
            value_item = QTableWidgetItem(value)
            value_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if color:
                value_item.setForeground(QColor(color))
            self.setItem(row, 0, name_item)
            self.setItem(row, 1, value_item)
        self.resizeColumnsToContents()


class PboPanel(QWidget):
    """Вкладка «PBO»: настройки CSCV + сводка + графики."""

    #: запрос на расчёт; main_window запускает PboWorker с этим n_splits
    run_requested = Signal(int)      # n_splits

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: Optional[PBOResult] = None

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # --- левая колонка: настройки + сводка ----------------------------#
        config_box = QGroupBox("Настройки CSCV")
        form = QFormLayout(config_box)
        self.spin_splits = QSpinBox(minimum=4, maximum=20, value=16)
        self.spin_splits.setSingleStep(2)        # S должно быть чётным
        self.spin_splits.setToolTip(
            "S — число блоков, на которые режется история испытаний.\n"
            "Перебираются все C(S, S/2) разбиений на IS/OOS.\n"
            "Каноническое S=16 (12870 комбинаций). Больше S — точнее, но\n"
            "нужно больше наблюдений (в каждом блоке должно быть ≥2 бара)."
        )
        form.addRow("Блоков (S):", self.spin_splits)

        self.btn_run = QPushButton("🔬  Рассчитать PBO")
        self.btn_run.clicked.connect(
            lambda: self.run_requested.emit(self._even_splits())
        )

        self.summary = PboSummaryTable()

        left_widget = QWidget()
        left_col = QVBoxLayout(left_widget)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.addWidget(config_box)
        left_col.addWidget(self.btn_run)
        left_col.addWidget(self.summary, stretch=1)

        scroll = QScrollArea()
        scroll.setWidget(left_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setFrameShape(QScrollArea.NoFrame)
        root.addWidget(scroll)

        # --- правая часть: графики ----------------------------------------#
        right = QVBoxLayout()
        self.lbl_hint = QLabel(
            "Сначала запустите оптимизацию (≥2 различных конфигурации), "
            "затем — PBO. Левее нуля по логиту — область переподгонки."
        )
        self.lbl_hint.setStyleSheet("color: gray;")
        right.addWidget(self.lbl_hint)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(BG)
        right.addWidget(self.glw, stretch=1)
        root.addLayout(right, stretch=1)

        self.logit_hist = self.glw.addPlot(row=0, col=0)
        self.logit_hist.setTitle(
            "Логиты OOS-ранга IS-лучшего: λ≤0 — переподгонка"
        )
        self.scatter = self.glw.addPlot(row=1, col=0)
        self.scatter.setTitle("IS vs OOS Sharpe отобранной конфигурации")
        for plot in (self.logit_hist, self.scatter):
            plot.setMenuEnabled(False)
            plot.addItem(make_grid_item())

    # ------------------------------------------------------------------ #
    def _even_splits(self) -> int:
        """S должно быть чётным (нужно ровно S/2 блоков на IS)."""
        s = self.spin_splits.value()
        return s if s % 2 == 0 else s - 1

    def set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)

    # ------------------------------------------------------------------ #
    def show_result(self, result: PBOResult) -> None:
        self._result = result
        self.summary.show_result(result)
        for plot in (self.logit_hist, self.scatter):
            plot.clear()
            plot.addItem(make_grid_item())
        self._draw_logits(result)
        self._draw_scatter(result)
        self.lbl_hint.setText(
            f"PBO = {result.pbo:.3f} по {result.n_trials} конфигурациям, "
            f"S={result.n_splits} блоков, {result.n_combinations:,} разбиений."
        )

    def _draw_logits(self, r: PBOResult) -> None:
        logits = r.logits[np.isfinite(r.logits)]
        if len(logits):
            counts, edges = np.histogram(logits, bins=40)
            self.logit_hist.addItem(pg.BarGraphItem(
                x0=edges[:-1], x1=edges[1:], height=counts,
                brush=pg.mkBrush(*HIST_BRUSH), pen=pg.mkPen(BG),
            ))
        self.logit_hist.addItem(pg.InfiniteLine(
            0.0, angle=90,
            pen=pg.mkPen(COLOR_BAD, width=2, style=QtCore.Qt.PenStyle.DashLine),
            label="переподгонка ←", labelOpts={"color": COLOR_BAD, "position": 0.92},
        ))
        self.logit_hist.setLabel("bottom", "логит относительного OOS-ранга")
        self.logit_hist.setLabel("left", "разбиений")

    def _draw_scatter(self, r: PBOResult) -> None:
        self.scatter.addItem(pg.ScatterPlotItem(
            x=r.is_perf_selected, y=r.oos_perf_selected,
            size=5, brush=pg.mkBrush(*SCATTER_BRUSH), pen=None,
        ))
        self.scatter.addItem(pg.InfiniteLine(
            0.0, angle=0,
            pen=pg.mkPen(COLOR_START_CASH, style=QtCore.Qt.PenStyle.DashLine),
        ))
        self.scatter.setLabel("bottom", "IS Sharpe отобранного")
        self.scatter.setLabel("left", "OOS Sharpe отобранного")
