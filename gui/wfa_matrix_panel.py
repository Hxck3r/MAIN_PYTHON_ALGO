"""Дашборд матричного WFO (WFA Cluster Analysis).

Тепловая карта: строки -- длина IS-окна, столбцы -- длина OOS-окна,
цвет -- выбранная метрика (по умолчанию WFE). Ищем ПЛАТО -- широкую
светлую область соседних ячеек: если она есть, стратегия устойчива к
выбору длин окон. Одинокая яркая ячейка среди тёмных -- переподгонка под
конкретное нарезание истории.

Зелёным кружком отмечена рекомендованная ячейка (центр плато) -- максимум
среднего по окрестности, а не глобальный пик. Двойной клик по любой
ячейке подставляет её длины IS/OOS в настройки Walk-Forward слева.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from backtester import MATRIX_FIELDS, WfaMatrixResult

from .chart_widget import BG, make_grid_item

CMAP = pg.colormap.get("viridis")
NAN_RGBA = (38, 38, 38, 255)          # ячейки без прогона
COLOR_ROBUST = "#16c784"              # маркер рекомендованной ячейки
COLOR_GOOD = "#16c784"
COLOR_WARN = "#ff9f43"
COLOR_BAD = "#ff4d4d"


def _fmt(value: float, suffix: str = "") -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.1f}{suffix}"


class WfaMatrixSummary(QTableWidget):
    """Двухколоночная сводка устойчивости матрицы."""

    def __init__(self, parent=None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Показатель", "Значение"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setFixedWidth(300)

    def show_result(self, result: WfaMatrixResult) -> None:
        st = result.stability()
        robust = result.robust_cell()
        best = result.best_cell()
        rows: list[tuple[str, str, Optional[str]]] = [
            ("Ячеек прогнано", f"{st.n_ran} / {st.n_cells}", None),
            ("Средняя WFE по сетке", _fmt(st.avg_wfe_pct, " %"),
             COLOR_GOOD if math.isfinite(st.avg_wfe_pct) and st.avg_wfe_pct >= 50
             else COLOR_WARN),
            ("Прибыльных ячеек", _fmt(st.pct_profitable_cells, " %"),
             COLOR_GOOD if math.isfinite(st.pct_profitable_cells)
             and st.pct_profitable_cells >= 60 else COLOR_WARN),
            ("Разброс цели (CV)", _fmt(st.target_cv_pct, " %"),
             COLOR_GOOD if math.isfinite(st.target_cv_pct)
             and st.target_cv_pct <= 40 else COLOR_WARN),
            ("Ширина плато (соседи)", _fmt(st.plateau_pct, " %"),
             COLOR_GOOD if math.isfinite(st.plateau_pct)
             and st.plateau_pct >= 60 else COLOR_WARN),
        ]
        if robust is not None:
            rows.append(("— Рекомендация —", "", None))
            rows.append((
                "Центр плато (IS / OOS)",
                f"{robust.is_days} / {robust.oos_days} дн", COLOR_ROBUST,
            ))
            rows.append(("  WFE", _fmt(robust.wfe_pct, " %"), None))
            rows.append(("  OOS доходность", _fmt(robust.oos_return_pct, " %"), None))
            rows.append(("  OOS Sharpe", _fmt(robust.oos_sharpe), None))
            rows.append(("  Окон", str(robust.n_windows), None))
        if best is not None and robust is not None and (
            (best.is_days, best.oos_days) != (robust.is_days, robust.oos_days)
        ):
            rows.append((
                "Глобальный пик (IS / OOS)",
                f"{best.is_days} / {best.oos_days} дн", COLOR_WARN,
            ))

        self.setRowCount(len(rows))
        for r, (name, value, color) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            value_item = QTableWidgetItem(value)
            value_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if color:
                value_item.setForeground(QColor(color))
            self.setItem(r, 0, name_item)
            self.setItem(r, 1, value_item)
        self.resizeColumnsToContents()


class WfaMatrixPanel(QWidget):
    """Тепловая карта матричного WFO + сводка устойчивости."""

    #: двойной клик по ячейке -> (is_days, oos_days) для подстановки в WFA
    cell_activated = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: Optional[WfaMatrixResult] = None
        # (центры X=OOS, центры Y=IS, матрица значений) -- для ховера/клика
        self._hover_info: Optional[tuple] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.summary = WfaMatrixSummary()
        layout.addWidget(self.summary)

        right = QVBoxLayout()
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Метрика:"))
        self.combo_field = QComboBox()
        for name, (label, _) in MATRIX_FIELDS.items():
            self.combo_field.addItem(label, name)
        self.combo_field.currentIndexChanged.connect(self._redraw)
        controls.addWidget(self.combo_field)
        self.lbl_hover = QLabel("")
        self.lbl_hover.setStyleSheet("color: gray;")
        controls.addSpacing(12)
        controls.addWidget(self.lbl_hover)
        controls.addStretch(1)
        self.lbl_hint = QLabel(
            "Светлое плато соседних ячеек — стратегия устойчива к длинам окон; "
            "одинокий пик — переподгонка.  Двойной клик — применить IS/OOS.  "
            "⬤ — рекомендация (центр плато)."
        )
        self.lbl_hint.setStyleSheet("color: gray;")
        self.lbl_hint.setWordWrap(True)
        right.addLayout(controls)
        right.addWidget(self.lbl_hint)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(BG)
        right.addWidget(self.glw)
        self.heat_plot = self.glw.addPlot(row=0, col=0)
        self.heat_plot.setMenuEnabled(False)
        self.heat_plot.setTitle("Матрица WFO: цвет — метрика по (IS × OOS)")
        self.heat_plot.setLabel("bottom", "OOS-окно, дн")
        self.heat_plot.setLabel("left", "IS-окно, дн")
        self.colorbar = pg.ColorBarItem(colorMap=CMAP, interactive=False, width=14)
        self.glw.addItem(self.colorbar, row=0, col=1)
        layout.addLayout(right, stretch=1)

        self.glw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.glw.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    # ------------------------------------------------------------------ #
    def show_result(self, result: WfaMatrixResult) -> None:
        self._result = result
        self.summary.show_result(result)
        # выбрать в комбобоксе целевую метрику расчёта
        idx = self.combo_field.findData(result.target)
        self.combo_field.blockSignals(True)
        self.combo_field.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_field.blockSignals(False)
        self._redraw()

    def _current_field(self) -> str:
        return self.combo_field.currentData() or "wfe_pct"

    def _redraw(self) -> None:
        self.heat_plot.clear()
        self._hover_info = None
        self.lbl_hover.setText("")
        for axis in ("bottom", "left"):
            self.heat_plot.getAxis(axis).setTicks(None)
        if self._result is None:
            return

        field = self._current_field()
        mat = self._result.matrix(field)              # строки=IS, столбцы=OOS
        if mat.empty:
            return
        y_centers = np.asarray(mat.index, dtype=float)   # IS
        x_centers = np.asarray(mat.columns, dtype=float)  # OOS
        # ImageItem col-major ждёт data[x, y] -> транспонируем (IS×OOS -> OOS×IS)
        matrix_xy = mat.to_numpy(dtype=float).T

        finite = matrix_xy[np.isfinite(matrix_xy)]
        if not len(finite):
            self.heat_plot.addItem(make_grid_item())
            return
        vmin, vmax = float(finite.min()), float(finite.max())
        norm = (
            (matrix_xy - vmin) / (vmax - vmin) if vmax > vmin
            else np.zeros_like(matrix_xy)
        )
        rgba = CMAP.map(np.nan_to_num(norm), mode="byte")
        rgba[np.isnan(matrix_xy)] = NAN_RGBA

        image = pg.ImageItem(axisOrder="col-major")
        image.setImage(rgba)
        image.setRect(QtCore.QRectF(0, 0, len(x_centers), len(y_centers)))
        self.heat_plot.addItem(image)
        self._apply_ticks("bottom", x_centers)
        self._apply_ticks("left", y_centers)
        self.colorbar.setVisible(True)
        self.colorbar.setLevels((vmin, vmax))

        label = MATRIX_FIELDS.get(field, (field, True))[0]
        self.heat_plot.setTitle(f"Матрица WFO — {label}")
        self._hover_info = (x_centers, y_centers, matrix_xy, label)

        # маркер рекомендованной ячейки (центр плато)
        robust = self._result.robust_cell()
        if robust is not None:
            if robust.oos_days in mat.columns and robust.is_days in mat.index:
                ix = list(mat.columns).index(robust.oos_days)
                iy = list(mat.index).index(robust.is_days)
                self.heat_plot.addItem(pg.ScatterPlotItem(
                    [ix + 0.5], [iy + 0.5], size=16, symbol="o",
                    brush=pg.mkBrush(None), pen=pg.mkPen(COLOR_ROBUST, width=3),
                ))

    def _apply_ticks(self, axis_name: str, centers: np.ndarray) -> None:
        step = max(1, math.ceil(len(centers) / 10))
        ticks = [
            (i + 0.5, f"{int(round(v))}") for i, v in enumerate(centers)
        ][::step]
        self.heat_plot.getAxis(axis_name).setTicks([ticks])

    # --- ховер: показать значение ячейки ----------------------------------- #
    def _cell_at(self, scene_pos) -> Optional[tuple[int, int]]:
        """Индексы (ix=OOS, iy=IS) ячейки под курсором сцены или None."""
        if self._hover_info is None:
            return None
        if not self.heat_plot.sceneBoundingRect().contains(scene_pos):
            return None
        pt = self.heat_plot.vb.mapSceneToView(scene_pos)
        x_centers, y_centers, matrix, _ = self._hover_info
        ix, iy = int(math.floor(pt.x())), int(math.floor(pt.y()))
        if 0 <= ix < len(x_centers) and 0 <= iy < len(y_centers):
            return ix, iy
        return None

    def _on_mouse_moved(self, scene_pos) -> None:
        cell = self._cell_at(scene_pos)
        if cell is None:
            self.lbl_hover.setText("")
            return
        ix, iy = cell
        x_centers, y_centers, matrix, label = self._hover_info
        value = matrix[ix, iy]
        oos, is_ = int(round(x_centers[ix])), int(round(y_centers[iy]))
        if math.isfinite(value):
            self.lbl_hover.setText(
                f"IS={is_}д, OOS={oos}д:  {label} = {value:.2f}"
            )
        else:
            self.lbl_hover.setText(f"IS={is_}д, OOS={oos}д:  не прогналась")

    def _on_mouse_clicked(self, event) -> None:
        if not event.double():
            return
        cell = self._cell_at(event.scenePos())
        if cell is None:
            return
        ix, iy = cell
        x_centers, y_centers, _, _ = self._hover_info
        self.cell_activated.emit(int(round(y_centers[iy])), int(round(x_centers[ix])))
