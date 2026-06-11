"""Графики результатов оптимизации: тепловая карта и распределение score.

* Тепловая карта -- ландшафт целевой функции по двум выбранным параметрам.
  Широкая ровная светлая область -- плато (устойчивые параметры),
  одиночная яркая ячейка среди тёмных -- нестабильный пик (переподгонка).
  Остальные параметры свёртываются агрегатом (среднее/максимум):
  среднее подчёркивает плато, максимум показывает лучший достижимый score.
* Гистограмма распределения score показывает, какую долю пространства
  параметров занимают хорошие результаты: если лучший score (зелёный
  пунктир) висит в длинном пустом правом хвосте -- это единичная удача,
  а не системное преимущество.

Особенности реализации:
    * score = +-inf (прогоны, не прошедшие фильтр мин. сделок) исключаются;
    * оси с числом значений больше MAX_CELLS биннингуются равными
      интервалами, иначе каждая ячейка -- конкретное значение параметра;
    * RGBA собирается вручную (ColorMap.map), чтобы ячейки без данных
      рисовались нейтральным серым, а не цветом минимума;
    * при единственном оптимизируемом параметре (или X == Y) вместо
      теплокарты рисуется кривая score-от-параметра.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from backtester import OptimizationResult

from .chart_widget import BG, COLOR_EQUITY, make_grid_item

MAX_CELLS = 25                       # максимум ячеек на ось теплокарты
NAN_RGBA = (38, 38, 38, 255)         # ячейки без испытаний
COLOR_BEST = "#16c784"
CMAP = pg.colormap.get("viridis")


def _fmt(value: float) -> str:
    return f"{value:g}"


class OptimizationChartsPanel(QWidget):
    """Тепловая карта (слева) + гистограмма распределения score (справа)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None
        self._best_score: float = math.nan
        # (имя X, имя Y, центры X, центры Y, матрица score) -- для ховера
        self._hover_info: Optional[tuple] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        controls = QHBoxLayout()
        self.combo_x = QComboBox()
        self.combo_y = QComboBox()
        self.combo_agg = QComboBox()
        self.combo_agg.addItems(["Среднее (плато)", "Максимум (пики)"])
        for combo in (self.combo_x, self.combo_y, self.combo_agg):
            combo.currentIndexChanged.connect(self._redraw)
        self.lbl_hover = QLabel("")
        self.lbl_hover.setStyleSheet("color: gray;")
        controls.addWidget(QLabel("X:"))
        controls.addWidget(self.combo_x)
        controls.addWidget(QLabel("Y:"))
        controls.addWidget(self.combo_y)
        controls.addWidget(QLabel("Свёртка:"))
        controls.addWidget(self.combo_agg)
        controls.addSpacing(16)
        controls.addWidget(self.lbl_hover)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(BG)
        layout.addWidget(self.glw)

        self.heat_plot = self.glw.addPlot(row=0, col=0)
        self.heat_plot.setMenuEnabled(False)
        self.colorbar = pg.ColorBarItem(colorMap=CMAP, interactive=False, width=14)
        self.glw.addItem(self.colorbar, row=0, col=1)
        self.hist_plot = self.glw.addPlot(row=0, col=2)
        self.hist_plot.setTitle("Распределение score")
        self.hist_plot.setLabel("bottom", "score")
        self.hist_plot.setLabel("left", "испытаний")
        self.glw.ci.layout.setColumnStretchFactor(0, 3)
        self.glw.ci.layout.setColumnStretchFactor(2, 2)

        # сцена общая на все панели layout-а -- фильтруем по heat_plot в хендлере
        self.glw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._clear_plots()

    # ------------------------------------------------------------------ #
    def show_result(self, opt_result: OptimizationResult) -> None:
        """Принимает итог оптимизации и перерисовывает оба графика."""
        self._best_score = opt_result.best_score
        df = opt_result.history_df()
        if not df.empty:
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
        if df.empty:
            self._df = None
            self._clear_plots()
            self.lbl_hover.setText("Нет валидных испытаний")
            return
        self._df = df

        param_cols = [c for c in df.columns if c != "score"]
        varied = [c for c in param_cols if df[c].nunique() > 1] or param_cols
        for combo in (self.combo_x, self.combo_y):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(varied)
        self.combo_x.setCurrentIndex(0)
        self.combo_y.setCurrentIndex(1 if len(varied) > 1 else 0)
        for combo in (self.combo_x, self.combo_y):
            combo.blockSignals(False)
        self._redraw()

    # ------------------------------------------------------------------ #
    def _clear_plots(self) -> None:
        self._hover_info = None
        self.lbl_hover.setText("")
        self.heat_plot.clear()
        self.hist_plot.clear()
        self.hist_plot.addItem(make_grid_item())
        for axis in ("bottom", "left"):
            self.heat_plot.getAxis(axis).setTicks(None)

    def _redraw(self) -> None:
        self._clear_plots()
        if self._df is None or self._df.empty:
            return
        self._draw_histogram()
        x_name = self.combo_x.currentText()
        y_name = self.combo_y.currentText()
        if not x_name:
            return
        if x_name == y_name:
            self._draw_curve(x_name)
        else:
            self._draw_heatmap(x_name, y_name)

    # --- тепловая карта -------------------------------------------------#
    @staticmethod
    def _bin(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Индекс ячейки для каждого испытания + значения центров ячеек."""
        values = series.to_numpy(dtype=float)
        unique = np.unique(values)
        if len(unique) <= MAX_CELLS:
            return np.searchsorted(unique, values), unique
        edges = np.linspace(values.min(), values.max(), MAX_CELLS + 1)
        idx = np.clip(np.digitize(values, edges) - 1, 0, MAX_CELLS - 1)
        centers = (edges[:-1] + edges[1:]) / 2
        return idx, centers

    def _draw_heatmap(self, x_name: str, y_name: str) -> None:
        df = self._df
        ix, x_centers = self._bin(df[x_name])
        iy, y_centers = self._bin(df[y_name])
        agg = "mean" if self.combo_agg.currentIndex() == 0 else "max"
        grouped = (
            pd.DataFrame({"ix": ix, "iy": iy, "score": df["score"].to_numpy()})
            .groupby(["ix", "iy"])["score"]
            .agg(agg)
        )
        matrix = np.full((len(x_centers), len(y_centers)), np.nan)
        matrix[
            grouped.index.get_level_values(0), grouped.index.get_level_values(1)
        ] = grouped.to_numpy()

        vmin = float(np.nanmin(matrix))
        vmax = float(np.nanmax(matrix))
        if vmax > vmin:
            norm = (matrix - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(matrix)
        rgba = CMAP.map(np.nan_to_num(norm), mode="byte")
        rgba[np.isnan(matrix)] = NAN_RGBA

        image = pg.ImageItem(axisOrder="col-major")  # data[x, y]
        image.setImage(rgba)
        image.setRect(QtCore.QRectF(0, 0, len(x_centers), len(y_centers)))
        self.heat_plot.addItem(image)
        self.heat_plot.setTitle("Ландшафт оптимизации")
        self.heat_plot.setLabel("bottom", x_name)
        self.heat_plot.setLabel("left", y_name)
        self._apply_ticks("bottom", x_centers)
        self._apply_ticks("left", y_centers)
        self.colorbar.setVisible(True)
        self.colorbar.setLevels((vmin, vmax))
        self._hover_info = (x_name, y_name, x_centers, y_centers, matrix)

    def _apply_ticks(self, axis_name: str, centers: np.ndarray) -> None:
        """Подписи осей -- значения параметров по центрам ячеек (не все подряд)."""
        step = max(1, math.ceil(len(centers) / 8))
        ticks = [
            (i + 0.5, _fmt(value)) for i, value in enumerate(centers)
        ][::step]
        self.heat_plot.getAxis(axis_name).setTicks([ticks])

    # --- кривая для одного параметра -------------------------------------#
    def _draw_curve(self, x_name: str) -> None:
        agg = "mean" if self.combo_agg.currentIndex() == 0 else "max"
        series = self._df.groupby(x_name)["score"].agg(agg).sort_index()
        self.heat_plot.addItem(make_grid_item())
        self.heat_plot.plot(
            series.index.to_numpy(dtype=float),
            series.to_numpy(),
            pen=pg.mkPen(COLOR_EQUITY, width=2),
            symbol="o",
            symbolSize=6,
            symbolBrush=COLOR_EQUITY,
            symbolPen=None,
        )
        self.heat_plot.setTitle("Score по параметру")
        self.heat_plot.setLabel("bottom", x_name)
        self.heat_plot.setLabel("left", "score")
        self.colorbar.setVisible(False)

    # --- гистограмма ------------------------------------------------------#
    def _draw_histogram(self) -> None:
        scores = self._df["score"].to_numpy()
        n_bins = min(40, max(10, int(round(math.sqrt(len(scores)) * 2))))
        hist, edges = np.histogram(scores, bins=n_bins)
        self.hist_plot.plot(
            edges,
            hist,
            stepMode="center",
            fillLevel=0,
            brush=(79, 156, 247, 80),
            pen=pg.mkPen(COLOR_EQUITY),
        )
        if math.isfinite(self._best_score):
            self.hist_plot.addItem(
                pg.InfiniteLine(
                    self._best_score,
                    angle=90,
                    pen=pg.mkPen(
                        COLOR_BEST, style=QtCore.Qt.PenStyle.DashLine, width=1.5
                    ),
                    label="лучший",
                    labelOpts={"color": COLOR_BEST, "position": 0.92},
                )
            )

    # --- ховер по теплокарте ----------------------------------------------#
    def _on_mouse_moved(self, scene_pos) -> None:
        if self._hover_info is None:
            return
        if not self.heat_plot.sceneBoundingRect().contains(scene_pos):
            return
        x_name, y_name, x_centers, y_centers, matrix = self._hover_info
        point = self.heat_plot.vb.mapSceneToView(scene_pos)
        ix, iy = int(point.x()), int(point.y())
        if not (0 <= ix < len(x_centers) and 0 <= iy < len(y_centers)):
            return
        value = matrix[ix, iy]
        score_text = "нет данных" if math.isnan(value) else f"score {value:.4f}"
        self.lbl_hover.setText(
            f"{x_name} = {_fmt(x_centers[ix])},  "
            f"{y_name} = {_fmt(y_centers[iy])}  →  {score_text}"
        )
