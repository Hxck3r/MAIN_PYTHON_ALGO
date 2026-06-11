"""Дашборд Walk-Forward Analysis.

Две вкладки:
    * «Дашборд» -- сводные метрики устойчивости (WFE и др.) + графики:
        - сшитая OOS-кривая эквити с подсветкой окон;
        - таймлайн окон (IS/OOS участки на оси времени);
        - IS vs OOS доходность по окнам (в % годовых -- общая шкала
          для участков разной длины);
        - эволюция лучших параметров от окна к окну;
        - Walk-Forward Efficiency каждого окна.
    * «Окна» -- таблица всех окон; двойной клик по строке подставляет
      параметры окна в таблицу «Параметры» (сигнал row_activated).

Трактовка WFE: ~100% -- OOS работает как IS; 50-70% -- умеренная
деградация; <=0 -- переподгонка.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from backtester import WalkForwardResult

from .chart_widget import BG, COLOR_EQUITY, COLOR_START_CASH, make_grid_item
from .results_panel import _num_item

COLOR_IS = "#4f9cf7"          # in-sample (синий)
COLOR_OOS = "#ff9f43"         # out-of-sample (оранжевый)
COLOR_GOOD = "#16c784"
COLOR_BAD = "#ff4d4d"
COLOR_WARN = "#ff9f43"
REGION_BRUSHES = ((79, 156, 247, 18), (255, 159, 67, 14))  # подсветка окон


def _ts_to_x(index: pd.DatetimeIndex) -> np.ndarray:
    """DatetimeIndex -> секунды эпохи (ось X для DateAxisItem).

    Не использует asi8: в pandas >= 2.0 индекс может иметь разрешение
    s/ms/us (а не только ns), и деление на фиксированную константу
    «уносило» кривую в 1970 год. Деление timedelta на Timedelta
    корректно при любом разрешении.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return ((idx - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy()


def _annualized(return_pct: float, days: float) -> float:
    """Доходность участка -> % годовых (общая шкала IS и OOS)."""
    return return_pct / days * 365.0 if days > 0 else math.nan


def _wfe_color(wfe: float) -> str:
    if not math.isfinite(wfe):
        return "gray"
    if wfe >= 60:
        return COLOR_GOOD
    if wfe >= 30:
        return COLOR_WARN
    return COLOR_BAD


class WfaSummaryTable(QTableWidget):
    """Двухколоночная сводка метрик устойчивости WFA."""

    def __init__(self, parent=None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Метрика", "Значение"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setFixedWidth(300)

    def show_result(self, result: WalkForwardResult) -> None:
        stats = result.oos_stats
        wfe_text = (
            f"{result.wfe_pct:.1f} %" if math.isfinite(result.wfe_pct) else "n/a"
        )
        rows: list[tuple[str, str, Optional[str]]] = [
            ("Walk-Forward Efficiency", wfe_text, _wfe_color(result.wfe_pct)),
            ("Окон (валидных / всего)",
             f"{sum(1 for w in result.windows if w.valid)} / {result.n_windows}",
             None),
            ("Окон пропущено (мало баров)", str(result.skipped_windows), None),
            ("Прибыльных окон",
             f"{result.pct_profitable_windows:.0f} %",
             COLOR_GOOD if result.pct_profitable_windows >= 60 else COLOR_WARN),
            ("Разброс OOS-доходностей (std)",
             f"{result.consistency_std:.2f} %", None),
            ("OOS доходность (сшитая)", f"{stats.total_return_pct:.2f} %", None),
            ("OOS CAGR", f"{stats.cagr_pct:.2f} %", None),
            ("OOS Sharpe", f"{stats.sharpe_ratio:.2f}", None),
            ("OOS Sortino", f"{stats.sortino_ratio:.2f}", None),
            ("OOS макс. просадка", f"{stats.max_drawdown_pct:.2f} %", None),
            ("OOS профит-фактор", f"{stats.profit_factor:.2f}", None),
            ("OOS сделок всего", str(stats.total_trades), None),
        ]
        for name, cv in result.param_stability.items():
            cv_text = f"{cv:.1f} %" if math.isfinite(cv) else "n/a"
            color = COLOR_GOOD if math.isfinite(cv) and cv <= 30 else COLOR_WARN
            rows.append((f"Стабильность «{name}» (CV)", cv_text, color))

        self.setRowCount(len(rows))
        for row, (name, value, color) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            value_item = QTableWidgetItem(value)
            value_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if color:
                value_item.setForeground(QColor(color))
            self.setItem(row, 0, name_item)
            self.setItem(row, 1, value_item)
        self.resizeColumnsToContents()


class WfaWindowsTable(QTableWidget):
    """Таблица окон WFA; двойной клик -> сигнал с параметрами окна."""

    row_activated = Signal(dict)

    BASE_COLUMNS = [
        "№", "IS с", "IS по", "OOS с", "OOS по",
        "IS score", "IS дох. %", "OOS дох. %",
        "OOS Sharpe", "OOS сделок", "WFE окна %",
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(0, 0, parent)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSortingEnabled(True)
        self.itemDoubleClicked.connect(self._on_double_click)

    def show_result(self, result: WalkForwardResult) -> None:
        param_names = sorted({
            name for w in result.windows for name in w.best_params
        })
        columns = self.BASE_COLUMNS + param_names

        self.setSortingEnabled(False)
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        self.setRowCount(len(result.windows))
        for row, w in enumerate(result.windows):
            first = _num_item(w.index)
            # параметры окна храним в самой строке -- двойной клик
            # работает корректно и после сортировки таблицы
            first.setData(Qt.UserRole, dict(w.best_params))
            cells = [
                first,
                self._date_item(w.is_start), self._date_item(w.is_end),
                self._date_item(w.oos_start), self._date_item(w.oos_end),
                _num_item(w.is_score),
                _num_item(w.is_stats.total_return_pct),
                _num_item(w.oos_stats.total_return_pct),
                _num_item(w.oos_stats.sharpe_ratio),
                _num_item(w.oos_stats.total_trades),
                _num_item(w.efficiency),
            ]
            for name in param_names:
                cells.append(_num_item(w.best_params.get(name, math.nan)))
            oos_ret = w.oos_stats.total_return_pct
            color = QColor(COLOR_GOOD if oos_ret > 0 else COLOR_BAD)
            cells[7].setForeground(color)
            if not w.valid:
                for cell in cells:
                    cell.setForeground(QColor("gray"))
            for col, cell in enumerate(cells):
                self.setItem(row, col, cell)
        self.setSortingEnabled(True)
        self.resizeColumnsToContents()

    @staticmethod
    def _date_item(ts: pd.Timestamp) -> QTableWidgetItem:
        item = QTableWidgetItem(f"{ts:%Y-%m-%d}")
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        return item

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        first = self.item(item.row(), 0)
        if first is None:
            return
        params = first.data(Qt.UserRole)
        if params:
            self.row_activated.emit(params)


class WfaPanel(QTabWidget):
    """Вкладка Walk-Forward: дашборд с графиками + таблица окон."""

    #: двойной клик по окну -> применить его параметры (пробрасывается наружу)
    params_activated = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: Optional[WalkForwardResult] = None

        # --- вкладка «Дашборд» ------------------------------------------- #
        dashboard = QWidget()
        dash_layout = QHBoxLayout(dashboard)
        dash_layout.setContentsMargins(4, 4, 4, 4)

        self.summary = WfaSummaryTable()
        dash_layout.addWidget(self.summary)

        right = QVBoxLayout()
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Параметр:"))
        self.combo_param = QComboBox()
        self.combo_param.currentIndexChanged.connect(self._redraw_param_plot)
        controls.addWidget(self.combo_param)
        self.lbl_hint = QLabel(
            "WFE ≈ 100% — OOS работает как IS;  50–70% — умеренная "
            "деградация;  ≤ 0 — переподгонка"
        )
        self.lbl_hint.setStyleSheet("color: gray;")
        controls.addSpacing(16)
        controls.addWidget(self.lbl_hint)
        controls.addStretch(1)
        right.addLayout(controls)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(BG)
        right.addWidget(self.glw)
        dash_layout.addLayout(right, stretch=1)

        # Сшитая эквити -- на всю ширину, остальные графики 2x2 под ней
        self.equity_plot = self.glw.addPlot(
            row=0, col=0, colspan=2, axisItems={"bottom": pg.DateAxisItem()}
        )
        self.equity_plot.setTitle("Сшитая OOS-эквити (компаундинг по окнам)")
        self.timeline_plot = self.glw.addPlot(
            row=1, col=0, axisItems={"bottom": pg.DateAxisItem()}
        )
        self.timeline_plot.setTitle("Окна: IS (синий) → OOS (оранжевый)")
        self.bars_plot = self.glw.addPlot(row=1, col=1)
        self.bars_plot.setTitle("Доходность, % годовых: IS (синий) vs OOS (оранжевый)")
        self.param_plot = self.glw.addPlot(row=2, col=0)
        self.param_plot.setTitle("Лучший параметр по окнам")
        self.eff_plot = self.glw.addPlot(row=2, col=1)
        self.eff_plot.setTitle("WFE окна, %")
        for plot in self._plots():
            plot.setMenuEnabled(False)
            plot.addItem(make_grid_item())
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 2)
        self.glw.ci.layout.setRowStretchFactor(2, 2)

        self.addTab(dashboard, "Дашборд")

        # --- вкладка «Окна» ------------------------------------------------#
        windows_widget = QWidget()
        win_layout = QVBoxLayout(windows_widget)
        win_layout.setContentsMargins(4, 4, 4, 4)
        hint = QLabel(
            "Двойной клик по строке — подставить параметры окна в таблицу "
            "«Параметры»."
        )
        hint.setStyleSheet("color: gray;")
        self.windows_table = WfaWindowsTable()
        self.windows_table.row_activated.connect(self.params_activated)
        win_layout.addWidget(hint)
        win_layout.addWidget(self.windows_table)
        self.addTab(windows_widget, "Окна")

    # ------------------------------------------------------------------ #
    def _plots(self) -> list[pg.PlotItem]:
        return [
            self.equity_plot, self.timeline_plot, self.bars_plot,
            self.param_plot, self.eff_plot,
        ]

    def show_result(self, result: WalkForwardResult) -> None:
        self._result = result
        self.summary.show_result(result)
        self.windows_table.show_result(result)

        for plot in self._plots():
            plot.clear()
            plot.addItem(make_grid_item())

        self._draw_equity(result)
        self._draw_timeline(result)
        self._draw_is_vs_oos(result)
        self._draw_efficiency(result)

        param_names = sorted({
            name for w in result.windows for name in w.best_params
        })
        self.combo_param.blockSignals(True)
        self.combo_param.clear()
        self.combo_param.addItems(param_names)
        self.combo_param.blockSignals(False)
        self._redraw_param_plot()

    # --- сшитая эквити ----------------------------------------------------#
    def _draw_equity(self, result: WalkForwardResult) -> None:
        eq = result.stitched_equity
        if eq.empty:
            return
        # Подсветка окон чередующимися полупрозрачными областями
        for i, w in enumerate(result.windows):
            region = pg.LinearRegionItem(
                values=(w.oos_start.timestamp(), w.oos_end.timestamp()),
                movable=False,
                brush=pg.mkBrush(*REGION_BRUSHES[i % 2]),
                pen=pg.mkPen(None),
            )
            region.setZValue(-50)
            self.equity_plot.addItem(region)

        x = _ts_to_x(eq.index)
        self.equity_plot.plot(
            x, eq.to_numpy(dtype=float), pen=pg.mkPen(COLOR_EQUITY, width=2)
        )
        initial = float(eq.iloc[0])
        self.equity_plot.addItem(pg.InfiniteLine(
            initial, angle=0,
            pen=pg.mkPen(COLOR_START_CASH, style=QtCore.Qt.PenStyle.DashLine),
        ))
        self.equity_plot.setLabel("left", "эквити")

    # --- таймлайн окон ------------------------------------------------------#
    def _draw_timeline(self, result: WalkForwardResult) -> None:
        is_x0, is_x1, oos_x0, oos_x1, ys = [], [], [], [], []
        for w in result.windows:
            ys.append(w.index)
            is_x0.append(w.is_start.timestamp())
            is_x1.append(w.is_end.timestamp())
            oos_x0.append(w.oos_start.timestamp())
            oos_x1.append(w.oos_end.timestamp())
        self.timeline_plot.addItem(pg.BarGraphItem(
            x0=is_x0, x1=is_x1, y=ys, height=0.6,
            brush=pg.mkBrush(COLOR_IS), pen=pg.mkPen(None),
        ))
        self.timeline_plot.addItem(pg.BarGraphItem(
            x0=oos_x0, x1=oos_x1, y=ys, height=0.6,
            brush=pg.mkBrush(COLOR_OOS), pen=pg.mkPen(None),
        ))
        self.timeline_plot.setLabel("left", "окно №")
        self.timeline_plot.invertY(True)   # окно 1 сверху

    # --- IS vs OOS, % годовых ------------------------------------------------#
    def _draw_is_vs_oos(self, result: WalkForwardResult) -> None:
        xs, is_vals, oos_vals = [], [], []
        for w in result.windows:
            if not w.valid:
                continue
            xs.append(w.index)
            is_vals.append(_annualized(w.is_stats.total_return_pct, w.is_days))
            oos_vals.append(_annualized(w.oos_stats.total_return_pct, w.oos_days))
        if not xs:
            return
        x = np.asarray(xs, dtype=float)
        self.bars_plot.addItem(pg.BarGraphItem(
            x=x - 0.18, width=0.34, height=is_vals,
            brush=pg.mkBrush(COLOR_IS), pen=pg.mkPen(None),
        ))
        self.bars_plot.addItem(pg.BarGraphItem(
            x=x + 0.18, width=0.34, height=oos_vals,
            brush=pg.mkBrush(COLOR_OOS), pen=pg.mkPen(None),
        ))
        self.bars_plot.addItem(pg.InfiniteLine(
            0, angle=0, pen=pg.mkPen("#888888")
        ))
        self.bars_plot.setLabel("bottom", "окно №")
        self.bars_plot.setLabel("left", "% годовых")

    # --- WFE по окнам ---------------------------------------------------------#
    def _draw_efficiency(self, result: WalkForwardResult) -> None:
        xs, heights, brushes = [], [], []
        for w in result.windows:
            if not math.isfinite(w.efficiency):
                continue
            xs.append(w.index)
            heights.append(w.efficiency)
            brushes.append(
                pg.mkBrush(COLOR_GOOD if w.efficiency > 0 else COLOR_BAD)
            )
        if xs:
            self.eff_plot.addItem(pg.BarGraphItem(
                x=xs, width=0.6, height=heights,
                brushes=brushes, pen=pg.mkPen(None),
            ))
        self.eff_plot.addItem(pg.InfiniteLine(
            100, angle=0,
            pen=pg.mkPen("#888888", style=QtCore.Qt.PenStyle.DashLine),
            label="100%", labelOpts={"color": "#888888", "position": 0.05},
        ))
        self.eff_plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#555555")))
        self.eff_plot.setLabel("bottom", "окно №")
        self.eff_plot.setLabel("left", "WFE, %")

    # --- эволюция параметра ----------------------------------------------------#
    def _redraw_param_plot(self) -> None:
        self.param_plot.clear()
        self.param_plot.addItem(make_grid_item())
        if self._result is None:
            return
        name = self.combo_param.currentText()
        if not name:
            return
        xs, values = [], []
        for w in self._result.windows:
            if w.valid and name in w.best_params:
                xs.append(w.index)
                values.append(float(w.best_params[name]))
        if not xs:
            return
        self.param_plot.plot(
            np.asarray(xs, dtype=float), np.asarray(values, dtype=float),
            pen=pg.mkPen(COLOR_EQUITY, width=2),
            symbol="o", symbolSize=7,
            symbolBrush=COLOR_OOS, symbolPen=None,
        )
        self.param_plot.setTitle(f"Лучший «{name}» по окнам")
        self.param_plot.setLabel("bottom", "окно №")
        self.param_plot.setLabel("left", name)
