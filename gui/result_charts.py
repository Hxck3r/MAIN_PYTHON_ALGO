"""Графики результатов одного бэктеста — для вкладки «Результаты».

Заполняют область рядом с таблицей метрик информативными диаграммами,
которые читаются «одним взглядом»:

    * кривая эквити с пунктиром стартового капитала;
    * «подводный» график просадки (% от пика) с заливкой — видно глубину
      и длительность каждого провала;
    * распределение прибыли сделок (%, гистограмма) с линией нуля —
      перекошено ли оно вправо (есть преимущество) или симметрично;
    * доходность по месяцам (зелёные/красные столбцы) — стабильность
      результата во времени, а не «весь профит за один удачный месяц».

Реализовано на pyqtgraph в той же тёмной теме, что и остальные панели
(см. mc_panel / opt_charts), без внешних зависимостей сверх проекта.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore
from PySide6.QtWidgets import QVBoxLayout, QWidget

from backtester import BacktestResult

from .chart_widget import (
    BG,
    COLOR_EQUITY,
    COLOR_LONG,
    COLOR_SHORT,
    COLOR_START_CASH,
    make_grid_item,
)

HIST_BRUSH = (58, 110, 165, 180)     # синий полупрозрачный, как в mc_panel
DD_BRUSH = (255, 77, 77, 70)         # красная заливка просадки
MAX_MONTH_LABELS = 16                # максимум подписей на оси месяцев


def _epoch(index: pd.DatetimeIndex) -> np.ndarray:
    """Время в секундах эпохи — ось X для pg.DateAxisItem."""
    return index.asi8 / 1e9


class ResultChartsPanel(QWidget):
    """Сетка диаграмм одного прогона: эквити, просадка, P&L, месяцы."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(BG)
        layout.addWidget(self.glw)

        self.equity_plot = self.glw.addPlot(
            row=0, col=0, colspan=2,
            axisItems={"bottom": pg.DateAxisItem()},
        )
        self.equity_plot.setTitle("Кривая эквити")
        self.dd_plot = self.glw.addPlot(
            row=1, col=0, colspan=2,
            axisItems={"bottom": pg.DateAxisItem()},
        )
        self.dd_plot.setTitle("Просадка от пика, %")
        self.dd_plot.setXLink(self.equity_plot)
        self.pnl_hist = self.glw.addPlot(row=2, col=0)
        self.pnl_hist.setTitle("Прибыль сделок, % (распределение)")
        self.month_plot = self.glw.addPlot(row=2, col=1)
        self.month_plot.setTitle("Доходность по месяцам, %")

        for plot in self._plots():
            plot.setMenuEnabled(False)
            plot.showGrid(x=False, y=False)
            plot.addItem(make_grid_item())
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 2)
        self.glw.ci.layout.setRowStretchFactor(2, 3)

    # ------------------------------------------------------------------ #
    def _plots(self) -> list[pg.PlotItem]:
        return [self.equity_plot, self.dd_plot, self.pnl_hist, self.month_plot]

    def clear(self) -> None:
        for plot in self._plots():
            plot.clear()
            plot.addItem(make_grid_item())

    def show_result(self, result: BacktestResult) -> None:
        self.clear()
        equity = result.equity_curve
        if equity is None or equity.empty:
            return
        self._draw_equity(equity, result.initial_cash)
        self._draw_drawdown(equity)
        self._draw_pnl_hist(result.trades_df())
        self._draw_monthly(equity, result.initial_cash)

    # ------------------------------------------------------------------ #
    def _draw_equity(self, equity: pd.Series, initial_cash: float) -> None:
        x = _epoch(equity.index)
        self.equity_plot.plot(
            x, equity.to_numpy(dtype=float),
            pen=pg.mkPen(COLOR_EQUITY, width=2),
        )
        self.equity_plot.addItem(pg.InfiniteLine(
            initial_cash, angle=0,
            pen=pg.mkPen(COLOR_START_CASH, style=QtCore.Qt.PenStyle.DashLine),
            label="старт", labelOpts={"color": COLOR_START_CASH, "position": 0.04},
        ))
        self.equity_plot.setLabel("left", "капитал")

    def _draw_drawdown(self, equity: pd.Series) -> None:
        values = equity.to_numpy(dtype=float)
        running_max = np.maximum.accumulate(values)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(running_max > 0, values / running_max - 1.0, 0.0) * 100.0
        x = _epoch(equity.index)
        curve = self.dd_plot.plot(x, dd, pen=pg.mkPen(COLOR_SHORT, width=1))
        curve.setFillLevel(0.0)
        curve.setFillBrush(pg.mkBrush(*DD_BRUSH))
        self.dd_plot.setLabel("left", "просадка, %")

    def _draw_pnl_hist(self, trades: pd.DataFrame) -> None:
        if trades.empty:
            return
        data = trades["pnl_pct"].to_numpy(dtype=float)
        data = data[np.isfinite(data)]
        if len(data) == 0:
            return
        bins = min(40, max(10, len(data) // 5))
        counts, edges = np.histogram(data, bins=bins)
        self.pnl_hist.addItem(pg.BarGraphItem(
            x0=edges[:-1], x1=edges[1:], height=counts,
            brush=pg.mkBrush(*HIST_BRUSH), pen=pg.mkPen(BG),
        ))
        self.pnl_hist.addItem(pg.InfiniteLine(
            0.0, angle=90,
            pen=pg.mkPen(COLOR_START_CASH, width=2,
                         style=QtCore.Qt.PenStyle.DashLine),
        ))
        self.pnl_hist.setLabel("bottom", "прибыль сделки, %")
        self.pnl_hist.setLabel("left", "сделок")

    def _draw_monthly(self, equity: pd.Series, initial_cash: float) -> None:
        if not isinstance(equity.index, pd.DatetimeIndex):
            return
        month_end = equity.groupby(
            [equity.index.year, equity.index.month]
        ).last()
        vals = month_end.to_numpy(dtype=float)
        if len(vals) == 0:
            return
        prev = np.concatenate(([float(initial_cash)], vals[:-1]))
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.where(prev > 0, vals / prev - 1.0, 0.0) * 100.0
        xs = np.arange(len(rets), dtype=float)
        brushes = [
            pg.mkBrush(COLOR_LONG if r >= 0 else COLOR_SHORT) for r in rets
        ]
        self.month_plot.addItem(pg.BarGraphItem(
            x=xs, height=rets, width=0.8, brushes=brushes, pen=pg.mkPen(BG),
        ))
        self.month_plot.addItem(pg.InfiniteLine(
            0.0, angle=0, pen=pg.mkPen(COLOR_START_CASH, width=1),
        ))
        # Прореживаем подписи месяцев, чтобы ось не превращалась в кашу.
        step = max(1, len(rets) // MAX_MONTH_LABELS)
        ticks = [
            (float(i), f"{y % 100:02d}-{mo:02d}")
            for i, (y, mo) in enumerate(month_end.index)
            if i % step == 0
        ]
        self.month_plot.getAxis("bottom").setTicks([ticks])
        self.month_plot.setLabel("left", "доходность, %")
