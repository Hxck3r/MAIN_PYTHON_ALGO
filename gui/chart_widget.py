"""Виджет графика: свечи + точки сделок + кривая эквити.

Построен на finplot (pyqtgraph): аппаратно-ускоренная отрисовка
с автоматическим прореживанием -- сотни тысяч свечей без лагов.

Особенности реализации:
    * тёмная тема задаётся через глобали finplot ДО создания осей,
      иначе оси рисуются чёрным по чёрному и «исчезают»;
    * master для finplot -- сам виджет (НЕ GraphicsLayoutWidget):
      в этом режиме finplot оборачивает каждую панель в собственный
      PlotWidget (ax.ax_widget), и панели можно разложить по QSplitter --
      границу цена/эквити можно таскать мышью, меняя высоту эквити
      (вплоть до полного сворачивания);
    * на каждую панель кладётся GridItem с пунктирным пером --
      полупрозрачная сетка (прозрачность GridItem подбирает сам по
      плотности линий); ax.reset() удаляет сетку, поэтому clear()
      добавляет её заново;
    * маркеры сделок передаются как серия, выровненная по индексу баров
      (NaN там, где сделки нет) -- так они жёстко привязаны к свечам
      и не «плавают» при зуме;
    * вход и выход каждой сделки соединены пунктиром -- все связки
      рисуются ОДНОЙ ломаной с разрывами NaN (быстро при тысячах сделок);
    * панель эквити отвязана от зума цены (ax.decouple()), при этом
      обе панели получают собственную шкалу времени снизу и цены справа.
"""
from __future__ import annotations

import locale
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

# finplot >= 1.9.4 при импорте молча вызывает os._exit(1), если
# locale.getdefaultlocale() возвращает ru/by. Подменяем функцию до импорта;
# на реальную локаль приложения это не влияет.
locale.getdefaultlocale = lambda *a: ("en_US", "UTF-8")

import finplot as fplt
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from backtester import BacktestResult

# --- Тёмная тема (применяется до создания осей) ------------------------- #
BG = "#000000"
fplt.background = BG
fplt.odd_plot_background = BG          # фон чётных/нечётных панелей одинаковый
fplt.foreground = "#c8c8c8"            # оси, подписи, легенда
fplt.cross_hair_color = "#ffffff60"
fplt.candle_bull_color = "#D3D3D3"
fplt.candle_bull_body_color = "#D3D3D3"  # залитые тела бычьих свечей
fplt.candle_bear_color = "#A9A9A9"
fplt.legend_border_color = "#444444"

COLOR_LONG = "#16c784"
COLOR_SHORT = "#ff4d4d"
COLOR_EXIT_WIN = "#9be564"
COLOR_EXIT_LOSS = "#ff9f43"
COLOR_LINK = "#aaaaaa"
COLOR_EQUITY = "#ffffff"
COLOR_START_CASH = "#888888"
COLOR_GRID = "#ffffff"


def make_grid_item() -> pg.GridItem:
    """Пунктирная сетка; полупрозрачность GridItem выставляет сам (alpha<=50)."""
    grid = pg.GridItem(
        pen=pg.mkPen(COLOR_GRID, style=QtCore.Qt.PenStyle.DashLine),
        textPen=None,
    )
    grid.setZValue(-100)  # позади свечей и линий
    return grid


class ChartWidget(QWidget):
    """Две панели в сплиттере: цена со сделками (сверху) и эквити (снизу)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # master -- не GraphicsLayoutWidget, поэтому finplot создаёт каждой
        # панели отдельный PlotWidget (ax.ax_widget) -- их кладём в сплиттер.
        axs = fplt.create_plot_widget(master=self, rows=2, init_zoom_periods=300)
        # finplot.refresh() обходит windows и ожидает атрибут .axs у мастера
        self.axs = list(axs) if isinstance(axs, (tuple, list)) else [axs]
        self.ax_price, self.ax_equity = self.axs

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(6)
        self.splitter.setStyleSheet(
            "QSplitter::handle { background: #1a1a1a; }"
            "QSplitter::handle:hover { background: #3a6ea5; }"
        )
        self.splitter.addWidget(self.ax_price.ax_widget)
        self.splitter.addWidget(self.ax_equity.ax_widget)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        layout.addWidget(self.splitter)

        # Эквити живёт в своём масштабе и не зумится вместе с ценой.
        self.ax_equity.decouple()
        for ax in self.axs:
            ax.addItem(make_grid_item())

    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        for ax in self.axs:
            ax.reset()                     # удаляет и сетку --
            ax.addItem(make_grid_item())   # возвращаем её на место

    def show_data(self, df: pd.DataFrame) -> None:
        """Только свечи (после загрузки данных, до бэктеста)."""
        self.clear()
        fplt.candlestick_ochl(
            df[["open", "close", "high", "low"]], ax=self.ax_price
        )
        fplt.refresh()

    def show_result(self, result: BacktestResult) -> None:
        """Свечи + маркеры сделок + связки вход-выход + эквити."""
        self.clear()
        df = result.data
        fplt.candlestick_ochl(
            df[["open", "close", "high", "low"]], ax=self.ax_price
        )

        trades = result.trades_df()
        if not trades.empty:
            self._plot_trade_links(df, trades)
            self._plot_trade_markers(df, trades)

        fplt.plot(
            result.equity_curve,
            ax=self.ax_equity,
            color=COLOR_EQUITY,
            legend="Эквити",
        )
        # Пунктир стартового капитала на панели эквити
        fplt.add_line(
            (result.equity_curve.index[0], result.initial_cash),
            (result.equity_curve.index[-1], result.initial_cash),
            color=COLOR_START_CASH,
            style="--",
            ax=self.ax_equity,
        )
        fplt.refresh()

    # ------------------------------------------------------------------ #
    def _bar_series(
        self, df: pd.DataFrame, bars: pd.Series, prices: pd.Series
    ) -> pd.Series:
        """Серия по всем барам: цена в барах сделок, NaN в остальных.

        Привязка по номеру бара гарантирует, что маркер стоит ровно
        на своей свече при любом зуме (не «плавает»).
        """
        values = np.full(len(df), np.nan)
        values[bars.to_numpy(dtype=int)] = prices.to_numpy(dtype=float)
        return pd.Series(values, index=df.index)

    def _plot_trade_markers(self, df: pd.DataFrame, trades: pd.DataFrame) -> None:
        longs = trades[trades["side"] == "BUY"]
        shorts = trades[trades["side"] == "SELL"]
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]

        marker_sets = [
            (longs["entry_bar"], longs["entry_price"],
             "^", COLOR_LONG, "Вход long"),
            (shorts["entry_bar"], shorts["entry_price"],
             "v", COLOR_SHORT, "Вход short"),
            (wins["exit_bar"], wins["exit_price"],
             "x", COLOR_EXIT_WIN, "Выход +"),
            (losses["exit_bar"], losses["exit_price"],
             "x", COLOR_EXIT_LOSS, "Выход -"),
        ]
        for bars, prices, style, color, legend in marker_sets:
            if len(bars):
                fplt.plot(
                    self._bar_series(df, bars, prices),
                    ax=self.ax_price,
                    style=style,
                    color=color,
                    legend=legend,
                    width=1.5,
                )

    def _plot_trade_links(self, df: pd.DataFrame, trades: pd.DataFrame) -> None:
        """Пунктирные линии вход->выход: одна ломаная с разрывами NaN.

        Рисуем напрямую через pyqtgraph в координатах номеров баров --
        это та же ось X, что у finplot (x_indexed), поэтому линии всегда
        совпадают со свечами. Одна ломаная вместо тысяч отдельных линий
        сохраняет скорость на больших историях.
        """
        xs: list[float] = []
        ys: list[float] = []
        for trade in trades.itertuples():
            xs.extend((float(trade.entry_bar), float(trade.exit_bar),
                       float(trade.exit_bar)))
            ys.extend((float(trade.entry_price), float(trade.exit_price),
                       np.nan))
        pen = pg.mkPen(
            color=COLOR_LINK, width=1, style=QtCore.Qt.PenStyle.DashLine
        )
        item = self.ax_price.plot(
            np.asarray(xs), np.asarray(ys), pen=pen, connect="finite"
        )
        item.setZValue(4)  # под маркерами (10), над свечами
