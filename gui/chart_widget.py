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

# Палитра для индикаторов без явного цвета (циклически по порядку объявления)
INDICATOR_PALETTE = ("#f5b041", "#5dade2", "#bb8fce", "#48c9b0", "#ec7063", "#f7dc6f")

#: Предел ячеек теплокарты: на каждую ячейку создаётся отдельный
#: QGraphicsRectItem, поэтому 1-минутный футпринт за месяцы (753k ячеек)
#: рисуется десятками минут и всё это время подвешивает главный поток.
#: Замеры: 20 000 ячеек -- 2.7 с, 60 000 -- 22.9 с (рост быстрее линейного).
MAX_FOOTPRINT_CELLS = 20_000


def make_grid_item() -> pg.GridItem:
    """Пунктирная сетка; полупрозрачность GridItem выставляет сам (alpha<=50)."""
    grid = pg.GridItem(
        pen=pg.mkPen(COLOR_GRID, style=QtCore.Qt.PenStyle.DashLine),
        textPen=None,
    )
    grid.setZValue(-100)  # позади свечей и линий
    return grid


def _coordinate_edges(values: np.ndarray, fallback_step: float) -> np.ndarray:
    """Infer cell edges from centers without stretching irregular gaps."""
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        half = fallback_step / 2
        return np.array([values[0] - half, values[0] + half])
    mids = (values[:-1] + values[1:]) / 2
    return np.concatenate(([values[0] - (mids[0] - values[0])], mids,
                           [values[-1] + (values[-1] - mids[-1])]))


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

    def show_data(self, df: pd.DataFrame, indicators=None) -> None:
        """Свечи и (опционально) оверлеи индикаторов.

        indicators -- список dict(name, [values | series], color, overlay, width),
        как в BacktestResult.indicators или в событии «indicators» live-движка.
        """
        self.clear()
        fplt.candlestick_ochl(
            df[["open", "close", "high", "low"]], ax=self.ax_price
        )
        self._plot_overlays(df, indicators)
        fplt.refresh()

    def show_footprint(self, footprint: pd.DataFrame, value: str | None = None) -> None:
        """Draw a time/price footprint heatmap on the price panel."""
        if footprint.empty:
            return
        if value is None:
            value = "combined_delta" if "combined_delta" in footprint.columns else "delta"
        if value not in footprint.columns:
            return
        frame = footprint.reset_index()
        time_col = "bar_start" if "bar_start" in frame else "bar"
        if "price_level" not in frame or time_col not in frame:
            raise ValueError("Footprint needs bar_start/bar and price_level columns")
        # Overlapping partitions (same bar loaded twice) must not draw or
        # colour-count the same cell twice.
        frame = frame.drop_duplicates(subset=[time_col, "price_level"], keep="last")
        times = pd.to_datetime(frame[time_col], utc=True)
        levels = pd.to_numeric(frame["price_level"], errors="raise")
        values = pd.to_numeric(frame[value], errors="coerce")
        x_values = pd.DatetimeIndex(times).sort_values().unique()
        x_codes = pd.Categorical(times, categories=x_values, ordered=True).codes
        y_values = np.sort(levels.unique())
        y_codes = pd.Categorical(levels, categories=y_values, ordered=True).codes
        x_seconds = x_values.asi8.astype(np.float64) / 1_000_000_000
        # ImageItem uses a single affine rectangle, so it cannot represent
        # nonuniform cells. Draw actual rectangles at each timestamp/price cell.
        x_edges = _coordinate_edges(x_seconds, 60.0)
        y_edges = _coordinate_edges(y_values.astype(float), 1.0)
        colormap = pg.colormap.get("CET-D1A")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        span = high - low or 1.0
        for (x_code, y_code), cell_value in zip(zip(x_codes, y_codes), values):
            if pd.isna(cell_value):
                continue
            color = colormap.map((float(cell_value) - low) / span, mode="byte")
            item = pg.QtWidgets.QGraphicsRectItem(
                float(x_edges[x_code]), float(y_edges[y_code]),
                float(x_edges[x_code + 1] - x_edges[x_code]),
                float(y_edges[y_code + 1] - y_edges[y_code]),
            )
            item.setPen(pg.mkPen(None))
            item.setBrush(pg.mkBrush(color))
            item.setZValue(-10)          # под свечами и линиями
            self.ax_price.addItem(item)
        fplt.refresh()

    def show_result(self, result: BacktestResult) -> None:
        """Свечи + индикаторы + маркеры сделок + связки вход-выход + эквити."""
        self.clear()
        df = result.data
        fplt.candlestick_ochl(
            df[["open", "close", "high", "low"]], ax=self.ax_price
        )
        # Оверлеи индикаторов рисуем сразу после свечей -- маркеры сделок лягут поверх
        self._plot_overlays(df, getattr(result, "indicators", None))
        footprint = getattr(result, "footprint", None)
        if footprint is not None and len(footprint) <= MAX_FOOTPRINT_CELLS:
            self.show_footprint(footprint)

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
    def _plot_overlays(self, df: pd.DataFrame, indicators) -> None:
        """Накладывает линии-индикаторы (overlay=True) поверх цены.

        Каждый индикатор -- dict с готовой ``series`` (pd.Series с DatetimeIndex)
        либо с массивом ``values`` длины len(df) (выравнивается по df.index).
        NaN-участки прогрева отбрасываются, чтобы линия начиналась с данных.
        Осцилляторы (overlay=False) пока пропускаются (отдельная панель -- TODO).
        """
        if not indicators:
            return
        for i, ind in enumerate(indicators):
            if not ind.get("overlay", True):
                continue
            series = ind.get("series")
            if series is None:
                values = np.asarray(ind["values"], dtype=float)
                n = min(len(values), len(df.index))
                series = pd.Series(values[:n], index=df.index[:n])
            series = series.dropna()
            if series.empty:
                continue
            color = ind.get("color") or INDICATOR_PALETTE[i % len(INDICATOR_PALETTE)]
            fplt.plot(
                series,
                ax=self.ax_price,
                color=color,
                legend=ind.get("name", f"ind {i + 1}"),
                width=ind.get("width", 1.5),
            )

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
