"""Дашборд Монте-Карло анализа.

Слева -- настройки симуляции и сводная таблица вероятностей/интервалов,
справа -- графики:
    * веер кривых эквити («спагетти» + перцентильные конверты 5/25/50/75/95
      + кривая исходного бэктеста);
    * гистограмма итоговых доходностей (с линиями: исходный бэктест, цель);
    * гистограмма максимальных просадок (с линиями: исходная, лимит);
    * гистограмма максимальных серий убытков подряд;
    * CDF итоговой доходности -- вероятность «не хуже X%» одним взглядом.

Кнопка «Запустить» шлёт сигнал run_requested(MonteCarloConfig) наружу --
расчёт выполняет MonteCarloWorker, результат возвращается в show_result().

Подсказки по трактовке:
    * метод «перестановка» не меняет итоговую прибыль -- он оценивает
      РАЗБРОС ПУТИ (просадки, серии убытков) при тех же сделках;
    * перцентиль бэктеста > 95% в облаке симуляций -- признак того, что
      красивая кривая получена удачным порядком сделок (переподгонка).
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
    QDoubleSpinBox,
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

from backtester import MonteCarloConfig, MonteCarloResult

from .chart_widget import BG, COLOR_EQUITY, COLOR_START_CASH, make_grid_item

COLOR_GOOD = "#16c784"
COLOR_BAD = "#ff4d4d"
COLOR_WARN = "#ff9f43"
COLOR_MEDIAN = "#f7d54f"          # медианный конверт
COLOR_ENVELOPE = "#4f9cf7"        # остальные перцентили
COLOR_ORIGINAL = "#ffffff"        # кривая исходного бэктеста
SPAGHETTI_RGBA = (79, 156, 247, 22)
HIST_BRUSH = (79, 156, 247, 160)

METHODS: dict[str, str] = {
    "Бутстрэп (с возвратом)": "resample",
    "Перестановка (без возврата)": "permute",
    "Блочный бутстрэп (~√N)": "block",
}
SIZINGS: dict[str, str] = {
    "Реинвестирование (% от счёта)": "percent",
    "Фиксированный размер ($)": "fixed",
}


def _prob_color(prob_pct: float, good_low: bool) -> str:
    """Цвет вероятности: good_low=True -- чем меньше, тем лучше."""
    bad = prob_pct if not good_low else 100.0 - prob_pct
    if bad >= 75:
        return COLOR_GOOD
    if bad >= 40:
        return COLOR_WARN
    return COLOR_BAD


class McSummaryTable(QTableWidget):
    """Двухколоночная сводка Монте-Карло: вероятности, интервалы, ранги."""

    def __init__(self, parent=None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Метрика", "Значение"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

    def show_result(self, r: MonteCarloResult) -> None:
        cfg = r.config
        ci_lo, ci_hi = r.expectancy_ci_pct
        rank_ret = r.original_return_rank_pct
        rank_dd = r.original_dd_rank_pct

        # (название, значение, цвет | None)
        rows: list[tuple[str, str, Optional[str]]] = [
            ("Симуляций / сделок на путь", f"{r.n_sims:,} / {r.horizon}", None),
            ("— Распределение доходности —", "", None),
            ("Исходный бэктест", f"{r.original_return_pct:+.2f} %", None),
            ("Медиана (P50)", f"{r.return_percentile(50):+.2f} %", None),
            ("90% интервал (P5 … P95)",
             f"{r.return_percentile(5):+.2f} … {r.return_percentile(95):+.2f} %",
             None),
            ("VaR 95% (худшие 5%)", f"{r.var95_pct:+.2f} %",
             COLOR_GOOD if r.var95_pct > 0 else COLOR_BAD),
            ("CVaR 95% (среднее хвоста)", f"{r.cvar95_pct:+.2f} %",
             COLOR_GOOD if r.cvar95_pct > 0 else COLOR_BAD),
            ("— Просадка —", "", None),
            ("Исходный бэктест", f"{r.original_max_dd_pct:.2f} %", None),
            ("Медиана (P50)", f"{r.dd_percentile(50):.2f} %", None),
            ("P95 (план «худший случай»)", f"{r.dd_percentile(95):.2f} %",
             COLOR_WARN),
            ("Худшая из симуляций", f"{r.dd_percentile(100):.2f} %", None),
            ("— Вероятности —", "", None),
            ("Прибыль (итог > 0)", f"{r.prob_profit_pct:.1f} %",
             _prob_color(r.prob_profit_pct, good_low=False)),
            (f"Достижение цели +{cfg.target_return_pct:g}%",
             f"{r.prob_target_pct:.1f} %",
             _prob_color(r.prob_target_pct, good_low=False)),
            (f"Просадка > {cfg.dd_limit_pct:g}%",
             f"{r.prob_dd_exceeded_pct:.1f} %",
             _prob_color(r.prob_dd_exceeded_pct, good_low=True)),
            (f"Разорение (−{cfg.ruin_dd_pct:g}% от старта)",
             f"{r.risk_of_ruin_pct:.2f} %",
             COLOR_GOOD if r.risk_of_ruin_pct < 1 else COLOR_BAD),
            ("— Серии убытков —", "", None),
            ("Исходный бэктест", str(r.original_max_streak), None),
            ("Медиана / P95",
             f"{r.streak_percentile(50):.0f} / {r.streak_percentile(95):.0f}",
             None),
            ("— Защита от переоптимизации —", "", None),
            ("Сценариев с убытком (p-value)", f"{r.p_loss_pct:.1f} %",
             COLOR_GOOD if r.p_loss_pct <= 5
             else (COLOR_WARN if r.p_loss_pct <= 25 else COLOR_BAD)),
            ("Перцентиль бэктеста (доходность)", f"{rank_ret:.1f} %",
             COLOR_BAD if rank_ret >= 95 else COLOR_GOOD),
            ("Симуляций с DD ≤ исходной", f"{rank_dd:.1f} %",
             COLOR_BAD if rank_dd <= 5 else COLOR_GOOD),
            ("Матожидание на сделку, 90% ДИ",
             f"{ci_lo:+.3f} … {ci_hi:+.3f} %",
             COLOR_GOOD if ci_lo > 0 else COLOR_WARN),
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


class McPanel(QWidget):
    """Вкладка «Монте-Карло»: настройки + сводка + графики."""

    #: запрос на расчёт; main_window запускает MonteCarloWorker
    run_requested = Signal(object)   # MonteCarloConfig

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: Optional[MonteCarloResult] = None

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # --- левая колонка: настройки + сводка ----------------------------#
        left = QVBoxLayout()

        config_box = QGroupBox("Настройки симуляции")
        form = QFormLayout(config_box)
        self.spin_sims = QSpinBox(minimum=100, maximum=100_000, value=2000)
        self.spin_sims.setSingleStep(500)
        self.combo_method = QComboBox()
        self.combo_method.addItems(list(METHODS))
        self.combo_method.setToolTip(
            "Бутстрэп: сделки тянутся с возвратом — меняется и итог, и путь.\n"
            "Перестановка: те же сделки в новом порядке — итог фиксирован,\n"
            "  оценивается разброс просадок и серий убытков.\n"
            "Блочный: блоки подряд идущих сделок — сохраняет кластеры\n"
            "  прибыльных/убыточных периодов."
        )
        self.combo_sizing = QComboBox()
        self.combo_sizing.addItems(list(SIZINGS))
        self.spin_horizon = QSpinBox(minimum=0, maximum=100_000, value=0)
        self.spin_horizon.setSpecialValueText("= бэктест")
        self.spin_horizon.setToolTip(
            "Сколько сделок в каждом сценарии будущего.\n"
            "0 — столько же, сколько в бэктесте.\n"
            "Для метода «перестановка» всегда равен числу сделок бэктеста."
        )
        self.spin_target = QDoubleSpinBox(
            minimum=0.1, maximum=10_000, value=20, decimals=1, suffix=" %"
        )
        self.spin_dd_limit = QDoubleSpinBox(
            minimum=0.1, maximum=100, value=20, decimals=1, suffix=" %"
        )
        self.spin_ruin = QDoubleSpinBox(
            minimum=1, maximum=100, value=50, decimals=0, suffix=" %"
        )
        self.spin_seed = QSpinBox(minimum=0, maximum=10_000, value=42)
        form.addRow("Симуляций:", self.spin_sims)
        form.addRow("Метод:", self.combo_method)
        form.addRow("Сайзинг:", self.combo_sizing)
        form.addRow("Сделок на путь:", self.spin_horizon)
        form.addRow("Цель по прибыли:", self.spin_target)
        form.addRow("Лимит просадки:", self.spin_dd_limit)
        form.addRow("Порог разорения:", self.spin_ruin)
        form.addRow("Seed:", self.spin_seed)

        self.btn_run = QPushButton("🎲  Запустить Монте-Карло")
        self.btn_run.clicked.connect(
            lambda: self.run_requested.emit(self.make_config())
        )

        self.summary = McSummaryTable()

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
        left.addWidget(scroll)
        root.addLayout(left)

        # --- правая часть: графики ----------------------------------------#
        right = QVBoxLayout()
        self.lbl_hint = QLabel(
            "Сначала запустите бэктест, затем — Монте-Карло. Веер кривых: "
            "облако переупорядоченных сделок; белая линия — исходный бэктест."
        )
        self.lbl_hint.setStyleSheet("color: gray;")
        right.addWidget(self.lbl_hint)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(BG)
        right.addWidget(self.glw, stretch=1)
        root.addLayout(right, stretch=1)

        self.fan_plot = self.glw.addPlot(row=0, col=0, colspan=2)
        self.fan_plot.setTitle(
            "Веер эквити: P5/P25/P50/P75/P95 + исходный бэктест (белый)"
        )
        self.ret_hist = self.glw.addPlot(row=1, col=0)
        self.ret_hist.setTitle("Итоговая доходность, % (распределение)")
        self.dd_hist = self.glw.addPlot(row=1, col=1)
        self.dd_hist.setTitle("Максимальная просадка, % (распределение)")
        self.streak_hist = self.glw.addPlot(row=2, col=0)
        self.streak_hist.setTitle("Макс. серия убытков подряд")
        self.cdf_plot = self.glw.addPlot(row=2, col=1)
        self.cdf_plot.setTitle("CDF доходности: P(итог ≤ X)")
        for plot in self._plots():
            plot.setMenuEnabled(False)
            plot.addItem(make_grid_item())
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 2)
        self.glw.ci.layout.setRowStretchFactor(2, 2)

    # ------------------------------------------------------------------ #
    def make_config(self) -> MonteCarloConfig:
        return MonteCarloConfig(
            n_sims=self.spin_sims.value(),
            method=METHODS[self.combo_method.currentText()],
            sizing=SIZINGS[self.combo_sizing.currentText()],
            horizon_trades=self.spin_horizon.value(),
            ruin_dd_pct=self.spin_ruin.value(),
            target_return_pct=self.spin_target.value(),
            dd_limit_pct=self.spin_dd_limit.value(),
            seed=self.spin_seed.value(),
        )

    def set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)

    def _plots(self) -> list[pg.PlotItem]:
        return [
            self.fan_plot, self.ret_hist, self.dd_hist,
            self.streak_hist, self.cdf_plot,
        ]

    # ------------------------------------------------------------------ #
    def show_result(self, result: MonteCarloResult) -> None:
        self._result = result
        self.summary.show_result(result)
        for plot in self._plots():
            plot.clear()
            plot.addItem(make_grid_item())
        self._draw_fan(result)
        self._draw_return_hist(result)
        self._draw_dd_hist(result)
        self._draw_streak_hist(result)
        self._draw_cdf(result)
        method_label = self.combo_method.currentText()
        extra = (
            "  •  Перестановка: итог фиксирован, оценивается разброс пути."
            if result.config.method == "permute" else ""
        )
        self.lbl_hint.setText(
            f"{result.n_sims:,} симуляций ({method_label}), "
            f"{result.horizon} сделок на путь, сделок в бэктесте: "
            f"{result.n_trades}.{extra}"
        )

    # --- веер эквити ----------------------------------------------------- #
    def _draw_fan(self, r: MonteCarloResult) -> None:
        curves = r.sample_curves
        k, steps = curves.shape
        # все «спагетти» одной ломаной с NaN-разрывами -- быстро при сотнях кривых
        xs = np.tile(np.append(np.arange(steps, dtype=float), np.nan), k)
        ys = np.hstack([curves, np.full((k, 1), np.nan)]).ravel()
        self.fan_plot.plot(
            xs, ys, pen=pg.mkPen(SPAGHETTI_RGBA), connect="finite"
        )

        x_env = np.arange(r.envelope.shape[1], dtype=float)
        for q, band in zip(r.envelope_q, r.envelope, strict=True):
            is_median = q == 50
            self.fan_plot.plot(
                x_env, band,
                pen=pg.mkPen(
                    COLOR_MEDIAN if is_median else COLOR_ENVELOPE,
                    width=2 if is_median else 1,
                    style=(QtCore.Qt.PenStyle.SolidLine if is_median
                           else QtCore.Qt.PenStyle.DashLine),
                ),
            )
        self.fan_plot.plot(
            np.arange(len(r.original_curve), dtype=float), r.original_curve,
            pen=pg.mkPen(COLOR_ORIGINAL, width=2),
        )
        self.fan_plot.addItem(pg.InfiniteLine(
            r.initial_equity, angle=0,
            pen=pg.mkPen(COLOR_START_CASH, style=QtCore.Qt.PenStyle.DashLine),
        ))
        ruin = r.initial_equity * (1.0 - r.config.ruin_dd_pct / 100.0)
        self.fan_plot.addItem(pg.InfiniteLine(
            ruin, angle=0,
            pen=pg.mkPen(COLOR_BAD, style=QtCore.Qt.PenStyle.DotLine),
            label="разорение", labelOpts={"color": COLOR_BAD, "position": 0.03},
        ))
        self.fan_plot.setLabel("bottom", "№ сделки")
        self.fan_plot.setLabel("left", "эквити")

    # --- гистограммы ------------------------------------------------------ #
    def _hist(self, plot: pg.PlotItem, data: np.ndarray, bins: int = 50) -> None:
        data = data[np.isfinite(data)]
        if len(data) == 0:
            return
        counts, edges = np.histogram(data, bins=bins)
        plot.addItem(pg.BarGraphItem(
            x0=edges[:-1], x1=edges[1:], height=counts,
            brush=pg.mkBrush(*HIST_BRUSH), pen=pg.mkPen(BG),
        ))
        plot.setLabel("left", "симуляций")

    @staticmethod
    def _vline(plot: pg.PlotItem, x: float, color: str, label: str) -> None:
        if not math.isfinite(x):
            return
        plot.addItem(pg.InfiniteLine(
            x, angle=90,
            pen=pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.DashLine),
            label=label, labelOpts={"color": color, "position": 0.92,
                                    "rotateAxis": (1, 0)},
        ))

    def _draw_return_hist(self, r: MonteCarloResult) -> None:
        self._hist(self.ret_hist, r.total_return_pct)
        self._vline(self.ret_hist, 0.0, COLOR_START_CASH, "0")
        self._vline(self.ret_hist, r.original_return_pct, COLOR_ORIGINAL, "бэктест")
        self._vline(self.ret_hist, r.config.target_return_pct, COLOR_GOOD, "цель")
        self.ret_hist.setLabel("bottom", "доходность, %")

    def _draw_dd_hist(self, r: MonteCarloResult) -> None:
        self._hist(self.dd_hist, r.max_dd_pct)
        self._vline(self.dd_hist, r.original_max_dd_pct, COLOR_ORIGINAL, "бэктест")
        self._vline(self.dd_hist, r.config.dd_limit_pct, COLOR_WARN, "лимит")
        self._vline(self.dd_hist, r.dd_percentile(95), COLOR_BAD, "P95")
        self.dd_hist.setLabel("bottom", "макс. просадка, %")

    def _draw_streak_hist(self, r: MonteCarloResult) -> None:
        streaks = r.max_losing_streak
        lo, hi = int(streaks.min()), int(streaks.max())
        bins = np.arange(lo - 0.5, hi + 1.5)  # целочисленные корзины
        counts, edges = np.histogram(streaks, bins=bins)
        self.streak_hist.addItem(pg.BarGraphItem(
            x0=edges[:-1], x1=edges[1:], height=counts,
            brush=pg.mkBrush(*HIST_BRUSH), pen=pg.mkPen(BG),
        ))
        self._vline(self.streak_hist, r.original_max_streak,
                    COLOR_ORIGINAL, "бэктест")
        self.streak_hist.setLabel("bottom", "убытков подряд")
        self.streak_hist.setLabel("left", "симуляций")

    def _draw_cdf(self, r: MonteCarloResult) -> None:
        rets = np.sort(r.total_return_pct)
        prob = (np.arange(len(rets)) + 1) / len(rets) * 100.0
        self.cdf_plot.plot(rets, prob, pen=pg.mkPen(COLOR_EQUITY, width=2))
        self._vline(self.cdf_plot, 0.0, COLOR_START_CASH, "0")
        self._vline(self.cdf_plot, r.config.target_return_pct, COLOR_GOOD, "цель")
        self.cdf_plot.addItem(pg.InfiniteLine(
            50, angle=0,
            pen=pg.mkPen("#888888", style=QtCore.Qt.PenStyle.DashLine),
        ))
        self.cdf_plot.setLabel("bottom", "доходность, %")
        self.cdf_plot.setLabel("left", "P(итог ≤ X), %")
