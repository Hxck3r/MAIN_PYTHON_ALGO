"""Главное окно десктопного «Тестера стратегий» (стиль тестера MT5).

Компоновка:
    * слева  -- настройки: данные, период, счёт, стратегия, критерий, метод;
    * центр  -- график (свечи + сделки + эквити, finplot/pyqtgraph);
    * снизу  -- вкладки: Параметры | Результаты | Сделки | Оптимизация | Журнал.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QDate
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from backtester import (
    MATRIX_FIELDS,
    BacktestResult,
    DataLoader,
    MonteCarloConfig,
    MonteCarloResult,
    OptimizationResult,
    WalkForwardConfig,
    WalkForwardResult,
    WfaMatrixConfig,
    WfaMatrixResult,
    generate_synthetic_data,
    span_values,
    window_bounds,
)
from backtester.metrics import CRITERIA
from strategies import STRATEGIES

from .chart_widget import ChartWidget
from .mc_panel import McPanel
from .opt_charts import OptimizationChartsPanel
from .param_table import ParamTable
from .results_panel import MetricsTable, OptimizationTable, TradesTable
from .wfa_matrix_panel import WfaMatrixPanel
from .wfa_panel import WfaPanel
from .workers import (
    BacktestWorker,
    MonteCarloWorker,
    OptimizationWorker,
    WalkForwardWorker,
    WfaMatrixWorker,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Тестер стратегий")
        self.resize(1500, 950)

        self._data: Optional[pd.DataFrame] = None        # полные данные
        self._worker: Optional[Any] = None               # активный поток
        self._last_opt: Optional[OptimizationResult] = None
        self._last_result: Optional[BacktestResult] = None  # для Монте-Карло

        self._build_ui()
        self._on_strategy_changed()
        self._generate_synthetic()

    # ================================================================== #
    # Построение интерфейса
    # ================================================================== #
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Настройки -- в прокручиваемой области: при нехватке высоты экрана
        # появляется вертикальный скролл, а не наложение виджетов друг на друга.
        settings_scroll = QScrollArea()
        settings_scroll.setWidget(self._build_settings_panel())
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_scroll.setFixedWidth(340)   # 320 контент + полоса прокрутки
        root.addWidget(settings_scroll, stretch=0)

        right = QSplitter(Qt.Vertical)
        self.chart = ChartWidget()
        right.addWidget(self.chart)
        right.addWidget(self._build_bottom_tabs())
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        root.addWidget(right, stretch=1)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Готов")

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(312)   # ширину задаёт область прокрутки-обёртка
        layout = QVBoxLayout(panel)

        # --- Данные ------------------------------------------------------#
        data_box = QGroupBox("Данные")
        data_layout = QVBoxLayout(data_box)
        self.radio_csv = QRadioButton("CSV-файл (в т.ч. экспорт MT5)")
        self.radio_synth = QRadioButton("Синтетические (для проверки)")
        self.radio_synth.setChecked(True)
        self.btn_open_csv = QPushButton("Открыть CSV...")
        self.btn_open_csv.clicked.connect(self._open_csv)
        self.lbl_data = QLabel("—")
        self.lbl_data.setWordWrap(True)

        synth_form = QFormLayout()
        self.spin_bars = QSpinBox(minimum=100, maximum=1_000_000, value=5000)
        self.spin_bars.setSingleStep(1000)
        self.spin_seed = QSpinBox(minimum=0, maximum=10_000, value=7)
        synth_form.addRow("Баров:", self.spin_bars)
        synth_form.addRow("Seed:", self.spin_seed)
        self.btn_generate = QPushButton("Сгенерировать")
        self.btn_generate.clicked.connect(self._generate_synthetic)

        data_layout.addWidget(self.radio_csv)
        data_layout.addWidget(self.btn_open_csv)
        data_layout.addWidget(self.radio_synth)
        data_layout.addLayout(synth_form)
        data_layout.addWidget(self.btn_generate)
        data_layout.addWidget(self.lbl_data)
        layout.addWidget(data_box)

        # --- Период -------------------------------------------------------#
        period_box = QGroupBox("Период теста")
        period_form = QFormLayout(period_box)
        self.date_from = QDateEdit(calendarPopup=True)
        self.date_to = QDateEdit(calendarPopup=True)
        period_form.addRow("С:", self.date_from)
        period_form.addRow("По:", self.date_to)
        layout.addWidget(period_box)

        # --- Счёт ----------------------------------------------------------#
        account_box = QGroupBox("Счёт и издержки")
        account_form = QFormLayout(account_box)
        self.spin_cash = QDoubleSpinBox(
            minimum=100, maximum=1e9, value=10_000, decimals=2
        )
        self.spin_cash.setSingleStep(1000)
        self.spin_commission = QDoubleSpinBox(
            minimum=0, maximum=5, value=0.05, decimals=3, suffix=" %"
        )
        self.spin_commission.setSingleStep(0.01)
        self.spin_slippage = QDoubleSpinBox(
            minimum=0, maximum=5, value=0.01, decimals=3, suffix=" %"
        )
        self.spin_slippage.setSingleStep(0.01)
        account_form.addRow("Капитал:", self.spin_cash)
        account_form.addRow("Комиссия:", self.spin_commission)
        account_form.addRow("Слиппедж:", self.spin_slippage)
        layout.addWidget(account_box)

        # --- Стратегия и оптимизация ---------------------------------------#
        strat_box = QGroupBox("Стратегия и оптимизация")
        strat_form = QFormLayout(strat_box)
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(list(STRATEGIES))
        self.combo_strategy.currentIndexChanged.connect(self._on_strategy_changed)
        self.combo_criterion = QComboBox()
        self.combo_criterion.addItems(list(CRITERIA))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Полный перебор (Grid)", "Генетический алгоритм"])
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        self.combo_method.currentIndexChanged.connect(self._update_grid_label)
        self.combo_method.currentIndexChanged.connect(self._update_wfa_label)
        self.spin_min_trades = QSpinBox(minimum=0, maximum=10_000, value=5)
        self.spin_jobs = QSpinBox(minimum=0, maximum=64, value=0)
        self.spin_jobs.setSpecialValueText("авто")
        self.spin_jobs.setToolTip(
            "Число процессов для параллельной оптимизации.\n"
            "авто = все ядра CPU минус одно; 1 = без параллельности."
        )
        self.spin_population = QSpinBox(minimum=5, maximum=1000, value=30)
        self.spin_generations = QSpinBox(minimum=1, maximum=500, value=15)
        self.spin_population.valueChanged.connect(self._update_grid_label)
        self.spin_generations.valueChanged.connect(self._update_grid_label)
        self.spin_population.valueChanged.connect(self._update_wfa_label)
        self.spin_generations.valueChanged.connect(self._update_wfa_label)
        strat_form.addRow("Стратегия:", self.combo_strategy)
        strat_form.addRow("Критерий:", self.combo_criterion)
        strat_form.addRow("Метод:", self.combo_method)
        strat_form.addRow("Мин. сделок:", self.spin_min_trades)
        strat_form.addRow("Процессы CPU:", self.spin_jobs)
        strat_form.addRow("Популяция:", self.spin_population)
        strat_form.addRow("Поколений:", self.spin_generations)
        layout.addWidget(strat_box)
        self._on_method_changed()

        # --- Walk-Forward ----------------------------------------------------#
        wfa_box = QGroupBox("Walk-Forward (WFA)")
        wfa_form = QFormLayout(wfa_box)
        self.spin_wfa_is = QSpinBox(minimum=1, maximum=3650, value=180)
        self.spin_wfa_is.setSuffix(" дн")
        self.spin_wfa_oos = QSpinBox(minimum=1, maximum=3650, value=45)
        self.spin_wfa_oos.setSuffix(" дн")
        self.spin_wfa_step = QSpinBox(minimum=0, maximum=3650, value=0)
        self.spin_wfa_step.setSuffix(" дн")
        self.spin_wfa_step.setSpecialValueText("= OOS")
        self.spin_wfa_step.setToolTip(
            "Смещение окон. 0 (= OOS) — классика: OOS-участки стыкуются\n"
            "без зазоров. Меньше длины OOS задать нельзя (перекрытие)."
        )
        self.chk_wfa_anchored = QCheckBox("Якорное IS (растущее окно)")
        self.lbl_wfa = QLabel()
        self.lbl_wfa.setWordWrap(True)
        wfa_form.addRow("Окно IS:", self.spin_wfa_is)
        wfa_form.addRow("Окно OOS:", self.spin_wfa_oos)
        wfa_form.addRow("Смещение:", self.spin_wfa_step)
        wfa_form.addRow(self.chk_wfa_anchored)
        wfa_form.addRow(self.lbl_wfa)
        layout.addWidget(wfa_box)
        for spin in (self.spin_wfa_is, self.spin_wfa_oos, self.spin_wfa_step):
            spin.valueChanged.connect(self._update_wfa_label)
        self.chk_wfa_anchored.toggled.connect(self._update_wfa_label)
        for date_widget in (self.date_from, self.date_to):
            date_widget.dateChanged.connect(self._update_wfa_label)
            date_widget.dateChanged.connect(self._update_matrix_label)

        # --- Матричный WFO (перебор длин окон) ------------------------------ #
        mx_box = QGroupBox("Матричный WFO (плато окон)")
        mx_form = QFormLayout(mx_box)
        self.spin_mx_is_min = QSpinBox(minimum=1, maximum=3650, value=90)
        self.spin_mx_is_max = QSpinBox(minimum=1, maximum=3650, value=360)
        self.spin_mx_is_n = QSpinBox(minimum=2, maximum=12, value=4)
        self.spin_mx_oos_min = QSpinBox(minimum=1, maximum=3650, value=20)
        self.spin_mx_oos_max = QSpinBox(minimum=1, maximum=3650, value=90)
        self.spin_mx_oos_n = QSpinBox(minimum=2, maximum=12, value=4)
        for spin in (self.spin_mx_is_min, self.spin_mx_is_max,
                     self.spin_mx_oos_min, self.spin_mx_oos_max):
            spin.setSuffix(" дн")
        self.combo_mx_target = QComboBox()
        for name, (label, _) in MATRIX_FIELDS.items():
            self.combo_mx_target.addItem(label, name)
        self.combo_mx_target.setToolTip(
            "Метрика, по которой ищется плато стабильности на тепловой карте."
        )
        is_row = QHBoxLayout()
        is_row.setContentsMargins(0, 0, 0, 0)
        is_row.addWidget(self.spin_mx_is_min)
        is_row.addWidget(QLabel("…"))
        is_row.addWidget(self.spin_mx_is_max)
        is_row.addWidget(QLabel("×"))
        is_row.addWidget(self.spin_mx_is_n)
        is_w = QWidget()
        is_w.setLayout(is_row)
        oos_row = QHBoxLayout()
        oos_row.setContentsMargins(0, 0, 0, 0)
        oos_row.addWidget(self.spin_mx_oos_min)
        oos_row.addWidget(QLabel("…"))
        oos_row.addWidget(self.spin_mx_oos_max)
        oos_row.addWidget(QLabel("×"))
        oos_row.addWidget(self.spin_mx_oos_n)
        oos_w = QWidget()
        oos_w.setLayout(oos_row)
        self.lbl_mx = QLabel()
        self.lbl_mx.setWordWrap(True)
        mx_form.addRow("IS мин…макс×N:", is_w)
        mx_form.addRow("OOS мин…макс×N:", oos_w)
        mx_form.addRow("Цель плато:", self.combo_mx_target)
        mx_form.addRow(self.lbl_mx)
        layout.addWidget(mx_box)
        for spin in (self.spin_mx_is_min, self.spin_mx_is_max, self.spin_mx_is_n,
                     self.spin_mx_oos_min, self.spin_mx_oos_max, self.spin_mx_oos_n):
            spin.valueChanged.connect(self._update_matrix_label)

        # --- Кнопки запуска -------------------------------------------------#
        self.btn_backtest = QPushButton("▶  Старт (бэктест)")
        self.btn_backtest.clicked.connect(self._run_backtest)
        self.btn_optimize = QPushButton("⚙  Оптимизация")
        self.btn_optimize.clicked.connect(self._run_optimization)
        self.btn_wfa = QPushButton("📈  Walk-Forward")
        self.btn_wfa.clicked.connect(self._run_wfa)
        self.btn_wfa_matrix = QPushButton("🔲  Матричный WFO")
        self.btn_wfa_matrix.setToolTip(
            "Перебор комбинаций длин окон IS/OOS — поиск плато стабильности.\n"
            "Тепловая карта во вкладке «Матрица WFO»."
        )
        self.btn_wfa_matrix.clicked.connect(self._run_wfa_matrix)
        self.btn_mc = QPushButton("🎲  Монте-Карло")
        self.btn_mc.setToolTip(
            "Анализ устойчивости последнего бэктеста: перемешивание сделок,\n"
            "распределение доходностей и просадок, вероятности целей.\n"
            "Настройки — во вкладке «Монте-Карло»."
        )
        self.btn_mc.clicked.connect(self._run_monte_carlo_clicked)
        layout.addWidget(self.btn_backtest)
        layout.addWidget(self.btn_optimize)
        layout.addWidget(self.btn_wfa)
        layout.addWidget(self.btn_wfa_matrix)
        layout.addWidget(self.btn_mc)
        layout.addStretch(1)
        return panel

    def _build_bottom_tabs(self) -> QWidget:
        self.tabs = QTabWidget()

        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        param_layout.setContentsMargins(0, 0, 0, 0)
        self.param_table = ParamTable()
        self.param_table.grid_changed.connect(self._update_grid_label)
        self.param_table.grid_changed.connect(self._update_wfa_label)
        self.lbl_grid = QLabel()
        self.lbl_grid.setWordWrap(True)
        param_layout.addWidget(self.param_table)
        param_layout.addWidget(self.lbl_grid)
        self.tabs.addTab(param_widget, "Параметры")

        self.metrics_table = MetricsTable()
        self.tabs.addTab(self.metrics_table, "Результаты")

        self.trades_table = TradesTable()
        self.tabs.addTab(self.trades_table, "Сделки")

        opt_widget = QWidget()
        opt_layout = QVBoxLayout(opt_widget)
        hint = QLabel(
            "Двойной клик по строке — подставить параметры в таблицу "
            "«Параметры» и запустить бэктест."
        )
        hint.setStyleSheet("color: gray;")
        self.opt_table = OptimizationTable()
        self.opt_table.row_activated.connect(self._apply_and_backtest)
        opt_layout.addWidget(hint)
        opt_layout.addWidget(self.opt_table)
        self.tabs.addTab(opt_widget, "Оптимизация")

        self.opt_charts = OptimizationChartsPanel()
        self.tabs.addTab(self.opt_charts, "Опт. графики")

        self.wfa_panel = WfaPanel()
        self.wfa_panel.params_activated.connect(self._apply_and_backtest)
        self.tabs.addTab(self.wfa_panel, "Walk-Forward")

        self.wfa_matrix_panel = WfaMatrixPanel()
        self.wfa_matrix_panel.cell_activated.connect(self._apply_matrix_cell)
        self.tabs.addTab(self.wfa_matrix_panel, "Матрица WFO")

        self.mc_panel = McPanel()
        self.mc_panel.run_requested.connect(self._run_monte_carlo)
        self.tabs.addTab(self.mc_panel, "Монте-Карло")

        self.log = QPlainTextEdit(readOnly=True)
        self.tabs.addTab(self.log, "Журнал")
        return self.tabs

    # ================================================================== #
    # Данные
    # ================================================================== #
    def _set_data(self, df: pd.DataFrame, label: str) -> None:
        self._data = df
        self.lbl_data.setText(f"{label}\n{len(df)} баров, "
                              f"{df.index[0]:%Y-%m-%d} — {df.index[-1]:%Y-%m-%d}")
        first = QDate(df.index[0].year, df.index[0].month, df.index[0].day)
        last = QDate(df.index[-1].year, df.index[-1].month, df.index[-1].day)
        for widget in (self.date_from, self.date_to):
            widget.setDateRange(first, last)
        self.date_from.setDate(first)
        self.date_to.setDate(last)
        self.chart.show_data(df)
        self._log(f"Данные загружены: {label} ({len(df)} баров)")

    def _generate_synthetic(self) -> None:
        self.radio_synth.setChecked(True)
        df = generate_synthetic_data(
            n_bars=self.spin_bars.value(), freq="1h", seed=self.spin_seed.value()
        )
        self._set_data(df, "Синтетические данные")

    def _open_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл истории", "", "CSV (*.csv *.txt);;Все файлы (*)"
        )
        if not path:
            return
        try:
            loader = DataLoader()
            df = loader.load_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
            return
        self.radio_csv.setChecked(True)
        self._set_data(df, path.split("/")[-1])
        if loader.last_report:
            self._log(str(loader.last_report))

    def _selected_data(self) -> Optional[pd.DataFrame]:
        """Данные, обрезанные по выбранному периоду."""
        if self._data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return None
        start = self.date_from.date().toPython()
        end = self.date_to.date().toPython()
        df = self._data.loc[str(start): str(end)]
        if len(df) < 50:
            QMessageBox.warning(
                self, "Мало данных",
                f"В выбранном периоде всего {len(df)} баров — расширьте период",
            )
            return None
        return df

    def _engine_kwargs(self) -> dict[str, float]:
        return dict(
            cash=self.spin_cash.value(),
            commission_pct=self.spin_commission.value() / 100.0,
            slippage_pct=self.spin_slippage.value() / 100.0,
        )

    def _strategy_cls(self):
        return STRATEGIES[self.combo_strategy.currentText()]

    # ================================================================== #
    # Запуск задач
    # ================================================================== #
    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.btn_backtest.setEnabled(not busy)
        self.btn_optimize.setEnabled(not busy)
        self.btn_wfa.setEnabled(not busy)
        self.btn_wfa_matrix.setEnabled(not busy)
        self.btn_mc.setEnabled(not busy)
        self.mc_panel.set_running(busy)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)  # «бегунок», пока нет точного прогресса
        self.statusBar().showMessage(message or ("Расчёт..." if busy else "Готов"))

    def _run_backtest(self) -> None:
        data = self._selected_data()
        if data is None:
            return
        try:
            params = self.param_table.fixed_values()
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры", str(exc))
            return

        self._set_busy(True, "Бэктест...")
        self._log(f"Бэктест: {self._strategy_cls().__name__} {params}")
        self._worker = BacktestWorker(
            data, self._strategy_cls(), params, self._engine_kwargs(), parent=self
        )
        self._worker.finished_ok.connect(self._on_backtest_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _run_optimization(self) -> None:
        data = self._selected_data()
        if data is None:
            return
        try:
            fixed, grid = self.param_table.optimization_setup()
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры", str(exc))
            return
        if not grid:
            QMessageBox.information(
                self, "Оптимизация",
                "Отметьте галочкой «Опт.» хотя бы один параметр "
                "во вкладке «Параметры»",
            )
            return

        use_genetic = self.combo_method.currentIndex() == 1
        total = int(np.prod([len(v) for v in grid.values()]))
        if not use_genetic and total > 20_000:
            answer = QMessageBox.question(
                self, "Большая сетка",
                f"Полный перебор: {total} комбинаций. Это может занять "
                f"много времени. Продолжить?",
            )
            if answer != QMessageBox.Yes:
                return

        self._set_busy(True, "Оптимизация...")
        self._log(
            f"Оптимизация ({self.combo_method.currentText()}): сетка {grid}, "
            f"критерий «{self.combo_criterion.currentText()}»"
        )
        self._worker = OptimizationWorker(
            data=data,
            strategy_cls=self._strategy_cls(),
            metric=CRITERIA[self.combo_criterion.currentText()],
            fixed_params=fixed,
            grid=grid,
            engine_kwargs=self._engine_kwargs(),
            min_trades=self.spin_min_trades.value(),
            use_genetic=use_genetic,
            population_size=self.spin_population.value(),
            generations=self.spin_generations.value(),
            n_jobs=self.spin_jobs.value(),
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_optimization_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    # ================================================================== #
    # Walk-Forward Analysis
    # ================================================================== #
    def _wfa_config(self) -> WalkForwardConfig:
        """Конфигурация WFA из дат периода и настроек панели."""
        start = pd.Timestamp(self.date_from.date().toPython())
        # конец -- исключительно: +1 день, чтобы захватить бары последней даты
        end = pd.Timestamp(self.date_to.date().toPython()) + pd.Timedelta(days=1)
        return WalkForwardConfig(
            start=start,
            end=end,
            in_sample_days=self.spin_wfa_is.value(),
            out_sample_days=self.spin_wfa_oos.value(),
            step_days=self.spin_wfa_step.value(),
            anchored=self.chk_wfa_anchored.isChecked(),
        )

    def _update_wfa_label(self) -> None:
        """Живой счётчик окон и оценка числа прогонов."""
        if not hasattr(self, "lbl_wfa") or not hasattr(self, "param_table"):
            return
        try:
            bounds = window_bounds(self._wfa_config())
        except ValueError as exc:
            self.lbl_wfa.setText(f"⚠ {exc}")
            self.lbl_wfa.setStyleSheet("color: #ff9f43;")
            return
        if not bounds:
            self.lbl_wfa.setText(
                "⚠ Окна не помещаются в период — уменьшите IS/OOS"
            )
            self.lbl_wfa.setStyleSheet("color: #ff9f43;")
            return

        grid_total, per_param = self.param_table.grid_summary()
        if self.combo_method.currentIndex() == 1:
            runs = self.spin_population.value() * self.spin_generations.value()
        else:
            runs = grid_total
        text = f"Окон: {len(bounds)}"
        if per_param and runs:
            text += f"  •  ≈ {len(bounds) * runs:,} прогонов"
        self.lbl_wfa.setText(text)
        self.lbl_wfa.setStyleSheet("color: gray;")

    def _run_wfa(self) -> None:
        data = self._selected_data()
        if data is None:
            return
        try:
            fixed, grid = self.param_table.optimization_setup()
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры", str(exc))
            return
        if not grid:
            QMessageBox.information(
                self, "Walk-Forward",
                "Отметьте галочкой «Опт.» хотя бы один параметр "
                "во вкладке «Параметры»",
            )
            return
        try:
            config = self._wfa_config()
            bounds = window_bounds(config)
        except ValueError as exc:
            QMessageBox.warning(self, "Walk-Forward", str(exc))
            return
        if not bounds:
            QMessageBox.warning(
                self, "Walk-Forward",
                "Окна не помещаются в выбранный период: уменьшите длины "
                "IS/OOS или расширьте даты",
            )
            return
        if len(bounds) < 2:
            QMessageBox.information(
                self, "Walk-Forward",
                "Получилось всего одно окно — для оценки устойчивости "
                "нужно несколько. Уменьшите смещение или расширьте период.",
            )
            return

        use_genetic = self.combo_method.currentIndex() == 1
        grid_total = int(np.prod([len(v) for v in grid.values()]))
        runs = (
            self.spin_population.value() * self.spin_generations.value()
            if use_genetic else grid_total
        )
        if len(bounds) * runs > 100_000:
            answer = QMessageBox.question(
                self, "Долгий расчёт",
                f"WFA: {len(bounds)} окон × ~{runs} прогонов = "
                f"~{len(bounds) * runs:,} бэктестов. Продолжить?",
            )
            if answer != QMessageBox.Yes:
                return

        self._set_busy(True, "Walk-Forward...")
        self._log(
            f"WFA: {len(bounds)} окон (IS {config.in_sample_days}д, "
            f"OOS {config.out_sample_days}д, сдвиг {config.effective_step_days}д"
            f"{', якорное' if config.anchored else ''}), сетка {grid}, "
            f"критерий «{self.combo_criterion.currentText()}»"
        )
        self._worker = WalkForwardWorker(
            data=data,
            strategy_cls=self._strategy_cls(),
            config=config,
            metric=CRITERIA[self.combo_criterion.currentText()],
            fixed_params=fixed,
            grid=grid,
            engine_kwargs=self._engine_kwargs(),
            min_trades=self.spin_min_trades.value(),
            use_genetic=use_genetic,
            population_size=self.spin_population.value(),
            generations=self.spin_generations.value(),
            n_jobs=self.spin_jobs.value(),
            parent=self,
        )
        self._worker.progress.connect(self._on_wfa_progress)
        self._worker.status.connect(
            lambda text: self.statusBar().showMessage(text)
        )
        self._worker.finished_ok.connect(self._on_wfa_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    # ================================================================== #
    # Монте-Карло
    # ================================================================== #
    def _run_monte_carlo_clicked(self) -> None:
        """Кнопка слева: открыть вкладку и запустить с её настройками."""
        self.tabs.setCurrentWidget(self.mc_panel)
        self._run_monte_carlo(self.mc_panel.make_config())

    def _run_monte_carlo(self, config: MonteCarloConfig) -> None:
        if self._last_result is None or not self._last_result.trades:
            QMessageBox.information(
                self, "Монте-Карло",
                "Сначала выполните бэктест (кнопка «Старт») — Монте-Карло "
                "перемешивает сделки последнего прогона.",
            )
            return
        n_trades = len(self._last_result.trades)
        if n_trades < 5:
            QMessageBox.warning(
                self, "Монте-Карло",
                f"В бэктесте всего {n_trades} сделок — минимум 5. "
                f"Расширьте период или смягчите параметры стратегии.",
            )
            return
        if n_trades < 30:
            self._log(
                f"Монте-Карло: внимание, сделок всего {n_trades} (<30) — "
                f"доверительные интервалы будут широкими и ненадёжными"
            )

        self._set_busy(True, "Монте-Карло...")
        self._log(
            f"Монте-Карло: {config.n_sims} симуляций, метод {config.method}, "
            f"сайзинг {config.sizing}, сделок {n_trades}, "
            f"цель +{config.target_return_pct:g}%, лимит DD {config.dd_limit_pct:g}%"
        )
        self._worker = MonteCarloWorker(self._last_result, config, parent=self)
        self._worker.progress.connect(self._on_progress_simple)
        self._worker.finished_ok.connect(self._on_mc_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress_simple(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _on_mc_done(self, result: MonteCarloResult) -> None:
        self._set_busy(False)
        self.mc_panel.show_result(result)
        self.tabs.setCurrentWidget(self.mc_panel)
        summary = (
            f"Монте-Карло завершён: медианная доходность "
            f"{result.return_percentile(50):+.2f}% "
            f"[P5 {result.return_percentile(5):+.2f}% … "
            f"P95 {result.return_percentile(95):+.2f}%], "
            f"P(прибыль) {result.prob_profit_pct:.0f}%, "
            f"P(цель) {result.prob_target_pct:.0f}%, "
            f"P95 просадка {result.dd_percentile(95):.2f}%, "
            f"риск разорения {result.risk_of_ruin_pct:.2f}%"
        )
        self._log(summary)
        if result.original_return_rank_pct >= 95:
            self._log(
                "⚠ Монте-Карло: исходный бэктест в топ-5% облака симуляций — "
                "кривая, вероятно, обязана удачному порядку сделок (переподгонка)"
            )
        self.statusBar().showMessage(summary)

    def _on_wfa_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _on_wfa_done(self, result: WalkForwardResult) -> None:
        self._set_busy(False)
        self.wfa_panel.show_result(result)
        self.tabs.setCurrentWidget(self.wfa_panel)
        wfe = (
            f"{result.wfe_pct:.1f}%" if math.isfinite(result.wfe_pct) else "n/a"
        )
        summary = (
            f"WFA завершён: окон {result.n_windows} "
            f"(пропущено {result.skipped_windows}), WFE {wfe}, "
            f"прибыльных окон {result.pct_profitable_windows:.0f}%, "
            f"OOS доходность {result.oos_stats.total_return_pct:.2f}%, "
            f"Sharpe {result.oos_stats.sharpe_ratio:.2f}, "
            f"MaxDD {result.oos_stats.max_drawdown_pct:.2f}%"
        )
        self._log(summary)
        self.statusBar().showMessage(summary)

    # ================================================================== #
    # Матричный WFO (WFA Cluster Analysis)
    # ================================================================== #
    def _matrix_config(self) -> WfaMatrixConfig:
        """Конфигурация матрицы из дат периода и диапазонов длин окон."""
        start = pd.Timestamp(self.date_from.date().toPython())
        end = pd.Timestamp(self.date_to.date().toPython()) + pd.Timedelta(days=1)
        return WfaMatrixConfig(
            start=start,
            end=end,
            is_days_list=span_values(
                self.spin_mx_is_min.value(), self.spin_mx_is_max.value(),
                self.spin_mx_is_n.value(),
            ),
            oos_days_list=span_values(
                self.spin_mx_oos_min.value(), self.spin_mx_oos_max.value(),
                self.spin_mx_oos_n.value(),
            ),
            anchored=self.chk_wfa_anchored.isChecked(),
            step_days=self.spin_wfa_step.value(),
        )

    def _matrix_runnable_cells(self, config: WfaMatrixConfig) -> int:
        """Сколько ячеек реально помещают >= 2 окон (для оценки объёма)."""
        runnable = 0
        for is_d, oos_d in config.cells():
            try:
                if len(window_bounds(config.window_config(is_d, oos_d))) >= 2:
                    runnable += 1
            except ValueError:
                continue
        return runnable

    def _update_matrix_label(self) -> None:
        """Живой счётчик ячеек и оценка числа прогонов матрицы."""
        if not hasattr(self, "lbl_mx") or not hasattr(self, "param_table"):
            return
        try:
            config = self._matrix_config()
            config.validate()
        except ValueError as exc:
            self.lbl_mx.setText(f"⚠ {exc}")
            self.lbl_mx.setStyleSheet("color: #ff9f43;")
            return
        n_is, n_oos = len(config.is_days_list), len(config.oos_days_list)
        runnable = self._matrix_runnable_cells(config)
        text = (
            f"Сетка {n_is}×{n_oos} = {n_is * n_oos} ячеек "
            f"(помещается {runnable})"
        )
        if runnable == 0:
            text += " — уменьшите длины окон"
            self.lbl_mx.setStyleSheet("color: #ff9f43;")
        else:
            self.lbl_mx.setStyleSheet("color: gray;")
        self.lbl_mx.setText(text)

    def _run_wfa_matrix(self) -> None:
        data = self._selected_data()
        if data is None:
            return
        try:
            fixed, grid = self.param_table.optimization_setup()
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры", str(exc))
            return
        if not grid:
            QMessageBox.information(
                self, "Матричный WFO",
                "Отметьте галочкой «Опт.» хотя бы один параметр "
                "во вкладке «Параметры»",
            )
            return
        try:
            config = self._matrix_config()
            config.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Матричный WFO", str(exc))
            return
        runnable = self._matrix_runnable_cells(config)
        if runnable == 0:
            QMessageBox.warning(
                self, "Матричный WFO",
                "Ни одна комбинация длин окон не помещает достаточно окон: "
                "уменьшите длины IS/OOS или расширьте период.",
            )
            return

        use_genetic = self.combo_method.currentIndex() == 1
        grid_total = int(np.prod([len(v) for v in grid.values()]))
        runs = (
            self.spin_population.value() * self.spin_generations.value()
            if use_genetic else grid_total
        )
        # грубая оценка: число прогонов на ячейку × ячейки (без точного числа окон)
        est = runnable * runs * 4
        if est > 200_000:
            answer = QMessageBox.question(
                self, "Долгий расчёт",
                f"Матричный WFO: ~{runnable} ячеек × {runs} прогонов на окно. "
                f"На слабом CPU это может занять много времени. Продолжить?",
            )
            if answer != QMessageBox.Yes:
                return

        target = self.combo_mx_target.currentData()
        self._set_busy(True, "Матричный WFO...")
        self._log(
            f"Матричный WFO: IS {list(config.is_days_list)}д × "
            f"OOS {list(config.oos_days_list)}д "
            f"({runnable} ячеек помещается), сетка {grid}, "
            f"критерий «{self.combo_criterion.currentText()}», "
            f"цель плато «{self.combo_mx_target.currentText()}»"
        )
        self._worker = WfaMatrixWorker(
            data=data,
            strategy_cls=self._strategy_cls(),
            matrix_config=config,
            metric=CRITERIA[self.combo_criterion.currentText()],
            target=target,
            fixed_params=fixed,
            grid=grid,
            engine_kwargs=self._engine_kwargs(),
            min_trades=self.spin_min_trades.value(),
            use_genetic=use_genetic,
            population_size=self.spin_population.value(),
            generations=self.spin_generations.value(),
            n_jobs=self.spin_jobs.value(),
            parent=self,
        )
        self._worker.progress.connect(self._on_wfa_progress)
        self._worker.status.connect(
            lambda text: self.statusBar().showMessage(text)
        )
        self._worker.finished_ok.connect(self._on_wfa_matrix_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _apply_matrix_cell(self, is_days: int, oos_days: int) -> None:
        """Двойной клик по ячейке матрицы -> подставить длины окон в WFA."""
        self.spin_wfa_is.setValue(int(is_days))
        self.spin_wfa_oos.setValue(int(oos_days))
        self._log(
            f"Из матрицы выбраны длины окон: IS={is_days}д, OOS={oos_days}д "
            f"(подставлены в Walk-Forward)"
        )
        self.statusBar().showMessage(
            f"Длины окон IS={is_days}д / OOS={oos_days}д применены к Walk-Forward"
        )

    def _on_wfa_matrix_done(self, result: WfaMatrixResult) -> None:
        self._set_busy(False)
        self.wfa_matrix_panel.show_result(result)
        self.tabs.setCurrentWidget(self.wfa_matrix_panel)
        st = result.stability()
        robust = result.robust_cell()
        rec = (
            f"IS={robust.is_days}д/OOS={robust.oos_days}д" if robust else "n/a"
        )
        summary = (
            f"Матричный WFO завершён: ячеек {st.n_ran}/{st.n_cells}, "
            f"avg WFE {st.avg_wfe_pct:.0f}%, прибыльных ячеек "
            f"{st.pct_profitable_cells:.0f}%, плато {st.plateau_pct:.0f}%, "
            f"рекомендация {rec}"
        )
        self._log(summary)
        self.statusBar().showMessage(summary)

    # ================================================================== #
    # Обработка результатов
    # ================================================================== #
    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.statusBar().showMessage(f"Оптимизация: {done}/{total}")

    def _on_backtest_done(self, result: BacktestResult) -> None:
        self._set_busy(False)
        self._last_result = result
        stats = result.stats()
        self.metrics_table.show_result(result)
        self.trades_table.show_result(result)
        self.chart.show_result(result)
        self.tabs.setCurrentWidget(self.metrics_table)
        self._log(
            f"Бэктест завершён: прибыль {stats.net_profit:,.2f} "
            f"({stats.total_return_pct:.2f}%), сделок {stats.total_trades}, "
            f"Sharpe {stats.sharpe_ratio:.2f}, MaxDD {stats.max_drawdown_pct:.2f}%"
        )
        self.statusBar().showMessage(
            f"Прибыль: {stats.net_profit:,.2f} | Сделок: {stats.total_trades} | "
            f"Sharpe: {stats.sharpe_ratio:.2f} | MaxDD: {stats.max_drawdown_pct:.2f}%"
        )

    def _on_optimization_done(self, opt_result: OptimizationResult) -> None:
        self._set_busy(False)
        self._last_opt = opt_result
        self.opt_table.show_result(opt_result)
        self.opt_charts.show_result(opt_result)
        self.tabs.setCurrentIndex(3)  # вкладка «Оптимизация»
        if opt_result.best_params and math.isfinite(opt_result.best_score):
            self._log(
                f"Оптимизация завершена: лучший score {opt_result.best_score:.4f} "
                f"при {opt_result.best_params} "
                f"({len(opt_result.history)} испытаний)"
            )
            if opt_result.deflated_sharpe is not None:
                d = opt_result.deflated_sharpe
                verdict = (
                    "значим с учётом перебора" if d.dsr >= 0.95
                    else "вероятен артефакт множественного тестирования"
                )
                self._log(
                    f"Deflated Sharpe: {d.dsr:.2f} — {verdict} "
                    f"(Sharpe {d.observed_sr:.3f} vs порог шума {d.expected_max_sr:.3f})"
                )
            self._apply_and_backtest(opt_result.best_params)
        else:
            self._log("Оптимизация: ни один прогон не прошёл фильтр мин. сделок")
            QMessageBox.information(
                self, "Оптимизация",
                "Ни один прогон не дал результата (мало сделок или ошибки). "
                "Уменьшите «Мин. сделок» или расширьте диапазоны.",
            )

    def _apply_and_backtest(self, params: dict[str, Any]) -> None:
        """Подставляет параметры в таблицу и запускает бэктест."""
        clean = {k: v for k, v in params.items() if v is not None}
        self.param_table.apply_values(clean)
        self._log(f"Применены параметры: {clean}")
        self._run_backtest()

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self._log(f"ОШИБКА: {message}")
        QMessageBox.critical(self, "Ошибка", message)

    # ================================================================== #
    def _on_strategy_changed(self) -> None:
        self.param_table.set_strategy(self._strategy_cls())

    def _on_method_changed(self) -> None:
        genetic = self.combo_method.currentIndex() == 1
        self.spin_population.setEnabled(genetic)
        self.spin_generations.setEnabled(genetic)

    def _update_grid_label(self) -> None:
        """Живой счётчик шагов оптимизации под таблицей параметров."""
        if not hasattr(self, "lbl_grid"):  # вызов до полной сборки UI
            return
        total, per_param = self.param_table.grid_summary()
        if not per_param:
            self.lbl_grid.setText(
                "Оптимизация: отметьте «Опт.» у нужных параметров "
                "и задайте Старт / Шаг / Стоп"
            )
            self.lbl_grid.setStyleSheet("color: gray; padding: 2px 6px;")
            return

        parts = " × ".join(f"{name}: {count}" for name, count in per_param)
        color = "gray"
        if total == 0:
            text = f"⚠ Проверьте Старт/Шаг/Стоп — получается 0 значений ({parts})"
            color = "#ff9f43"
        elif self.combo_method.currentIndex() == 1:  # генетика
            evals = self.spin_population.value() * self.spin_generations.value()
            text = (
                f"Сетка: {total:,} точек ({parts})  •  "
                f"генетика: до {evals} прогонов"
            )
        else:
            text = f"Полный перебор: {total:,} комбинаций ({parts})"
            if total > 20_000:
                color = "#ff5c5c"
        self.lbl_grid.setText(text)
        self.lbl_grid.setStyleSheet(f"color: {color}; padding: 2px 6px;")

    def _log(self, message: str) -> None:
        self.log.appendPlainText(f"[{datetime.now():%H:%M:%S}] {message}")
