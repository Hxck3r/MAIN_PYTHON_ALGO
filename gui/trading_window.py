"""Окно live-торговли на криптобиржах (отдельно от тестера стратегий).

Открывается кнопкой из главного окна. Здесь можно:
    * настроить подключение к бирже (Binance — CEX или Lighter — DEX), демо/реал;
    * перенести стратегию из тестера (или выбрать другую) и её параметры;
    * выбрать символ -- появляется график цены;
    * запустить автоматическую торговлю по сигналам стратегии.

Архитектура: вся работа с биржей -- в QThread-воркерах (gui/trading_workers.py);
торговый цикл -- live.engine.LiveTradingEngine; исполнение сигналов -- live.broker.
Окно лишь собирает настройки и отображает события движка.
"""
from __future__ import annotations

from typing import Any, Optional, Type

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from backtester import BaseStrategy
from strategies import STRATEGIES

import live
from live.broker import ExecutionConfig, RiskLimits, SizingConfig
from live.engine import EngineConfig, EngineEvent
from live.exchange import EXCHANGES, MODE_DEMO, MODE_REAL, TIMEFRAMES

from .chart_widget import ChartWidget
from .param_table import ParamTable
from .trading_workers import (
    ConnectionSpec,
    ConnectionTestWorker,
    OhlcvWorker,
    SymbolsWorker,
    TradingEngineThread,
)

# id биржи (ключ EXCHANGES) <-> отображаемая метка
_EX_IDS = list(EXCHANGES)


class TradingWindow(QMainWindow):
    """Отдельное окно реальной/демо торговли."""

    def __init__(
        self,
        parent=None,
        initial_strategy: Optional[str] = None,
        initial_params: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Торговля на бирже (демо / реал)")
        self.resize(1400, 900)

        self._spec: Optional[ConnectionSpec] = None       # текущее подключение
        self._connected = False
        self._worker: Optional[Any] = None                # активный фоновый воркер
        self._engine_thread: Optional[TradingEngineThread] = None
        self._real_acknowledged = False                   # подтверждён ли реал-режим
        self._live_df = None                              # последние свечи для графика
        self._live_indicators = None                      # последние оверлеи индикаторов

        if not live.CCXT_AVAILABLE:
            self._build_install_hint()
            return

        self._build_ui()
        self._on_exchange_changed()
        # Перенос стратегии из тестера
        if initial_strategy and initial_strategy in STRATEGIES:
            self.combo_strategy.setCurrentText(initial_strategy)
        self._on_strategy_changed()
        if initial_params:
            self.param_table.apply_values(initial_params)

    # ================================================================== #
    # Заглушка, если ccxt не установлена
    # ================================================================== #
    def _build_install_hint(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        lbl = QLabel(live.INSTALL_HINT)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 14px; padding: 40px;")
        layout.addWidget(lbl)
        self.setCentralWidget(central)

    # ================================================================== #
    # Построение интерфейса
    # ================================================================== #
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        settings_scroll = QScrollArea()
        settings_scroll.setWidget(self._build_settings_panel())
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_scroll.setFixedWidth(360)
        root.addWidget(settings_scroll, stretch=0)

        right = QSplitter(Qt.Vertical)
        self.chart = ChartWidget()
        right.addWidget(self.chart)
        right.addWidget(self._build_bottom_tabs())
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        root.addWidget(right, stretch=1)

        self.statusBar().showMessage("Не подключено")

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(332)
        layout = QVBoxLayout(panel)

        # --- Подключение -------------------------------------------------- #
        conn_box = QGroupBox("Подключение")
        conn_form = QFormLayout(conn_box)
        self.combo_exchange = QComboBox()
        for ex_id in _EX_IDS:
            self.combo_exchange.addItem(EXCHANGES[ex_id].label, ex_id)
        self.combo_exchange.currentIndexChanged.connect(self._on_exchange_changed)

        self.radio_demo = QRadioButton("Демо / тестнет")
        self.radio_real = QRadioButton("РЕАЛ ⚠")
        self.radio_demo.setChecked(True)
        self.radio_demo.toggled.connect(self._on_mode_changed)
        self.radio_real.toggled.connect(self._on_mode_changed)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.radio_demo)
        mode_row.addWidget(self.radio_real)
        mode_w = QWidget()
        mode_w.setLayout(mode_row)

        # Поля ключей строятся динамически под схему выбранной биржи:
        # CEX (Binance) — API Key + Secret; DEX (Lighter) — приватный ключ API
        # и индексы аккаунта (Account Index / API Key Index).
        self._cred_edits: dict[str, QLineEdit] = {}
        self.cred_form = QFormLayout()
        self.cred_form.setContentsMargins(0, 0, 0, 0)
        cred_w = QWidget()
        cred_w.setLayout(self.cred_form)

        self.spin_leverage = QSpinBox(minimum=1, maximum=125, value=5)
        self.spin_leverage.setSuffix("×")
        self.combo_margin = QComboBox()
        self.combo_margin.addItems(["isolated", "cross"])

        conn_form.addRow("Биржа:", self.combo_exchange)
        conn_form.addRow("Режим:", mode_w)
        conn_form.addRow(cred_w)
        conn_form.addRow("Плечо:", self.spin_leverage)
        conn_form.addRow("Маржа:", self.combo_margin)

        self.lbl_warn = QLabel()
        self.lbl_warn.setWordWrap(True)
        conn_form.addRow(self.lbl_warn)
        self.lbl_ex_note = QLabel()
        self.lbl_ex_note.setWordWrap(True)
        self.lbl_ex_note.setStyleSheet("color: gray; font-size: 11px;")
        conn_form.addRow(self.lbl_ex_note)

        btn_row = QHBoxLayout()
        self.btn_test = QPushButton("Проверить подключение")
        self.btn_test.clicked.connect(self._test_connection)
        self.btn_save_keys = QPushButton("Сохранить ключи")
        self.btn_save_keys.clicked.connect(self._save_keys)
        btn_row.addWidget(self.btn_test)
        btn_row.addWidget(self.btn_save_keys)
        btn_w = QWidget()
        btn_w.setLayout(btn_row)
        conn_form.addRow(btn_w)
        layout.addWidget(conn_box)

        # --- Стратегия (из тестера) -------------------------------------- #
        strat_box = QGroupBox("Стратегия (из тестера)")
        strat_layout = QVBoxLayout(strat_box)
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(list(STRATEGIES))
        self.combo_strategy.currentIndexChanged.connect(self._on_strategy_changed)
        self.param_table = ParamTable()
        self.param_table.setMaximumHeight(220)
        strat_layout.addWidget(self.combo_strategy)
        strat_layout.addWidget(self.param_table)
        layout.addWidget(strat_box)

        # --- Символ ------------------------------------------------------- #
        sym_box = QGroupBox("Символ и таймфрейм")
        sym_form = QFormLayout(sym_box)
        self.combo_symbol = QComboBox()
        self.combo_symbol.setEditable(True)
        self.combo_symbol.setInsertPolicy(QComboBox.NoInsert)
        self.combo_symbol.setEnabled(False)
        self.combo_symbol.currentIndexChanged.connect(self._on_symbol_changed)
        self.btn_reload_symbols = QPushButton("Обновить список символов")
        self.btn_reload_symbols.setEnabled(False)
        self.btn_reload_symbols.clicked.connect(self._load_symbols)
        self.combo_timeframe = QComboBox()
        self.combo_timeframe.addItems(list(TIMEFRAMES))
        self.combo_timeframe.setCurrentText("1h")
        self.combo_timeframe.currentIndexChanged.connect(self._on_symbol_changed)
        sym_form.addRow("Символ:", self.combo_symbol)
        sym_form.addRow(self.btn_reload_symbols)
        sym_form.addRow("Таймфрейм:", self.combo_timeframe)
        layout.addWidget(sym_box)

        # --- Сайзинг ------------------------------------------------------ #
        size_box = QGroupBox("Размер позиции")
        size_form = QFormLayout(size_box)
        self.radio_notional = QRadioButton("Фиксированный ноционал (USDT)")
        self.radio_pct = QRadioButton("% от баланса")
        self.radio_notional.setChecked(True)
        self.radio_notional.toggled.connect(self._on_sizing_changed)
        self.spin_notional = QDoubleSpinBox(
            minimum=1, maximum=1e7, value=100, decimals=2
        )
        self.spin_notional.setSuffix(" USDT")
        self.spin_pct = QDoubleSpinBox(minimum=0.1, maximum=100, value=5, decimals=1)
        self.spin_pct.setSuffix(" %")
        size_form.addRow(self.radio_notional, self.spin_notional)
        size_form.addRow(self.radio_pct, self.spin_pct)
        self._on_sizing_changed()
        layout.addWidget(size_box)

        # --- Исполнение (экономия на комиссии) --------------------------- #
        exec_box = QGroupBox("Исполнение")
        exec_form = QFormLayout(exec_box)
        self.chk_entry_limit = QCheckBox("Лимитные входы (мейкер)")
        self.chk_entry_limit.setToolTip(
            "Вход не рыночным, а лимитным ордером (комиссия мейкера). Лимит может "
            "не залиться: если к следующему бару не исполнился — снимается как "
            "устаревший, позиция не открывается."
        )
        self.chk_post_only = QCheckBox("Только мейкер (post-only)")
        self.chk_post_only.setChecked(True)
        self.chk_post_only.setToolTip(
            "Биржа отклонит лимитный вход, если он исполнился бы тейкером — "
            "гарантия комиссии мейкера."
        )
        self.spin_limit_offset = QDoubleSpinBox(
            minimum=0.0, maximum=100.0, value=1.0, decimals=1
        )
        self.spin_limit_offset.setSuffix(" б.п.")
        self.spin_limit_offset.setToolTip(
            "Отступ лимит-цены в свою пользу от close (если стратегия не задала "
            "цену): buy ниже, sell выше. 1 б.п. = 0.01%."
        )
        self.chk_tp_maker = QCheckBox("Тейк-профит мейкером (лимит)")
        self.chk_tp_maker.setToolTip(
            "TP как reduce-only лимит (мейкер) вместо тейк-маркета. Стоп-лосс "
            "всегда остаётся стоп-маркетом — гарантия выхода важнее комиссии."
        )
        self.spin_max_slippage = QDoubleSpinBox(
            minimum=0.0, maximum=100.0, value=1.0, decimals=2
        )
        self.spin_max_slippage.setSuffix(" %")
        self.spin_max_slippage.setSpecialValueText("выкл")
        self.spin_max_slippage.setToolTip(
            "Максимальное проскальзывание для рыночного входа. Если к моменту "
            "исполнения цена ушла против входа дальше этого процента — ордер не "
            "выставляется (вход пропускается). 0 = выкл. Закрытия и защитные "
            "ордера никогда не блокируются."
        )
        self.chk_entry_limit.toggled.connect(self._on_execution_changed)
        exec_form.addRow(self.chk_entry_limit)
        exec_form.addRow(self.chk_post_only)
        exec_form.addRow("Отступ лимита:", self.spin_limit_offset)
        exec_form.addRow(self.chk_tp_maker)
        exec_form.addRow("Макс. проскальзывание:", self.spin_max_slippage)
        self._on_execution_changed()
        layout.addWidget(exec_box)

        # --- Предохранители («рубильник») -------------------------------- #
        risk_box = QGroupBox("Предохранители (рубильник)")
        risk_box.setToolTip("0 в поле = предохранитель выключен.")
        risk_form = QFormLayout(risk_box)
        self.spin_max_notional = QDoubleSpinBox(
            minimum=0.0, maximum=1e8, value=0.0, decimals=2
        )
        self.spin_max_notional.setSuffix(" USDT")
        self.spin_max_notional.setSpecialValueText("выкл")
        self.spin_max_notional.setToolTip(
            "Потолок ноционала одного входного ордера (защита от «толстого "
            "пальца»/бага сайзинга). Ордер сверх лимита отклоняется."
        )
        self.spin_max_orders_hour = QSpinBox(minimum=0, maximum=10000, value=0)
        self.spin_max_orders_hour.setSpecialValueText("выкл")
        self.spin_max_orders_hour.setToolTip(
            "Максимум входных ордеров за скользящий час (анти-зацикливание)."
        )
        self.spin_max_daily_loss = QDoubleSpinBox(
            minimum=0.0, maximum=100.0, value=0.0, decimals=1
        )
        self.spin_max_daily_loss.setSuffix(" %")
        self.spin_max_daily_loss.setSpecialValueText("выкл")
        self.spin_max_daily_loss.setToolTip(
            "Дневной стоп-убыток: при просадке эквити от старта суток (UTC) на N% "
            "позиция закрывается и торговля останавливается."
        )
        risk_form.addRow("Ноционал/ордер ≤", self.spin_max_notional)
        risk_form.addRow("Входов в час ≤", self.spin_max_orders_hour)
        risk_form.addRow("Убыток за сутки ≤", self.spin_max_daily_loss)
        layout.addWidget(risk_box)

        # --- Управление --------------------------------------------------- #
        self.chk_dry = QCheckBox("Сухой прогон (не отправлять ордера)")
        self.chk_dry.setToolTip(
            "Цикл работает как обычно, но ордера не отправляются на биржу — "
            "только показываются в журнале. Безопасная проверка стратегии."
        )
        layout.addWidget(self.chk_dry)

        self.btn_start = QPushButton("▶  Старт торговли")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._start_trading)
        self.btn_stop = QPushButton("■  Стоп")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("color: #ff5c5c;")
        self.btn_stop.clicked.connect(self._stop_trading)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addStretch(1)
        return panel

    def _build_bottom_tabs(self) -> QWidget:
        self.tabs = QTabWidget()

        # Позиция / Баланс
        state_w = QWidget()
        state_form = QFormLayout(state_w)
        self.lbl_balance = QLabel("—")
        self.lbl_position = QLabel("нет (плоско)")
        self.lbl_position.setWordWrap(True)
        for lbl in (self.lbl_balance, self.lbl_position):
            lbl.setStyleSheet("font-size: 13px;")
        state_form.addRow("Баланс:", self.lbl_balance)
        state_form.addRow("Позиция:", self.lbl_position)
        self.tabs.addTab(state_w, "Позиция / Баланс")

        # Ордера
        self.orders_log = QPlainTextEdit(readOnly=True)
        self.tabs.addTab(self.orders_log, "Ордера")

        # Журнал
        self.log = QPlainTextEdit(readOnly=True)
        self.tabs.addTab(self.log, "Журнал")
        return self.tabs

    # ================================================================== #
    # Реакции на изменение настроек
    # ================================================================== #
    def _current_exchange_id(self) -> str:
        return self.combo_exchange.currentData()

    def _current_mode(self) -> str:
        return MODE_DEMO if self.radio_demo.isChecked() else MODE_REAL

    def _on_exchange_changed(self) -> None:
        ex_id = self._current_exchange_id()
        meta = EXCHANGES[ex_id]
        self.lbl_ex_note.setText(meta.note)
        # Демо доступно только если биржа его поддерживает
        self.radio_demo.setEnabled(meta.supports_demo)
        if not meta.supports_demo and self.radio_demo.isChecked():
            self.radio_real.setChecked(True)
        self._connected = False
        self.btn_start.setEnabled(False)
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        is_real = self.radio_real.isChecked()
        if is_real:
            self.lbl_warn.setText(
                "⚠ РЕАЛЬНЫЙ СЧЁТ: ордера ставятся настоящими средствами. "
                "Движок торгует автоматически по сигналам стратегии."
            )
            self.lbl_warn.setStyleSheet(
                "color: #ff5c5c; font-weight: bold; padding: 4px;"
            )
        else:
            self.lbl_warn.setText("Демо/тестнет: средства виртуальные.")
            self.lbl_warn.setStyleSheet("color: #16c784; padding: 4px;")
        # Перестраиваем поля ключей под схему биржи и подгружаем сохранённые
        self._rebuild_cred_fields()
        self._connected = False
        self.btn_start.setEnabled(False)

    def _rebuild_cred_fields(self) -> None:
        """Пересобрать поля ключей под текущую биржу и заполнить сохранённым.

        У CEX это API Key + Secret, у DEX Lighter — приватный ключ API и индексы
        аккаунта (см. EXCHANGES[...].cred_fields).
        """
        while self.cred_form.rowCount():
            self.cred_form.removeRow(0)
        self._cred_edits.clear()
        meta = EXCHANGES[self._current_exchange_id()]
        creds = live.credentials.load(self._current_exchange_id(), self._current_mode())
        trading = (
            self._engine_thread is not None and self._engine_thread.isRunning()
        )
        for f in meta.cred_fields:
            edit = QLineEdit()
            if f.secret:
                edit.setEchoMode(QLineEdit.Password)
            edit.setPlaceholderText(f.placeholder or f.label)
            edit.setText(creds.get(f.key))
            edit.setEnabled(not trading)
            self.cred_form.addRow(f.label + ":", edit)
            self._cred_edits[f.key] = edit

    def _on_strategy_changed(self) -> None:
        cls = self._strategy_cls()
        self.param_table.set_strategy(cls)

    def _strategy_cls(self) -> Type[BaseStrategy]:
        return STRATEGIES[self.combo_strategy.currentText()]

    def _on_sizing_changed(self) -> None:
        notional = self.radio_notional.isChecked()
        self.spin_notional.setEnabled(notional)
        self.spin_pct.setEnabled(not notional)

    def _on_execution_changed(self) -> None:
        # post-only и отступ лимита имеют смысл только при лимитных входах
        limit = self.chk_entry_limit.isChecked()
        self.chk_post_only.setEnabled(limit)
        self.spin_limit_offset.setEnabled(limit)

    def _on_symbol_changed(self) -> None:
        if not self._connected:
            return
        symbol = self.combo_symbol.currentText().strip()
        if not symbol:
            return
        self._load_chart(symbol, self.combo_timeframe.currentText())

    # ================================================================== #
    # Подключение
    # ================================================================== #
    def _make_spec(self) -> ConnectionSpec:
        return ConnectionSpec(
            exchange_id=self._current_exchange_id(),
            mode=self._current_mode(),
            creds={k: e.text().strip() for k, e in self._cred_edits.items()},
        )

    def _test_connection(self) -> None:
        if self._busy_with_worker():
            return
        spec = self._make_spec()
        meta = EXCHANGES[spec.exchange_id]
        missing = [
            f.label for f in meta.cred_fields
            if f.required and not spec.creds.get(f.key, "").strip()
        ]
        if missing:
            QMessageBox.warning(
                self, "Ключи",
                "Заполните поля для подключения: " + ", ".join(missing) + ".",
            )
            return
        self._spec = spec
        self.btn_test.setEnabled(False)
        self.statusBar().showMessage("Проверка подключения…")
        self._log("Проверка подключения…")
        self._worker = ConnectionTestWorker(spec, parent=self)
        self._worker.finished_ok.connect(self._on_connected)
        self._worker.failed.connect(self._on_conn_failed)
        self._worker.start()

    def _on_connected(self, summary: dict) -> None:
        self.btn_test.setEnabled(True)
        self._connected = True
        bal = summary.get("balance_usdt", {})
        cur = bal.get("currency", "USDT")
        msg = (
            f"Подключено: {summary['exchange']} [{summary['mode']}] • "
            f"баланс {bal.get('total', 0):.2f} {cur} • "
            f"символов {summary.get('symbols_count', 0)}"
        )
        self.statusBar().showMessage(msg)
        self._log("✓ " + msg)
        self.lbl_balance.setText(self._fmt_balance(bal))
        self.combo_symbol.setEnabled(True)
        self.btn_reload_symbols.setEnabled(True)
        self.btn_start.setEnabled(True)
        self._load_symbols()

    @staticmethod
    def _fmt_balance(bal: dict) -> str:
        cur = bal.get("currency", "USDT")
        return f"{bal.get('total', 0):.2f} {cur} (свободно {bal.get('free', 0):.2f})"

    def _on_conn_failed(self, message: str) -> None:
        self.btn_test.setEnabled(True)
        self._connected = False
        self.statusBar().showMessage("Подключение не удалось")
        self._log("✗ Подключение не удалось: " + message)
        QMessageBox.critical(self, "Подключение", message)

    def _save_keys(self) -> None:
        creds = live.Credentials(
            values={k: e.text().strip() for k, e in self._cred_edits.items()}
        )
        path = live.credentials.save(
            self._current_exchange_id(), self._current_mode(), creds
        )
        self._log(f"Ключи сохранены: {path}")
        self.statusBar().showMessage(f"Ключи сохранены ({path})")

    # ================================================================== #
    # Символы и график
    # ================================================================== #
    def _load_symbols(self) -> None:
        if self._spec is None or self._busy_with_worker():
            return
        self.btn_reload_symbols.setEnabled(False)
        self.statusBar().showMessage("Загрузка списка символов…")
        self._worker = SymbolsWorker(self._spec, parent=self)
        self._worker.finished_ok.connect(self._on_symbols_loaded)
        self._worker.failed.connect(
            lambda m: (self._log("Символы: " + m),
                       self.btn_reload_symbols.setEnabled(True))
        )
        self._worker.start()

    def _on_symbols_loaded(self, symbols: list) -> None:
        self.btn_reload_symbols.setEnabled(True)
        current = self.combo_symbol.currentText()
        self.combo_symbol.blockSignals(True)
        self.combo_symbol.clear()
        self.combo_symbol.addItems(symbols)
        # популярный дефолт: точное совпадение BTC, иначе первый символ BTC/*
        chosen = -1
        for default in ("BTC/USDT:USDT", "BTC/USDT"):
            chosen = self.combo_symbol.findText(default)
            if chosen >= 0:
                break
        if chosen < 0:
            chosen = next(
                (i for i, s in enumerate(symbols) if s.startswith("BTC/")), -1
            )
        if chosen >= 0:
            self.combo_symbol.setCurrentIndex(chosen)
        if current:
            idx = self.combo_symbol.findText(current)
            if idx >= 0:
                self.combo_symbol.setCurrentIndex(idx)
        self.combo_symbol.blockSignals(False)
        self.statusBar().showMessage(f"Загружено символов: {len(symbols)}")
        self._on_symbol_changed()

    def _load_chart(self, symbol: str, timeframe: str) -> None:
        if self._spec is None or self._busy_with_worker():
            return
        self.statusBar().showMessage(f"Загрузка графика {symbol} ({timeframe})…")
        self._worker = OhlcvWorker(self._spec, symbol, timeframe, limit=500, parent=self)
        self._worker.finished_ok.connect(self._on_chart_loaded)
        self._worker.failed.connect(lambda m: self._log("График: " + m))
        self._worker.start()

    def _on_chart_loaded(self, df, symbol: str, timeframe: str) -> None:
        self._live_df = df
        self._live_indicators = None   # новый символ/таймфрейм -> прежние оверлеи неактуальны
        try:
            self._redraw_live_chart()
        except Exception as exc:  # noqa: BLE001
            self._log(f"Не удалось отрисовать график: {exc}")
            return
        self.statusBar().showMessage(
            f"{symbol} • {timeframe} • {len(df)} баров"
        )

    def _set_live_indicators(self, data: dict) -> None:
        """Строит series-оверлеи из события «indicators» движка и перерисовывает."""
        import pandas as pd

        plots = data.get("plots") or []
        index = data.get("index")
        inds = []
        if index is not None:
            for p in plots:
                vals = p["values"]
                n = min(len(vals), len(index))
                inds.append({
                    "name": p.get("name", ""),
                    "series": pd.Series(vals[:n], index=index[:n]),
                    "color": p.get("color"),
                    "overlay": p.get("overlay", True),
                    "width": p.get("width", 1.5),
                })
        self._live_indicators = inds or None
        self._redraw_live_chart()

    def _redraw_live_chart(self) -> None:
        """Перерисовывает свечи + текущие оверлеи индикаторов."""
        if self._live_df is None:
            return
        try:
            self.chart.show_data(self._live_df, self._live_indicators)
        except Exception:  # noqa: BLE001 - график не должен ронять торговлю
            pass

    # ================================================================== #
    # Запуск / остановка торговли
    # ================================================================== #
    def _sizing_config(self) -> SizingConfig:
        if self.radio_notional.isChecked():
            return SizingConfig(mode="notional", notional_usdt=self.spin_notional.value())
        return SizingConfig(mode="balance_pct", balance_pct=self.spin_pct.value())

    def _execution_config(self) -> ExecutionConfig:
        return ExecutionConfig(
            entry_limit=self.chk_entry_limit.isChecked(),
            post_only=self.chk_post_only.isChecked(),
            limit_offset_bps=self.spin_limit_offset.value(),
            tp_maker=self.chk_tp_maker.isChecked(),
            max_slippage_pct=self.spin_max_slippage.value(),
        )

    def _risk_limits(self) -> RiskLimits:
        return RiskLimits(
            max_notional_per_order=self.spin_max_notional.value(),
            max_orders_per_hour=self.spin_max_orders_hour.value(),
            max_daily_loss_pct=self.spin_max_daily_loss.value(),
        )

    def _start_trading(self) -> None:
        if not self._connected or self._spec is None:
            QMessageBox.warning(self, "Торговля", "Сначала проверьте подключение.")
            return
        if self._engine_thread is not None and self._engine_thread.isRunning():
            return
        symbol = self.combo_symbol.currentText().strip()
        if not symbol:
            QMessageBox.warning(self, "Торговля", "Выберите символ для торговли.")
            return
        try:
            params = self.param_table.fixed_values()
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры стратегии", str(exc))
            return

        dry = self.chk_dry.isChecked()
        is_real = self._current_mode() == MODE_REAL
        # Однократное осознанное подтверждение включения РЕАЛА (часть «тумблера»).
        if is_real and not dry and not self._real_acknowledged:
            ans = QMessageBox.warning(
                self, "Реальная торговля",
                f"Запуск АВТОМАТИЧЕСКОЙ торговли РЕАЛЬНЫМИ средствами на "
                f"{EXCHANGES[self._current_exchange_id()].label}.\n\n"
                f"Символ: {symbol}\nСтратегия: {self.combo_strategy.currentText()}\n"
                f"Плечо: {self.spin_leverage.value()}×\n\n"
                f"Движок будет сам ставить ордера. Продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
            self._real_acknowledged = True

        config = EngineConfig(
            symbol=symbol,
            timeframe=self.combo_timeframe.currentText(),
            strategy_cls=self._strategy_cls(),
            strategy_params=params,
            sizing=self._sizing_config(),
            execution=self._execution_config(),
            risk=self._risk_limits(),
            leverage=self.spin_leverage.value(),
            margin_mode=self.combo_margin.currentText(),
            dry_run=dry,
        )
        self._set_trading_ui(True)
        self.orders_log.clear()
        self._log("=== Запуск торгового цикла ===")
        self._engine_thread = TradingEngineThread(self._spec, config, parent=self)
        self._engine_thread.event.connect(self._on_engine_event)
        self._engine_thread.failed.connect(self._on_engine_failed)
        self._engine_thread.finished.connect(self._on_engine_finished)
        self._engine_thread.start()

    def _stop_trading(self) -> None:
        if self._engine_thread is not None and self._engine_thread.isRunning():
            self.btn_stop.setEnabled(False)
            self.statusBar().showMessage("Остановка торгового цикла…")
            self._log("Запрошена остановка…")
            self._engine_thread.stop_engine()

    def _on_engine_failed(self, message: str) -> None:
        self._log("✗ " + message)
        QMessageBox.critical(self, "Торговый цикл", message)

    def _on_engine_finished(self) -> None:
        self._set_trading_ui(False)
        self.statusBar().showMessage("Торговля остановлена")

    def _set_trading_ui(self, trading: bool) -> None:
        self.btn_start.setEnabled(not trading)
        self.btn_stop.setEnabled(trading)
        # На время торговли блокируем смену подключения/стратегии/символа
        for w in (
            self.combo_exchange, self.radio_demo, self.radio_real,
            self.btn_test, self.btn_save_keys,
            self.combo_strategy, self.param_table, self.combo_symbol,
            self.combo_timeframe, self.btn_reload_symbols, self.chk_dry,
            self.spin_leverage, self.combo_margin,
            self.chk_entry_limit, self.chk_post_only, self.spin_limit_offset,
            self.chk_tp_maker, self.spin_max_slippage,
            self.spin_max_notional, self.spin_max_orders_hour,
            self.spin_max_daily_loss,
        ):
            w.setEnabled(not trading)
        for edit in self._cred_edits.values():
            edit.setEnabled(not trading)

    # ================================================================== #
    # События движка
    # ================================================================== #
    def _on_engine_event(self, evt: EngineEvent) -> None:
        kind = evt.kind
        if kind == "candles":
            df = (evt.data or {}).get("df")
            if df is not None and not df.empty:
                self._live_df = df
                self._redraw_live_chart()
            return
        if kind == "indicators":
            self._set_live_indicators(evt.data or {})
            return
        if kind == "balance":
            self.lbl_balance.setText(self._fmt_balance(evt.data or {}))
            return
        if kind == "position":
            self.lbl_position.setText(evt.message.replace("Позиция: ", ""))
            return
        if kind == "order":
            self.orders_log.appendPlainText(evt.message)
            self._log(evt.message)
            return
        if kind == "status":
            running = (evt.data or {}).get("running")
            if running is False:
                self._set_trading_ui(False)
            self.statusBar().showMessage(evt.message)

        # info / signal / error / status -> журнал
        prefix = {"error": "✗ ", "signal": "• ", "status": "» "}.get(kind, "")
        self._log(prefix + evt.message)
        if kind == "error":
            self._highlight_last_log_red()
        if kind == "error" and (evt.data or {}).get("fatal"):
            self._stop_trading()

    # ================================================================== #
    # Журнал и служебное
    # ================================================================== #
    def _log(self, message: str) -> None:
        from datetime import datetime

        self.log.appendPlainText(f"[{datetime.now():%H:%M:%S}] {message}")

    def _highlight_last_log_red(self) -> None:
        # лёгкая подсветка: пробрасываем цвет для последней строки через формат
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log.setTextCursor(cursor)

    def _busy_with_worker(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # ================================================================== #
    def closeEvent(self, event) -> None:
        # Останавливаем торговый цикл, чтобы не оставить открытый поток/ордера без присмотра
        if self._engine_thread is not None and self._engine_thread.isRunning():
            ans = QMessageBox.question(
                self, "Закрыть окно",
                "Идёт торговля. Остановить и закрыть окно?\n"
                "(Открытые позиции на бирже НЕ закрываются автоматически.)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                event.ignore()
                return
            self._engine_thread.stop_engine()
            self._engine_thread.wait(5000)
        event.accept()
