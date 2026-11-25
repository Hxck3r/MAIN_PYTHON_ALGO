"""Окно «Тиковые данные»: загрузка архивов бирж и сборка футпринта.

Здесь нет ни одной строчки сетевой логики: диалог собирает
:class:`marketdata.archives.DatasetRequest` из полей формы, отдаёт его
``ArchiveDownloadWorker`` и показывает прогресс. Вся загрузка, сборка футпринта
и свечей выполняется в ``marketdata.archives`` — тот же код, что и у команды
``algo-trades archive`` и у скрипта ``fetch_ticks.py``.
"""
from __future__ import annotations

from datetime import date, timedelta

from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from marketdata.archives import (
    DATASETS,
    DEFAULT_DATA_DIR,
    EXCHANGES,
    KLINE_INTERVALS,
    DatasetRequest,
    DatasetResult,
)

from .workers import ArchiveDownloadWorker

#: (подпись, pandas-частота, интервал Binance) — футпринт и свечи обязаны
#: совпадать по бару, иначе признаки не совместятся с барами бэктеста.
TIMEFRAMES = (
    ("1 минута", "1min", "1m"),
    ("5 минут", "5min", "5m"),
    ("15 минут", "15min", "15m"),
    ("1 час", "1h", "1h"),
    ("4 часа", "4h", "4h"),
    ("1 день", "1D", "1d"),
)

EXCHANGE_LABELS = {
    "binance": "Binance (USD-M)",
    "bybit": "Bybit (linear)",
    "okx": "OKX (swap)",
}


class TickDataDialog(QDialog):
    """Выбор бирж/инструмента/периода -> скачивание -> тики, футпринт и свечи.

    ``loaded`` испускается с :class:`DatasetResult`, когда данные готовы и
    пользователь просил сразу открыть их в тестере.
    """

    loaded = Signal(object)   # DatasetResult

    def __init__(self, parent=None, *, data_dir: str = DEFAULT_DATA_DIR,
                 symbol: str = "BTC/USDT:USDT") -> None:
        super().__init__(parent)
        self.setWindowTitle("Тиковые данные: архивы бирж -> футпринт")
        self.setMinimumWidth(620)
        self._worker: ArchiveDownloadWorker | None = None
        self._result: DatasetResult | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_form(data_dir, symbol))

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(150)
        self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log, stretch=1)

        buttons = QDialogButtonBox()
        self.btn_start = QPushButton("Скачать")
        self.btn_start.setDefault(True)
        self.btn_start.clicked.connect(self.start_download)
        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_download)
        self.btn_load = QPushButton("Открыть в тестере")
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self._emit_loaded)
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.reject)
        for button in (self.btn_start, self.btn_stop, self.btn_load):
            buttons.addButton(button, QDialogButtonBox.ActionRole)
        buttons.addButton(self.btn_close, QDialogButtonBox.RejectRole)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ #
    # Форма
    # ------------------------------------------------------------------ #
    def _build_form(self, data_dir: str, symbol: str) -> QGroupBox:
        box = QGroupBox("Что скачать")
        form = QFormLayout(box)

        self.exchange_checks: dict[str, QCheckBox] = {}
        exchanges_row = QHBoxLayout()
        for name in EXCHANGES:
            check = QCheckBox(EXCHANGE_LABELS.get(name, name))
            check.setChecked(name == "binance")
            check.toggled.connect(self._on_exchange_toggled)
            self.exchange_checks[name] = check
            exchanges_row.addWidget(check)
        form.addRow("Биржи:", exchanges_row)

        self.txt_symbol = QLineEdit(symbol)
        self.txt_symbol.setToolTip("Формат CCXT: BASE/QUOTE:SETTLE, например BTC/USDT:USDT")
        form.addRow("Инструмент:", self.txt_symbol)

        self.combo_dataset = QComboBox()
        for name in DATASETS:
            self.combo_dataset.addItem(
                "aggTrades (рекомендуется)" if name == "aggTrades" else "trades", name
            )
        self.combo_dataset.setToolTip(
            "Binance отдаёт оба потока; aggTrades — та же агрессия и объём, но втрое меньше"
        )
        form.addRow("Данные Binance:", self.combo_dataset)

        today = date.today()
        self.date_since = QDateEdit(calendarPopup=True)
        self.date_until = QDateEdit(calendarPopup=True)
        for widget, value in (
            (self.date_since, today - timedelta(days=7)),
            # Архивы публикуются до T-1, поэтому «по» по умолчанию — сегодня:
            # верхняя граница не включается, и запрос покрывает вчерашний день.
            (self.date_until, today),
        ):
            widget.setDate(QDate(value.year, value.month, value.day))
        form.addRow("С (UTC):", self.date_since)
        form.addRow("По (UTC, не включая):", self.date_until)

        self.combo_timeframe = QComboBox()
        for label, pandas_freq, _interval in TIMEFRAMES:
            self.combo_timeframe.addItem(label, pandas_freq)
        form.addRow("Таймфрейм баров:", self.combo_timeframe)

        self.spin_price_step = QDoubleSpinBox()
        self.spin_price_step.setRange(0.01, 100_000.0)
        self.spin_price_step.setDecimals(2)
        self.spin_price_step.setValue(10.0)
        self.spin_price_step.setToolTip(
            "Размер ценового уровня футпринта в единицах котировки (для BTC — доллары)"
        )
        form.addRow("Шаг цены:", self.spin_price_step)

        dir_row = QHBoxLayout()
        self.txt_data_dir = QLineEdit(data_dir)
        btn_browse = QPushButton("...")
        btn_browse.setMaximumWidth(40)
        btn_browse.clicked.connect(self._choose_dir)
        dir_row.addWidget(self.txt_data_dir)
        dir_row.addWidget(btn_browse)
        form.addRow("Хранилище:", dir_row)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 16)
        self.spin_workers.setValue(4)
        self.spin_workers.setToolTip("Параллельных загрузок; 4 хватает, больше — вежливее к бирже")
        form.addRow("Потоков:", self.spin_workers)

        self.chk_footprint = QCheckBox("Собрать футпринт Parquet")
        self.chk_footprint.setChecked(True)
        form.addRow("", self.chk_footprint)

        self.chk_candles = QCheckBox("Скачать свечи Binance (нужны для бэктеста)")
        self.chk_candles.setChecked(True)
        self.chk_candles.setToolTip(
            "Бары для тестера. Публикует только Binance — у Bybit и OKX "
            "bulk-архива свечей нет"
        )
        form.addRow("", self.chk_candles)

        self.chk_open = QCheckBox("Открыть результат в тестере после загрузки")
        self.chk_open.setChecked(True)
        form.addRow("", self.chk_open)

        hint = QLabel(
            "Данные пишутся в каноническое хранилище (Parquet-партиции по дням), "
            "поэтому те же дни повторно не скачиваются. Уже загруженное можно "
            "пересобрать командой algo-trades footprint."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
        form.addRow("", hint)
        return box

    def _on_exchange_toggled(self) -> None:
        """Свечи есть только у Binance — подсказываем это галочкой."""
        if "binance" not in self.selected_exchanges():
            self.chk_candles.setChecked(False)
            self.chk_candles.setEnabled(False)
        else:
            self.chk_candles.setEnabled(True)

    def _choose_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка хранилища", self.txt_data_dir.text())
        if path:
            self.txt_data_dir.setText(path)

    # ------------------------------------------------------------------ #
    # Запрос
    # ------------------------------------------------------------------ #
    def selected_exchanges(self) -> list[str]:
        return [name for name, check in self.exchange_checks.items() if check.isChecked()]

    def build_request(self) -> DatasetRequest:
        label, pandas_freq, interval = TIMEFRAMES[self.combo_timeframe.currentIndex()]
        return DatasetRequest(
            exchanges=tuple(self.selected_exchanges()),
            symbol=self.txt_symbol.text().strip(),
            since=self.date_since.date().toString("yyyy-MM-dd"),
            until=self.date_until.date().toString("yyyy-MM-dd"),
            dataset=self.combo_dataset.currentData(),
            data_dir=self.txt_data_dir.text().strip() or DEFAULT_DATA_DIR,
            workers=self.spin_workers.value(),
            with_footprint=self.chk_footprint.isChecked(),
            price_step=self.spin_price_step.value(),
            timeframe=pandas_freq,
            with_candles=self.chk_candles.isChecked(),
            candles_interval=interval,
        )

    # ------------------------------------------------------------------ #
    # Запуск
    # ------------------------------------------------------------------ #
    def start_download(self) -> None:
        try:
            request = self.build_request()
            request.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте параметры", str(exc))
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self.log.clear()
        self._append(f"Запрос: {', '.join(request.exchanges)} {request.symbol}, "
                     f"{request.since} .. {request.until}")
        if request.with_footprint:
            self._append(f"Футпринт: {request.timeframe} x {request.price_step:g}")
        self._set_running(True)
        self._worker = ArchiveDownloadWorker(request, self)
        self._worker.log.connect(self._append)
        self._worker.status.connect(lambda text: self.progress.setFormat(f"{text}  %p%"))
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.start()

    def stop_download(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._append("Останавливаю: текущий файл дописывается, дальше продолжится с него")

    def wait_for_finish(self, timeout_ms: int = 10_000) -> bool:
        """Дождаться окончания загрузки (для тестов и закрытия окна)."""
        if self._worker is None:
            return True
        return bool(self._worker.wait(timeout_ms))

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat(f"%p%  ({done}/{total})")

    def _on_finished(self, result: DatasetResult) -> None:
        self._result = result
        self._set_running(False)
        for failure in result.failures:
            self._append(f"! {failure}")
        self._append("Готово: " + result.summary())
        if result.footprint_path:
            self._append(f"Футпринт: {result.footprint_path}")
        if result.candles_path:
            self._append(f"Свечи: {result.candles_path}")
        self.btn_load.setEnabled(result.footprint is not None or result.candles is not None)
        if result.footprint is None and result.candles is None:
            QMessageBox.warning(
                self, "Нет данных",
                "Архивы скачаны, но футпринт/свечи не собраны.\n" + "\n".join(result.failures),
            )
            return
        if self.chk_open.isChecked():
            self._emit_loaded()

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._append(f"Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка загрузки", message)

    def _on_stopped(self) -> None:
        self._set_running(False)
        self._append("Загрузка остановлена. Уже скачанные дни сохранены — "
                     "повторный запуск продолжит с них.")

    def _emit_loaded(self) -> None:
        if self._result is not None:
            self.loaded.emit(self._result)
            self.accept()

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_close.setEnabled(not running)
        self.progress.setVisible(running or self.progress.value() > 0)
        if running:
            self.progress.setRange(0, 0)      # «бегунок», пока нет плана файлов
            self.progress.setFormat("подготовка...")
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)

    def _append(self, message: str) -> None:
        self.log.appendPlainText(message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._worker is not None and self._worker.isRunning():
            answer = QMessageBox.question(
                self, "Загрузка идёт",
                "Остановить загрузку и закрыть окно?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._worker.stop()
            self._worker.wait(10_000)
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Enter запускает загрузку, но не закрывает окно (как в диалогах MT5)."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.start_download()
            return
        super().keyPressEvent(event)


__all__ = ["TickDataDialog", "TIMEFRAMES", "KLINE_INTERVALS"]
