"""Offscreen smoke-тест тикового слоя в GUI: footprint -> признаки -> прогон.

Открывает реальное окно в режиме QT_QPA_PLATFORM=offscreen, подсовывает
footprint Parquet (как это делает кнопка «Открыть footprint Parquet...»)
и проверяет, что признаки ордерфлоу дошли до движка, а теплокарта нарисована.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
pytest.importorskip("PySide6")
pytest.importorskip("pyarrow")

from marketdata import aggregate_footprint, normalize_trades  # noqa: E402

FOOTPRINT_STRATEGY = "Footprint delta (event-driven)"

#: QApplication создаётся один раз и живёт до конца сессии pytest: второй
#: singleton поднять нельзя, а сборка мусора уронила бы первый.
_APP = None


def _synthetic_trades(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Trades that print inside the given bars, with a rising delta."""
    rows = []
    for position, bar in enumerate(index):
        ms = int(bar.timestamp() * 1000)
        for offset in (0, 500):
            side = "buy" if position % 3 else "sell"
            rows.append({
                "id": f"{position}-{offset}",
                "timestamp": ms + offset,
                "price": 100.0 + position,
                "amount": 0.5 if side == "buy" else 0.4,
                "side": side,
            })
    return normalize_trades(rows, "binance")


def _wait(app, window, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if window._worker is not None and not window._worker.isRunning():
            for _ in range(30):
                app.processEvents()
            return
        time.sleep(0.02)
    raise TimeoutError("воркер не завершился за отведённое время")


def _wait_footprint(app, window, timeout_s: float = 120.0) -> None:
    """Дождаться фоновой загрузки footprint (read_parquet + признаки)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        worker = window._footprint_worker
        if worker is None or not worker.isRunning():
            for _ in range(30):
                app.processEvents()
            return
        time.sleep(0.02)
    raise TimeoutError("загрузка footprint не завершилась за отведённое время")


@pytest.fixture
def window(monkeypatch):
    """Окно в offscreen-режиме; модальные диалоги собираются в ``window.dialogs``."""
    global _APP
    from PySide6.QtWidgets import QApplication

    from gui import main_window as main_window_module
    from gui.main_window import MainWindow

    _APP = QApplication.instance() or QApplication([])
    dialogs: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox, "critical",
        lambda *args, **kwargs: dialogs.append(str(args[-1])),
    )
    win = MainWindow()
    win.show()
    _APP.processEvents()
    win.dialogs = dialogs
    yield win
    # Закрываем через finplot: он чистит свои глобальные ссылки на панели.
    # Прямой win.close() удаляет C++-объекты, оставляя их в fplt.windows --
    # следующее окно тогда создаётся уже поверх удалённых ViewBox.
    import finplot as fplt

    fplt.close()


def test_footprint_loads_drives_strategy_and_draws_heatmap(window, tmp_path) -> None:
    index = window._data.index[:60]
    footprint = aggregate_footprint(
        _synthetic_trades(index), timeframe="1h", price_step=1.0
    )
    path = tmp_path / "footprint.parquet"
    footprint.to_parquet(path)

    assert window.load_marketdata_files([str(path)]) is True
    assert window._footprint is not None
    assert "rolling_delta" in window._market_features.columns
    assert window._market_features.index.is_monotonic_increasing

    window.combo_strategy.setCurrentIndex(
        window.combo_strategy.findText(FOOTPRINT_STRATEGY)
    )
    window._run_backtest()
    _wait(_APP, window)

    result = window._last_result
    assert result is not None, "бэктест не выполнился"
    assert "rolling_delta" in result.market_features.columns, \
        "признаки ордерфлоу не дошли до результата"
    # Теплокарта рисуется прямоугольниками поверх панели цены.
    from pyqtgraph import QtWidgets

    rectangles = [
        item for item in window.chart.ax_price.items
        if isinstance(item, QtWidgets.QGraphicsRectItem)
    ]
    assert rectangles, "футпринт не отрисован на графике"


def test_large_footprint_is_not_drawn_after_backtest(window) -> None:
    """Выше предела ячеек теплокарта не рисуется -- иначе окно висит минуты.

    На каждую ячейку создаётся QGraphicsRectItem: 20 000 ячеек = ~2.7 с, а
    1-минутный футпринт за месяцы (753k) -- минуты в главном потоке.
    """
    import numpy as np
    from pyqtgraph import QtWidgets

    from backtester import BacktestEngine
    from gui.chart_widget import MAX_FOOTPRINT_CELLS
    from strategies import STRATEGIES

    bars = window._data.index[:60]
    n_levels = MAX_FOOTPRINT_CELLS // len(bars) + 2       # заведомо выше предела
    footprint = pd.DataFrame(
        {"buy": 1.0, "sell": 0.5, "delta": 0.5},
        index=pd.MultiIndex.from_arrays(
            [np.repeat(bars, n_levels), np.tile(np.arange(n_levels, dtype=float), len(bars))],
            names=["bar_start", "price_level"],
        ),
    )
    assert len(footprint) > MAX_FOOTPRINT_CELLS

    result = BacktestEngine(
        window._data, STRATEGIES["SMA Crossover (пересечение скользящих)"],
        strategy_params={},
        footprint=footprint,
    ).run()
    window.chart.show_result(result)

    rectangles = [
        item for item in window.chart.ax_price.items
        if isinstance(item, QtWidgets.QGraphicsRectItem)
    ]
    assert not rectangles, "футпринт выше предела не должен рисоваться"


def test_async_footprint_load_does_not_block_the_caller(window, tmp_path) -> None:
    """Запуск чтения возвращается сразу: работа идёт в фоновом потоке."""
    index = window._data.index[:60]
    footprint = aggregate_footprint(
        _synthetic_trades(index), timeframe="1h", price_step=1.0
    )
    path = tmp_path / "footprint.parquet"
    footprint.to_parquet(path)

    assert window.load_marketdata_files_async([str(path)]) is True
    assert window._footprint_worker is not None, "работа должна идти в отдельном потоке"
    assert window._footprint is None, "данные приходят сигналом, а не синхронно"

    _wait_footprint(_APP, window)
    assert window._footprint is not None
    assert "rolling_delta" in window._market_features.columns


def test_broken_footprint_file_leaves_no_partial_state(window, tmp_path) -> None:
    broken = tmp_path / "broken.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(broken)

    assert window.load_marketdata_files([str(broken)]) is False
    assert window._footprint is None and window._market_features is None
    assert window.dialogs and "bar_start" in window.dialogs[0]


def test_engine_accepts_footprint_and_bar_features() -> None:
    """Footprint и bar-признаки доходят до BacktestResult без lookahead."""
    from backtester import BacktestEngine
    from marketdata import bar_features_from_footprint
    from strategies import STRATEGIES

    index = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
    footprint = aggregate_footprint(
        _synthetic_trades(index), timeframe="1h", price_step=1.0
    )
    candles = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 5.0,
    }, index=index)
    result = BacktestEngine(
        candles, STRATEGIES[FOOTPRINT_STRATEGY],
        market_features=bar_features_from_footprint(footprint),
        footprint=footprint,
    ).run()
    assert result.footprint is footprint
    assert "volume_imbalance" in result.market_features.columns


# --------------------------------------------------------------------------- #
# Кнопка «Скачать тиковые данные (архивы)»: диалог и воркер
# --------------------------------------------------------------------------- #
IMBALANCE_STRATEGY = "Footprint imbalance + absorption (order flow)"


def _qt_app():
    """QApplication — singleton, поэтому создаём один раз на всю сессию."""
    global _APP
    from PySide6.QtWidgets import QApplication

    _APP = QApplication.instance() or QApplication([])
    return _APP


def _synth_candles(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = [100.0 + position for position in range(len(index))]
    return pd.DataFrame({
        "open": close,
        "high": [value + 1.0 for value in close],
        "low": [value - 1.0 for value in close],
        "close": close,
        "volume": [10.0] * len(index),
    }, index=index)


def _dataset_result(tmp_path):
    """(DatasetResult, признаки) — ровно то, что отдаёт воркер, вместе с файлами."""
    from marketdata import bar_features_from_footprint
    from marketdata.archives import DatasetResult

    index = pd.date_range("2024-01-02", periods=60, freq="1h", tz="UTC")
    footprint = aggregate_footprint(
        _synthetic_trades(index), timeframe="1h", price_step=1.0
    )
    candles = _synth_candles(index)
    footprint_path = tmp_path / "BTC_USDT_USDT_1h_step1.parquet"
    candles_path = tmp_path / "BTCUSDT-1h.csv"
    footprint.to_parquet(footprint_path)
    candles.to_csv(candles_path)
    return DatasetResult(
        store=str(tmp_path),
        rows={"binance": 120, "bybit": 118},
        files={"binance": 1, "bybit": 1},
        footprint=footprint,
        footprint_path=str(footprint_path),
        candles=candles,
        candles_path=str(candles_path),
    ), bar_features_from_footprint(footprint)


def test_tick_dialog_keeps_footprint_timeframe_and_candles_in_step(tmp_path) -> None:
    """Таймфрейм футпринта и интервал свечей берутся из одной пары."""
    _qt_app()
    from gui.tick_data import TIMEFRAMES, TickDataDialog

    dialog = TickDataDialog(data_dir=str(tmp_path))
    assert dialog.selected_exchanges() == ["binance"]
    assert dialog.build_request().candles_interval == "1m"

    for position, (_label, pandas_freq, interval) in enumerate(TIMEFRAMES):
        dialog.combo_timeframe.setCurrentIndex(position)
        request = dialog.build_request()
        assert request.timeframe == pandas_freq
        assert request.candles_interval == interval
        request.validate()          # рассогласованная пара упала бы здесь
        assert dialog.spin_price_step.value() == request.price_step

    # Свечи публикует только Binance: без него галочка гаснет.
    dialog.exchange_checks["bybit"].setChecked(True)
    dialog.exchange_checks["binance"].setChecked(False)
    assert dialog.chk_candles.isChecked() is False
    assert dialog.chk_candles.isEnabled() is False
    assert dialog.build_request().with_candles is False

    dialog.exchange_checks["binance"].setChecked(True)
    assert dialog.chk_candles.isEnabled() is True
    dialog.close()


def test_tick_dialog_refuses_an_empty_range_without_starting_worker(
    tmp_path, monkeypatch
) -> None:
    """Диалог проверяет запрос до запуска потока и объясняет отказ."""
    app = _qt_app()
    from PySide6.QtCore import QDate

    from gui import tick_data as tick_data_module
    from gui.tick_data import TickDataDialog

    warnings: list[str] = []
    monkeypatch.setattr(
        tick_data_module.QMessageBox, "warning",
        lambda *args, **kwargs: warnings.append(str(args[-1])),
    )
    dialog = TickDataDialog(data_dir=str(tmp_path))
    dialog.date_since.setDate(QDate(2024, 3, 10))
    dialog.date_until.setDate(QDate(2024, 3, 10))
    dialog.start_download()
    app.processEvents()

    assert dialog._worker is None, "поток не должен запускаться на плохом запросе"
    assert warnings and "раньше" in warnings[-1]
    assert dialog.btn_start.isEnabled() and not dialog.btn_stop.isEnabled()
    dialog.close()


def test_tick_dialog_reports_worker_progress_and_loaded_result(
    window, tmp_path, monkeypatch
) -> None:
    """Воркер гоняет ``DatasetRequest.download`` и отдаёт результат в тестер."""
    app = _qt_app()
    from PySide6.QtCore import QDate

    from gui.tick_data import TickDataDialog
    from marketdata.archives import ArchiveProgress, DatasetRequest

    result, _features = _dataset_result(tmp_path)
    seen: list[tuple[int, int]] = []

    def fake_download(self, progress=None, should_stop=None):  # noqa: ANN001
        progress(ArchiveProgress(event="plan", exchange="binance", done=0, total=2))
        for day in ("2024-01-02", "2024-01-03"):
            assert should_stop() is False
            progress(ArchiveProgress(event="file", exchange="binance",
                                     label=day, rows=60))
        progress(ArchiveProgress(event="footprint", label="1h", rows=len(result.footprint)))
        return result

    monkeypatch.setattr(DatasetRequest, "download", fake_download)
    dialog = TickDataDialog(data_dir=str(tmp_path), symbol="BTC/USDT:USDT")
    dialog.date_since.setDate(QDate(2024, 1, 2))
    dialog.date_until.setDate(QDate(2024, 1, 4))
    dialog.chk_open.setChecked(False)      # не открывать окно модально
    emitted: list[object] = []
    dialog.loaded.connect(emitted.append)
    # ``dialog.progress`` — это QProgressBar, поэтому слушаем сам обработчик.
    handler = dialog._on_progress
    dialog._on_progress = lambda done, total: (seen.append((done, total)),
                                               handler(done, total))[0]

    dialog.start_download()
    assert dialog.btn_stop.isEnabled() and not dialog.btn_start.isEnabled()
    assert dialog.wait_for_finish(10_000) is True
    for _ in range(50):
        app.processEvents()

    assert seen[-1] == (2, 2), f"прогресс файлов не дошёл: {seen}"
    text = dialog.log.toPlainText()
    assert "Готово" in text
    assert "2 file(s) to fetch" in text and "2024-01-02: 60 rows" in text, text
    assert dialog.btn_load.isEnabled(), "результат есть — можно открыть в тестере"
    assert dialog.btn_start.isEnabled() and not dialog.btn_stop.isEnabled()
    assert not emitted, "при снятой галочке результат не открывается сам"

    dialog._emit_loaded()
    assert emitted == [dialog._result]

    # Сводка того же объекта попадает и в журнал главного окна.
    window._apply_dataset(result)
    _wait_footprint(app, window)          # футпринт читается в фоне
    assert window._footprint is not None
    assert "poc_price" in window._market_features.columns
    assert len(window._data) == len(result.candles)


def test_downloaded_dataset_opens_the_order_flow_strategy(window, tmp_path) -> None:
    """Скачанные свечи + футпринт запускают ордерфлоу-стратегию из GUI."""
    result, _features = _dataset_result(tmp_path)
    window._apply_dataset(result)
    _wait_footprint(_APP, window)         # read_parquet + признаки -- в фоне

    from pyqtgraph import QtWidgets

    rectangles = [
        item for item in window.chart.ax_price.items
        if isinstance(item, QtWidgets.QGraphicsRectItem)
    ]
    assert rectangles, "теплокарта скачанного футпринта не отрисована"

    window.combo_strategy.setCurrentIndex(
        window.combo_strategy.findText(IMBALANCE_STRATEGY)
    )
    assert window.combo_strategy.currentText() == IMBALANCE_STRATEGY
    window._run_backtest()
    _wait(_APP, window)

    result = window._last_result
    assert result is not None, "бэктест на скачанных данных не выполнился"
    assert "poc_price" in result.market_features.columns
    assert "delta_zscore" in result.market_features.columns
    assert result.footprint is window._footprint
