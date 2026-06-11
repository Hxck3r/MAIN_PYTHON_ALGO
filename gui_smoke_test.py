"""Smoke-тест десктопного GUI (без показа окна, offscreen).

Запуск:
    python gui_smoke_test.py

Прогоняет основные сценарии: создание окна, бэктест, оптимизация,
применение параметров из таблицы результатов. Запускайте после правок
в gui/, чтобы быстро убедиться, что интерфейс не сломан.
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.param_table import COL_OPT, COL_START, COL_STEP, COL_STOP


def wait_worker(app: QApplication, window: MainWindow, timeout_s: float = 120) -> None:
    """Ждёт завершения активного воркера, прокачивая очередь событий."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if window._worker is not None and not window._worker.isRunning():
            for _ in range(30):
                app.processEvents()
            # завершившийся воркер мог запустить следующий (бэктест лучших)
            if window._worker is not None and window._worker.isRunning():
                continue
            return
        time.sleep(0.02)
    raise TimeoutError("воркер не завершился за отведённое время")


def main() -> None:
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    print("OK: окно создано, синтетика загружена, свечи отрисованы")

    window._run_backtest()
    wait_worker(app, window)
    assert window.metrics_table.rowCount() > 0, "таблица метрик пуста"
    assert window.trades_table.rowCount() > 0, "таблица сделок пуста"
    print(f"OK: бэктест, сделок в таблице: {window.trades_table.rowCount()}")

    # два параметра с компактной сеткой -- проверяем и теплокарту (2D)
    for row, (start, step, stop) in ((0, (5, 5, 20)), (1, (30, 10, 60))):
        window.param_table.item(row, COL_OPT).setCheckState(Qt.Checked)
        window.param_table.item(row, COL_START).setText(str(start))
        window.param_table.item(row, COL_STEP).setText(str(step))
        window.param_table.item(row, COL_STOP).setText(str(stop))
    # счётчик шагов: 4 значения (5,10,15,20) × 4 (30,40,50,60) = 16 комбинаций
    total, per = window.param_table.grid_summary()
    assert total == 16, f"ожидалось 16 комбинаций, получено {total} ({per})"
    assert "16" in window.lbl_grid.text(), f"в подписи нет счётчика: {window.lbl_grid.text()!r}"
    print(f"OK: счётчик шагов: {window.lbl_grid.text()}")

    window._run_optimization()
    wait_worker(app, window)
    assert window.opt_table.rowCount() > 0, "таблица оптимизации пуста"
    print(f"OK: оптимизация, испытаний: {window.opt_table.rowCount()}")

    charts = window.opt_charts
    assert charts.combo_x.count() == 2, "в комбобоксах не два параметра"
    assert charts._hover_info is not None, "теплокарта не построена"
    assert len(charts.hist_plot.listDataItems()) > 0, "гистограмма пуста"
    charts.combo_y.setCurrentIndex(charts.combo_x.currentIndex())  # режим 1D
    assert len(charts.heat_plot.listDataItems()) > 0, "кривая 1D пуста"
    charts.combo_agg.setCurrentIndex(1)  # свёртка «максимум»
    print("OK: теплокарта, гистограмма и кривая 1D построены")

    window._apply_and_backtest({"fast_period": 7})
    wait_worker(app, window)
    print("OK: применение параметров из таблицы оптимизации")

    # --- Walk-Forward: синтетика ~208 дней -> окна 60/20 дней ----------- #
    window.spin_wfa_is.setValue(60)
    window.spin_wfa_oos.setValue(20)
    assert "Окон" in window.lbl_wfa.text(), \
        f"счётчик окон не обновился: {window.lbl_wfa.text()!r}"
    print(f"OK: счётчик WFA: {window.lbl_wfa.text()}")

    window._run_wfa()
    wait_worker(app, window)
    panel = window.wfa_panel
    assert panel.summary.rowCount() > 0, "сводка WFA пуста"
    assert panel.windows_table.rowCount() > 1, "таблица окон WFA пуста"
    assert len(panel.equity_plot.listDataItems()) > 0, "сшитая эквити пуста"
    assert panel.combo_param.count() == 2, "нет эволюции параметров"
    assert len(panel.param_plot.listDataItems()) > 0, "график параметра пуст"
    print(f"OK: WFA, окон в таблице: {panel.windows_table.rowCount()}")

    # двойной клик по окну -> параметры применяются и запускается бэктест
    panel.windows_table._on_double_click(panel.windows_table.item(0, 0))
    wait_worker(app, window)
    print("OK: применение параметров из окна WFA")

    # --- Монте-Карло: все методы по сделкам последнего бэктеста --------- #
    assert window._last_result is not None, "нет сохранённого бэктеста для МК"
    mc = window.mc_panel
    mc.spin_sims.setValue(500)
    for i in range(mc.combo_method.count()):
        mc.combo_method.setCurrentIndex(i)
        window._run_monte_carlo(mc.make_config())
        wait_worker(app, window)
        assert mc.summary.rowCount() > 0, "сводка Монте-Карло пуста"
        assert len(mc.fan_plot.listDataItems()) > 0, "веер эквити пуст"
        assert len(mc.ret_hist.items) > 0, "гистограмма доходностей пуста"
        assert len(mc.dd_hist.items) > 0, "гистограмма просадок пуста"
        assert len(mc.cdf_plot.listDataItems()) > 0, "CDF пуста"
        print(f"OK: Монте-Карло «{mc.combo_method.currentText()}», "
              f"строк сводки: {mc.summary.rowCount()}")

    # кнопка слева запускает МК и открывает вкладку
    window._run_monte_carlo_clicked()
    wait_worker(app, window)
    assert window.tabs.currentWidget() is mc, "вкладка Монте-Карло не активна"
    print("OK: кнопка «Монте-Карло» на панели слева")

    print("ВСЕ ПРОВЕРКИ ДЕСКТОПНОГО GUI ПРОЙДЕНЫ")


if __name__ == "__main__":
    main()
