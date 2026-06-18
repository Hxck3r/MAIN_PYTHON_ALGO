"""Таблицы результатов: метрики, сделки, испытания оптимизации.

Все таблицы только для чтения, с сортировкой по клику на заголовок.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from backtester import BacktestResult, OptimizationResult

# Подписи метрик на русском (порядок = порядок строк в таблице)
METRIC_LABELS: dict[str, str] = {
    "initial_equity": "Начальный капитал",
    "final_equity": "Конечный капитал",
    "net_profit": "Чистая прибыль",
    "total_return_pct": "Доходность, %",
    "cagr_pct": "CAGR (годовых), %",
    "max_drawdown": "Макс. просадка",
    "max_drawdown_pct": "Макс. просадка, %",
    "max_drawdown_duration": "Длительность просадки, баров",
    "sharpe_ratio": "Коэф. Шарпа",
    "sortino_ratio": "Коэф. Сортино",
    "recovery_factor": "Фактор восстановления",
    "total_trades": "Всего сделок",
    "win_rate_pct": "Прибыльных, %",
    "profit_factor": "Профит-фактор",
    "expectancy": "Матожидание на сделку",
    "avg_win": "Средняя прибыльная",
    "avg_loss": "Средняя убыточная",
    "largest_win": "Лучшая сделка",
    "largest_loss": "Худшая сделка",
    "max_consecutive_wins": "Макс. серия прибыльных",
    "max_consecutive_losses": "Макс. серия убыточных",
    "avg_trade_duration_bars": "Средняя длительность, баров",
    "total_commission": "Комиссии всего",
    "complex_criterion": "Комплексный критерий",
}

# Направление «лучше»: для каких метрик РОСТ значения — это ухудшение
# (просадка, волатильность, серии убытков, комиссии). Для остальных по
# умолчанию рост = улучшение. Перечисленные в NEUTRAL метрики не красятся:
# для них «лучше/хуже» не определено (стартовый капитал, число сделок и т. п.).
LOWER_IS_BETTER: frozenset[str] = frozenset({
    "max_drawdown",
    "max_drawdown_pct",
    "max_drawdown_duration",
    "annualized_volatility_pct",
    "max_consecutive_losses",
    "total_commission",
})
NEUTRAL_METRICS: frozenset[str] = frozenset({
    "initial_equity",
    "total_trades",
    "avg_trade_duration_bars",
    "exposure_pct",
})

# Цвета колонки сравнения (тон в тон с тёмной темой графиков).
_COLOR_BETTER = QColor("#16c784")
_COLOR_WORSE = QColor("#ff4d4d")
_COLOR_NEUTRAL = QColor("#9e9e9e")


def _delta_item(key: str, current: Any, previous: Any) -> QTableWidgetItem:
    """Ячейка «насколько метрика изменилась к прошлому бэктесту».

    Показывает относительное изменение в процентах со стрелкой и красит
    его зелёным/красным в зависимости от того, улучшение это или ухудшение
    (с учётом направления метрики), серым — для нейтральных метрик.
    """
    item = QTableWidgetItem()
    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    if not isinstance(current, (int, float)) or not isinstance(previous, (int, float)):
        return item
    if isinstance(current, float) and (math.isnan(current) or math.isinf(current)):
        return item
    if isinstance(previous, float) and (math.isnan(previous) or math.isinf(previous)):
        return item

    diff = float(current) - float(previous)
    if abs(diff) < 1e-12:
        item.setText("= 0%")
        item.setForeground(QBrush(_COLOR_NEUTRAL))
        return item

    if abs(previous) > 1e-12:
        pct = diff / abs(float(previous)) * 100.0
        text = f"{'▲' if diff > 0 else '▼'} {pct:+,.2f}%"
    else:
        # Прошлое значение было нулём — процент не определён, показываем дельту.
        text = f"{'▲' if diff > 0 else '▼'} {diff:+,.4f}".rstrip("0").rstrip(".")
    item.setText(text)

    if key in NEUTRAL_METRICS:
        item.setForeground(QBrush(_COLOR_NEUTRAL))
    else:
        improved = (diff < 0) if key in LOWER_IS_BETTER else (diff > 0)
        item.setForeground(QBrush(_COLOR_BETTER if improved else _COLOR_WORSE))
    return item


def _num_item(value: Any) -> QTableWidgetItem:
    """Ячейка с числом: красивый текст + корректная числовая сортировка."""
    item = QTableWidgetItem()
    if isinstance(value, float):
        text = "" if math.isnan(value) else f"{value:,.4f}".rstrip("0").rstrip(".")
        if isinstance(value, float) and math.isinf(value):
            text = "inf" if value > 0 else "-inf"
        item.setData(Qt.EditRole, value)
        item.setText(text)
    elif isinstance(value, (int,)):
        item.setData(Qt.EditRole, value)
        item.setText(str(value))
    else:
        item.setText(str(value))
    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    return item


class MetricsTable(QTableWidget):
    """Таблица «метрика -- значение -- изменение к прошлому бэктесту».

    Третья колонка автоматически сравнивает каждую метрику с предыдущим
    показанным прогоном: видно, что улучшилось, а что просело после смены
    параметров — без скриншотов и запоминания чисел.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Метрика", "Значение", "Δ к прошлому"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self._prev_stats: dict[str, Any] | None = None

    def reset_baseline(self) -> None:
        """Забыть прошлый прогон (следующий бэктест покажется без сравнения)."""
        self._prev_stats = None

    def show_result(self, result: BacktestResult) -> None:
        stats = result.stats().to_dict()
        prev = self._prev_stats
        self.setRowCount(len(stats))
        for row, (key, value) in enumerate(stats.items()):
            label = METRIC_LABELS.get(key, key)
            name_item = QTableWidgetItem(label)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.setItem(row, 0, name_item)
            self.setItem(row, 1, _num_item(value))
            if prev is not None and key in prev:
                self.setItem(row, 2, _delta_item(key, value, prev[key]))
            else:
                self.setItem(row, 2, QTableWidgetItem())
        self.resizeColumnsToContents()
        self._prev_stats = stats


class TradesTable(QTableWidget):
    """Список сделок завершённого бэктеста."""

    COLUMNS = [
        ("id", "№"),
        ("side", "Тип"),
        ("size", "Объём"),
        ("entry_time", "Вход (время)"),
        ("entry_price", "Вход (цена)"),
        ("exit_time", "Выход (время)"),
        ("exit_price", "Выход (цена)"),
        ("pnl", "Прибыль"),
        ("pnl_pct", "Прибыль, %"),
        ("exit_reason", "Причина выхода"),
        ("duration_bars", "Баров"),
        ("commission", "Комиссия"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(self.COLUMNS), parent)
        self.setHorizontalHeaderLabels([title for _, title in self.COLUMNS])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSortingEnabled(True)

    def show_result(self, result: BacktestResult) -> None:
        df = result.trades_df()
        self.setSortingEnabled(False)
        self.setRowCount(len(df))
        for row, (_, trade) in enumerate(df.iterrows()):
            for col, (key, _) in enumerate(self.COLUMNS):
                value = trade[key]
                if isinstance(value, pd.Timestamp):
                    item = QTableWidgetItem(value.strftime("%Y-%m-%d %H:%M"))
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                else:
                    item = _num_item(value)
                self.setItem(row, col, item)
        self.setSortingEnabled(True)
        self.resizeColumnsToContents()


class OptimizationTable(QTableWidget):
    """Все испытания оптимизации; двойной клик по строке -> сигнал с параметрами."""

    row_activated = Signal(dict)   # параметры выбранной строки

    def __init__(self, parent=None) -> None:
        super().__init__(0, 0, parent)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.itemDoubleClicked.connect(self._on_double_click)
        self._param_names: list[str] = []

    def show_result(self, opt_result: OptimizationResult) -> None:
        df = opt_result.history_df()
        if df.empty:
            self.setRowCount(0)
            return
        self._param_names = [c for c in df.columns if c != "score"]
        columns = ["score"] + self._param_names
        df = df.sort_values("score", ascending=False)

        self.setSortingEnabled(False)
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        self.setRowCount(len(df))
        for row, (_, rec) in enumerate(df.iterrows()):
            for col, name in enumerate(columns):
                self.setItem(row, col, _num_item(rec[name]))
        self.setSortingEnabled(True)
        self.resizeColumnsToContents()

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        row = item.row()
        params: dict[str, Any] = {}
        for col in range(1, self.columnCount()):
            name = self.horizontalHeaderItem(col).text()
            cell = self.item(row, col)
            if cell is not None:
                params[name] = cell.data(Qt.EditRole)
        self.row_activated.emit(params)
