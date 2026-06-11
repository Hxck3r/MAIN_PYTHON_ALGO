"""Таблица параметров стратегии в стиле тестера MT5.

Колонки: Параметр | Значение | Опт. (галочка) | Старт | Шаг | Стоп.
Значение используется бэктестом и как фиксированное при оптимизации;
Старт/Шаг/Стоп задают перебор для отмеченных галочкой параметров.

Типы параметров:
    * int / float -- редактируемое число, оптимизация по сетке Старт..Стоп;
    * bool        -- галочка в колонке «Значение»; при отмеченной «Опт.»
                     перебираются оба значения (false/true), как в MT5;
    * str         -- редактируемый текст, оптимизация недоступна.
"""
from __future__ import annotations

from typing import Any, Type

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from backtester import BaseStrategy

COL_NAME, COL_VALUE, COL_OPT, COL_START, COL_STEP, COL_STOP = range(6)
HEADERS = ["Параметр", "Значение", "Опт.", "Старт", "Шаг", "Стоп"]


def numeric_params(strategy_cls: Type[BaseStrategy]) -> dict[str, float | int]:
    """Числовые параметры стратегии (доступны для сеточной оптимизации)."""
    return {
        name: value
        for name, value in strategy_cls.params.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _fmt(value: float | int) -> str:
    return str(value) if isinstance(value, int) else f"{value:g}"


def _readonly_item(text: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemIsEnabled)
    return item


class ParamTable(QTableWidget):
    """Редактируемая таблица параметров с галочками оптимизации."""

    #: сетка изменилась (правка ячейки или галочки «Опт.») -- для счётчика шагов
    grid_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(HEADERS), parent)
        self.setHorizontalHeaderLabels(HEADERS)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self._defaults: dict[str, Any] = {}
        self.cellChanged.connect(lambda *_: self.grid_changed.emit())

    # ------------------------------------------------------------------ #
    def set_strategy(self, strategy_cls: Type[BaseStrategy]) -> None:
        """Заполняет таблицу всеми параметрами стратегии.

        Для чисел -- эвристика старт/шаг/стоп; bool -- галочка; str -- текст.
        """
        self._defaults = dict(strategy_cls.params)
        self.blockSignals(True)  # не пересчитываем счётчик на каждой ячейке
        self.setRowCount(len(self._defaults))
        for row, (name, value) in enumerate(self._defaults.items()):
            self.setItem(row, COL_NAME, _readonly_item(name))

            opt_item = QTableWidgetItem()
            opt_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            opt_item.setCheckState(Qt.Unchecked)

            if isinstance(value, bool):
                # bool: галочка в «Значении», перебор false/true при «Опт.»
                value_item = QTableWidgetItem()
                value_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                value_item.setCheckState(Qt.Checked if value else Qt.Unchecked)
                self.setItem(row, COL_VALUE, value_item)
                self.setItem(row, COL_OPT, opt_item)
                self.setItem(row, COL_START, _readonly_item("false"))
                self.setItem(row, COL_STEP, _readonly_item("—"))
                self.setItem(row, COL_STOP, _readonly_item("true"))
            elif isinstance(value, str):
                # str: текст без оптимизации
                opt_item.setFlags(Qt.ItemIsEnabled)  # галочка недоступна
                self.setItem(row, COL_VALUE, QTableWidgetItem(value))
                self.setItem(row, COL_OPT, opt_item)
                for col in (COL_START, COL_STEP, COL_STOP):
                    self.setItem(row, col, _readonly_item("—"))
            else:
                # число: редактируемое значение + сетка Старт/Шаг/Стоп
                if isinstance(value, int):
                    start, step, stop = max(1, value // 2), 1, value * 2
                else:
                    start, step, stop = value / 2, value / 4, value * 2
                self.setItem(row, COL_OPT, opt_item)
                for col, val in (
                    (COL_VALUE, value),
                    (COL_START, start),
                    (COL_STEP, step),
                    (COL_STOP, stop),
                ):
                    self.setItem(row, col, QTableWidgetItem(_fmt(val)))
        self.resizeColumnsToContents()
        self.blockSignals(False)
        self.grid_changed.emit()

    # ------------------------------------------------------------------ #
    def _cast(self, name: str, value: float) -> float | int:
        """Приводит число к типу параметра по умолчанию."""
        return int(round(value)) if isinstance(self._defaults[name], int) else value

    def _cell_float(self, row: int, col: int) -> float:
        item = self.item(row, col)
        text = (item.text() if item else "").strip().replace(",", ".")
        try:
            return float(text)
        except ValueError:
            name = self.item(row, COL_NAME).text()
            raise ValueError(
                f"{name}: «{text}» в колонке «{HEADERS[col]}» -- не число"
            ) from None

    def _cell_value(self, row: int) -> Any:
        """Значение из колонки «Значение» с учётом типа параметра."""
        name = self.item(row, COL_NAME).text()
        default = self._defaults[name]
        if isinstance(default, bool):
            return self.item(row, COL_VALUE).checkState() == Qt.Checked
        if isinstance(default, str):
            return self.item(row, COL_VALUE).text().strip()
        return self._cast(name, self._cell_float(row, COL_VALUE))

    def fixed_values(self) -> dict[str, Any]:
        """Колонка «Значение» для всех параметров (для одиночного бэктеста)."""
        return {
            self.item(row, COL_NAME).text(): self._cell_value(row)
            for row in range(self.rowCount())
        }

    def optimization_setup(self) -> tuple[dict[str, Any], dict[str, list]]:
        """Возвращает (фиксированные параметры, сетка для отмеченных).

        Сетка: имя -> список значений от Старт до Стоп включительно с Шагом;
        для bool-параметра -- [False, True].
        """
        fixed: dict[str, Any] = {}
        grid: dict[str, list] = {}
        for row in range(self.rowCount()):
            name = self.item(row, COL_NAME).text()
            default = self._defaults[name]
            optimize = self.item(row, COL_OPT).checkState() == Qt.Checked
            if not optimize:
                fixed[name] = self._cell_value(row)
            elif isinstance(default, bool):
                grid[name] = [False, True]
            else:
                start = self._cell_float(row, COL_START)
                step = self._cell_float(row, COL_STEP)
                stop = self._cell_float(row, COL_STOP)
                if step <= 0:
                    raise ValueError(f"{name}: шаг должен быть положительным")
                if stop < start:
                    raise ValueError(f"{name}: стоп меньше старта")
                raw = np.arange(start, stop + step * 0.5, step)
                values = [self._cast(name, float(v)) for v in raw]
                grid[name] = sorted(dict.fromkeys(values))
        return fixed, grid

    def grid_summary(self) -> tuple[int, list[tuple[str, int]]]:
        """Сводка сетки для отмеченных «Опт.» параметров.

        Возвращает (всего комбинаций, [(имя, число значений), ...]).
        Вызывается на каждое нажатие клавиши, поэтому не бросает исключений:
        недописанное/некорректное значение (нечисло, шаг<=0, стоп<старт) даёт
        0 значений у параметра и, как следствие, total == 0 -- сигнал «проверь».
        """
        per_param: list[tuple[str, int]] = []
        for row in range(self.rowCount()):
            opt_item = self.item(row, COL_OPT)
            name_item = self.item(row, COL_NAME)
            if opt_item is None or name_item is None:
                continue
            if opt_item.checkState() != Qt.Checked:
                continue
            name = name_item.text()
            if isinstance(self._defaults.get(name), bool):
                per_param.append((name, 2))
                continue
            try:
                start = self._cell_float(row, COL_START)
                step = self._cell_float(row, COL_STEP)
                stop = self._cell_float(row, COL_STOP)
            except ValueError:
                per_param.append((name, 0))
                continue
            if step <= 0 or stop < start:
                per_param.append((name, 0))
                continue
            raw = np.arange(start, stop + step * 0.5, step)
            count = len({self._cast(name, float(v)) for v in raw})
            per_param.append((name, count))

        total = 0
        if per_param:
            total = 1
            for _, count in per_param:
                total *= count
        return total, per_param

    def apply_values(self, params: dict[str, Any]) -> None:
        """Записывает значения (например, лучшие из оптимизации) в «Значение»."""
        for row in range(self.rowCount()):
            name = self.item(row, COL_NAME).text()
            if name not in params:
                continue
            value = params[name]
            default = self._defaults.get(name)
            if isinstance(default, bool):
                # генетика может вернуть 0/1 вместо False/True
                self.item(row, COL_VALUE).setCheckState(
                    Qt.Checked if bool(value) else Qt.Unchecked
                )
            elif isinstance(default, str):
                self.item(row, COL_VALUE).setText(str(value))
            else:
                self.item(row, COL_VALUE).setText(_fmt(value))
