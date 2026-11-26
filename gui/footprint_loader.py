"""Чтение footprint Parquet и барные признаки ордерфлоу -- вне GUI-потока.

Обе операции недешёвые: на 1-минутном футпринте за три месяца (753k ячеек)
``bar_features_from_footprint`` считается ~50 с, из них ~45 с -- цикл
``cell_order_flow_features`` по 123 840 барам. В обработчике кнопки это
блокировало бы главный поток: Qt не успевает обрабатывать события, и ОС
показывает «приложение не отвечает». Поэтому обе операции выполняются в
:class:`gui.workers.FootprintLoadWorker`, а здесь живёт общий код чтения,
которым пользуются и воркер, и синхронный :meth:`MainWindow.load_marketdata_files`.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from marketdata.features import bar_features_from_footprint


def read_footprint(paths: Sequence[str]) -> pd.DataFrame:
    """Собрать footprint из Parquet-файлов и проверить его индекс.

    Индекс обязан быть ``(bar_start, price_level)``: на нём держатся и барные
    признаки, и тепловая карта. Ошибка -- ``ValueError`` (её текст показывается
    пользователю в диалоге).
    """
    if not paths:
        raise ValueError("Не выбрано ни одного файла footprint")
    footprint = pd.concat([pd.read_parquet(path) for path in paths]).sort_index()
    if not isinstance(footprint.index, pd.MultiIndex):
        raise ValueError(
            "Footprint Parquet должен иметь MultiIndex bar_start/price_level"
        )
    if list(footprint.index.names) != ["bar_start", "price_level"]:
        raise ValueError("Индекс footprint должен быть (bar_start, price_level)")
    return footprint


def footprint_features(footprint: pd.DataFrame) -> pd.DataFrame:
    """Барные признаки ордерфлоу из ячеек футпринта (см. ``marketdata``)."""
    return bar_features_from_footprint(footprint)


__all__ = ["footprint_features", "read_footprint"]
