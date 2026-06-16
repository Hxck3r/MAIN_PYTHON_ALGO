"""Доменные структуры ML-пайплайна — без зависимости от Qt и backtester.

``FeatureSet`` и ``ModelResult`` тестируются headless. Результат бэктеста здесь
НЕ дублируется: ML-стратегия исполняется штатным движком MAIN и возвращает
``backtester.BacktestResult`` — поэтому переиспользуются метрики / Monte-Carlo /
PBO / Walk-Forward без изменений.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureSet:
    """Результат инженерии признаков."""

    frame: pd.DataFrame                 # исходные OHLCV + признаки + target (выровнено)
    feature_columns: list[str]          # имена столбцов-признаков
    target_column: str                  # имя столбца целевой переменной
    horizon: int = 1                    # на сколько баров вперёд смотрит target (для purge/embargo)
    overlay_indicators: list[str] = field(default_factory=list)   # рисуются поверх цены
    rsi_column: Optional[str] = None
    macd_columns: Optional[tuple[str, str]] = None
    # Конфиг признаков как словарь (для воспроизводимости; без импорта engineering).
    config_meta: dict = field(default_factory=dict)
    # Позиционный индекс последнего бара, который покрывает метка строки (длина «жизни»
    # метки) — для весов уникальности выборки (AFML гл. 4). None → веса недоступны.
    event_end: Optional[np.ndarray] = None


@dataclass
class ModelResult:
    """Результат обучения и оценки модели на out-of-sample сегменте."""

    model: Any
    model_name: str
    feature_columns: list[str]
    metrics: dict[str, float]
    confusion: list[list[int]]
    importances: dict[str, float]
    # Предсказания на тестовом (out-of-sample) сегменте:
    test_index: pd.Index
    y_true: pd.Series
    y_pred: pd.Series
    y_proba: pd.Series
    # Способ валидации и порог решения.
    validation: str = "holdout"          # "holdout" | "walk_forward"
    threshold: float = 0.5               # порог вероятности для класса 1
    calibrated: bool = False
    # Метрики по фолдам walk-forward (пусто для holdout): список словарей метрик.
    fold_metrics: list[dict] = field(default_factory=list)
    # Честная важность признаков перестановкой на OOS (MDA, AFML гл. 8); пусто, если
    # не считалась. Если непуста — предпочтительнее импьюрити-важности ``importances``.
    importances_mda: dict[str, float] = field(default_factory=dict)
    # OOS-доходность простого «лонг/флэт» сигнала по текущему порогу (диагностика;
    # реальная эквити — из бэктеста MLStrategy). Индекс выровнен по OOS-барам.
    oos_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # «Сырая» доходность следующего бара на каждом OOS-баре — чтобы интерактивно
    # пересчитывать pnl/Sharpe при сдвиге порога без переобучения.
    oos_fwd: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
