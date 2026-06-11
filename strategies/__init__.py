"""Пользовательские стратегии (наследники backtester.BaseStrategy).

Чтобы новая стратегия появилась в GUI (desktop_app.py), добавьте её в STRATEGIES.
"""
from .sma_cross import SmaCrossStrategy
from .kama_strategy import KamaStrategy

#: реестр стратегий: отображаемое имя -> класс
STRATEGIES = {
    "SMA Crossover (пересечение скользящих)": SmaCrossStrategy,
    "KAMA Strategy v2 (адаптивная средняя Кауфмана)": KamaStrategy,
}

__all__ = ["SmaCrossStrategy", "KamaStrategy", "STRATEGIES"]
