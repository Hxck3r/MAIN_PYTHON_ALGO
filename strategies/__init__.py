"""Пользовательские стратегии (наследники backtester.BaseStrategy).

Чтобы новая стратегия появилась в GUI (desktop_app.py), добавьте её в STRATEGIES.
"""
from .sma_cross import SmaCrossStrategy

#: реестр стратегий: отображаемое имя -> класс
STRATEGIES = {
    "SMA Crossover (пересечение скользящих)": SmaCrossStrategy,
}

__all__ = ["SmaCrossStrategy", "STRATEGIES"]

# Сигнальные стратегии исполняются быстрым numba-ядром (десятки крат быстрее
# на оптимизации/WFA). Регистрируются только если установлена numba -- иначе
# пакет остаётся работоспособным на чистом событийном движке.
try:
    from .sma_cross_signal import SmaCrossSignal
except ImportError:
    pass
else:
    STRATEGIES["SMA Crossover (быстрый, numba)"] = SmaCrossSignal
    __all__.append("SmaCrossSignal")
