"""Пользовательские стратегии (наследники backtester.BaseStrategy).

Чтобы новая стратегия появилась в GUI (desktop_app.py), добавьте её в STRATEGIES.
"""
from .sma_cross import SmaCrossStrategy

#: реестр стратегий: отображаемое имя -> класс
STRATEGIES = {
    "SMA Crossover (пересечение скользящих)": SmaCrossStrategy,
}

__all__ = ["STRATEGIES", "SmaCrossStrategy"]

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

# ML-стратегия (мост к пакету ml/). Импорт лёгкий (не тянет sklearn/lightgbm);
# тяжёлые зависимости подтягиваются лишь при загрузке обученной модели в init().
try:
    from .ml_strategy import MLStrategy
except ImportError:
    pass
else:
    STRATEGIES["ML модель (классификатор)"] = MLStrategy
    __all__.append("MLStrategy")

    # Мета-лейблинг (López de Prado): первичный сигнал × фильтр мета-модели.
    from .meta_strategy import MetaLabelStrategy

    STRATEGIES["ML мета-модель (López de Prado)"] = MetaLabelStrategy
    __all__.append("MetaLabelStrategy")
