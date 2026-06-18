"""Пользовательские стратегии (наследники backtester.BaseStrategy).

Чтобы новая стратегия появилась в GUI (desktop_app.py), добавьте её в STRATEGIES.
"""
from .sma_cross import SmaCrossStrategy
from .kama_cross import KamaCrossStrategy

#: реестр стратегий: отображаемое имя -> класс
STRATEGIES = {
    "SMA Crossover (пересечение скользящих)": SmaCrossStrategy,
    "KAMA Crossover (адаптивные Кауфмана)": KamaCrossStrategy,
}

__all__ = ["SmaCrossStrategy", "KamaCrossStrategy", "STRATEGIES"]

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

# TA Adaptive Trend Lite -- сигнальная (быстрое numba-ядро); регистрируется
# только при установленной numba (как и прочие SignalStrategy выше).
try:
    from .adaptive_trend_lite import TAAdaptiveTrendLite
except ImportError:
    pass
else:
    STRATEGIES["TA Adaptive Trend Lite (быстрый, numba)"] = TAAdaptiveTrendLite
    __all__.append("TAAdaptiveTrendLite")

# TA Adaptive Trend (ATR + частичные закрытия) -- событийная стратегия,
# работает без numba. Регистрируем ДОБАВЛЕНИЕМ ключа, а не переприсваиванием
# STRATEGIES (иначе стёрли бы все стратегии, добавленные выше).
from .adaptive_trend_atr import TAAdaptiveTrendATR  # noqa: E402

STRATEGIES["TA Adaptive Trend (ATR + Частичные закрытия)"] = TAAdaptiveTrendATR
__all__.append("TAAdaptiveTrendATR")