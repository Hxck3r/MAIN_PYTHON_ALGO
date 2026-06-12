"""
Абстрактный базовый класс торговой стратегии.

Пользовательская стратегия наследуется от BaseStrategy и реализует:
    init()     -- одноразовая подготовка (предрасчёт индикаторов);
    next(bar)  -- торговая логика, вызывается на каждом баре.

Стратегия общается с рынком только через методы брокера
(buy / sell / close_position / cancel_all_orders) -- это позволяет
подменять исполнение (бэктест, walk-forward, в перспективе live)
без изменения кода стратегии.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # только для аннотаций, избегаем циклического импорта
    from .backtest_engine import Broker, Order, Position


@dataclass(frozen=True, slots=True)
class Bar:
    """Один бар рыночных данных, передаётся в Strategy.next()."""

    index: int                 # порядковый номер бара (0..n-1)
    datetime: pd.Timestamp     # время бара
    open: float
    high: float
    low: float
    close: float
    volume: float


class BaseStrategy(ABC):
    """Базовый класс всех торговых стратегий.

    Параметры стратегии объявляются в словаре класса ``params``
    и могут быть переопределены через kwargs конструктора -- это
    единая точка входа для оптимизатора.

    Пример::

        class SmaCross(BaseStrategy):
            params = {"fast": 10, "slow": 30}

            def init(self) -> None:
                close = pd.Series(self.full_close)
                self.fast_sma = close.rolling(self.p["fast"]).mean().to_numpy()
                self.slow_sma = close.rolling(self.p["slow"]).mean().to_numpy()

            def next(self, bar: Bar) -> None:
                i = bar.index
                if self.fast_sma[i] > self.slow_sma[i] and not self.position:
                    self.buy()
    """

    #: параметры по умолчанию; переопределяются в наследниках
    params: ClassVar[dict[str, Any]] = {}

    def __init__(self, **param_overrides: Any) -> None:
        unknown = set(param_overrides) - set(type(self).params)
        if unknown:
            raise ValueError(
                f"Неизвестные параметры стратегии {type(self).__name__}: {sorted(unknown)}"
            )
        #: действующий набор параметров экземпляра
        self.p: dict[str, Any] = {**type(self).params, **param_overrides}

        # Заполняются движком через _bind() перед запуском
        self._broker: Optional["Broker"] = None
        self._data: Optional[pd.DataFrame] = None
        self._i: int = -1
        self._open: np.ndarray = np.empty(0)
        self._high: np.ndarray = np.empty(0)
        self._low: np.ndarray = np.empty(0)
        self._close: np.ndarray = np.empty(0)
        self._volume: np.ndarray = np.empty(0)

    # ------------------------------------------------------------------ #
    # Контракт стратегии
    # ------------------------------------------------------------------ #
    @abstractmethod
    def init(self) -> None:
        """Одноразовая инициализация перед прогоном (предрасчёт индикаторов).

        Здесь доступны полные массивы данных (full_close и т.д.).
        ВАЖНО: предрассчитанный индикатор в точке i должен использовать
        только данные до бара i включительно (rolling/ewm это гарантируют),
        иначе возникнет заглядывание в будущее (lookahead bias).
        """

    @abstractmethod
    def next(self, bar: Bar) -> None:
        """Торговая логика; вызывается движком на каждом баре по порядку."""

    # ------------------------------------------------------------------ #
    # Привязка к движку (вызывается ядром, не пользователем)
    # ------------------------------------------------------------------ #
    def _bind(self, broker: "Broker", data: pd.DataFrame) -> None:
        self._broker = broker
        self._data = data
        self._open = data["open"].to_numpy()
        self._high = data["high"].to_numpy()
        self._low = data["low"].to_numpy()
        self._close = data["close"].to_numpy()
        self._volume = data["volume"].to_numpy()
        self._i = -1

    def _set_index(self, i: int) -> None:
        self._i = i

    # ------------------------------------------------------------------ #
    # Доступ к данным
    # ------------------------------------------------------------------ #
    @property
    def i(self) -> int:
        """Индекс текущего бара."""
        return self._i

    @property
    def data(self) -> pd.DataFrame:
        """Полный DataFrame данных (используйте в init; в next -- осторожно)."""
        assert self._data is not None, "Стратегия не привязана к движку"
        return self._data

    # Срезы «по текущий бар включительно» -- безопасны от lookahead в next()
    @property
    def open(self) -> np.ndarray:
        return self._open[: self._i + 1]

    @property
    def high(self) -> np.ndarray:
        return self._high[: self._i + 1]

    @property
    def low(self) -> np.ndarray:
        return self._low[: self._i + 1]

    @property
    def close(self) -> np.ndarray:
        return self._close[: self._i + 1]

    @property
    def volume(self) -> np.ndarray:
        return self._volume[: self._i + 1]

    # Полные массивы -- для предрасчёта индикаторов в init()
    @property
    def full_open(self) -> np.ndarray:
        return self._open

    @property
    def full_high(self) -> np.ndarray:
        return self._high

    @property
    def full_low(self) -> np.ndarray:
        return self._low

    @property
    def full_close(self) -> np.ndarray:
        return self._close

    @property
    def full_volume(self) -> np.ndarray:
        return self._volume

    # ------------------------------------------------------------------ #
    # Торговый API (делегирование брокеру)
    # ------------------------------------------------------------------ #
    @property
    def broker(self) -> "Broker":
        assert self._broker is not None, "Стратегия не привязана к движку"
        return self._broker

    @property
    def position(self) -> Optional["Position"]:
        """Текущая открытая позиция или None."""
        return self.broker.position

    @property
    def equity(self) -> float:
        """Текущая стоимость счёта (баланс + плавающая прибыль)."""
        return self.broker.equity

    @property
    def cash(self) -> float:
        """Текущий баланс (реализованный)."""
        return self.broker.cash

    def buy(
        self,
        size: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_pct: Optional[float] = None,
        tag: str = "",
    ) -> "Order":
        """Покупка. Тип ордера (взаимоисключающие варианты):
            * без limit_price/stop_price -- рыночный (исполнение на открытии
              следующего бара);
            * limit_price -- лимитный (исполнение при касании уровня снизу);
            * stop_price  -- buy-stop (исполнение при пробое уровня вверх).

        Сайзинг (взаимоисключающие варианты):
            * size       -- явный размер в единицах актива;
            * risk_pct   -- от риска: размер подбирается так, чтобы убыток
                            при выходе ровно по stop_loss составил risk_pct
                            от эквити (требует stop_loss);
            * ничего     -- автосайзинг: default_size_pct от эквити.
        """
        return self.broker.buy(
            size=size,
            limit_price=limit_price,
            stop_price=stop_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_pct=risk_pct,
            tag=tag,
        )

    def sell(
        self,
        size: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_pct: Optional[float] = None,
        tag: str = "",
    ) -> "Order":
        """Продажа (открытие шорта или разворот). Семантика как у buy();
        stop_price здесь -- sell-stop (исполнение при пробое уровня вниз)."""
        return self.broker.sell(
            size=size,
            limit_price=limit_price,
            stop_price=stop_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_pct=risk_pct,
            tag=tag,
        )

    def close_position(self, tag: str = "") -> Optional["Order"]:
        """Закрыть текущую позицию рыночным ордером (если она есть)."""
        return self.broker.close_position(tag=tag)

    def cancel_all_orders(self) -> None:
        """Отменить все отложенные (ещё не исполненные) ордера."""
        self.broker.cancel_all_orders()
