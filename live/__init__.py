"""Пакет live-торговли на криптобиржах (BingX / MEXC / Binance).

Запускает существующие стратегии «вживую» на демо (тестнет) и реальных счетах
USDT-перпетуалов через единый адаптер ccxt. Архитектура повторяет бэктестер:
стратегия общается с рынком только через брокера, поэтому LiveBroker исполняет
её сигналы реальными ордерами без изменения кода стратегии.

ccxt -- опциональная зависимость: пакет импортируется и без неё (CCXT_AVAILABLE
покажет False), окно торговли тогда выводит подсказку по установке. Установка::

    pip install -e ".[live]"
"""
from __future__ import annotations

try:
    import ccxt  # noqa: F401
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

#: текст подсказки для UI, когда ccxt не установлена
INSTALL_HINT = (
    "Для торговли на биржах нужна библиотека ccxt.\n"
    "Установите её командой:\n\n"
    "    pip install -e \".[live]\"\n\n"
    "(или: pip install ccxt)"
)

# Тяжёлые/ccxt-зависимые символы импортируем лениво, чтобы `import live` работал
# без ccxt. Доступ к ним -- через функции ниже или прямой импорт подмодулей.
from .credentials import Credentials, credentials_path  # noqa: E402

__all__ = [
    "CCXT_AVAILABLE",
    "INSTALL_HINT",
    "Credentials",
    "credentials_path",
    "get_exchanges",
    "make_client",
]


def get_exchanges() -> dict:
    """Реестр поддерживаемых бирж (требует ccxt-подмодуль, но не самой ccxt)."""
    from .exchange import EXCHANGES

    return EXCHANGES


def make_client(exchange_id: str, mode: str, api_key: str = "", secret: str = ""):
    """Фабрика ExchangeClient (бросит понятную ошибку, если ccxt не установлена)."""
    from .exchange import ExchangeClient

    return ExchangeClient(exchange_id, mode=mode, api_key=api_key, secret=secret)
