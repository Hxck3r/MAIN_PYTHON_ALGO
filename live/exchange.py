"""Адаптер криптобирж поверх ccxt (USDT-перпетуалы / swap).

Единый интерфейс к Binance (CEX) и Lighter (DEX): данные (OHLCV, тикер, список
символов), счёт (баланс, позиция) и исполнение (рыночные ордера, защитные
стоп-ордера, плечо). Всё приводится к каноническому виду фреймворка, чтобы
существующие стратегии и график работали без изменений.

ccxt импортируется лениво: пакет live/ должен импортироваться и без ccxt
(окно торговли тогда покажет подсказку по установке) -- как сделано для
numba/sklearn в остальном проекте.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

try:  # ленивый «мягкий» импорт -- наличие проверяется в live/__init__.py
    import ccxt
except ImportError:  # pragma: no cover - окружение без ccxt
    ccxt = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Схемы аутентификации и поля ключей
# --------------------------------------------------------------------------- #
#: классический CEX -- пара apiKey + secret
AUTH_KEY_SECRET = "key_secret"
#: DEX -- подпись приватным ключом API + индексы аккаунта (как у Lighter)
AUTH_DEX_KEY = "dex_key"


@dataclass(frozen=True)
class CredField:
    """Поле ввода ключа для UI: что хранить, как подписать и маскировать ли ввод."""

    key: str                 # ключ в хранилище и в конфиге ccxt
    label: str               # подпись поля в окне торговли
    secret: bool = True      # маскировать ввод (echo = Password)
    placeholder: str = ""    # подсказка в пустом поле
    required: bool = True     # обязательно ли для подключения


#: поля для обычных CEX (Binance и т.п.)
_CEX_FIELDS: tuple[CredField, ...] = (
    CredField("api_key", "API Key", True, "API Key"),
    CredField("secret", "API Secret", True, "API Secret"),
)
#: поля для DEX Lighter. Используем УЖЕ созданный в приложении Lighter ключ API
#: (его приватный ключ) + индексы аккаунта + путь к нативной библиотеке-подписанту.
#: Это безопаснее сырого L1-ключа кошелька (ключ API можно отозвать, он не даёт
#: доступа к самому кошельку) и не требует ончейн-регистрации ключа.
_LIGHTER_FIELDS: tuple[CredField, ...] = (
    CredField("private_key", "API Private Key", True,
              "Приватный ключ API из приложения Lighter"),
    CredField("account_index", "Account Index", False,
              "Индекс аккаунта (целое число)"),
    CredField("api_key_index", "API Key Index", False,
              "Индекс API-ключа, целое 4–254"),
    CredField("library_path", "Lighter Signer (.so)", False,
              "Путь к библиотеке-подписанту .so для вашей ОС"),
)


# --------------------------------------------------------------------------- #
# Метаданные поддерживаемых бирж
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExchangeMeta:
    """Описание биржи: id для ccxt, поддержка демо, схема ключей и оговорки."""

    ccxt_id: str
    label: str
    supports_demo: bool          # есть ли тестнет/демо через sandbox-режим ccxt
    futures_orders_ok: bool      # надёжно ли ставятся ордера на фьючерсы через ccxt
    note: str = ""               # заметка для UI (особенности/ограничения)
    is_dex: bool = False         # децентрализованная биржа (иной способ ключей)
    auth: str = AUTH_KEY_SECRET  # схема аутентификации (AUTH_*)
    cred_fields: tuple[CredField, ...] = _CEX_FIELDS  # какие поля ключей собирать


#: реестр поддерживаемых бирж (ключ -- наш внутренний id)
EXCHANGES: dict[str, ExchangeMeta] = {
    "binance": ExchangeMeta(
        ccxt_id="binance",
        label="Binance",
        supports_demo=True,
        futures_orders_ok=True,
        note="Полноценный фьючерсный testnet (testnet.binancefuture.com).",
    ),
    "lighter": ExchangeMeta(
        ccxt_id="lighter",
        label="Lighter (DEX)",
        supports_demo=True,
        futures_orders_ok=True,
        is_dex=True,
        auth=AUTH_DEX_KEY,
        cred_fields=_LIGHTER_FIELDS,
        note=(
            "Децентрализованная биржа бессрочных контрактов (расчёт в USDC). "
            "Создайте ключ API в приложении Lighter и введите его ПРИВАТНЫЙ ключ "
            "API + индексы (Account Index / API Key Index) — НЕ сырой ключ кошелька "
            "(так безопаснее: ключ API можно отозвать). ТАКЖЕ нужна нативная "
            "библиотека-подписант (.so) — скачайте файл для вашей ОС с "
            "github.com/elliottech/lighter-python (папка lighter/signers) и "
            "укажите путь к нему. ВНИМАНИЕ: демо — отдельный тестнет, ключи и "
            "индексы основной сети там не работают. Биржевые стоп/тейк-ордера через "
            "ccxt у DEX могут поддерживаться ограниченно."
        ),
    ),
}

#: режимы счёта
MODE_DEMO = "demo"
MODE_REAL = "real"

#: таймфреймы для UI -> множитель в секундах
TIMEFRAMES: dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


def ccxt_available() -> bool:
    return ccxt is not None


def timeframe_seconds(timeframe: str) -> int:
    """Длительность бара в секундах (для определения закрытого бара)."""
    if timeframe in TIMEFRAMES:
        return TIMEFRAMES[timeframe]
    unit = timeframe[-1]
    qty = int(timeframe[:-1])
    return qty * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


# --------------------------------------------------------------------------- #
# Нормализованная позиция
# --------------------------------------------------------------------------- #
@dataclass
class LivePosition:
    """Открытая позиция на бирже, приведённая к нейтральному виду."""

    side: str                    # "long" | "short"
    size: float                  # размер в контрактах/единицах базового актива
    entry_price: float
    notional: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: Optional[float] = None
    raw: dict = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    @property
    def signed_size(self) -> float:
        """Размер со знаком: + для long, − для short."""
        return self.size if self.is_long else -self.size


class ExchangeError(RuntimeError):
    """Ошибка работы с биржей (обёртка над исключениями ccxt + наши проверки)."""


# --------------------------------------------------------------------------- #
# Клиент биржи
# --------------------------------------------------------------------------- #
class ExchangeClient:
    """Обёртка над ccxt-инстансом, настроенным на USDT-перпетуалы (swap).

    Параметры
    ---------
    exchange_id: ключ из EXCHANGES ("binance"/"lighter").
    mode:        MODE_DEMO (тестнет/демо) или MODE_REAL (реальный счёт).
    creds:       словарь полей ключей (по схеме биржи, см. ExchangeMeta.cred_fields):
                 у CEX -- {"api_key", "secret"}; у DEX Lighter --
                 {"private_key", "account_index", "api_key_index"}.
    api_key, secret: устаревший способ передать ключи CEX (для совместимости).
                 Для публичных данных всё можно оставить пустым.
    """

    def __init__(
        self,
        exchange_id: str,
        mode: str = MODE_DEMO,
        creds: Optional[dict[str, str]] = None,
        api_key: str = "",
        secret: str = "",
    ) -> None:
        if ccxt is None:
            raise ExchangeError(
                "Библиотека ccxt не установлена. Установите: pip install -e \".[live]\""
            )
        if exchange_id not in EXCHANGES:
            raise ExchangeError(f"Неизвестная биржа: {exchange_id}")
        self.meta = EXCHANGES[exchange_id]
        self.exchange_id = exchange_id
        self.mode = mode
        self.is_demo = mode == MODE_DEMO

        creds = dict(creds or {})
        # Обратная совместимость со старым вызовом по api_key/secret
        if api_key:
            creds.setdefault("api_key", api_key)
        if secret:
            creds.setdefault("secret", secret)
        self.creds = creds

        klass = getattr(ccxt, self.meta.ccxt_id)
        self.ex = klass(self._build_ccxt_config(creds))
        if self.is_demo:
            if not self.meta.supports_demo:
                raise ExchangeError(
                    f"{self.meta.label}: демо/тестнет-фьючерсы недоступны через ccxt. "
                    f"Выберите реальный счёт или другую биржу."
                )
            try:
                self.ex.set_sandbox_mode(True)
            except Exception as exc:  # noqa: BLE001
                raise ExchangeError(
                    f"{self.meta.label}: не удалось включить демо-режим: {exc}"
                ) from exc

        self._markets_loaded = False

    @staticmethod
    def _dex_index(label: str, raw: Optional[str]) -> Optional[int]:
        """Разобрать индекс аккаунта/ключа DEX как целое число (или None, если пусто)."""
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise ExchangeError(
                f"Lighter: «{label}» должен быть целым числом, получено: {raw!r}."
            ) from exc

    def _build_ccxt_config(self, creds: dict[str, str]) -> dict[str, Any]:
        """Конфиг ccxt с учётом схемы аутентификации биржи.

        CEX (Binance) -- apiKey + secret. DEX (Lighter) -- приватный ключ API +
        индексы аккаунта + нативная библиотека-подписант (см. _configure_lighter).
        """
        config: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},  # линейные перпетуалы
        }
        if self.meta.auth == AUTH_DEX_KEY:
            self._configure_lighter(config, creds)
        else:
            config["apiKey"] = creds.get("api_key") or None
            config["secret"] = creds.get("secret") or None
        return config

    def _configure_lighter(self, config: dict[str, Any], creds: dict[str, str]) -> None:
        """Настроить ccxt-конфиг для Lighter по УЖЕ созданному ключу API.

        Пользователь создаёт ключ API в приложении Lighter и вводит его приватный
        ключ + индексы (Account Index / API Key Index). Мы кладём этот ключ прямо
        в options['auths'][acc][aki]['lighterPrivateKey'] -- тогда ccxt подписывает
        им запросы через нативную библиотеку, не требуя сырого L1-ключа кошелька и
        не регистрируя ключ ончейн. ``privateKey`` дублируем тем же ключом, чтобы
        пройти проверку обязательных учётных данных ccxt (requiredCredentials).

        Индексы обязаны быть ЦЕЛЫМИ: ccxt сравнивает apiKeyIndex численно
        (`< 4` / `> 254`), и строка дала бы TypeError «'<' between str and int».
        """
        opts = config["options"]
        api_private_key = (creds.get("private_key") or "").strip()
        account_index = self._dex_index("Account Index", creds.get("account_index"))
        api_key_index = self._dex_index("API Key Index", creds.get("api_key_index"))
        library_path = (creds.get("library_path") or "").strip()

        if account_index is not None:
            opts["accountIndex"] = account_index
        if api_key_index is not None:
            opts["apiKeyIndex"] = api_key_index
        if library_path:
            if not os.path.isfile(library_path):
                raise ExchangeError(
                    f"Lighter: файл библиотеки-подписанта не найден: {library_path}"
                )
            opts["libraryPath"] = library_path

        if not api_private_key:
            return  # только публичные данные (без ключей)

        # Для подписи нужны индексы и библиотека-подписант -- проверяем понятно.
        if account_index is None or api_key_index is None:
            raise ExchangeError(
                "Lighter: для ключа API укажите Account Index и API Key Index "
                "(целые числа из приложения Lighter)."
            )
        if not library_path:
            raise ExchangeError(
                "Lighter: для подписи запросов нужна нативная библиотека-подписант. "
                "Скачайте файл для вашей ОС (например lighter-signer-linux-amd64.so) "
                "из github.com/elliottech/lighter-python (папка lighter/signers) и "
                "укажите путь к нему в поле «Lighter Signer (.so)»."
            )

        config["privateKey"] = api_private_key
        opts["auths"] = {
            str(account_index): {
                str(api_key_index): {
                    "signer": None,
                    "lighterPrivateKey": api_private_key,
                    "deadline": None,
                    "token": None,
                }
            }
        }

    # ------------------------------------------------------------------ #
    # Служебное
    # ------------------------------------------------------------------ #
    def load_markets(self, reload: bool = False) -> None:
        try:
            self.ex.load_markets(reload)
        except Exception as exc:  # noqa: BLE001
            raise ExchangeError(f"Не удалось загрузить рынки: {exc}") from exc
        self._markets_loaded = True

    def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            self.load_markets()

    def market(self, symbol: str) -> dict:
        self._ensure_markets()
        try:
            return self.ex.market(symbol)
        except Exception as exc:  # noqa: BLE001
            raise ExchangeError(f"Неизвестный символ {symbol}: {exc}") from exc

    def contract_size(self, symbol: str) -> float:
        """Размер одного контракта в единицах базового актива (по умолчанию 1)."""
        return float(self.market(symbol).get("contractSize") or 1.0)

    # ------------------------------------------------------------------ #
    # Подключение / счёт
    # ------------------------------------------------------------------ #
    def check_connection(self) -> dict[str, Any]:
        """Проверка ключей: грузит рынки и баланс. Возвращает сводку."""
        self.load_markets(reload=True)
        balance = self.fetch_quote_balance()
        symbols = self.list_swap_symbols()
        return {
            "exchange": self.meta.label,
            "mode": self.mode,
            "balance_usdt": balance,
            "symbols_count": len(symbols),
        }

    def fetch_balance(self) -> dict:
        try:
            return self.ex.fetch_balance()
        except Exception as exc:  # noqa: BLE001
            raise ExchangeError(f"Не удалось получить баланс: {exc}") from exc

    def fetch_quote_balance(self, quote: str = "USDT") -> dict[str, Any]:
        """Свободный и общий баланс в валюте расчёта счёта.

        Валюта определяется автоматически: обычно USDT, но на демо/тестнете
        счёт у некоторых бирж ведётся в виртуальной валюте (VST/VUSDT). Поэтому
        перебираем предпочтительные коды, а если их нет -- берём валюту с
        наибольшим балансом. Возвращает также 'currency'.
        """
        bal = self.fetch_balance()
        totals = bal.get("total") or {}
        frees = bal.get("free") or {}

        # Порядок предпочтения валюты расчёта: запрошенная, затем USDC (DEX
        # Lighter ведёт счёт в USDC); для демо добавляем виртуальные VST/VUSDT.
        prefer = [quote, "USDC"]
        if self.is_demo:
            prefer += ["VST", "VUSDT"]

        def _has(cur: str) -> bool:
            return (
                float(totals.get(cur) or 0.0) != 0.0
                or float(frees.get(cur) or 0.0) != 0.0
                or cur in bal
            )

        currency: Optional[str] = next((c for c in prefer if _has(c)), None)
        if currency is None and totals:
            # запасной путь: валюта с наибольшим суммарным балансом
            currency = max(totals, key=lambda k: float(totals.get(k) or 0.0))
        if currency is None:
            currency = quote

        free = frees.get(currency)
        total = totals.get(currency)
        if free is None and currency in bal:  # некоторые биржи кладут под bal[cur]
            free = bal[currency].get("free")
            total = bal[currency].get("total")
        return {
            "free": float(free or 0.0),
            "total": float(total or 0.0),
            "currency": currency,
        }

    def fetch_position(self, symbol: str) -> Optional[LivePosition]:
        """Текущая позиция по символу или None (если плоско)."""
        try:
            positions = self.ex.fetch_positions([symbol])
        except Exception as exc:  # noqa: BLE001
            raise ExchangeError(f"Не удалось получить позицию: {exc}") from exc
        for p in positions:
            contracts = float(p.get("contracts") or 0.0)
            if contracts == 0.0:
                continue
            side = p.get("side") or ("long" if contracts > 0 else "short")
            return LivePosition(
                side=str(side),
                size=abs(contracts),
                entry_price=float(p.get("entryPrice") or 0.0),
                notional=float(p.get("notional") or 0.0),
                unrealized_pnl=float(p.get("unrealizedPnl") or 0.0),
                leverage=(float(p["leverage"]) if p.get("leverage") else None),
                raw=p,
            )
        return None

    @staticmethod
    def _is_benign(exc: Exception) -> bool:
        """True для «безобидных» отказов биржи: значение уже выставлено / не нужно.

        Биржи часто отвечают ошибкой, когда плечо/режим/позиционный режим уже
        такие, как просим (Binance: -4046 «no need to change leverage», -4059
        «no need to change position side» и т.п.) -- это не повод тревожить
        пользователя. Также сюда попадает NotSupported (например, у DEX, где
        концепции hedge-режима нет).
        """
        if ccxt is not None and isinstance(exc, getattr(ccxt, "NotSupported", ())):
            return True
        s = str(exc).lower()
        benign = (
            "no need", "not modified", "no change", "cannot be changed",
            "-4046", "-4059", "already", "unchanged",
        )
        return any(b in s for b in benign)

    def set_leverage(
        self, leverage: int, symbol: str, margin_mode: Optional[str] = None
    ) -> list[str]:
        """Best-effort: выставить плечо и режим маржи.

        Возвращает список предупреждений (пустой, если всё ок). Тихий провал
        опасен: торговля шла бы на ДЕФОЛТНОМ плече/режиме аккаунта, что меняет
        требования к марже и риск ликвидации. Поэтому о реальных сбоях сообщаем
        наверх (движок выводит их в журнал), но торговлю не блокируем.
        """
        warnings: list[str] = []
        if margin_mode:
            try:
                self.ex.set_margin_mode(margin_mode, symbol)
            except Exception as exc:  # noqa: BLE001
                if not self._is_benign(exc):
                    warnings.append(
                        f"Не удалось выставить режим маржи «{margin_mode}» для "
                        f"{symbol}: {exc}. Торгую на текущем режиме аккаунта — "
                        f"проверьте его вручную."
                    )
        try:
            self.ex.set_leverage(leverage, symbol)
        except Exception as exc:  # noqa: BLE001
            if not self._is_benign(exc):
                warnings.append(
                    f"Не удалось выставить плечо x{leverage} для {symbol}: {exc}. "
                    f"Торгую на ТЕКУЩЕМ плече аккаунта — проверьте его вручную, "
                    f"иначе риск/маржа будут не те."
                )
        return warnings

    def ensure_one_way(self, symbol: Optional[str] = None) -> list[str]:
        """Best-effort перевод аккаунта в One-way (нетто) режим позиций.

        Вся логика брокера рассчитана на one-way: разворот через ноль, reduceOnly
        закрытие, единственная позиция по символу. В Hedge-режиме это сломалось бы.
        Многие площадки (в т.ч. DEX Lighter) one-way по умолчанию / не знают про
        hedge -- такой отказ безобиден и проглатывается. Возвращает список
        предупреждений (непустой только при настоящей проблеме).
        """
        try:
            self.ex.set_position_mode(False, symbol)  # hedged=False -> one-way
            return []
        except Exception as exc:  # noqa: BLE001
            if self._is_benign(exc):
                return []
            return [
                f"Не удалось подтвердить One-way (нетто) режим: {exc}. "
                f"Логика торговли рассчитана на one-way — убедитесь, что "
                f"Hedge-режим на аккаунте ВЫКЛЮЧЕН, иначе учёт позиции разойдётся."
            ]

    # ------------------------------------------------------------------ #
    # Рыночные данные
    # ------------------------------------------------------------------ #
    def list_swap_symbols(self, quote: str = "USDT") -> list[str]:
        """Унифицированные символы линейных перпетуалов, отсортированы.

        Предпочитает контракты в валюте ``quote`` (обычно USDT). Если в ней
        перпетуалов нет (например, DEX Lighter рассчитывает контракты в USDC) --
        возвращает все линейные перпетуалы биржи, чтобы список не оказался пустым.
        """
        self._ensure_markets()
        linear_swaps = [
            sym for sym, m in self.ex.markets.items()
            if m.get("active", True) and m.get("swap") and m.get("linear")
        ]
        preferred = [
            sym for sym in linear_swaps if self.ex.markets[sym].get("quote") == quote
        ]
        return sorted(preferred or linear_swaps)

    def fetch_ticker_last(self, symbol: str) -> float:
        try:
            return float(self.ex.fetch_ticker(symbol)["last"])
        except Exception as exc:  # noqa: BLE001
            raise ExchangeError(f"Не удалось получить цену {symbol}: {exc}") from exc

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 500
    ) -> pd.DataFrame:
        """OHLCV в каноническом виде: DatetimeIndex(UTC) + open/high/low/close/volume.

        Последняя свеча может быть ещё формирующейся (незакрытой) -- отсев
        делает вызывающая сторона через drop_forming_candle().
        """
        self._ensure_markets()
        try:
            raw = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:  # noqa: BLE001
            raise ExchangeError(f"Не удалось получить свечи {symbol}: {exc}") from exc
        if not raw:
            raise ExchangeError(f"Биржа вернула пустую историю по {symbol}")
        arr = np.asarray(raw, dtype=np.float64)
        index = pd.to_datetime(arr[:, 0], unit="ms", utc=True)
        df = pd.DataFrame(
            {
                "open": arr[:, 1],
                "high": arr[:, 2],
                "low": arr[:, 3],
                "close": arr[:, 4],
                "volume": arr[:, 5],
            },
            index=pd.DatetimeIndex(index, name="datetime"),
        )
        return df[~df.index.duplicated(keep="last")].sort_index()

    # ------------------------------------------------------------------ #
    # Исполнение ордеров
    # ------------------------------------------------------------------ #
    def amount_to_precision(self, symbol: str, amount: float) -> float:
        self._ensure_markets()
        try:
            return float(self.ex.amount_to_precision(symbol, amount))
        except Exception:  # noqa: BLE001
            return float(amount)

    def min_amount(self, symbol: str) -> float:
        limits = self.market(symbol).get("limits", {}) or {}
        amount = (limits.get("amount") or {}).get("min")
        return float(amount or 0.0)

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        reduce_only: bool = False,
        params: Optional[dict] = None,
    ) -> dict:
        """Рыночный ордер. side: 'buy'|'sell'. amount -- в контрактах."""
        p = dict(params or {})
        if reduce_only:
            p["reduceOnly"] = True
        try:
            return self.ex.create_order(symbol, "market", side, amount, None, p)
        except Exception as exc:  # noqa: BLE001
            raise ExchangeError(
                f"{self.meta.label}: ордер отклонён ({side} {amount} {symbol}): {exc}"
            ) from exc

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        reduce_only: bool = False,
        post_only: bool = False,
        params: Optional[dict] = None,
    ) -> dict:
        """Лимитный ордер (мейкер). side: 'buy'|'sell'. amount -- в контрактах.

        post_only=True просит биржу отклонить ордер, если он исполнится тейкером
        (гарантия комиссии мейкера). reduce_only=True -- ордер только сокращает
        позицию (для лимитного тейк-профита).
        """
        p = dict(params or {})
        if reduce_only:
            p["reduceOnly"] = True
        if post_only:
            p["postOnly"] = True
        px = float(self.ex.price_to_precision(symbol, price))
        try:
            return self.ex.create_order(symbol, "limit", side, amount, px, p)
        except Exception as exc:  # noqa: BLE001
            raise ExchangeError(
                f"{self.meta.label}: лимитный ордер отклонён "
                f"({side} {amount} {symbol} @ {px}): {exc}"
            ) from exc

    def cancel_order(self, order_id: str, symbol: str) -> None:
        """Отменить один ордер по id (мягко: ошибки не критичны)."""
        try:
            self.ex.cancel_order(order_id, symbol)
        except Exception:  # noqa: BLE001 - ордер мог уже исполниться/исчезнуть
            pass

    def create_stop_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        kind: str = "stop",
    ) -> dict:
        """Защитный reduce-only ордер (stop-loss / take-profit) по триггеру.

        kind: 'stop' (стоп-маркет, SL) или 'take_profit' (тейк-маркет, TP).
        Используется унифицированный параметр ccxt ``triggerPrice``.
        """
        price = self.ex.price_to_precision(symbol, stop_price)
        params = {"reduceOnly": True, "triggerPrice": float(price)}
        order_type = "takeProfitMarket" if kind == "take_profit" else "stopMarket"
        try:
            return self.ex.create_order(symbol, order_type, side, amount, None, params)
        except Exception as exc:  # noqa: BLE001
            # Не все биржи принимают эти типы одинаково -- пробуем generic 'market'
            # с триггером как запасной путь.
            try:
                return self.ex.create_order(
                    symbol, "market", side, amount, None, params
                )
            except Exception:  # noqa: BLE001
                raise ExchangeError(
                    f"{self.meta.label}: не удалось выставить защитный ордер "
                    f"({kind}) для {symbol}: {exc}"
                ) from exc

    def cancel_all_orders(self, symbol: str) -> None:
        try:
            self.ex.cancel_all_orders(symbol)
        except Exception:  # noqa: BLE001 - может не поддерживаться/нет ордеров
            try:
                for o in self.ex.fetch_open_orders(symbol):
                    self.ex.cancel_order(o["id"], symbol)
            except Exception:  # noqa: BLE001
                pass


def drop_forming_candle(df: pd.DataFrame, timeframe: str, now_ms: Optional[int] = None) -> pd.DataFrame:
    """Отсекает последнюю незакрытую свечу.

    Свеча с меткой T закрыта, когда now >= T + длительность таймфрейма. Так мы
    не действуем по ещё формирующемуся бару (репэйнт) независимо от того,
    включает ли биржа текущую свечу в ответ.
    """
    if df.empty:
        return df
    if now_ms is None:
        now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    tf_ms = timeframe_seconds(timeframe) * 1000
    last_ts_ms = int(df.index[-1].timestamp() * 1000)
    if last_ts_ms + tf_ms > now_ms:           # последняя свеча ещё не закрылась
        return df.iloc[:-1]
    return df
