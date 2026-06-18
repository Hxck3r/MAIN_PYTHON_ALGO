"""Адаптер криптобирж поверх ccxt (USDT-перпетуалы / swap).

Единый интерфейс к BingX, MEXC и Binance: данные (OHLCV, тикер, список
символов), счёт (баланс, позиция) и исполнение (рыночные ордера, защитные
стоп-ордера, плечо). Всё приводится к каноническому виду фреймворка, чтобы
существующие стратегии и график работали без изменений.

ccxt импортируется лениво: пакет live/ должен импортироваться и без ccxt
(окно торговли тогда покажет подсказку по установке) -- как сделано для
numba/sklearn в остальном проекте.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

try:  # ленивый «мягкий» импорт -- наличие проверяется в live/__init__.py
    import ccxt
except ImportError:  # pragma: no cover - окружение без ccxt
    ccxt = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Метаданные поддерживаемых бирж
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExchangeMeta:
    """Описание биржи: id для ccxt, поддержка демо и оговорки по ордерам."""

    ccxt_id: str
    label: str
    supports_demo: bool          # есть ли тестнет/демо через sandbox-режим ccxt
    futures_orders_ok: bool      # надёжно ли ставятся ордера на фьючерсы через ccxt
    note: str = ""               # заметка для UI (особенности/ограничения)


#: реестр поддерживаемых бирж (ключ -- наш внутренний id)
EXCHANGES: dict[str, ExchangeMeta] = {
    "mexc": ExchangeMeta(
        ccxt_id="mexc",
        label="MEXC",
        supports_demo=False,
        futures_orders_ok=False,
        note=(
            "Данные, баланс и позиции — ок. ВНИМАНИЕ: выставление ордеров на "
            "фьючерсы у MEXC через API исторически ограничено самой биржей — "
            "реальные ордера могут отклоняться. Демо-фьючерсов в ccxt нет."
        ),
    ),
    "bingx": ExchangeMeta(
        ccxt_id="bingx",
        label="BingX",
        supports_demo=True,
        futures_orders_ok=True,
        note="Демо-перпетуалы (VST) через sandbox-режим, если доступны на аккаунте.",
    ),
    "binance": ExchangeMeta(
        ccxt_id="binance",
        label="Binance",
        supports_demo=True,
        futures_orders_ok=True,
        note="Полноценный фьючерсный testnet (testnet.binancefuture.com).",
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
    exchange_id: ключ из EXCHANGES ("mexc"/"bingx"/"binance").
    mode:        MODE_DEMO (тестнет/демо) или MODE_REAL (реальный счёт).
    api_key, secret: ключи API (для публичных данных можно оставить пустыми).
    """

    def __init__(
        self,
        exchange_id: str,
        mode: str = MODE_DEMO,
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

        klass = getattr(ccxt, self.meta.ccxt_id)
        self.ex = klass({
            "apiKey": api_key or None,
            "secret": secret or None,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},  # USDT-перпетуалы
        })
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
        BingX счёт ведётся в VST («виртуальные USDT»), а у некоторых бирж --
        в другой валюте. Поэтому перебираем предпочтительные коды, а если их
        нет -- берём валюту с наибольшим балансом. Возвращает также 'currency'.
        """
        bal = self.fetch_balance()
        totals = bal.get("total") or {}
        frees = bal.get("free") or {}

        # Порядок предпочтения валюты расчёта (для демо добавляем VST/VUSDT)
        prefer = [quote]
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

    def set_leverage(
        self, leverage: int, symbol: str, margin_mode: Optional[str] = None
    ) -> None:
        """Best-effort: выставить плечо и режим маржи. Ошибки не критичны."""
        if margin_mode:
            try:
                self.ex.set_margin_mode(margin_mode, symbol)
            except Exception:  # noqa: BLE001 - часть бирж не поддерживает/уже выставлено
                pass
        try:
            self.ex.set_leverage(leverage, symbol)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Рыночные данные
    # ------------------------------------------------------------------ #
    def list_swap_symbols(self, quote: str = "USDT") -> list[str]:
        """Унифицированные символы линейных USDT-перпетуалов, отсортированы."""
        self._ensure_markets()
        out = []
        for sym, m in self.ex.markets.items():
            if not m.get("active", True):
                continue
            if m.get("swap") and m.get("linear") and m.get("quote") == quote:
                out.append(sym)
        return sorted(out)

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
