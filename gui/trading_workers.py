"""Фоновые потоки окна торговли.

Все обращения к бирже (ccxt) выполняются в QThread, чтобы окно не «замерзало».
Каждый воркер строит собственный ExchangeClient из спецификации подключения --
ccxt-инстанс не делится между потоками (он не потокобезопасен при конкурентных
запросах). Результаты возвращаются в GUI-поток через сигналы Qt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import QThread, Signal

from live.engine import EngineConfig, LiveTradingEngine
from live.exchange import ExchangeClient, ExchangeError


@dataclass
class ConnectionSpec:
    """Всё необходимое для построения ExchangeClient.

    creds -- словарь полей ключей по схеме биржи (api_key/secret у CEX либо
    private_key/account_index/api_key_index у DEX Lighter).
    """

    exchange_id: str
    mode: str
    creds: dict = field(default_factory=dict)

    def build_client(self) -> ExchangeClient:
        return ExchangeClient(self.exchange_id, mode=self.mode, creds=self.creds)


class ConnectionTestWorker(QThread):
    """Проверка ключей: строит клиент, грузит рынки и баланс."""

    finished_ok = Signal(object)   # dict со сводкой
    failed = Signal(str)

    def __init__(self, spec: ConnectionSpec, parent=None) -> None:
        super().__init__(parent)
        self.spec = spec

    def run(self) -> None:
        try:
            client = self.spec.build_client()
            summary = client.check_connection()
        except ExchangeError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Непредвиденная ошибка: {exc}")
            return
        self.finished_ok.emit(summary)


class SymbolsWorker(QThread):
    """Загрузка списка USDT-перпетуалов биржи."""

    finished_ok = Signal(object)   # list[str]
    failed = Signal(str)

    def __init__(self, spec: ConnectionSpec, parent=None) -> None:
        super().__init__(parent)
        self.spec = spec

    def run(self) -> None:
        try:
            client = self.spec.build_client()
            symbols = client.list_swap_symbols()
        except ExchangeError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Непредвиденная ошибка: {exc}")
            return
        self.finished_ok.emit(symbols)


class OhlcvWorker(QThread):
    """Загрузка истории свечей по символу для графика."""

    finished_ok = Signal(object, str, str)   # df, symbol, timeframe
    failed = Signal(str)

    def __init__(
        self, spec: ConnectionSpec, symbol: str, timeframe: str,
        limit: int = 500, parent=None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit

    def run(self) -> None:
        try:
            client = self.spec.build_client()
            df = client.fetch_ohlcv(self.symbol, self.timeframe, self.limit)
        except ExchangeError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Непредвиденная ошибка: {exc}")
            return
        self.finished_ok.emit(df, self.symbol, self.timeframe)


class TradingEngineThread(QThread):
    """Запускает LiveTradingEngine; события движка переизлучает Qt-сигналом.

    Колбэк on_event движка вызывается в этом потоке, а `event.emit(...)` Qt
    автоматически ставит доставку в очередь GUI-потока (queued connection),
    поэтому обработчики в окне выполняются безопасно в GUI-потоке.
    """

    event = Signal(object)         # EngineEvent
    failed = Signal(str)

    def __init__(self, spec: ConnectionSpec, config: EngineConfig, parent=None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.config = config
        self._engine: Optional[LiveTradingEngine] = None

    def run(self) -> None:
        try:
            client = self.spec.build_client()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Не удалось подключиться: {exc}")
            return
        self._engine = LiveTradingEngine(
            client, self.config, on_event=self.event.emit
        )
        try:
            self._engine.run()
        except Exception as exc:  # noqa: BLE001 - последний рубеж, цикл не должен ронять GUI
            self.failed.emit(f"Сбой торгового цикла: {exc}")

    def stop_engine(self) -> None:
        if self._engine is not None:
            self._engine.stop()
