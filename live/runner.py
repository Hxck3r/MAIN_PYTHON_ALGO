"""Headless-раннер live-торговли (без GUI) -- для VPS/сервера.

Запускает тот же live.engine.LiveTradingEngine, что и окно торговли, но без Qt:
конфиг берётся из JSON-файла, ключи -- из общего хранилища (live.credentials) или
переменных окружения, события движка пишутся в stdout (его подхватывает journald)
и опционально дублируются алертом в Telegram. Цикл аккуратно останавливается по
SIGINT/SIGTERM (systemd `stop`), состояние сессии переживает рестарт (live.state).

Запуск::

    algo-live --config trade.json            # после `pip install -e ".[live]"`
    python -m live.runner --config trade.json

Безопасность реала: при mode="real" и без dry_run раннер ТРЕБУЕТ явного
подтверждения -- флаг --yes (для systemd/неинтерактивного запуска) либо ответ в
терминале. Без этого торговля реальными деньгами не стартует.

Формат конфига (JSON), пример -- live/trade.example.json::

    {
      "exchange": "lighter",          # id из live.exchange.EXCHANGES
      "mode": "real",                 # "demo" | "real"
      "symbol": "BTC/USDC:USDC",
      "timeframe": "1h",
      "strategy": "SMA Crossover (пересечение скользящих)",  # имя в реестре или класс
      "strategy_params": {"fast_period": 10, "slow_period": 30},
      "leverage": 5,
      "margin_mode": "isolated",
      "window_bars": 800,
      "poll_seconds": 10,
      "dry_run": false,
      "sizing":    {"mode": "notional", "notional_usdt": 100},
      "execution": {"require_protective": true, "max_slippage_pct": 0.3},
      "risk":      {"max_daily_loss_pct": 5, "max_orders_per_hour": 10}
    }

Ключи в конфиг НЕ кладём. Раннер берёт их так:
    1) из переменных окружения ALGO_CRED_<ПОЛЕ> (в верхнем регистре) -- удобно для
       systemd EnvironmentFile, секреты не лежат на диске в открытом виде;
    2) иначе из общего хранилища live.credentials (тот же файл, что заполняет GUI).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional, Type

from backtester.base_strategy import BaseStrategy

from . import credentials
from .broker import ExecutionConfig, RiskLimits, SizingConfig
from .engine import EngineConfig, EngineEvent, LiveTradingEngine
from .exchange import EXCHANGES, MODE_DEMO, MODE_REAL, ExchangeClient, ExchangeError

log = logging.getLogger("algo.live")


# --------------------------------------------------------------------------- #
# Telegram-алерты (опционально, без сторонних зависимостей)
# --------------------------------------------------------------------------- #
class TelegramAlerter:
    """Шлёт важные события в Telegram в фоне (бест-эффорт, не роняет цикл).

    Включается переменными окружения TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID. Если
    их нет -- алертер «пустой» (enabled=False) и ничего не делает. Отправка идёт
    в отдельном потоке через очередь, чтобы сетевой запрос не тормозил торговый
    цикл; сбои отправки молча игнорируются (журнал важнее доставки в чат).
    """

    def __init__(self, token: str = "", chat_id: str = "") -> None:
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def send(self, text: str) -> None:
        if self.enabled:
            self._q.put(text)

    def close(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=5.0)

    def _worker(self) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        while True:
            text = self._q.get()
            if text is None:
                return
            try:
                data = urllib.parse.urlencode(
                    {"chat_id": self.chat_id, "text": text}
                ).encode()
                urllib.request.urlopen(url, data=data, timeout=10.0).read()
            except Exception:  # noqa: BLE001 - доставка алерта не критична
                pass


# --------------------------------------------------------------------------- #
# Сборка конфигурации из JSON
# --------------------------------------------------------------------------- #
class ConfigError(RuntimeError):
    """Ошибка пользовательского конфига (понятная, без трейсбэка)."""


def load_run_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Конфиг не найден: {path}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Конфиг не разобран как JSON ({path}): {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Конфиг должен быть JSON-объектом {…}")
    return data


def resolve_strategy(name: str) -> Type[BaseStrategy]:
    """Найти класс стратегии по имени в реестре или по имени класса."""
    from strategies import STRATEGIES

    if name in STRATEGIES:
        return STRATEGIES[name]
    for cls in STRATEGIES.values():
        if cls.__name__ == name:
            return cls
    available = "\n  - ".join(STRATEGIES)
    raise ConfigError(
        f"Стратегия «{name}» не найдена. Доступные:\n  - {available}"
    )


def _section(cfg: dict, key: str) -> dict:
    val = cfg.get(key) or {}
    if not isinstance(val, dict):
        raise ConfigError(f"Поле «{key}» должно быть объектом {{…}}")
    return val


def build_engine_config(cfg: dict[str, Any]) -> EngineConfig:
    """Собрать EngineConfig из словаря конфига (повторяет сборку в GUI)."""
    symbol = str(cfg.get("symbol") or "").strip()
    if not symbol:
        raise ConfigError("В конфиге не задан «symbol»")
    strategy = str(cfg.get("strategy") or "").strip()
    if not strategy:
        raise ConfigError("В конфиге не задана «strategy»")

    sizing_raw = _section(cfg, "sizing")
    sizing = SizingConfig(
        mode=str(sizing_raw.get("mode", "notional")),
        notional_usdt=float(sizing_raw.get("notional_usdt", 100.0)),
        balance_pct=float(sizing_raw.get("balance_pct", 5.0)),
    )
    exec_raw = _section(cfg, "execution")
    execution = ExecutionConfig(
        entry_limit=bool(exec_raw.get("entry_limit", False)),
        post_only=bool(exec_raw.get("post_only", True)),
        limit_offset_bps=float(exec_raw.get("limit_offset_bps", 1.0)),
        tp_maker=bool(exec_raw.get("tp_maker", False)),
        max_slippage_pct=float(exec_raw.get("max_slippage_pct", 0.0)),
        require_protective=bool(exec_raw.get("require_protective", True)),
    )
    risk_raw = _section(cfg, "risk")
    risk = RiskLimits(
        max_notional_per_order=float(risk_raw.get("max_notional_per_order", 0.0)),
        max_orders_per_hour=int(risk_raw.get("max_orders_per_hour", 0)),
        max_daily_loss_pct=float(risk_raw.get("max_daily_loss_pct", 0.0)),
    )
    params = cfg.get("strategy_params") or {}
    if not isinstance(params, dict):
        raise ConfigError("«strategy_params» должно быть объектом {{…}}")

    return EngineConfig(
        symbol=symbol,
        timeframe=str(cfg.get("timeframe", "1h")),
        strategy_cls=resolve_strategy(strategy),
        strategy_params=params,
        sizing=sizing,
        execution=execution,
        risk=risk,
        leverage=int(cfg.get("leverage", 5)),
        margin_mode=str(cfg.get("margin_mode", "isolated")),
        window_bars=int(cfg.get("window_bars", 800)),
        poll_seconds=float(cfg.get("poll_seconds", 10.0)),
        dry_run=bool(cfg.get("dry_run", False)),
        persist_state=bool(cfg.get("persist_state", True)),
    )


def resolve_credentials(exchange_id: str, mode: str) -> dict[str, str]:
    """Собрать ключи: сперва из env ALGO_CRED_<ПОЛЕ>, иначе из хранилища GUI."""
    if exchange_id not in EXCHANGES:
        raise ConfigError(f"Неизвестная биржа: {exchange_id}")
    meta = EXCHANGES[exchange_id]
    stored = credentials.load(exchange_id, mode).values
    creds: dict[str, str] = {}
    for f in meta.cred_fields:
        env_val = os.environ.get(f"ALGO_CRED_{f.key.upper()}", "").strip()
        creds[f.key] = env_val or str(stored.get(f.key, "") or "").strip()

    missing = [f.label for f in meta.cred_fields if f.required and not creds[f.key]]
    if missing:
        raise ConfigError(
            f"Не заданы ключи для {meta.label} [{mode}]: {', '.join(missing)}.\n"
            f"Заполните их в GUI (окно торговли) либо через переменные окружения "
            f"ALGO_CRED_<ПОЛЕ> (например ALGO_CRED_"
            f"{meta.cred_fields[0].key.upper()})."
        )
    return creds


# --------------------------------------------------------------------------- #
# Журналирование событий движка
# --------------------------------------------------------------------------- #
#: какие виды событий движка дублировать алертом в Telegram
_ALERT_KINDS = {"status", "error"}


def make_event_handler(alerter: TelegramAlerter):
    """Колбэк on_event: пишет в журнал и шлёт важное в Telegram."""

    def on_event(evt: EngineEvent) -> None:
        data = evt.data or {}
        fatal = bool(data.get("fatal"))
        if evt.kind == "error":
            log.error("%s", evt.message)
        elif evt.kind in ("status", "signal", "order", "info"):
            log.info("[%s] %s", evt.kind, evt.message)
        else:  # balance/position/candles/indicators -- шум для журнала, в debug
            log.debug("[%s] %s", evt.kind, evt.message)

        # Алертим старт/стоп, ошибки и всё фатальное/риск-события.
        if alerter.enabled and (fatal or evt.kind in _ALERT_KINDS):
            tag = "🛑" if fatal else ("⚠️" if evt.kind == "error" else "ℹ️")
            alerter.send(f"{tag} {evt.message}")

    return on_event


def _setup_logging(log_file: Optional[str], verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def _confirm_real(cfg_mode: str, dry_run: bool, assume_yes: bool) -> bool:
    """Гейт реала: реальные деньги -- только с явным подтверждением."""
    if cfg_mode != MODE_REAL or dry_run:
        return True
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        log.error(
            "Отказ: запуск РЕАЛЬНОЙ торговли в неинтерактивном режиме без --yes. "
            "Добавьте флаг --yes (например, в systemd-юните), если это осознанно."
        )
        return False
    ans = input("Запуск торговли РЕАЛЬНЫМИ средствами. Продолжить? [y/N] ").strip()
    return ans.lower() in ("y", "yes", "д", "да")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="algo-live",
        description="Headless live-торговля (без GUI) по JSON-конфигу.",
    )
    parser.add_argument("--config", "-c", required=True, help="путь к JSON-конфигу")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="форсировать сухой прогон (без реальных ордеров), переопределяет конфиг",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="подтвердить реальную торговлю без интерактивного запроса (для systemd)",
    )
    parser.add_argument("--log-file", help="дополнительно писать журнал в файл")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="подробный журнал (DEBUG)"
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_file, args.verbose)

    try:
        cfg = load_run_config(args.config)
        exchange_id = str(cfg.get("exchange") or "").strip()
        mode = str(cfg.get("mode") or MODE_DEMO).strip()
        if mode not in (MODE_DEMO, MODE_REAL):
            raise ConfigError(f"«mode» должен быть '{MODE_DEMO}' или '{MODE_REAL}'")
        engine_cfg = build_engine_config(cfg)
        if args.dry_run:
            engine_cfg.dry_run = True
        creds = resolve_credentials(exchange_id, mode)
    except ConfigError as exc:
        log.error("Конфигурация: %s", exc)
        return 2

    if not _confirm_real(mode, engine_cfg.dry_run, args.yes):
        return 3

    try:
        client = ExchangeClient(exchange_id, mode=mode, creds=creds)
    except ExchangeError as exc:
        log.error("Подключение: %s", exc)
        return 4

    alerter = TelegramAlerter()
    engine = LiveTradingEngine(client, engine_cfg, on_event=make_event_handler(alerter))

    # Аккуратная остановка по сигналам (systemd stop шлёт SIGTERM).
    def _handle_signal(signum, _frame) -> None:
        log.info("Получен сигнал %s -- останавливаю торговый цикл…", signum)
        engine.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    label = EXCHANGES[exchange_id].label
    real = mode == MODE_REAL and not engine_cfg.dry_run
    banner = (
        f"algo-live: {label} [{mode}]"
        f"{' (сухой прогон)' if engine_cfg.dry_run else ''} • "
        f"{engine_cfg.symbol} • {engine_cfg.timeframe} • "
        f"{engine_cfg.strategy_cls.__name__}"
    )
    log.info("=== %s ===", banner)
    if real:
        log.warning("РЕАЛЬНЫЕ СРЕДСТВА: движок будет ставить ордера сам.")
    alerter.send(f"🚀 Старт {banner}")

    exit_code = 0
    try:
        engine.run()
    except Exception as exc:  # noqa: BLE001 - последний рубеж: логируем и выходим
        log.exception("Сбой торгового цикла: %s", exc)
        alerter.send(f"🛑 Сбой торгового цикла: {exc}")
        exit_code = 1
    finally:
        log.info("Торговый цикл завершён.")
        alerter.send("⏹️ algo-live остановлен.")
        alerter.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
