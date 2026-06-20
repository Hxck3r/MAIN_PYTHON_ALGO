"""Персистентность состояния торгового цикла между перезапусками.

Зачем: при работе на VPS с автоперезапуском (systemd Restart=always) процесс
может упасть и подняться заново. Без сохранённого состояния движок:
    * не помнит намерения по SL/TP -> на первом баре sync_protective_orders
      снял бы биржевые защитные ордера у живой позиции (она осталась бы голой);
    * забыл бы про незалитый лимитный вход;
    * сбросил бы базис дневного убытка -> «рубильник» обнулялся бы рестартом;
    * мог бы повторно обработать уже отработанный бар.

Состояние -- маленький JSON в каталоге конфигурации (как credentials), свой файл
на каждую «личность» сессии (биржа+режим+символ+таймфрейм). Запись атомарная.
Это БЕСТ-ЭФФОРТ кэш ради устойчивости, а не источник истины: позиция и баланс
всегда перечитываются с биржи. Поэтому устаревшее/битое состояние безопасно --
оно лишь подсказывает SL/TP и базис убытка, всё остальное сверяется с биржей.
"""
from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from .credentials import config_dir

STATE_DIR_NAME = "live_state"


def _sanitize(text: str) -> str:
    """Безопасное имя файла из произвольной строки (символы биржи -> _)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "x"


def state_path(exchange_id: str, mode: str, symbol: str, timeframe: str) -> Path:
    """Путь к файлу состояния для конкретной торговой сессии."""
    name = "_".join(_sanitize(p) for p in (exchange_id, mode, symbol, timeframe))
    return config_dir() / STATE_DIR_NAME / f"{name}.json"


class StateStore:
    """Чтение/запись состояния одной сессии (атомарно, мягко к ошибкам)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def for_session(
        cls, exchange_id: str, mode: str, symbol: str, timeframe: str
    ) -> StateStore:
        return cls(state_path(exchange_id, mode, symbol, timeframe))

    def load(self) -> dict[str, Any]:
        """Прочитать состояние; {} если файла нет или он повреждён."""
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8")) or {}
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, state: dict[str, Any]) -> None:
        """Атомарно записать состояние (через временный файл + replace)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            try:
                os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600
            except OSError:
                pass
            os.replace(tmp, self.path)
        except OSError:
            pass  # не роняем торговлю из-за сбоя записи кэша состояния

    def clear(self) -> None:
        """Удалить файл состояния (например, при чистом старте сессии)."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
