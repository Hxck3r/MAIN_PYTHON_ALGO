"""Хранение API-ключей бирж вне репозитория.

Ключи лежат в каталоге конфигурации пользователя (НЕ в проекте), в одном
JSON-файле с правами доступа 0600 (чтение/запись только владельцу). Это
осознанно простой вариант для десктоп-инструмента: ключи хранятся в открытом
виде на машине пользователя. Для повышенной безопасности в будущем сюда можно
подключить системный keyring -- API (`load`/`save`/`delete`) при этом не меняется.

Структура файла::

    {
      "bingx": {"demo": {"api_key": "...", "secret": "..."}, "real": {...}},
      "lighter": {"real": {"private_key": "...", "account_index": "0",
                           "api_key_index": "0"}},
      ...
    }

Ключ верхнего уровня -- id биржи (как в live.exchange.EXCHANGES); второй уровень --
режим: "demo" (тестнет/демо) или "real" (реальный счёт). Набор полей третьего
уровня зависит от схемы биржи (CEX -- api_key/secret; DEX -- private_key и индексы
аккаунта), см. live.exchange.ExchangeMeta.cred_fields.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

APP_DIR_NAME = "main_python_algo"
CRED_FILENAME = "credentials.json"


def config_dir() -> Path:
    """Каталог конфигурации приложения (платформозависимый), без создания."""
    if os.name == "nt":  # Windows
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    else:                # Linux / macOS
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_DIR_NAME


def credentials_path() -> Path:
    return config_dir() / CRED_FILENAME


@dataclass
class Credentials:
    """Набор полей ключей для биржи и режима (состав зависит от схемы биржи).

    Значения хранятся в словаре ``values`` с ключами из ExchangeMeta.cred_fields
    (например ``api_key``/``secret`` у CEX или ``private_key``/``account_index``/
    ``api_key_index`` у DEX Lighter).
    """

    values: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> str:
        return str(self.values.get(key, "") or "")

    @property
    def api_key(self) -> str:
        return self.get("api_key")

    @property
    def secret(self) -> str:
        return self.get("secret")

    def is_filled(self, required: Iterable[str] = ("api_key", "secret")) -> bool:
        """Все ли перечисленные поля заполнены непустым значением."""
        return all(self.get(k).strip() for k in required)


def _read_all() -> dict:
    path = credentials_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_all(data: dict) -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Пишем во временный файл и атомарно подменяем -- чтобы сбой записи не
    # оставил повреждённый JSON с ключами.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass  # на некоторых ФС (например, Windows) chmod бессмысленен
    os.replace(tmp, path)


def load(exchange_id: str, mode: str) -> Credentials:
    """Загрузить ключи для (биржа, режим). Пустые, если не сохранены."""
    section = _read_all().get(exchange_id, {}).get(mode, {})
    if not isinstance(section, dict):
        section = {}
    return Credentials(values={str(k): str(v) for k, v in section.items()})


def save(exchange_id: str, mode: str, creds: Credentials) -> Path:
    """Сохранить ключи для (биржа, режим). Возвращает путь к файлу."""
    data = _read_all()
    data.setdefault(exchange_id, {})[mode] = {
        str(k): str(v).strip() for k, v in creds.values.items()
    }
    _write_all(data)
    return credentials_path()


def delete(exchange_id: str, mode: Optional[str] = None) -> None:
    """Удалить ключи биржи целиком или только для одного режима."""
    data = _read_all()
    if exchange_id not in data:
        return
    if mode is None:
        data.pop(exchange_id, None)
    else:
        data[exchange_id].pop(mode, None)
        if not data[exchange_id]:
            data.pop(exchange_id, None)
    _write_all(data)
