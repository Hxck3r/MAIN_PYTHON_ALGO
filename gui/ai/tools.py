"""Инструменты агента: чтение/запись файлов и проверка стратегий.

Инструменты описаны в формате OpenAI (function calling) и исполняются
``ToolRunner`` строго в пределах рабочего пространства проекта
(``ProjectContext``). Запись файла проходит через callback подтверждения —
GUI спрашивает пользователя, прежде чем что-то менять на диске.
"""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Callable, Optional

from .config import PROJECT_ROOT
from .project_context import ProjectContext, WorkspaceError

#: Схемы инструментов (передаются в поле ``tools`` запроса к LLM).
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "Показать содержимое каталога внутри проекта (файлы и подкаталоги). "
                "Используй, чтобы сориентироваться в структуре перед чтением."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Путь относительно корня проекта, напр. 'strategies' или '.'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Прочитать текстовый файл проекта (исходники backtester, стратегии, "
                "конфиги, отчёты). Всегда читай реальный код перед тем, как о нём "
                "рассуждать или его править — не выдумывай API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Путь к файлу относительно корня проекта.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Создать или перезаписать файл проекта (обычно strategies/<имя>.py "
                "или строку регистрации в strategies/__init__.py). Перед записью "
                "пользователь подтверждает изменение вручную. Передавай ПОЛНОЕ "
                "содержимое файла, а не патч."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Путь к файлу относительно корня проекта.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Полное новое содержимое файла.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_strategy",
            "description": (
                "Прогнать файл стратегии в изолированном подпроцессе: быстрый бэктест "
                "на синтетике + проверка причинности (анти-lookahead) для сигнальных "
                "стратегий. Обязательно вызывай это после создания/правки стратегии, "
                "чтобы доказать, что она запускается и не заглядывает в будущее."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Путь к файлу стратегии, напр. 'strategies/my_strategy.py'.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


#: callback подтверждения записи: (path, content) -> разрешено ли. Блокирующий.
ConfirmWrite = Callable[[str, str], bool]


class ToolRunner:
    """Исполняет вызовы инструментов в песочнице проекта."""

    def __init__(
        self,
        context: ProjectContext,
        confirm_write: Optional[ConfirmWrite] = None,
        on_activity: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.ctx = context
        self.confirm_write = confirm_write
        self.on_activity = on_activity

    # ------------------------------------------------------------------ #
    def run(self, name: str, arguments: dict[str, Any]) -> str:
        """Выполнить инструмент по имени; вернуть текстовый результат для модели."""
        try:
            if name == "list_directory":
                self._note(f"📁 list_directory({arguments.get('path', '.')})")
                return self.ctx.list_directory(arguments.get("path", "."))
            if name == "read_file":
                self._note(f"📄 read_file({arguments.get('path', '')})")
                return self.ctx.read_file(arguments["path"])
            if name == "write_file":
                return self._write(arguments["path"], arguments.get("content", ""))
            if name == "validate_strategy":
                return self._validate(arguments["path"])
            return f"Неизвестный инструмент: {name}"
        except KeyError as exc:
            return f"Ошибка: не передан обязательный аргумент {exc}."
        except WorkspaceError as exc:
            return f"Отказано: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Ошибка выполнения инструмента {name}: {exc}"

    # ------------------------------------------------------------------ #
    def _write(self, path: str, content: str) -> str:
        # сначала проверяем, что путь вообще внутри проекта — до запроса юзеру.
        self.ctx.resolve(path)
        if self.confirm_write is not None and not self.confirm_write(path, content):
            return (
                "Пользователь ОТКЛОНИЛ запись файла. Не повторяй её без явной "
                "просьбы; при необходимости объясни изменения текстом."
            )
        self._note(f"💾 write_file({path})")
        return self.ctx.write_file(path, content)

    def _validate(self, path: str) -> str:
        target = self.ctx.resolve(path)  # проверка границ песочницы
        rel = self.ctx.rel(target)
        self._note(f"🧪 validate_strategy({rel})")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "gui.ai._validate", rel],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return (
                "Проверка прервана по таймауту (120 с). Вероятно, в стратегии "
                "бесконечный цикл или слишком тяжёлый расчёт."
            )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        verdict = "ПРОВЕРКА ПРОЙДЕНА" if proc.returncode == 0 else "ЕСТЬ ПРОБЛЕМЫ"
        report = f"[{verdict}]\n{out}"
        if err:
            report += f"\n--- stderr ---\n{err[:2000]}"
        return report

    def _note(self, text: str) -> None:
        if self.on_activity:
            self.on_activity(text)


def parse_arguments(raw: str) -> dict[str, Any]:
    """Безопасный разбор JSON-аргументов инструмента (модель иногда шлёт мусор)."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
