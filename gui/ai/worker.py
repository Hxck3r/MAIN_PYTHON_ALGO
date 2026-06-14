"""Фоновый агентный цикл ИИ-ассистента (QThread).

Один экземпляр обслуживает один «ход» диалога: отправляет историю модели,
стримит текст, при необходимости исполняет инструменты и повторяет, пока
модель не перестанет их вызывать. Сетевые/файловые операции идут в фоне,
GUI не блокируется. Подтверждение записи файла запрашивается у главного
потока через сигнал + блокирующее ожидание ответа.
"""
from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QThread, Signal

from .llm_client import LLMClient, LLMError, StreamStopped
from .tools import TOOL_SCHEMAS, ToolRunner, parse_arguments

#: предохранитель от зацикливания инструментов в одном ходе.
_MAX_TOOL_ROUNDS = 12


class ChatWorker(QThread):
    """Выполняет один ход диалога (возможно, с цепочкой вызовов инструментов)."""

    #: текстовая дельта ответа ассистента (для живого отображения).
    delta = Signal(str)
    #: краткая отметка активности инструмента ("📄 read_file(...)").
    tool_activity = Signal(str)
    #: запрос подтверждения записи файла: (path, content).
    confirm_write = Signal(str, str)
    #: ход завершён успешно (полный текст последнего ответа ассистента).
    finished_ok = Signal(str)
    #: ошибка (текст для показа пользователю).
    failed = Signal(str)
    #: ход прерван пользователем.
    stopped = Signal(str)

    def __init__(
        self,
        client: LLMClient,
        messages: list[dict[str, Any]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        #: общий список сообщений в формате OpenAI (мутируется по месту).
        self.messages = messages
        self._tool_runner = ToolRunner(
            client_context(client),
            confirm_write=self._ask_confirm,
            on_activity=self.tool_activity.emit,
        )
        # Механизм блокирующего подтверждения записи из главного потока.
        self._confirm_event = threading.Event()
        self._confirm_result = False

    # ------------------------------------------------------------------ #
    def run(self) -> None:  # noqa: D401 — точка входа потока
        try:
            self._run_loop()
        except StreamStopped:
            self.stopped.emit("Генерация остановлена.")
        except LLMError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Внутренняя ошибка ассистента: {exc}")

    def _run_loop(self) -> None:
        last_text = ""
        for _ in range(_MAX_TOOL_ROUNDS):
            if self.isInterruptionRequested():
                raise StreamStopped()
            assistant = self._client.stream_completion(
                self._api_messages(),
                tools=TOOL_SCHEMAS,
                on_delta=self.delta.emit,
                should_stop=self.isInterruptionRequested,
            )
            finish = assistant.pop("_finish_reason", "stop")
            self.messages.append(assistant)
            if assistant.get("content"):
                last_text = assistant["content"]

            tool_calls = assistant.get("tool_calls")
            if not tool_calls or finish != "tool_calls":
                self.finished_ok.emit(last_text)
                return

            # Если перед вызовом инструментов была преамбула — отделим её от
            # последующего текста пустой строкой, чтобы они не слиплись.
            if assistant.get("content"):
                self.delta.emit("\n\n")

            # Исполняем все запрошенные инструменты и докладываем результаты.
            for call in tool_calls:
                if self.isInterruptionRequested():
                    raise StreamStopped()
                self._dispatch_tool(call)

        # Достигнут лимит раундов — не зацикливаемся.
        self.messages.append({
            "role": "system",
            "content": "Достигнут лимит вызовов инструментов за один ход. "
                       "Подведи итог тем, что уже выяснено.",
        })
        self.finished_ok.emit(last_text)

    # ------------------------------------------------------------------ #
    def _dispatch_tool(self, call: dict[str, Any]) -> None:
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = parse_arguments(fn.get("arguments", ""))
        result = self._tool_runner.run(name, args)
        self.messages.append({
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "content": result,
        })

    def _api_messages(self) -> list[dict[str, Any]]:
        """Сообщения без служебных полей (_finish_reason мог остаться где-то)."""
        clean = []
        for msg in self.messages:
            clean.append({k: v for k, v in msg.items() if not k.startswith("_")})
        return clean

    # ------------------------------------------------------------------ #
    # Подтверждение записи: вызывается из фонового потока, ждёт главный.
    # ------------------------------------------------------------------ #
    def _ask_confirm(self, path: str, content: str) -> bool:
        self._confirm_event.clear()
        self.confirm_write.emit(path, content)
        self._confirm_event.wait()
        return self._confirm_result

    def resolve_confirm(self, allowed: bool) -> None:
        """Главный поток сообщает решение пользователя по записи файла."""
        self._confirm_result = allowed
        self._confirm_event.set()


def client_context(client: LLMClient):
    """ProjectContext привязан к окну; воркер получает его через клиент-обёртку.

    Чтобы не плодить зависимости, ProjectContext кладётся на клиента атрибутом
    окном при создании. Если его нет — собираем дефолтный (корень проекта).
    """
    ctx = getattr(client, "project_context", None)
    if ctx is not None:
        return ctx
    from .project_context import ProjectContext
    return ProjectContext()
