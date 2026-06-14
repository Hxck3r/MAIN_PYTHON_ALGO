"""Тонкий клиент к OpenAI-совместимому Chat Completions API.

Работает с любым эндпоинтом формата OpenAI:
    * OpenRouter (облако, ключ в заголовке Authorization);
    * локальная LLM — Ollama / LM Studio / llama.cpp server (без ключа).

Зависимостей нет — только стандартная библиотека (``urllib``), потоковая
(SSE) выдача токенов читается построчно. Клиент не знает про Qt: его методы
вызываются из рабочего потока (``gui.ai.worker.ChatWorker``).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from .config import AssistantConfig


class LLMError(RuntimeError):
    """Ошибка обращения к LLM (сеть, HTTP, неверный ответ)."""


class StreamStopped(Exception):
    """Внутренний сигнал: пользователь прервал генерацию."""


# Тип callback'ов, передаваемых в stream_completion.
DeltaCallback = Callable[[str], None]
StopPredicate = Callable[[], bool]


class LLMClient:
    """Минимальный клиент Chat Completions с потоковой выдачей и tool-calling."""

    def __init__(self, config: AssistantConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    def _endpoint(self) -> str:
        base = self.config.base_url.strip().rstrip("/")
        if not base:
            raise LLMError("Не задан адрес эндпоинта (Base URL).")
        # base_url может уже включать /v1 — добавляем только хвост.
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        key = self.config.resolved_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        # OpenRouter использует эти заголовки для атрибуции (необязательны).
        if "openrouter" in self.config.base_url.lower():
            headers["HTTP-Referer"] = "https://localhost/algotester"
            headers["X-Title"] = "AlgoTester Quant Assistant"
        # ngrok на free-тарифе вставляет HTML-страницу предупреждения для
        # браузерных запросов — этот заголовок её отключает (иначе вместо
        # JSON прилетит HTML и разбор упадёт).
        if "ngrok" in self.config.base_url.lower():
            headers["ngrok-skip-browser-warning"] = "true"
        return headers

    def _payload(
        self, messages: list[dict[str, Any]], tools: Optional[list[dict]]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": float(self.config.temperature),
        }
        if self.config.max_tokens and self.config.max_tokens > 0:
            payload["max_tokens"] = int(self.config.max_tokens)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    # ------------------------------------------------------------------ #
    def stream_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        on_delta: Optional[DeltaCallback] = None,
        should_stop: Optional[StopPredicate] = None,
    ) -> dict[str, Any]:
        """Один обмен с моделью с потоковой выдачей.

        Возвращает собранное сообщение ассистента в формате OpenAI:
        ``{"role": "assistant", "content": str, "tool_calls": [...]?}``.
        Текстовые дельты отдаются в ``on_delta`` по мере поступления.
        ``should_stop`` опрашивается между чанками — при True генерация
        прерывается (StreamStopped).
        """
        data = json.dumps(self._payload(messages, tools)).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(), data=data, headers=self._headers(), method="POST"
        )

        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: Optional[str] = None

        try:
            response = urllib.request.urlopen(
                request, timeout=self.config.request_timeout
            )
        except urllib.error.HTTPError as exc:
            raise LLMError(self._format_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Не удалось подключиться к {self.config.base_url}: {exc.reason}. "
                f"Проверьте адрес эндпоинта и (для локальной LLM) что сервер запущен."
            ) from exc

        try:
            for raw_line in response:
                if should_stop and should_stop():
                    raise StreamStopped()
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                fr = self._consume_chunk(chunk, content_parts, tool_calls, on_delta)
                if fr:
                    finish_reason = fr
        finally:
            response.close()

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if tool_calls:
            message["tool_calls"] = [
                tool_calls[i] for i in sorted(tool_calls)
            ]
            message.setdefault("content", "")
        message["_finish_reason"] = finish_reason or (
            "tool_calls" if tool_calls else "stop"
        )
        return message

    # ------------------------------------------------------------------ #
    @staticmethod
    def _consume_chunk(
        chunk: dict[str, Any],
        content_parts: list[str],
        tool_calls: dict[int, dict[str, Any]],
        on_delta: Optional[DeltaCallback],
    ) -> Optional[str]:
        """Разбирает один SSE-чанк, накапливая текст и вызовы инструментов."""
        choices = chunk.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        delta = choice.get("delta") or {}

        text = delta.get("content")
        if text:
            content_parts.append(text)
            if on_delta:
                on_delta(text)

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_calls.setdefault(
                idx,
                {"id": "", "type": "function",
                 "function": {"name": "", "arguments": ""}},
            )
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]

        return choice.get("finish_reason")

    @staticmethod
    def _format_http_error(exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            msg = parsed.get("error", {})
            detail = msg.get("message") if isinstance(msg, dict) else str(msg)
        except Exception:
            detail = ""
        hint = ""
        if exc.code == 401:
            hint = " Похоже, неверный или отсутствующий API-ключ."
        elif exc.code == 402:
            hint = " Недостаточно средств на счёте провайдера."
        elif exc.code == 404:
            hint = " Проверьте имя модели и адрес эндпоинта."
        elif exc.code == 429:
            hint = " Превышен лимит запросов — подождите и повторите."
        return f"HTTP {exc.code}: {detail or exc.reason}.{hint}"
