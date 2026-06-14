"""Настройки ИИ-ассистента: провайдер, эндпоинт, ключ, модель.

Хранятся в `.ai_assistant.json` в корне проекта (файл в .gitignore — ключ
не попадает в репозиторий). API-ключ можно не писать в файл, а взять из
переменной окружения (для OpenRouter — ``OPENROUTER_API_KEY``).

Локальная LLM (Ollama, LM Studio, llama.cpp server) подключается так же,
как облако, — это OpenAI-совместимый HTTP-эндпоинт, только без ключа.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: корень проекта (…/gui/ai/config.py -> parents[2])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / ".ai_assistant.json"

#: пресеты провайдеров: подсказывают base_url и типовые модели в окне настроек.
#: Любой OpenAI-совместимый сервис добавляется через «Другой».
PRESETS: dict[str, dict] = {
    "OpenRouter (облако)": {
        "base_url": "https://openrouter.ai/api/v1",
        "needs_key": True,
        "models": [
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-opus-4.1",
            "openai/gpt-5",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-chat",
            "qwen/qwen-2.5-coder-32b-instruct",
        ],
    },
    "Локальная — Ollama": {
        "base_url": "http://localhost:11434/v1",
        "needs_key": False,
        "models": [
            "qwen2.5-coder:14b",
            "qwen2.5-coder:32b",
            "llama3.1:8b",
            "deepseek-r1:14b",
        ],
    },
    "Локальная — LM Studio": {
        "base_url": "http://localhost:1234/v1",
        "needs_key": False,
        "models": ["local-model"],
    },
    "Google Colab — ngrok": {
        "base_url": "https://ВАШ-АДРЕС.ngrok-free.app/v1",
        "needs_key": False,
        "models": ["qwen2.5-coder:7b", "local-model"],
        "hint": (
            "LLM запущена в Colab (Ollama/vLLM/llama.cpp), наружу проброшена через "
            "ngrok. Вставьте сюда публичный https-адрес ngrok и допишите /v1. "
            "Ключ не нужен (если сервер сам его не требует)."
        ),
    },
    "Google Colab — Cloudflare": {
        "base_url": "https://ВАШ-АДРЕС.trycloudflare.com/v1",
        "needs_key": False,
        "models": ["qwen2.5-coder:7b", "local-model"],
        "hint": (
            "LLM запущена в Colab, наружу проброшена через Cloudflare Tunnel "
            "(cloudflared / trycloudflare). Вставьте выданный https-адрес и "
            "допишите /v1. Ключ не нужен."
        ),
    },
    "Другой (OpenAI-совместимый)": {
        "base_url": "",
        "needs_key": False,
        "models": [],
    },
}


@dataclass
class AssistantConfig:
    """Параметры подключения к LLM и поведения ассистента."""

    provider: str = "OpenRouter (облако)"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""               # пусто => берём из переменной окружения
    model: str = "anthropic/claude-sonnet-4.5"
    temperature: float = 0.3
    max_tokens: int = 0             # 0 => не ограничивать (решает сервер)
    include_rules: bool = True      # вшивать AI_STRATEGY_RULES.md в системный промпт
    request_timeout: int = 600      # сек на потоковый ответ

    # Поля, которые не сериализуем в файл (служебные), можно добавить здесь.
    _extra: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    def resolved_key(self) -> str:
        """Ключ: из конфига, иначе из переменных окружения провайдера."""
        if self.api_key.strip():
            return self.api_key.strip()
        for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
            value = os.environ.get(var, "").strip()
            if value:
                return value
        return ""

    def is_local(self) -> bool:
        host = self.base_url.lower()
        return "localhost" in host or "127.0.0.1" in host or "0.0.0.0" in host

    def requires_key(self) -> bool:
        """Нужен ли API-ключ. Определяется пресетом провайдера; для туннелей
        (Colab/ngrok/Cloudflare) и локальных серверов — нет."""
        preset = PRESETS.get(self.provider)
        if preset is not None:
            return bool(preset.get("needs_key", False))
        # Неизвестный/кастомный провайдер: ключ нужен только если адрес похож
        # на публичное облако (не localhost).
        return not self.is_local()

    def is_ready(self) -> bool:
        """Готов к запросам: есть эндпоинт, модель и (если нужен) ключ."""
        if not self.base_url.strip() or not self.model.strip():
            return False
        if self.requires_key():
            return bool(self.resolved_key())
        return True

    def missing_reason(self) -> str:
        if not self.base_url.strip():
            return "Не задан адрес эндпоинта (Base URL)."
        if "ВАШ-АДРЕС" in self.base_url:
            return "Замените ВАШ-АДРЕС в Base URL на реальный адрес туннеля."
        if not self.model.strip():
            return "Не задано имя модели."
        if self.requires_key() and not self.resolved_key():
            return (
                "Не задан API-ключ. Введите его в настройках или задайте "
                "переменную окружения OPENROUTER_API_KEY."
            )
        return ""


# ---------------------------------------------------------------------- #
# Загрузка / сохранение
# ---------------------------------------------------------------------- #
_PERSISTED = (
    "provider", "base_url", "api_key", "model", "temperature",
    "max_tokens", "include_rules", "request_timeout",
)


def load_config() -> AssistantConfig:
    """Читает конфиг из файла; при отсутствии/ошибке — значения по умолчанию."""
    cfg = AssistantConfig()
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cfg
        for key in _PERSISTED:
            if key in raw:
                setattr(cfg, key, raw[key])
    return cfg


def save_config(cfg: AssistantConfig) -> None:
    """Сохраняет только публичные поля; служебные (_extra) не пишем."""
    data = {k: v for k, v in asdict(cfg).items() if k in _PERSISTED}
    CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
