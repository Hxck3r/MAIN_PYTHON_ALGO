"""Диалог настроек подключения к LLM (провайдер, эндпоинт, ключ, модель)."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .config import PRESETS, AssistantConfig, save_config
from .llm_client import LLMClient, LLMError


class _PingWorker(QThread):
    """Пробный запрос к модели (проверка, что эндпоинт/ключ/модель рабочие)."""

    done = Signal(bool, str)

    def __init__(self, config: AssistantConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:
        try:
            client = LLMClient(self._config)
            msg = client.stream_completion(
                [{"role": "user", "content": "ping"}],
            )
            text = (msg.get("content") or "").strip()
            self.done.emit(True, text[:80] or "(пустой ответ, но связь есть)")
        except LLMError as exc:
            self.done.emit(False, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, f"{exc}")


class SettingsDialog(QDialog):
    """Редактирование AssistantConfig с пресетами провайдеров и тестом связи."""

    def __init__(self, config: AssistantConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки ИИ-ассистента")
        self.setMinimumWidth(560)
        self._config = config
        self._ping: _PingWorker | None = None
        self._build_ui()
        self._load(config)

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.combo_provider = QComboBox()
        self.combo_provider.addItems(list(PRESETS))
        self.combo_provider.currentTextChanged.connect(self._on_preset)
        form.addRow("Провайдер:", self.combo_provider)

        self.edit_base = QLineEdit()
        self.edit_base.setPlaceholderText("https://openrouter.ai/api/v1")
        form.addRow("Base URL:", self.edit_base)

        self.edit_key = QLineEdit()
        self.edit_key.setEchoMode(QLineEdit.Password)
        self.edit_key.setPlaceholderText(
            "пусто = взять из переменной окружения OPENROUTER_API_KEY"
        )
        form.addRow("API-ключ:", self.edit_key)

        self.combo_model = QComboBox()
        self.combo_model.setEditable(True)
        form.addRow("Модель:", self.combo_model)

        self.spin_temp = QDoubleSpinBox(minimum=0.0, maximum=2.0, value=0.3)
        self.spin_temp.setSingleStep(0.1)
        self.spin_temp.setDecimals(2)
        form.addRow("Температура:", self.spin_temp)

        self.spin_max_tokens = QSpinBox(minimum=0, maximum=200_000, value=0)
        self.spin_max_tokens.setSingleStep(512)
        self.spin_max_tokens.setSpecialValueText("без лимита")
        form.addRow("Макс. токенов:", self.spin_max_tokens)

        self.spin_timeout = QSpinBox(minimum=30, maximum=3600, value=600)
        self.spin_timeout.setSuffix(" с")
        form.addRow("Таймаут ответа:", self.spin_timeout)

        self.chk_rules = QCheckBox(
            "Вшивать контракт стратегий (AI_STRATEGY_RULES.md) в системный промпт"
        )
        form.addRow(self.chk_rules)
        layout.addLayout(form)

        # Подсказка по выбранному провайдеру (заполняется из пресета).
        self.lbl_preset_hint = QLabel("")
        self.lbl_preset_hint.setWordWrap(True)
        self.lbl_preset_hint.setStyleSheet(
            "color: #7aa; background: #222; border-radius: 6px; padding: 6px;"
        )
        self.lbl_preset_hint.setVisible(False)
        layout.addWidget(self.lbl_preset_hint)

        # Строка теста подключения.
        test_row = QHBoxLayout()
        self.btn_test = QPushButton("Проверить подключение")
        self.btn_test.clicked.connect(self._on_test)
        self.lbl_test = QLabel("")
        self.lbl_test.setWordWrap(True)
        test_row.addWidget(self.btn_test)
        test_row.addWidget(self.lbl_test, stretch=1)
        layout.addLayout(test_row)

        hint = QLabel(
            "OpenRouter: введите ключ из openrouter.ai/keys. Локальная LLM "
            "(Ollama/LM Studio) ключа не требует — запустите сервер и укажите "
            "его адрес. Google Colab: LLM в ноутбуке + публичный туннель ngrok "
            "или Cloudflare; вставьте выданный https-адрес и допишите /v1. "
            "Ключ хранится в .ai_assistant.json (в .gitignore)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ #
    def _load(self, config: AssistantConfig) -> None:
        if config.provider in PRESETS:
            self.combo_provider.setCurrentText(config.provider)
        self._show_hint(config.provider)
        self._fill_models(config.provider)
        self.edit_base.setText(config.base_url)
        self.edit_key.setText(config.api_key)
        self.combo_model.setCurrentText(config.model)
        self.spin_temp.setValue(config.temperature)
        self.spin_max_tokens.setValue(config.max_tokens)
        self.spin_timeout.setValue(config.request_timeout)
        self.chk_rules.setChecked(config.include_rules)

    def _fill_models(self, provider: str) -> None:
        preset = PRESETS.get(provider, {})
        current = self.combo_model.currentText()
        self.combo_model.clear()
        self.combo_model.addItems(preset.get("models", []))
        if current:
            self.combo_model.setCurrentText(current)

    def _on_preset(self, provider: str) -> None:
        preset = PRESETS.get(provider, {})
        if preset.get("base_url"):
            self.edit_base.setText(preset["base_url"])
        self._fill_models(provider)
        if preset.get("models"):
            self.combo_model.setCurrentText(preset["models"][0])
        self._show_hint(provider)

    def _show_hint(self, provider: str) -> None:
        hint = PRESETS.get(provider, {}).get("hint", "")
        self.lbl_preset_hint.setText(hint)
        self.lbl_preset_hint.setVisible(bool(hint))

    # ------------------------------------------------------------------ #
    def _draft(self) -> AssistantConfig:
        return AssistantConfig(
            provider=self.combo_provider.currentText(),
            base_url=self.edit_base.text().strip(),
            api_key=self.edit_key.text().strip(),
            model=self.combo_model.currentText().strip(),
            temperature=self.spin_temp.value(),
            max_tokens=self.spin_max_tokens.value(),
            include_rules=self.chk_rules.isChecked(),
            request_timeout=self.spin_timeout.value(),
        )

    def _on_test(self) -> None:
        draft = self._draft()
        reason = draft.missing_reason()
        if reason:
            self.lbl_test.setText(f"⚠ {reason}")
            self.lbl_test.setStyleSheet("color: #ff9f43;")
            return
        self.btn_test.setEnabled(False)
        self.lbl_test.setText("Проверяю…")
        self.lbl_test.setStyleSheet("color: gray;")
        self._ping = _PingWorker(draft, parent=self)
        self._ping.done.connect(self._on_test_done)
        self._ping.start()

    def _on_test_done(self, ok: bool, message: str) -> None:
        self.btn_test.setEnabled(True)
        if ok:
            self.lbl_test.setText(f"✓ Связь есть: {message}")
            self.lbl_test.setStyleSheet("color: #2ecc71;")
        else:
            self.lbl_test.setText(f"✗ {message}")
            self.lbl_test.setStyleSheet("color: #ff5c5c;")

    def _on_save(self) -> None:
        self._config = self._draft()
        save_config(self._config)
        self.accept()

    def result_config(self) -> AssistantConfig:
        return self._config
