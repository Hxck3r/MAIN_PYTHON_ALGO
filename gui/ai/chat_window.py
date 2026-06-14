"""Окно чата ИИ-ассистента «Квант».

Отдельное окно (открывается кнопкой главного окна). Поддерживает:
    * переписку с потоковым выводом и Markdown-разметкой;
    * вложения — текстовые файлы и изображения (для vision-моделей);
    * исполнение инструментов агента в файлах проекта с подтверждением записи;
    * настройку подключения к OpenRouter или локальной LLM.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .config import AssistantConfig, load_config
from .llm_client import LLMClient
from .project_context import ProjectContext
from .settings_dialog import SettingsDialog
from .system_prompt import build_system_prompt
from .worker import ChatWorker

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_MAX_ATTACH_TEXT = 60_000  # символов на текстовое вложение


# --------------------------------------------------------------------------- #
class _AutoTextBrowser(QTextBrowser):
    """QTextBrowser, который подгоняет высоту под содержимое (пузырь чата)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def set_markdown(self, text: str) -> None:
        self.setMarkdown(text)
        self._adjust_height()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._adjust_height()

    def _adjust_height(self) -> None:
        doc = self.document()
        doc.setTextWidth(self.viewport().width())
        height = int(doc.size().height()) + 8
        self.setFixedHeight(max(height, 28))


class _MessageBubble(QFrame):
    """Один пузырь сообщения: заголовок роли + Markdown-текст."""

    _STYLES = {
        "user": ("Вы", "#2d4a63"),
        "assistant": ("Квант", "#2f3b2f"),
        "tool": ("Инструмент", "#3a3320"),
        "error": ("Ошибка", "#4a2a2a"),
    }

    def __init__(self, role: str, parent=None) -> None:
        super().__init__(parent)
        title, color = self._STYLES.get(role, ("", "#333"))
        self.setStyleSheet(
            f"_MessageBubble {{ background: {color}; border-radius: 8px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(2)
        if title:
            header = QLabel(title)
            f = QFont()
            f.setBold(True)
            f.setPointSize(max(f.pointSize() - 1, 8))
            header.setFont(f)
            header.setStyleSheet("color: #aab; background: transparent;")
            layout.addWidget(header)
        self.body = _AutoTextBrowser()
        self.body.setStyleSheet("background: transparent;")
        layout.addWidget(self.body)
        self._text = ""

    def set_text(self, text: str) -> None:
        self._text = text
        self.body.set_markdown(text)

    def append_text(self, chunk: str) -> None:
        self._text += chunk
        self.body.set_markdown(self._text)

    def text(self) -> str:
        return self._text


# --------------------------------------------------------------------------- #
class AssistantWindow(QWidget):
    """Главное окно чата ассистента."""

    def __init__(
        self,
        state_provider: Optional[Callable[[], str]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ИИ-ассистент «Квант»")
        self.resize(820, 880)
        # окно живёт самостоятельно поверх приложения, но не модально.
        self.setWindowFlag(Qt.Window, True)

        self._config: AssistantConfig = load_config()
        self._context = ProjectContext(state_provider=state_provider)
        self._client = self._make_client()
        self._messages: list[dict[str, Any]] = []
        self._attachments: list[dict[str, Any]] = []
        self._worker: Optional[ChatWorker] = None
        self._stream_bubble: Optional[_MessageBubble] = None

        self._build_ui()
        self._refresh_header()
        self._check_ready()

    # ------------------------------------------------------------------ #
    def _make_client(self) -> LLMClient:
        client = LLMClient(self._config)
        # воркер берёт песочницу проекта с клиента (см. worker.client_context).
        client.project_context = self._context  # type: ignore[attr-defined]
        return client

    # ------------------------------------------------------------------ #
    # Построение интерфейса
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # --- Шапка --------------------------------------------------------#
        header = QHBoxLayout()
        self.lbl_model = QLabel()
        self.lbl_model.setStyleSheet("color: gray;")
        btn_settings = QPushButton("⚙ Настройки")
        btn_settings.clicked.connect(self._open_settings)
        btn_clear = QPushButton("🗑 Новый чат")
        btn_clear.clicked.connect(self._new_chat)
        header.addWidget(self.lbl_model, stretch=1)
        header.addWidget(btn_clear)
        header.addWidget(btn_settings)
        root.addLayout(header)

        # --- Лента сообщений ---------------------------------------------#
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self._transcript = QWidget()
        self._transcript_layout = QVBoxLayout(self._transcript)
        self._transcript_layout.setAlignment(Qt.AlignTop)
        self._transcript_layout.setSpacing(8)
        self.scroll.setWidget(self._transcript)
        root.addWidget(self.scroll, stretch=1)

        # --- Активность инструмента --------------------------------------#
        self.lbl_activity = QLabel("")
        self.lbl_activity.setStyleSheet("color: #d6a; padding: 0 4px;")
        self.lbl_activity.setWordWrap(True)
        root.addWidget(self.lbl_activity)

        # --- Вложения -----------------------------------------------------#
        self.lbl_attach = QLabel("")
        self.lbl_attach.setStyleSheet("color: #7aa; padding: 0 4px;")
        self.lbl_attach.setWordWrap(True)
        root.addWidget(self.lbl_attach)

        # --- Ввод ---------------------------------------------------------#
        input_row = QHBoxLayout()
        self.input = _ChatInput(self._send)
        self.input.setPlaceholderText(
            "Спросите про стратегию, метрики, переобучение… "
            "Enter — отправить, Shift+Enter — перенос строки."
        )
        self.input.setMaximumHeight(120)
        input_row.addWidget(self.input, stretch=1)

        buttons = QVBoxLayout()
        btn_file = QPushButton("📎 Файл")
        btn_file.clicked.connect(lambda: self._attach(image=False))
        btn_img = QPushButton("🖼 Картинка")
        btn_img.clicked.connect(lambda: self._attach(image=True))
        self.btn_send = QPushButton("Отправить")
        self.btn_send.clicked.connect(self._send)
        self.btn_stop = QPushButton("■ Стоп")
        self.btn_stop.setStyleSheet("color: #ff5c5c;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        buttons.addWidget(btn_file)
        buttons.addWidget(btn_img)
        buttons.addWidget(self.btn_send)
        buttons.addWidget(self.btn_stop)
        input_row.addLayout(buttons)
        root.addLayout(input_row)

    # ------------------------------------------------------------------ #
    def _refresh_header(self) -> None:
        where = "локально" if self._config.is_local() else self._config.provider
        self.lbl_model.setText(f"Модель: {self._config.model or '—'}  ·  {where}")

    def _check_ready(self) -> None:
        reason = self._config.missing_reason()
        if reason:
            self._add_system_note(
                f"⚙ Ассистент ещё не настроен: {reason}\n\n"
                f"Нажмите «⚙ Настройки», выберите OpenRouter (введите API-ключ) "
                f"или локальную LLM (Ollama / LM Studio) и укажите модель."
            )

    # ------------------------------------------------------------------ #
    # Отправка сообщения
    # ------------------------------------------------------------------ #
    def _send(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        text = self.input.toPlainText().strip()
        if not text and not self._attachments:
            return
        reason = self._config.missing_reason()
        if reason:
            QMessageBox.warning(self, "Не настроено", reason)
            self._open_settings()
            return

        # Системное сообщение со свежим снимком состояния — всегда индекс 0.
        system_text = build_system_prompt(self._config, self._context)
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = system_text
        else:
            self._messages.insert(0, {"role": "system", "content": system_text})

        display_text, api_content = self._build_user_content(text)
        self._messages.append({"role": "user", "content": api_content})

        user_bubble = self._add_bubble("user")
        user_bubble.set_text(display_text)

        self.input.clear()
        self._clear_attachments()

        self._stream_bubble = self._add_bubble("assistant")
        self._stream_bubble.set_text("…")
        self._set_busy(True)

        self._worker = ChatWorker(self._client, self._messages, parent=self)
        self._worker.delta.connect(self._on_delta)
        self._worker.tool_activity.connect(self._on_activity)
        self._worker.confirm_write.connect(self._on_confirm_write)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.start()

    def _build_user_content(self, text: str):
        """Готовит (текст для показа, content для API) с учётом вложений."""
        text_blocks = [text] if text else []
        image_parts: list[dict] = []
        shown: list[str] = []
        for att in self._attachments:
            if att["kind"] == "image":
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": att["data_url"]},
                })
                shown.append(f"🖼 {att['name']}")
            else:
                text_blocks.append(
                    f"\n--- Прикреплённый файл: {att['name']} ---\n"
                    f"```\n{att['text']}\n```"
                )
                shown.append(f"📎 {att['name']}")

        joined = "\n".join(text_blocks)
        display = text or "(вложения)"
        if shown:
            display += "\n\n_" + ", ".join(shown) + "_"

        if image_parts:
            content: Any = [{"type": "text", "text": joined or "(см. изображение)"}]
            content.extend(image_parts)
        else:
            content = joined
        return display, content

    # ------------------------------------------------------------------ #
    # Сигналы воркера
    # ------------------------------------------------------------------ #
    def _on_delta(self, chunk: str) -> None:
        if self._stream_bubble is None:
            return
        if self._stream_bubble.text() == "…":
            self._stream_bubble.set_text("")
        self._stream_bubble.append_text(chunk)
        self._scroll_to_bottom()

    def _on_activity(self, note: str) -> None:
        self.lbl_activity.setText(note)

    def _on_confirm_write(self, path: str, content: str) -> None:
        preview = content if len(content) <= 4000 else content[:4000] + "\n…"
        box = QMessageBox(self)
        box.setWindowTitle("Подтвердите запись файла")
        box.setIcon(QMessageBox.Question)
        box.setText(
            f"Ассистент хочет записать файл:\n\n{path}\n\n"
            f"({len(content)} символов). Разрешить?"
        )
        box.setDetailedText(preview)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        allowed = box.exec() == QMessageBox.Yes
        if self._worker is not None:
            self._worker.resolve_confirm(allowed)
        self.lbl_activity.setText(
            f"💾 Запись {path}: {'разрешена' if allowed else 'отклонена'}"
        )

    def _on_finished(self, text: str) -> None:
        # Текст уже накоплен по дельтам (включая преамбулы между вызовами
        # инструментов). Перезаписываем пузырь только если ничего не стримилось.
        if self._stream_bubble is not None and self._stream_bubble.text() in ("…", ""):
            self._stream_bubble.set_text(text or "_(нет текстового ответа)_")
        self._stream_bubble = None
        self.lbl_activity.setText("")
        self._set_busy(False)
        self._scroll_to_bottom()

    def _on_failed(self, message: str) -> None:
        if self._stream_bubble is not None and self._stream_bubble.text() == "…":
            # пустой пузырь-заготовку убираем, ошибку показываем отдельно.
            self._stream_bubble.set_text("_(прервано)_")
        self._stream_bubble = None
        self._add_bubble("error").set_text(f"**Ошибка:** {message}")
        self.lbl_activity.setText("")
        self._set_busy(False)
        self._scroll_to_bottom()

    def _on_stopped(self, message: str) -> None:
        if self._stream_bubble is not None:
            cur = self._stream_bubble.text()
            self._stream_bubble.set_text(
                (cur if cur != "…" else "") + "\n\n_(остановлено)_"
            )
        self._stream_bubble = None
        self.lbl_activity.setText("")
        self._set_busy(False)

    # ------------------------------------------------------------------ #
    def _stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self.btn_stop.setEnabled(False)
            self.lbl_activity.setText("Останавливаю…")

    def _set_busy(self, busy: bool) -> None:
        self.btn_send.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        self.input.setEnabled(not busy)

    # ------------------------------------------------------------------ #
    # Вложения
    # ------------------------------------------------------------------ #
    def _attach(self, image: bool) -> None:
        if image:
            flt = "Изображения (*.png *.jpg *.jpeg *.gif *.webp *.bmp)"
        else:
            flt = "Текст и данные (*.py *.md *.txt *.csv *.json *.log);;Все файлы (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Прикрепить", "", flt)
        for path in paths:
            self._add_attachment(Path(path))
        self._refresh_attachments()

    def _add_attachment(self, path: Path) -> None:
        suffix = path.suffix.lower()
        try:
            if suffix in _IMAGE_SUFFIXES:
                raw = path.read_bytes()
                mime = "jpeg" if suffix in (".jpg", ".jpeg") else suffix[1:]
                b64 = base64.b64encode(raw).decode("ascii")
                self._attachments.append({
                    "kind": "image",
                    "name": path.name,
                    "data_url": f"data:image/{mime};base64,{b64}",
                })
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                if len(text) > _MAX_ATTACH_TEXT:
                    text = text[:_MAX_ATTACH_TEXT] + "\n… (обрезано)"
                self._attachments.append({
                    "kind": "text", "name": path.name, "text": text,
                })
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Вложение", f"Не удалось прочитать {path.name}: {exc}")

    def _refresh_attachments(self) -> None:
        if not self._attachments:
            self.lbl_attach.setText("")
            return
        names = ", ".join(
            f"{'🖼' if a['kind'] == 'image' else '📎'} {a['name']}"
            for a in self._attachments
        )
        self.lbl_attach.setText(f"Вложения: {names}  (отправятся со следующим сообщением)")

    def _clear_attachments(self) -> None:
        self._attachments = []
        self.lbl_attach.setText("")

    # ------------------------------------------------------------------ #
    # Лента сообщений
    # ------------------------------------------------------------------ #
    def _add_bubble(self, role: str) -> _MessageBubble:
        bubble = _MessageBubble(role)
        self._transcript_layout.addWidget(bubble)
        self._scroll_to_bottom()
        return bubble

    def _add_system_note(self, text: str) -> None:
        note = QLabel(text)
        note.setWordWrap(True)
        note.setStyleSheet(
            "color: #ccc; background: #333; border-radius: 6px; padding: 8px;"
        )
        self._transcript_layout.addWidget(note)

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

    # ------------------------------------------------------------------ #
    # Настройки и новый чат
    # ------------------------------------------------------------------ #
    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._config, parent=self)
        if dialog.exec():
            self._config = dialog.result_config()
            self._client = self._make_client()
            self._refresh_header()
            # системный промпт пересоберётся при следующей отправке.
            if self._messages and self._messages[0].get("role") == "system":
                self._messages[0]["content"] = build_system_prompt(
                    self._config, self._context
                )

    def _new_chat(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._messages = []
        self._clear_attachments()
        while self._transcript_layout.count():
            item = self._transcript_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._stream_bubble = None
        self.lbl_activity.setText("")
        self._check_ready()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
class _ChatInput(QPlainTextEdit):
    """Поле ввода: Enter отправляет, Shift+Enter — перенос строки."""

    def __init__(self, on_submit: Callable[[], None]) -> None:
        super().__init__()
        self._on_submit = on_submit

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
            event.modifiers() & Qt.ShiftModifier
        ):
            event.accept()
            self._on_submit()
            return
        super().keyPressEvent(event)
