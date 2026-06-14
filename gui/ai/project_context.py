"""Рабочее пространство агента: файловый доступ строго внутри корня проекта.

Агент по требованию пользователя обязан жить **только** в каталоге этого
проекта. ``ProjectContext`` инкапсулирует это ограничение: любой путь
резолвится относительно корня и проверяется на выход за его пределы
(защита от ``../`` и симлинков). Здесь же собирается «карта проекта» и
снимок текущего состояния GUI, которые подмешиваются в системный промпт.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .config import PROJECT_ROOT

#: каталоги/файлы, которые агенту видеть незачем (шум и приватное).
_IGNORED_NAMES = {
    "__pycache__", ".git", ".venv", "venv", ".pytest_cache", ".mypy_cache",
    ".idea", ".vscode", "node_modules", ".directory",
}
#: расширения, которые читаем как текст (остальное — бинарь, не отдаём целиком).
_TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".csv", ".json", ".toml", ".cfg", ".ini",
    ".yaml", ".yml", ".html", ".rst", ".sh",
}
#: верхняя граница чтения текстового файла (символов), чтобы не раздуть контекст.
_MAX_READ_CHARS = 120_000


class WorkspaceError(Exception):
    """Попытка выйти за пределы рабочего пространства или иная ошибка доступа."""


class ProjectContext:
    """Песочница файлового доступа и поставщик контекста проекта."""

    def __init__(
        self,
        root: Path = PROJECT_ROOT,
        state_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        self.root = root.resolve()
        #: вызывается окном чата — возвращает текстовый снимок состояния GUI.
        self.state_provider = state_provider

    # ------------------------------------------------------------------ #
    # Безопасный резолвинг путей
    # ------------------------------------------------------------------ #
    def resolve(self, rel_path: str) -> Path:
        """Превращает относительный путь в абсолютный внутри корня.

        Бросает WorkspaceError при попытке выйти за пределы проекта.
        """
        raw = (rel_path or "").strip().strip("\"'")
        if raw in ("", ".", "./"):
            return self.root
        candidate = (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(
                f"Путь «{rel_path}» выходит за пределы проекта — доступ запрещён. "
                f"Агент работает только внутри {self.root.name}."
            ) from exc
        return candidate

    def rel(self, path: Path) -> str:
        """Путь относительно корня (для красивого вывода)."""
        try:
            return str(path.resolve().relative_to(self.root)) or "."
        except ValueError:
            return str(path)

    # ------------------------------------------------------------------ #
    # Файловые операции (используются инструментами агента)
    # ------------------------------------------------------------------ #
    def list_directory(self, rel_path: str = ".") -> str:
        target = self.resolve(rel_path)
        if not target.exists():
            raise WorkspaceError(f"Каталог «{rel_path}» не найден.")
        if not target.is_dir():
            raise WorkspaceError(f"«{rel_path}» — это файл, а не каталог.")
        entries = []
        for child in sorted(
            target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        ):
            if child.name in _IGNORED_NAMES:
                continue
            if child.is_dir():
                entries.append(f"{self.rel(child)}/")
            else:
                size = child.stat().st_size
                entries.append(f"{self.rel(child)}  ({_human_size(size)})")
        if not entries:
            return f"(пусто) {self.rel(target)}"
        return "\n".join(entries)

    def read_file(self, rel_path: str) -> str:
        target = self.resolve(rel_path)
        if not target.exists():
            raise WorkspaceError(f"Файл «{rel_path}» не найден.")
        if target.is_dir():
            raise WorkspaceError(
                f"«{rel_path}» — каталог. Используйте list_directory."
            )
        if target.suffix.lower() not in _TEXT_SUFFIXES:
            raise WorkspaceError(
                f"«{rel_path}» не текстовый ({target.suffix}); читать целиком нельзя."
            )
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > _MAX_READ_CHARS:
            text = (
                text[:_MAX_READ_CHARS]
                + f"\n\n… (файл обрезан: показаны первые {_MAX_READ_CHARS} символов)"
            )
        return text

    def write_file(self, rel_path: str, content: str) -> str:
        target = self.resolve(rel_path)
        if target.is_dir():
            raise WorkspaceError(f"«{rel_path}» — каталог, в него нельзя писать.")
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, encoding="utf-8")
        verb = "перезаписан" if existed else "создан"
        return f"Файл {self.rel(target)} {verb} ({len(content)} символов)."

    # ------------------------------------------------------------------ #
    # Карта проекта и снимок состояния (для системного промпта)
    # ------------------------------------------------------------------ #
    def file_tree(self, max_depth: int = 2) -> str:
        """Сжатое дерево проекта до заданной глубины (без шумных каталогов)."""
        lines: list[str] = []

        def walk(directory: Path, depth: int) -> None:
            if depth > max_depth:
                return
            for child in sorted(
                directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
            ):
                if child.name in _IGNORED_NAMES or child.name.startswith("."):
                    continue
                indent = "  " * depth
                if child.is_dir():
                    lines.append(f"{indent}{child.name}/")
                    walk(child, depth + 1)
                else:
                    lines.append(f"{indent}{child.name}")

        walk(self.root, 0)
        return "\n".join(lines)

    def registered_strategies(self) -> str:
        """Список зарегистрированных стратегий (имя в GUI -> класс/файл)."""
        try:
            import strategies as strat_pkg  # noqa: PLC0415
            mapping = getattr(strat_pkg, "STRATEGIES", {})
        except Exception as exc:  # noqa: BLE001
            return f"(не удалось импортировать strategies: {exc})"
        if not mapping:
            return "(реестр STRATEGIES пуст)"
        rows = []
        for label, cls in mapping.items():
            module = getattr(cls, "__module__", "?")
            rows.append(f"- «{label}» → {cls.__name__}  ({module})")
        return "\n".join(rows)

    def app_state(self) -> str:
        """Снимок текущего состояния GUI (данные, стратегия, последние результаты)."""
        if self.state_provider is None:
            return "(приложение не передало состояние)"
        try:
            return self.state_provider() or "(пусто)"
        except Exception as exc:  # noqa: BLE001
            return f"(не удалось получить состояние приложения: {exc})"


def _human_size(num: int) -> str:
    value = float(num)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.0f} {unit}" if unit == "Б" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"
