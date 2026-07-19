"""Общие правила поиска файлов проекта независимо от рабочего каталога."""

from __future__ import annotations

from pathlib import Path


def resolve_project_file(path: str | Path) -> Path:
    """Ищет относительный файл сначала в текущем каталоге, затем в корне проекта.

    Абсолютные пути сохраняют пользовательский приоритет. Если файл не найден ни
    в одном ожидаемом месте, возвращается абсолютный путь относительно текущего
    каталога: вызывающий загрузчик сможет показать его в понятном исключении.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()

    project_root = Path(__file__).resolve().parents[2]
    project_candidate = project_root / candidate
    if project_candidate.exists():
        return project_candidate.resolve()

    return candidate.resolve()
