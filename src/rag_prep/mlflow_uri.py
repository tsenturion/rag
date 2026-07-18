from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def resolve_mlflow_tracking_uri(uri: str, *, base_dir: Path) -> str:
    """Разрешает локальный MLflow URI относительно корня конфигурации."""
    value = uri.strip()
    if value.startswith("sqlite:///"):
        database = value.removeprefix("sqlite:///")
        if database == ":memory:":
            return value
        path = Path(database).expanduser()
        if not path.is_absolute() and not path.drive:
            path = base_dir / path
        return f"sqlite:///{path.resolve().as_posix()}"

    if urlparse(value).scheme:
        return value

    path = Path(value).expanduser()
    if not path.is_absolute() and not path.drive:
        path = base_dir / path
    return str(path.resolve())


def ensure_mlflow_tracking_parent(uri: str) -> None:
    """Создаёт каталог локальной SQLite БД перед подключением MLflow."""
    if not uri.startswith("sqlite:///"):
        return
    database = uri.removeprefix("sqlite:///")
    if database == ":memory:":
        return
    Path(database).parent.mkdir(parents=True, exist_ok=True)
