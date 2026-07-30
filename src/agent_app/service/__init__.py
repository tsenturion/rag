"""Публичный интерфейс для HTTP-сервиса поддержки."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_app.service.app import create_app

__all__ = ["create_app"]


def __getattr__(name: str):
    """Откладывает создание зависимостей FastAPI до запроса фабрики приложения."""
    if name != "create_app":
        raise AttributeError(name)
    from agent_app.service.app import create_app

    return create_app
