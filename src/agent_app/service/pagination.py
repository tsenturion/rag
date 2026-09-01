"""Непрозрачные курсоры для стабильной пагинации HTTP API."""

from __future__ import annotations

import base64
import json
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SessionCursor(BaseModel):
    """Фиксирует позицию после последней сессии выданной страницы."""

    model_config = ConfigDict(extra="forbid")

    updated_at: datetime
    session_id: str


def encode_session_cursor(cursor: SessionCursor) -> str:
    """Кодирует позицию страницы в URL-safe строку без раскрытия формата API."""
    payload = cursor.model_dump_json().encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_session_cursor(value: str) -> SessionCursor:
    """Декодирует и валидирует курсор, отклоняя повреждённые значения."""
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = base64.b64decode(
            padded,
            altchars=b"-_",
            validate=True,
        ).decode("utf-8")
        return SessionCursor.model_validate(json.loads(payload))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Некорректный cursor списка сессий") from exc
