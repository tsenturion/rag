"""Типизированные модели данных для безопасности и ручного контроля."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Возвращает текущее время в UTC с информацией о часовом поясе."""
    return datetime.now(timezone.utc)


class GuardrailAction(StrEnum):
    """Определяет уровни действий системы безопасности, регулируя обработку потенциально опасного контента от разрешения до блокировки."""

    ALLOW = "allow"
    REDACT = "redact"
    REVIEW = "review"
    BLOCK = "block"


class GuardrailFinding(BaseModel):
    """Описывает выявленные проблемы безопасности с указанием категории и степени серьезности для информирования и принятия мер."""

    code: str
    category: Literal["prompt_injection", "privacy", "output_safety"]
    severity: Literal["low", "medium", "high", "critical"]
    description: str


class GuardrailResult(BaseModel):
    """Гарантирует, что результат проверки guardrail содержит стадию, действие, текст и находки для трассировки и принятия решений."""

    stage: Literal["input", "context", "tool_output", "output"]
    action: GuardrailAction = GuardrailAction.ALLOW
    text: str
    findings: list[GuardrailFinding] = Field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """Проверяет, что результат guardrail требует блокировки дальнейшей обработки."""
        return self.action == GuardrailAction.BLOCK


class SecurityAuditEvent(BaseModel):
    """Гарантирует целостность и полноту данных о событии безопасности для аудита и расследования инцидентов."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = Field(default_factory=utc_now)
    event_type: str
    action: str
    principal_id: str | None = None
    role: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    details: dict[str, object] = Field(default_factory=dict)


class HumanReviewRecord(BaseModel):
    """Гарантирует воспроизводимость истории ручной модерации с фиксацией статуса, причин и идентификаторов."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: Literal["pending", "approved", "rejected"] = "pending"
    user_id: str
    session_id: str
    request_id: str | None = None
    trace_id: str | None = None
    prompt: str
    answer: str
    reason: str
    reviewer_id: str | None = None
    comment: str | None = None
