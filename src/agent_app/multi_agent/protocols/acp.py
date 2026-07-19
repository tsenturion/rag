"""Реализация компонентов для межагентных протоколов."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_app.multi_agent.models import AgentEnvelope, MessageKind


class ACPMessage(BaseModel):
    """Учебная legacy-модель ACP; новые интеграции используют A2A."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    role: Literal["user", "agent"]
    parts: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ACPProtocolAdapter:
    """Преобразует legacy ACP-сообщения во внутренний и A2A-форматы."""

    @staticmethod
    def to_internal(message: ACPMessage, *, recipient: str) -> AgentEnvelope:
        """Гарантирует преобразование ACP-сообщения в внутренний конверт агента с сохранением идентификаторов и полезной нагрузки."""
        return AgentEnvelope(
            correlation_id=message.run_id,
            sender=message.agent_name,
            recipient=recipient,
            kind=MessageKind.REQUEST,
            payload={
                "legacy_protocol": "ACP",
                "parts": message.parts,
                "metadata": message.metadata,
            },
        )

    @staticmethod
    def to_a2a_message(message: ACPMessage) -> dict[str, object]:
        """Гарантирует преобразование ACP-сообщения в публичный формат A2A с сохранением метаданных и структуры частей."""
        text_parts = [
            {"text": str(part.get("content", part.get("text", "")))}
            for part in message.parts
        ]
        return {
            "messageId": message.run_id,
            "role": "ROLE_USER" if message.role == "user" else "ROLE_AGENT",
            "parts": text_parts,
            "metadata": {
                **message.metadata,
                "migratedFrom": "ACP",
                "agentName": message.agent_name,
            },
        }
