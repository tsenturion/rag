from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_app.models import AgentResponse
from agent_app.multi_agent.models import (
    MultiAgentComparisonReport,
    MultiAgentResponse,
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": (
                        "Какие обязательные поля нужно указать в заявке и что "
                        "делать, если данных недостаточно?"
                    ),
                    "user_id": "engineer-1",
                    "session_id": "incident-42",
                }
            ]
        }
    )

    message: str = Field(
        min_length=1,
        max_length=1_000_000,
        description="Запрос инженера к агенту.",
    )
    user_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[\w.@+-]+$",
        description="Проверенный идентификатор пользователя для изоляции памяти.",
        examples=["engineer-1"],
    )
    session_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[\w.@+-]+$",
        description="Идентификатор текущего диалога или расследования.",
        examples=["incident-42"],
    )

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        return value.strip()


class ChatResponse(AgentResponse):
    request_id: str = Field(description="Корреляционный идентификатор HTTP-запроса.")
    duration_ms: float = Field(
        description="Полная длительность обработки запроса в миллисекундах."
    )


class MultiAgentChatResponse(MultiAgentResponse):
    request_id: str = Field(description="Корреляционный идентификатор HTTP-запроса.")
    duration_ms: float = Field(
        description="Полная длительность supervisor-графа в миллисекундах."
    )
    run_dir: str | None = Field(
        default=None,
        description="Каталог воспроизводимых артефактов запуска.",
    )


class MultiAgentCompareRequest(ChatRequest):
    expected_terms: list[str] = Field(
        default_factory=list,
        description="Термины для детерминированной оценки качества обоих режимов.",
    )
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Tools, которые должен вызвать multi-agent режим.",
    )
    require_citations: bool = Field(
        default=False,
        description="Требовать citations в single- и multi-agent ответах.",
    )


class MultiAgentCompareResponse(MultiAgentComparisonReport):
    request_id: str = Field(description="Корреляционный идентификатор HTTP-запроса.")
    duration_ms: float = Field(
        description="Длительность двух запусков в миллисекундах."
    )


class SessionResponse(BaseModel):
    user_id: str = Field(description="Владелец сессии.")
    session_id: str = Field(description="Идентификатор сессии.")
    memory: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Доступные пользователю записи долговременной памяти.",
    )
    incidents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Инциденты пользователя, связанные с этой сессией.",
    )
    multi_agent_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Последние сообщения persistent multi-agent checkpoint этой сессии."
        ),
    )


class DeleteSessionResponse(BaseModel):
    user_id: str = Field(description="Владелец очищенной сессии.")
    session_id: str = Field(description="Идентификатор очищенной сессии.")
    deleted_memory_count: int = Field(
        description="Количество удалённых session-scoped записей памяти."
    )
    runner_removed: bool = Field(
        description="Удалён ли AgentRunner из in-process session cache."
    )
    multi_agent_checkpoint_deleted: bool = Field(
        default=False,
        description="Удалён ли persistent checkpoint мультиагентного диалога.",
    )


class HealthResponse(BaseModel):
    status: str = Field(description="Текущее состояние сервиса.")
    service: str = "engineer-support-agent"
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Диагностика компонентов; заполняется readiness-проверкой.",
    )


class ApiError(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": "http_error",
                    "message": "Некорректный API key.",
                    "request_id": "f9c85fd1-59b0-4c48-a57e-b67d4019aa19",
                }
            ]
        }
    )

    error: str = Field(description="Машиночитаемый код ошибки.")
    message: str = Field(description="Безопасное описание ошибки.")
    request_id: str | None = Field(
        default=None,
        description="Корреляционный идентификатор запроса.",
    )
