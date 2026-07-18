from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_app.models import AgentResponse
from agent_app.multi_agent.models import (
    MultiAgentComparisonReport,
    MultiAgentResponse,
)
from agent_app.orchestration.models import (
    JobPriority,
    OrchestrationJob,
    OrchestrationPattern,
    utc_now,
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


class OrchestrationJobRequest(ChatRequest):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": (
                        "Проверь инцидент с недоступностью API, оцени риски "
                        "и предложи порядок восстановления."
                    ),
                    "user_id": "engineer-1",
                    "session_id": "incident-42",
                    "pattern": "parallel",
                    "priority": "high",
                    "risk_level": "high",
                    "deadline_seconds": 300,
                    "idempotency_key": "incident-42-analysis-v1",
                }
            ]
        }
    )

    pattern: OrchestrationPattern = Field(
        default=OrchestrationPattern.SEQUENTIAL,
        description=(
            "Паттерн выполнения: последовательный, параллельный, условный, "
            "кворум или динамическое перепланирование."
        ),
    )
    priority: JobPriority = Field(
        default=JobPriority.NORMAL,
        description="Приоритет broker-очереди.",
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Детерминированный вход для условной ветки.",
    )
    quorum_size: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Число успешных голосов для кворума из трёх агентов.",
    )
    idempotency_key: str | None = Field(
        default=None,
        min_length=8,
        max_length=200,
        description="Ключ защиты от повторной постановки одного задания.",
    )
    deadline_seconds: int | None = Field(
        default=None,
        ge=1,
        le=86_400,
        description="Срок выполнения относительно момента постановки.",
    )
    max_plan_revisions: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Максимальное число динамических перепланирований.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Дополнительный контекст задания без секретов.",
    )

    def to_job(self) -> OrchestrationJob:
        return OrchestrationJob(
            user_id=self.user_id,
            session_id=self.session_id,
            message=self.message,
            pattern=self.pattern,
            priority=self.priority,
            risk_level=self.risk_level,
            quorum_size=self.quorum_size,
            idempotency_key=self.idempotency_key,
            deadline_at=(
                utc_now() + timedelta(seconds=self.deadline_seconds)
                if self.deadline_seconds is not None
                else None
            ),
            max_plan_revisions=self.max_plan_revisions,
            metadata=self.metadata,
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
