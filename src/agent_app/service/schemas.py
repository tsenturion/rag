"""Контракты HTTP API для HTTP-сервиса поддержки."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_app.models import AgentResponse, ConversationMessage
from agent_app.guardrails.models import HumanReviewRecord, SecurityAuditEvent
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
    """Валидирует и нормализует входные данные запроса к агенту, гарантируя корректность идентификаторов и содержимого сообщения для обработки."""

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
        """Гарантирует, что входное сообщение пользователя очищено от лишних пробелов для корректной обработки запроса."""
        return value.strip()


class ChatResponse(AgentResponse):
    """Гарантирует вызывающему коду воспроизводимый результат диалога с агентом с трассировкой времени, идентификатором запроса и статусом guardrail."""

    request_id: str = Field(description="Корреляционный идентификатор HTTP-запроса.")
    duration_ms: float = Field(
        description="Полная длительность обработки запроса в миллисекундах."
    )
    guardrail_action: str = Field(
        default="allow", description="Решение выходного guardrail."
    )
    review_id: str | None = Field(
        default=None, description="Human-review задача, если ответ требует проверки."
    )


class MultiAgentChatResponse(MultiAgentResponse):
    """Гарантирует вызывающему коду полный отчёт о запуске supervisor-графа с идентификатором запроса, временем выполнения и артефактами."""

    request_id: str = Field(description="Корреляционный идентификатор HTTP-запроса.")
    duration_ms: float = Field(
        description="Полная длительность supervisor-графа в миллисекундах."
    )
    run_dir: str | None = Field(
        default=None,
        description=(
            "Публичный идентификатор каталога артефактов без локального пути сервера."
        ),
    )
    guardrail_action: str = "allow"
    review_id: str | None = None


class HumanReviewDecisionRequest(BaseModel):
    """Обеспечивает проверку решения человека по обзору с обязательным статусом одобрения и опциональным комментарием для прозрачности."""

    approved: bool
    comment: str | None = Field(default=None, max_length=2000)


class HumanReviewResponse(HumanReviewRecord):
    """Гарантирует вызывающему коду доступ к результату human-review задачи в согласованном формате."""

    pass


class SecurityAuditResponse(BaseModel):
    """Гарантирует вызывающему коду полный список событий аудита безопасности для последующего анализа или отображения."""

    events: list[SecurityAuditEvent]


class AppFeatureFlags(BaseModel):
    """Описывает включённые backend-возможности для построения интерфейса."""

    chat: bool = True
    streaming: bool = True
    rag: bool
    multi_agent: bool
    orchestration: bool
    human_review: bool
    a2a: bool
    mcp: bool


class AppAuthenticationConfig(BaseModel):
    """Сообщает frontend допустимые способы входа без раскрытия секретов."""

    api_key_enabled: bool
    jwt_enabled: bool
    user_scope_enforced: bool
    api_key_header: str = "X-API-Key"
    bearer_scheme: str = "Bearer"


class AppLimitsConfig(BaseModel):
    """Публикует ограничения, необходимые клиентской валидации запросов."""

    request_max_chars: int
    max_history_messages: int
    rate_limit_enabled: bool
    rate_limit_requests_per_minute: int | None = None
    rate_limit_burst: int | None = None


class AppConfigResponse(BaseModel):
    """Предоставляет безопасный bootstrap-конфиг для web-приложения."""

    api_version: str = "v1"
    service: str = "engineer-support-agent"
    provider: str
    model: str
    features: AppFeatureFlags
    authentication: AppAuthenticationConfig
    limits: AppLimitsConfig
    openapi_url: str = "/openapi.json"
    docs_url: str = "/docs"


class CurrentPrincipalResponse(BaseModel):
    """Возвращает проверенную identity и права текущего HTTP-клиента."""

    subject: str
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    auth_method: str


class MultiAgentCompareRequest(ChatRequest):
    """Расширяет запрос чата для сравнения ответов нескольких агентов, гарантируя наличие критериев оценки и требований к цитированию."""

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
    """Гарантирует вызывающему коду воспроизводимый отчёт о сравнении двух запусков multi-agent сценариев с идентификатором запроса и временем."""

    request_id: str = Field(description="Корреляционный идентификатор HTTP-запроса.")
    duration_ms: float = Field(
        description="Длительность двух запусков в миллисекундах."
    )


class OrchestrationJobRequest(ChatRequest):
    """Определяет параметры задания оркестрации с валидацией приоритетов, паттернов и ограничений, обеспечивая корректное создание и управление задачами."""

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
        """Гарантирует преобразование пользовательского запроса в формат задания для оркестрации с сохранением всех параметров и дедлайна."""
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
    """Гарантирует вызывающему коду согласованный снимок пользовательской сессии с памятью, инцидентами и историей multi-agent диалога."""

    user_id: str = Field(description="Владелец сессии.")
    session_id: str = Field(description="Идентификатор сессии.")
    updated_at: datetime | None = Field(
        default=None,
        description="Время последнего сохранённого хода диалога.",
    )
    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Сохранённые пользовательские и агентские сообщения.",
    )
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


class SessionSummary(BaseModel):
    """Представляет одну строку списка диалогов для навигации frontend."""

    session_id: str
    updated_at: datetime
    message_count: int = Field(ge=0)
    preview: str


class SessionListResponse(BaseModel):
    """Возвращает страницу диалогов пользователя и непрозрачный cursor."""

    user_id: str
    items: list[SessionSummary] = Field(default_factory=list)
    next_cursor: str | None = None
    limit: int = Field(ge=1, le=100)


class DeleteSessionResponse(BaseModel):
    """Гарантирует вызывающему коду подтверждение удаления пользовательской сессии с деталями по памяти и состоянию multi-agent checkpoint."""

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
    """Гарантирует вызывающему коду актуальное состояние сервиса и диагностику компонентов для health-check и мониторинга."""

    status: str = Field(description="Текущее состояние сервиса.")
    service: str = "engineer-support-agent"
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Диагностика компонентов; заполняется readiness-проверкой.",
    )


class ApiValidationDetail(BaseModel):
    """Описывает одно поле некорректного HTTP-запроса без исходного значения."""

    field: str
    message: str
    type: str


class ApiError(BaseModel):
    """Структурирует информацию об ошибках API, предоставляя машиночитаемый код, описание и корреляционный идентификатор для отладки."""

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
    details: list[ApiValidationDetail] = Field(
        default_factory=list,
        description="Ошибки отдельных полей; заполнены для validation_error.",
    )
