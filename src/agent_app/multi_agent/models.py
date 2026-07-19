"""Типизированные модели данных для мультиагентной системы."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, NotRequired, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agent_app.models import AgentResponse, utc_now
from agent_app.rag.models import RagCitation


class AgentRunState(StrEnum):
    """Определяет стадии жизненного цикла агента в мультиагентной системе, обеспечивая контроль и управление состоянием выполнения задач."""

    RECEIVED = "received"
    DECOMPOSED = "decomposed"
    DELEGATED = "delegated"
    RUNNING = "running"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskExecutionState(StrEnum):
    """Отражает текущее состояние выполнения задачи агентом, гарантируя прозрачность и отслеживаемость прогресса."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class MessageKind(StrEnum):
    """Классифицирует типы сообщений между агентами для корректной обработки и маршрутизации коммуникаций."""

    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"


class MessageDeliveryState(StrEnum):
    """Отслеживает статус доставки сообщений, обеспечивая надежность и контроль обмена информацией между агентами."""

    CREATED = "created"
    SENT = "sent"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    DUPLICATE = "duplicate"


class AgentCapability(BaseModel):
    """Гарантирует, что описание отдельной способности агента доступно для декларативной маршрутизации и валидации возможностей в мультиагентной системе."""

    name: str
    description: str


class AgentDefinition(BaseModel):
    """Гарантирует, что описание агента содержит все необходимые атрибуты для корректного выбора, ограничения и маршрутизации его задач в мультиагентной системе."""

    name: str
    title: str
    goal: str
    capabilities: list[AgentCapability]
    tool_allowlist: list[str] = Field(default_factory=list)
    memory_access: Literal["none", "read", "read_write"] = "none"
    use_llm: bool = True

    @property
    def capability_names(self) -> set[str]:
        """Гарантирует получение полного множества имён поддерживаемых агентом возможностей для проверки совместимости и маршрутизации."""
        return {capability.name for capability in self.capabilities}


class AgentTask(BaseModel):
    """Моделирует задачу агента с необходимыми атрибутами для управления её жизненным циклом и распределения в мультиагентной системе."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    capability: str
    title: str
    instruction: str
    required_tools: list[str] = Field(default_factory=list)
    assigned_to: str | None = None
    state: TaskExecutionState = TaskExecutionState.PENDING
    position: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


def _merge_currency_amounts(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    """Суммирует суммы только внутри одинаковых валютных кодов."""
    merged = dict(left)
    for currency, amount in right.items():
        merged[currency] = round(merged.get(currency, 0.0) + amount, 8)
    return merged


def _add_optional_costs(left: float | None, right: float | None) -> float | None:
    """Не выдаёт частичную RUB-сумму за полную при ошибке одной конвертации."""
    if left is None or right is None:
        return None
    return round(left + right, 8)


def _subtract_optional_costs(
    current: float | None,
    previous: float | None,
) -> float | None:
    """Вычисляет разницу RUB-сумм только при наличии обеих оценок."""
    if current is None or previous is None:
        return None
    return round(current - previous, 8)


def _display_cost(
    costs: dict[str, float],
    rub_cost: float | None,
) -> tuple[float, str]:
    """Выбирает скаляр для совместимости, не складывая разные валюты напрямую."""
    if len(costs) == 1:
        currency, amount = next(iter(costs.items()))
        return amount, currency
    if len(costs) > 1:
        return (rub_cost or 0.0), ("RUB" if rub_cost is not None else "MIXED")
    return (rub_cost or 0.0), "RUB"


class UsageMetrics(BaseModel):
    """Хранит токены, исходные расходы по валютам и их эквивалент в RUB."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_tokens: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0
    estimated_cost: float = 0.0
    estimated_cost_currency: str = "RUB"
    costs_by_currency: dict[str, float] = Field(default_factory=dict)
    estimated_cost_rub: float | None = 0.0
    exchange_rates_to_rub: dict[str, float] = Field(default_factory=dict)
    exchange_rate_dates: dict[str, str] = Field(default_factory=dict)
    exchange_rate_source: str | None = None
    exchange_rate_stale: bool = False
    currency_conversion_errors: list[str] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Гарантирует корректный подсчёт общего числа токенов, использованных агентом, для учёта затрат."""
        return self.input_tokens + self.output_tokens

    def add(self, other: UsageMetrics) -> UsageMetrics:
        """Суммирует usage без смешивания исходных сумм разных валют."""
        costs = _merge_currency_amounts(
            self.costs_by_currency,
            other.costs_by_currency,
        )
        rub_cost = _add_optional_costs(
            self.estimated_cost_rub,
            other.estimated_cost_rub,
        )
        display_cost, display_currency = _display_cost(costs, rub_cost)
        return UsageMetrics(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            estimated_tokens=self.estimated_tokens + other.estimated_tokens,
            tool_calls=self.tool_calls + other.tool_calls,
            duration_ms=self.duration_ms + other.duration_ms,
            estimated_cost=display_cost,
            estimated_cost_currency=display_currency,
            costs_by_currency=costs,
            estimated_cost_rub=rub_cost,
            exchange_rates_to_rub={
                **self.exchange_rates_to_rub,
                **other.exchange_rates_to_rub,
            },
            exchange_rate_dates={
                **self.exchange_rate_dates,
                **other.exchange_rate_dates,
            },
            exchange_rate_source=(
                other.exchange_rate_source or self.exchange_rate_source
            ),
            exchange_rate_stale=(self.exchange_rate_stale or other.exchange_rate_stale),
            currency_conversion_errors=list(
                dict.fromkeys(
                    [
                        *self.currency_conversion_errors,
                        *other.currency_conversion_errors,
                    ]
                )
            ),
        )

    def subtract(self, before: UsageMetrics) -> UsageMetrics:
        """Вычисляет delta usage, сохраняя валютную структуру и рублёвую оценку."""
        costs = {
            currency: round(
                amount - before.costs_by_currency.get(currency, 0.0),
                8,
            )
            for currency, amount in self.costs_by_currency.items()
            if amount - before.costs_by_currency.get(currency, 0.0) != 0
        }
        rub_cost = _subtract_optional_costs(
            self.estimated_cost_rub,
            before.estimated_cost_rub,
        )
        display_cost, display_currency = _display_cost(costs, rub_cost)
        return UsageMetrics(
            llm_calls=self.llm_calls - before.llm_calls,
            input_tokens=self.input_tokens - before.input_tokens,
            output_tokens=self.output_tokens - before.output_tokens,
            estimated_tokens=self.estimated_tokens - before.estimated_tokens,
            tool_calls=self.tool_calls - before.tool_calls,
            duration_ms=round(self.duration_ms - before.duration_ms, 3),
            estimated_cost=display_cost,
            estimated_cost_currency=display_currency,
            costs_by_currency=costs,
            estimated_cost_rub=rub_cost,
            exchange_rates_to_rub=dict(self.exchange_rates_to_rub),
            exchange_rate_dates=dict(self.exchange_rate_dates),
            exchange_rate_source=self.exchange_rate_source,
            exchange_rate_stale=self.exchange_rate_stale,
            currency_conversion_errors=list(self.currency_conversion_errors),
        )


class AgentTaskResult(BaseModel):
    """Гарантирует воспроизводимость и трассируемость результата выполнения задачи агентом, включая состояние, метрики и ошибки."""

    task_id: str
    agent_name: str
    capability: str
    state: TaskExecutionState
    content: str
    tool_calls: list[str] = Field(default_factory=list)
    citations: list[RagCitation] = Field(default_factory=list)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    error: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)


class AgentEnvelope(BaseModel):
    """Гарантирует доставку сообщения между агентами с уникальной идентификацией, контролем времени жизни и обязательным адресатом или темой."""

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    correlation_id: str
    causation_id: str | None = None
    sender: str
    recipient: str | None = None
    topic: str | None = None
    kind: MessageKind
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    ttl_seconds: float = Field(default=60.0, gt=0)
    delivery_state: MessageDeliveryState = MessageDeliveryState.CREATED
    error: str | None = None

    @model_validator(mode="after")
    def require_recipient_or_topic(self) -> AgentEnvelope:
        """Проверяет, что сообщение всегда адресовано либо получателю, либо теме, и выбрасывает ошибку при нарушении этого инварианта."""
        if not self.recipient and not self.topic:
            raise ValueError("Сообщению нужен recipient или topic")
        return self


class LifecycleEvent(BaseModel):
    """Гарантирует фиксирование значимых изменений состояния мультиагентного процесса с возможностью трассировки деталей."""

    state: AgentRunState
    created_at: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class QualityAssessment(BaseModel):
    """Гарантирует прозрачную оценку качества результата мультиагентной работы с возможностью автоматизированной проверки критериев."""

    score: float = Field(ge=0.0, le=1.0)
    checks: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class LLMRouteInfo(BaseModel):
    """Гарантирует однозначную идентификацию маршрута LLM для каждой роли в мультиагентной сессии."""

    role: str
    profile: str
    provider: str
    model: str
    cost_currency: str = "RUB"


class MultiAgentResponse(BaseModel):
    """Гарантирует целостное представление результата мультиагентного запуска для автоматизации, аудита и пользовательских сценариев."""

    run_id: str
    answer: str
    user_id: str
    session_id: str
    selected_agents: list[str] = Field(default_factory=list)
    tasks: list[AgentTask] = Field(default_factory=list)
    task_results: list[AgentTaskResult] = Field(default_factory=list)
    citations: list[RagCitation] = Field(default_factory=list)
    review: str = ""
    history_messages_used: int = 0
    summary_used: bool = False
    llm_routes: list[LLMRouteInfo] = Field(default_factory=list)
    lifecycle: list[LifecycleEvent] = Field(default_factory=list)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    quality: QualityAssessment | None = None
    execution_mode: Literal["sequential", "parallel"] = "sequential"
    degraded: bool = False


class MultiAgentRunResult(BaseModel):
    """Гарантирует воспроизводимость и полноту результатов запуска мультиагентной сессии, включая сообщения, dead letters и путь к артефактам."""

    response: MultiAgentResponse
    messages: list[AgentEnvelope] = Field(default_factory=list)
    dead_letters: list[AgentEnvelope] = Field(default_factory=list)
    run_dir: str | None = None


class AgentModeResult(BaseModel):
    """Фиксирует все существенные метрики и параметры ответа агента в выбранном режиме для последующего анализа и сравнения."""

    mode: Literal["single", "multi"]
    answer: str
    citations_count: int = 0
    tool_calls: list[str] = Field(default_factory=list)
    selected_agents: list[str] = Field(default_factory=list)
    quality: QualityAssessment
    usage: UsageMetrics
    run_id: str | None = None


class ComparisonCaseResult(BaseModel):
    """Гарантирует целостное сравнение работы агентов в разных режимах по качеству, времени, токенам и стоимости для одного запроса."""

    id: str
    title: str
    request: str
    single: AgentModeResult
    multi: AgentModeResult
    quality_delta: float
    duration_delta_ms: float
    token_delta: int
    cost_delta: float
    cost_delta_rub: float | None = None


class MultiAgentComparisonReport(BaseModel):
    """Обеспечивает агрегированный отчёт о сравнении режимов работы агентов с инвариантом полноты и идентифицируемости эксперимента."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=utc_now)
    provider: str
    model: str
    cases: list[ComparisonCaseResult] = Field(min_length=1)
    average_single_quality: float
    average_multi_quality: float
    quality_delta: float
    total_single_cost: float
    total_multi_cost: float
    total_cost_delta: float
    cost_currency: str = "RUB"
    total_single_costs_by_currency: dict[str, float] = Field(default_factory=dict)
    total_multi_costs_by_currency: dict[str, float] = Field(default_factory=dict)
    total_single_cost_rub: float | None = None
    total_multi_cost_rub: float | None = None
    total_cost_delta_rub: float | None = None
    llm_routes: list[LLMRouteInfo] = Field(default_factory=list)
    run_dir: str | None = None


class ComparisonScenario(BaseModel):
    """Определяет контракт сценария сравнения с ожидаемыми условиями и ограничениями для корректной автоматизации тестов."""

    id: str
    title: str
    request: str
    expected_terms: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    require_citations: bool = False
    max_agents: int | None = Field(default=None, ge=1)


class ComparisonScenarioSuite(BaseModel):
    """Обеспечивает контейнер для набора сценариев сравнения, гарантируя наличие хотя бы одного сценария для анализа."""

    scenarios: list[ComparisonScenario] = Field(min_length=1)


class MultiAgentGraphState(TypedDict):
    """Хранит текущее состояние мультиагентного процесса, включая задачи, результаты и историю, для обеспечения целостности и восстановления."""

    run_id: str
    user_id: str
    session_id: str
    request: str
    history: Annotated[list[BaseMessage], add_messages]
    tasks: list[AgentTask]
    task_results: list[AgentTaskResult]
    review: str
    answer: str
    citations: list[RagCitation]
    round_number: int
    delegations: int
    degraded: bool
    error: NotRequired[str]


def single_response_answer(response: AgentResponse) -> str:
    """Гарантирует получение финального текстового ответа из структуры ответа агента для дальнейшей обработки."""
    return response.answer
