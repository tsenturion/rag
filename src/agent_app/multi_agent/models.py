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
    RECEIVED = "received"
    DECOMPOSED = "decomposed"
    DELEGATED = "delegated"
    RUNNING = "running"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskExecutionState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class MessageKind(StrEnum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"


class MessageDeliveryState(StrEnum):
    CREATED = "created"
    SENT = "sent"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    DUPLICATE = "duplicate"


class AgentCapability(BaseModel):
    name: str
    description: str


class AgentDefinition(BaseModel):
    name: str
    title: str
    goal: str
    capabilities: list[AgentCapability]
    tool_allowlist: list[str] = Field(default_factory=list)
    memory_access: Literal["none", "read", "read_write"] = "none"
    use_llm: bool = True

    @property
    def capability_names(self) -> set[str]:
        return {capability.name for capability in self.capabilities}


class AgentTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    capability: str
    title: str
    instruction: str
    required_tools: list[str] = Field(default_factory=list)
    assigned_to: str | None = None
    state: TaskExecutionState = TaskExecutionState.PENDING
    position: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class UsageMetrics(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_tokens: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0
    estimated_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: UsageMetrics) -> UsageMetrics:
        return UsageMetrics(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            estimated_tokens=self.estimated_tokens + other.estimated_tokens,
            tool_calls=self.tool_calls + other.tool_calls,
            duration_ms=self.duration_ms + other.duration_ms,
            estimated_cost=self.estimated_cost + other.estimated_cost,
        )


class AgentTaskResult(BaseModel):
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
        if not self.recipient and not self.topic:
            raise ValueError("Сообщению нужен recipient или topic")
        return self


class LifecycleEvent(BaseModel):
    state: AgentRunState
    created_at: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class QualityAssessment(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    checks: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class LLMRouteInfo(BaseModel):
    role: str
    profile: str
    provider: str
    model: str


class MultiAgentResponse(BaseModel):
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
    response: MultiAgentResponse
    messages: list[AgentEnvelope] = Field(default_factory=list)
    dead_letters: list[AgentEnvelope] = Field(default_factory=list)
    run_dir: str | None = None


class AgentModeResult(BaseModel):
    mode: Literal["single", "multi"]
    answer: str
    citations_count: int = 0
    tool_calls: list[str] = Field(default_factory=list)
    selected_agents: list[str] = Field(default_factory=list)
    quality: QualityAssessment
    usage: UsageMetrics
    run_id: str | None = None


class ComparisonCaseResult(BaseModel):
    id: str
    title: str
    request: str
    single: AgentModeResult
    multi: AgentModeResult
    quality_delta: float
    duration_delta_ms: float
    token_delta: int
    cost_delta: float


class MultiAgentComparisonReport(BaseModel):
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
    llm_routes: list[LLMRouteInfo] = Field(default_factory=list)
    run_dir: str | None = None


class ComparisonScenario(BaseModel):
    id: str
    title: str
    request: str
    expected_terms: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    require_citations: bool = False
    max_agents: int | None = Field(default=None, ge=1)


class ComparisonScenarioSuite(BaseModel):
    scenarios: list[ComparisonScenario] = Field(min_length=1)


class MultiAgentGraphState(TypedDict):
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
    return response.answer
