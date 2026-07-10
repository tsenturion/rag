from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


MemoryType = Literal["fact", "preference", "task", "summary", "note"]
MemorySource = Literal["user", "assistant", "tool", "system"]


class MemoryRecord(BaseModel):
    id: str
    user_id: str
    session_id: str | None = None
    memory_type: MemoryType = "fact"
    key: str
    value: str
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)
    source: MemorySource = "user"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_accessed_at: datetime | None = None
    access_count: int = 0
    ttl_seconds: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    records: list[MemoryRecord] = Field(default_factory=list)
    query: str | None = None
    count: int = 0


class AgentTraceState(BaseModel):
    name: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentToolResult(BaseModel):
    name: str | None = None
    content: str
    is_error: bool = False


class AgentTrace(BaseModel):
    user_request: str
    start_state: AgentTraceState
    intermediate_states: list[AgentTraceState] = Field(default_factory=list)
    final_state: AgentTraceState
    transition_rules: list[str] = Field(default_factory=list)
    decision_points: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    tool_results: list[AgentToolResult] = Field(default_factory=list)
    memory_created_ids: list[str] = Field(default_factory=list)
    memory_updated_ids: list[str] = Field(default_factory=list)
    memory_deleted_ids: list[str] = Field(default_factory=list)
    loop_guard_triggered: bool = False
    recursion_limit: int


class AgentResponse(BaseModel):
    answer: str
    user_id: str
    session_id: str
    tool_calls: list[str] = Field(default_factory=list)
    trace: AgentTrace | None = None


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    loop_guard_triggered: NotRequired[bool]
