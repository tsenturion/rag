from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

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


class AgentResponse(BaseModel):
    answer: str
    user_id: str
    session_id: str
    tool_calls: list[str] = Field(default_factory=list)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
