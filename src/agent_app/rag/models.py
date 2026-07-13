from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RagRetrievedChunk(BaseModel):
    point_id: str
    chunk_id: str
    document_id: str | None = None
    text: str
    source: str | None = None
    section: str | None = None
    position: int | None = None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagCitation(BaseModel):
    reference: str
    point_id: str
    chunk_id: str
    document_id: str | None = None
    source: str | None = None
    section: str | None = None
    position: int | None = None
    score: float
    excerpt: str


class RagRetrievalResult(BaseModel):
    status: str = "ok"
    query: str
    context: str = ""
    citations: list[RagCitation] = Field(default_factory=list)
    retrieved_count: int = 0
    used_count: int = 0
    context_tokens: int = 0
    provider: str | None = None
    model: str | None = None
    collection_name: str | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.status == "ok"


class RagReadiness(BaseModel):
    enabled: bool
    ready: bool
    collection_name: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    vector_size: int | None = None
    error: str | None = None
