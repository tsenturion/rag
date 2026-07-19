"""Типизированные модели данных для онлайн-RAG."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RagRetrievedChunk(BaseModel):
    """Гарантирует целостное описание извлечённого фрагмента знаний с метаданными для последующей трассировки и оценки релевантности."""

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
    """Содержит метаданные и оценку релевантности фрагмента источника для обеспечения точной атрибуции в онлайн-RAG."""

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
    """Гарантирует вызывающему коду полный контракт результата поиска: статус, исходный запрос, контекст, цитаты, параметры поиска и возможную ошибку."""

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
    guardrail_findings: int = 0

    @property
    def available(self) -> bool:
        """Проверяет, что результат поиска пригоден для использования, гарантируя отсутствие ошибок в статусе."""
        return self.status == "ok"


class RagReadiness(BaseModel):
    """Гарантирует вызывающему коду прозрачную диагностику готовности подсистемы RAG и её параметров для мониторинга и автоматизации."""

    enabled: bool
    ready: bool
    collection_name: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    vector_size: int | None = None
    error: str | None = None
