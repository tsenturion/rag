"""Пакет подготовки данных, чанкинга и embeddings для RAG."""

from typing import TYPE_CHECKING

from rag_prep.config import (
    ChunkingPipelineConfig,
    EmbeddingPipelineConfig,
    PipelineConfig,
    VectorStorePipelineConfig,
    load_chunking_config,
    load_config,
    load_embedding_config,
    load_vector_store_config,
)

if TYPE_CHECKING:
    from rag_prep.pipeline import (
        RagChunkingPipeline,
        RagEmbeddingPipeline,
        RagPreparationPipeline,
        RagVectorStorePipeline,
    )

__all__ = [
    "ChunkingPipelineConfig",
    "EmbeddingPipelineConfig",
    "PipelineConfig",
    "VectorStorePipelineConfig",
    "RagChunkingPipeline",
    "RagEmbeddingPipeline",
    "RagPreparationPipeline",
    "RagVectorStorePipeline",
    "load_chunking_config",
    "load_config",
    "load_embedding_config",
    "load_vector_store_config",
]


def __getattr__(name: str):
    if name not in {
        "RagChunkingPipeline",
        "RagEmbeddingPipeline",
        "RagPreparationPipeline",
        "RagVectorStorePipeline",
    }:
        raise AttributeError(name)
    from rag_prep import pipeline

    return getattr(pipeline, name)
