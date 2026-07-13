from __future__ import annotations

import logging
import random

from rag_prep.chunking_stages import (
    ChunkExportStage,
    ChunkSplittingStage,
    ChunkValidationStage,
    PreparedDocumentLoadingStage,
)
from rag_prep.config import (
    ChunkingPipelineConfig,
    EmbeddingPipelineConfig,
    PipelineConfig,
    VectorStorePipelineConfig,
)
from rag_prep.embedding_stages import (
    ChunkLoadingStage,
    EmbeddingExportStage,
    EmbeddingValidationStage,
    build_embedding_counts,
    build_embedding_diagnostics,
    build_embedding_stage,
)
from rag_prep.models import (
    ChunkingPipelineResult,
    EmbeddingPipelineResult,
    PipelineResult,
    VectorStorePipelineResult,
)
from rag_prep.stages import (
    DeduplicationStage,
    ExportStage,
    LlamaIndexLoadingStage,
    LlamaIndexStructuringStage,
    TextCleaningStage,
    TextNormalizationStage,
    UnstructuredParsingStage,
)
from rag_prep.tracking import MLflowTracker
from rag_prep.utils import new_run_id
from rag_prep.vector_store_stages import (
    EmbeddingLoadingStage,
    QdrantIndexingStage,
    QdrantSearchStage,
    QdrantValidationStage,
    VectorStoreExportStage,
    build_vector_store_counts,
    build_vector_store_diagnostics,
    qdrant_client_context,
)

LOGGER = logging.getLogger(__name__)


class RagPreparationPipeline:
    """ООП-фасад над стадиями подготовки данных."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.loader = LlamaIndexLoadingStage(config.loader)
        self.parser = UnstructuredParsingStage(
            config.parser, default_section=config.structuring.default_section
        )
        self.cleaner = TextCleaningStage(config.cleaning)
        self.normalizer = TextNormalizationStage(config.normalization)
        self.deduplicator = DeduplicationStage(config.deduplication)
        self.structurer = LlamaIndexStructuringStage(config.structuring)
        self.exporter = ExportStage(config)
        self.tracker = MLflowTracker(config)

    def run(self) -> PipelineResult:
        random.seed(self.config.run.seed)
        run_id = new_run_id()
        LOGGER.info("Старт запуска подготовки данных RAG: %s", run_id)

        sources = self.loader.run(self.config.paths.input_dir)
        parse_result = self.parser.run(sources)
        raw_elements = parse_result.elements
        cleaned = self.cleaner.run(raw_elements)
        normalized = self.normalizer.run(cleaned)
        dedupe_result = self.deduplicator.run(normalized)
        documents = self.structurer.run(dedupe_result.elements, run_id=run_id)
        llama_documents = self.structurer.to_llama_documents(documents)

        counts = {
            "sources_count": len(sources),
            "raw_elements_count": len(raw_elements),
            "parse_failed_sources_count": len(parse_result.failures),
            "cleaned_elements_count": len(cleaned),
            "normalized_elements_count": len(normalized),
            "deduplicated_elements_count": len(dedupe_result.elements),
            "duplicates_removed": dedupe_result.duplicates_removed,
            "exact_duplicates_removed": dedupe_result.exact_duplicates_removed,
            "near_duplicates_removed": dedupe_result.near_duplicates_removed,
            "prepared_documents_count": len(documents),
            "llama_index_documents_count": len(llama_documents),
        }
        diagnostics = {
            "parse_failures": [
                failure.model_dump(mode="json") for failure in parse_result.failures
            ]
        }
        export = self.exporter.run(
            documents,
            run_id=run_id,
            counts=counts,
            diagnostics=diagnostics,
        )
        self.tracker.log_run(counts, export)

        LOGGER.info("Запуск %s завершён; документов: %d", run_id, len(documents))
        return PipelineResult(
            run_id=run_id,
            sources_count=len(sources),
            raw_elements_count=len(raw_elements),
            parse_failed_sources_count=len(parse_result.failures),
            prepared_documents_count=len(documents),
            duplicates_removed=dedupe_result.duplicates_removed,
            export=export,
        )


class RagChunkingPipeline:
    """ООП-фасад над стадиями чанкинга."""

    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config
        self.loader = PreparedDocumentLoadingStage()
        self.splitter = ChunkSplittingStage(config.chunking)
        self.validator = ChunkValidationStage(config.chunking)
        self.exporter = ChunkExportStage(config)
        self.tracker = MLflowTracker(config)

    def run(self) -> ChunkingPipelineResult:
        random.seed(self.config.run.seed)
        run_id = new_run_id()
        LOGGER.info("Старт запуска чанкинга RAG: %s", run_id)

        documents = self.loader.run(self.config.paths.input_jsonl)
        chunks = self.splitter.run(documents)
        validation = self.validator.run(chunks)

        token_counts = [chunk.metadata.chunk_token_count for chunk in chunks]
        structure_scores = [
            float(chunk.metadata.quality["structure_score"])
            for chunk in chunks
            if chunk.metadata.quality.get("structure_score") is not None
        ]
        unique_block_ids = {
            block_id
            for chunk in chunks
            for block_id in chunk.metadata.semantic_block_ids
        }
        counts = {
            "documents_count": len(documents),
            "chunks_count": len(chunks),
            "semantic_blocks_count": len(unique_block_ids),
            "multi_block_chunks_count": sum(
                1 for chunk in chunks if len(chunk.metadata.semantic_block_ids) > 1
            ),
            "avg_chunk_tokens": round(sum(token_counts) / len(token_counts), 3)
            if token_counts
            else 0.0,
            "max_chunk_tokens": max(token_counts) if token_counts else 0,
            "min_chunk_tokens": min(token_counts) if token_counts else 0,
            "avg_chunk_structure_score": round(
                sum(structure_scores) / len(structure_scores), 3
            )
            if structure_scores
            else 0.0,
            "empty_chunks_count": validation.empty_chunks_count,
            "undersized_chunks_count": validation.undersized_chunks_count,
            "oversized_chunks_count": validation.oversized_chunks_count,
            "estimated_offsets_count": validation.estimated_offsets_count,
            "missing_parent_count": validation.missing_parent_count,
            "missing_lineage_count": validation.missing_lineage_count,
            "low_quality_chunks_count": validation.low_quality_chunks_count,
        }
        diagnostics = {"validation": validation.model_dump(mode="json")}
        export = self.exporter.run(
            chunks,
            run_id=run_id,
            counts=counts,
            diagnostics=diagnostics,
        )
        self.tracker.log_run(counts, export)

        LOGGER.info("Запуск чанкинга %s завершён; чанков: %d", run_id, len(chunks))
        return ChunkingPipelineResult(
            run_id=run_id,
            documents_count=len(documents),
            chunks_count=len(chunks),
            validation=validation,
            export=export,
        )


class RagEmbeddingPipeline:
    """ООП-фасад над стадиями embeddings."""

    def __init__(self, config: EmbeddingPipelineConfig):
        self.config = config
        self.loader = ChunkLoadingStage()
        self.embedder = build_embedding_stage(config.embedding)
        self.validator = EmbeddingValidationStage(config.embedding)
        self.exporter = EmbeddingExportStage(config)
        self.tracker = MLflowTracker(config)

    def run(self) -> EmbeddingPipelineResult:
        random.seed(self.config.run.seed)
        run_id = new_run_id()
        LOGGER.info("Старт запуска embeddings RAG: %s", run_id)

        chunks = self.loader.run(self.config.paths.input_jsonl)
        embedded_chunks = self.embedder.run(chunks, run_id=run_id)
        validation = self.validator.run(chunks, embedded_chunks)

        counts = build_embedding_counts(
            self.config,
            chunks,
            embedded_chunks,
            validation,
        )
        diagnostics = build_embedding_diagnostics(validation)
        export = self.exporter.run(
            embedded_chunks,
            run_id=run_id,
            counts=counts,
            diagnostics=diagnostics,
        )
        self.tracker.log_run(counts, export)

        LOGGER.info(
            "Запуск embeddings %s завершён; embeddings: %d",
            run_id,
            len(embedded_chunks),
        )
        return EmbeddingPipelineResult(
            run_id=run_id,
            chunks_count=len(chunks),
            embeddings_count=len(embedded_chunks),
            validation=validation,
            export=export,
        )


class RagVectorStorePipeline:
    """ООП-фасад над стадиями индексации vector store."""

    def __init__(self, config: VectorStorePipelineConfig):
        self.config = config
        self.loader = EmbeddingLoadingStage()
        self.indexer = QdrantIndexingStage(config.vector_store)
        self.validator = QdrantValidationStage(config.vector_store)
        self.searcher = QdrantSearchStage(config.vector_store)
        self.exporter = VectorStoreExportStage(config)
        self.tracker = MLflowTracker(config)

    def run(self) -> VectorStorePipelineResult:
        random.seed(self.config.run.seed)
        run_id = new_run_id()
        LOGGER.info("Старт запуска vector store RAG: %s", run_id)

        embedded_chunks = self.loader.run(self.config.paths.input_jsonl)
        with qdrant_client_context(self.config.vector_store) as client:
            index = self.indexer.run(embedded_chunks, client=client)
            validation = self.validator.run(embedded_chunks, client=client)
            search_results = self.searcher.run(embedded_chunks, client=client)

        counts = build_vector_store_counts(
            self.config,
            embedded_chunks,
            index,
            validation,
            search_results,
        )
        diagnostics = build_vector_store_diagnostics(validation, search_results)
        export = self.exporter.run(
            index=index,
            validation=validation,
            search_results=search_results,
            run_id=run_id,
            counts=counts,
            diagnostics=diagnostics,
        )
        self.tracker.log_run(counts, export)

        LOGGER.info(
            "Запуск vector store %s завершён; Qdrant points: %d",
            run_id,
            index.collection_points_count,
        )
        return VectorStorePipelineResult(
            run_id=run_id,
            embeddings_count=len(embedded_chunks),
            points_count=index.collection_points_count,
            search_results_count=len(search_results),
            validation=validation,
            export=export,
        )
