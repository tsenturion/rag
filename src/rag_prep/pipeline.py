from __future__ import annotations

import logging
import random
from uuid import uuid4

from rag_prep.chunking_stages import (
    ChunkExportStage,
    ChunkSplittingStage,
    ChunkValidationStage,
    PreparedDocumentLoadingStage,
)
from rag_prep.config import ChunkingPipelineConfig, PipelineConfig
from rag_prep.models import ChunkingPipelineResult, PipelineResult
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

LOGGER = logging.getLogger(__name__)


class RagPreparationPipeline:
    """OO facade around the preparation stages."""

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
        run_id = uuid4().hex
        LOGGER.info("Starting RAG data preparation run %s", run_id)

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

        LOGGER.info("Finished run %s with %d documents", run_id, len(documents))
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
    """OO facade around the chunking stages."""

    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config
        self.loader = PreparedDocumentLoadingStage()
        self.splitter = ChunkSplittingStage(config.chunking)
        self.validator = ChunkValidationStage(config.chunking)
        self.exporter = ChunkExportStage(config)
        self.tracker = MLflowTracker(config)

    def run(self) -> ChunkingPipelineResult:
        random.seed(self.config.run.seed)
        run_id = uuid4().hex
        LOGGER.info("Starting RAG chunking run %s", run_id)

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

        LOGGER.info("Finished chunking run %s with %d chunks", run_id, len(chunks))
        return ChunkingPipelineResult(
            run_id=run_id,
            documents_count=len(documents),
            chunks_count=len(chunks),
            validation=validation,
            export=export,
        )
