from __future__ import annotations

import logging
import random
from uuid import uuid4

from rag_prep.config import PipelineConfig
from rag_prep.models import PipelineResult
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
