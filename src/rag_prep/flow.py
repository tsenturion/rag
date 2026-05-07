from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

from prefect import flow, task
from prefect.logging import get_run_logger

from rag_prep.config import PipelineConfig, load_config
from rag_prep.models import ExportResult, PipelineResult, PreparedDocument, ProcessedElement, RawElement, SourceFile
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
from rag_prep.utils import setup_logging


@task(name="load_sources")
def load_sources_task(config: PipelineConfig) -> list[SourceFile]:
    sources = LlamaIndexLoadingStage(config.loader).run(config.paths.input_dir)
    get_run_logger().info("Loaded %d sources", len(sources))
    return sources


@task(name="parse_documents")
def parse_documents_task(config: PipelineConfig, sources: list[SourceFile]) -> list[RawElement]:
    elements = UnstructuredParsingStage(
        config.parser, default_section=config.structuring.default_section
    ).run(sources)
    get_run_logger().info("Parsed %d raw elements", len(elements))
    return elements


@task(name="clean_text")
def clean_text_task(config: PipelineConfig, elements: list[RawElement]) -> list[ProcessedElement]:
    cleaned = TextCleaningStage(config.cleaning).run(elements)
    get_run_logger().info("Cleaned %d elements", len(cleaned))
    return cleaned


@task(name="normalize_text")
def normalize_text_task(
    config: PipelineConfig, elements: list[ProcessedElement]
) -> list[ProcessedElement]:
    normalized = TextNormalizationStage(config.normalization).run(elements)
    get_run_logger().info("Normalized %d elements", len(normalized))
    return normalized


@task(name="deduplicate_text")
def deduplicate_text_task(
    config: PipelineConfig, elements: list[ProcessedElement]
) -> tuple[list[ProcessedElement], int]:
    result = DeduplicationStage(config.deduplication).run(elements)
    get_run_logger().info("Removed %d duplicates", result.duplicates_removed)
    return result.elements, result.duplicates_removed


@task(name="structure_documents")
def structure_documents_task(
    config: PipelineConfig, elements: list[ProcessedElement], run_id: str
) -> list[PreparedDocument]:
    stage = LlamaIndexStructuringStage(config.structuring)
    documents = stage.run(elements, run_id=run_id)
    stage.to_llama_documents(documents)
    get_run_logger().info("Structured %d documents", len(documents))
    return documents


@task(name="export_documents")
def export_documents_task(
    config: PipelineConfig,
    documents: list[PreparedDocument],
    run_id: str,
    counts: dict[str, int],
) -> ExportResult:
    export = ExportStage(config).run(documents, run_id=run_id, counts=counts)
    get_run_logger().info("Exported %d documents", export.documents_count)
    return export


@task(name="log_mlflow")
def log_mlflow_task(config: PipelineConfig, counts: dict[str, int], export: ExportResult) -> None:
    MLflowTracker(config).log_run(counts, export)


@flow(name="rag-data-preparation")
def rag_data_preparation_flow(config_path: str = "config/default.yaml") -> PipelineResult:
    import random
    from uuid import uuid4

    config = load_config(Path(config_path))
    setup_logging(config.logging.level)
    random.seed(config.run.seed)
    run_id = uuid4().hex
    logger = get_run_logger()
    logger.info("Starting RAG data preparation run %s", run_id)

    sources = load_sources_task(config)
    raw = parse_documents_task(config, sources)
    cleaned = clean_text_task(config, raw)
    normalized = normalize_text_task(config, cleaned)
    deduped, duplicates_removed = deduplicate_text_task(config, normalized)
    documents = structure_documents_task(config, deduped, run_id)

    counts = {
        "sources_count": len(sources),
        "raw_elements_count": len(raw),
        "cleaned_elements_count": len(cleaned),
        "normalized_elements_count": len(normalized),
        "deduplicated_elements_count": len(deduped),
        "duplicates_removed": duplicates_removed,
        "prepared_documents_count": len(documents),
        "llama_index_documents_count": len(documents),
    }
    export = export_documents_task(config, documents, run_id, counts)
    log_mlflow_task(config, counts, export)

    logger.info("Finished run %s", run_id)
    return PipelineResult(
        run_id=run_id,
        sources_count=len(sources),
        raw_elements_count=len(raw),
        prepared_documents_count=len(documents),
        duplicates_removed=duplicates_removed,
        export=export,
    )
