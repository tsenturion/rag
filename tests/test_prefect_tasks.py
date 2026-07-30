"""Проверки Prefect task-функций без запуска orchestration backend и событий."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import rag_prep.flow as flow_module
from rag_prep.models import (
    ChunkingValidationResult,
    EmbeddingValidationResult,
    VectorStoreValidationResult,
)


def _stage(result):
    """Создаёт stage-double с заранее определённым результатом метода run."""
    stage = Mock()
    stage.run.return_value = result
    return stage


def test_preparation_prefect_tasks_call_domain_stages(monkeypatch) -> None:
    """Проверяет адаптеры загрузки, обработки, структурирования, экспорта и MLflow."""
    logger = Mock()
    monkeypatch.setattr(flow_module, "get_run_logger", lambda: logger)
    config = SimpleNamespace(
        paths=SimpleNamespace(input_dir=Path("raw")),
        loader=object(),
        parser=object(),
        structuring=SimpleNamespace(default_section="full"),
        cleaning=object(),
        normalization=object(),
        deduplication=object(),
    )
    sources = [object()]
    raw = [object(), object()]
    processed = [object()]
    documents = [object()]
    parse_result = SimpleNamespace(elements=raw, failures=[])
    loader = _stage(sources)
    parser = _stage(parse_result)
    cleaner = _stage(processed)
    normalizer = _stage(processed)
    deduplicator = _stage(
        SimpleNamespace(
            elements=processed,
            duplicates_removed=1,
            exact_duplicates_removed=1,
            near_duplicates_removed=0,
        )
    )
    structurer = _stage(documents)
    structurer.to_llama_documents.return_value = [object(), object()]
    export = SimpleNamespace(documents_count=1)
    exporter = _stage(export)
    tracker = Mock()
    monkeypatch.setattr(flow_module, "LlamaIndexLoadingStage", lambda *_: loader)
    monkeypatch.setattr(
        flow_module, "UnstructuredParsingStage", lambda *_a, **_k: parser
    )
    monkeypatch.setattr(flow_module, "TextCleaningStage", lambda *_: cleaner)
    monkeypatch.setattr(flow_module, "TextNormalizationStage", lambda *_: normalizer)
    monkeypatch.setattr(flow_module, "DeduplicationStage", lambda *_: deduplicator)
    monkeypatch.setattr(
        flow_module, "LlamaIndexStructuringStage", lambda *_: structurer
    )
    monkeypatch.setattr(flow_module, "ExportStage", lambda *_: exporter)
    monkeypatch.setattr(flow_module, "MLflowTracker", lambda *_: tracker)

    assert flow_module.load_sources_task.fn(config) == sources
    assert flow_module.parse_documents_task.fn(config, sources) is parse_result
    assert flow_module.clean_text_task.fn(config, raw) == processed
    assert flow_module.normalize_text_task.fn(config, processed) == processed
    assert flow_module.deduplicate_text_task.fn(config, processed) == (
        processed,
        1,
        1,
        0,
    )
    assert flow_module.structure_documents_task.fn(config, processed, "run") == (
        documents,
        2,
    )
    counts = {"documents_count": 1}
    diagnostics = {"validation": "ok"}
    assert (
        flow_module.export_documents_task.fn(
            config, documents, "run", counts, diagnostics
        )
        is export
    )
    flow_module.log_mlflow_task.fn(config, counts, export)

    exporter.run.assert_called_once_with(
        documents, run_id="run", counts=counts, diagnostics=diagnostics
    )
    tracker.log_run.assert_called_once_with(counts, export)
    assert logger.info.call_count >= 7


def test_chunking_and_embedding_prefect_tasks(monkeypatch) -> None:
    """Проверяет task-адаптеры чанкинга и embeddings, включая полную диагностику."""
    logger = Mock()
    monkeypatch.setattr(flow_module, "get_run_logger", lambda: logger)
    chunk_config = SimpleNamespace(
        paths=SimpleNamespace(input_jsonl=Path("documents.jsonl")),
        chunking=object(),
    )
    documents = [object()]
    chunks = [object(), object()]
    chunk_validation = ChunkingValidationResult()
    chunk_export = SimpleNamespace(chunks_count=2)
    document_loader = _stage(documents)
    splitter = _stage(chunks)
    chunk_validator = _stage(chunk_validation)
    chunk_exporter = _stage(chunk_export)
    tracker = Mock()
    monkeypatch.setattr(
        flow_module, "PreparedDocumentLoadingStage", lambda: document_loader
    )
    monkeypatch.setattr(flow_module, "ChunkSplittingStage", lambda *_: splitter)
    monkeypatch.setattr(flow_module, "ChunkValidationStage", lambda *_: chunk_validator)
    monkeypatch.setattr(flow_module, "ChunkExportStage", lambda *_: chunk_exporter)
    monkeypatch.setattr(flow_module, "MLflowTracker", lambda *_: tracker)

    assert flow_module.load_prepared_documents_task.fn(chunk_config) == documents
    assert (
        flow_module.split_chunks_task.fn(chunk_config, documents, "chunk-run") == chunks
    )
    assert flow_module.validate_chunks_task.fn(chunk_config, chunks) is chunk_validation
    counts = {"chunks_count": 2}
    diagnostics = {"validation": {}}
    assert (
        flow_module.export_chunks_task.fn(
            chunk_config, chunks, "chunk-run", counts, diagnostics
        )
        is chunk_export
    )
    flow_module.log_chunking_mlflow_task.fn(chunk_config, counts, chunk_export)

    embedding_config = SimpleNamespace(
        paths=SimpleNamespace(input_jsonl=Path("chunks.jsonl")), embedding=object()
    )
    embedded = [object(), object()]
    embedding_validation = EmbeddingValidationResult()
    embedding_export = SimpleNamespace(embeddings_count=2)
    chunk_loader = _stage(chunks)
    embedder = _stage(embedded)
    embedding_validator = _stage(embedding_validation)
    embedding_exporter = _stage(embedding_export)
    monkeypatch.setattr(flow_module, "ChunkLoadingStage", lambda: chunk_loader)
    monkeypatch.setattr(flow_module, "build_embedding_stage", lambda *_: embedder)
    monkeypatch.setattr(
        flow_module, "EmbeddingValidationStage", lambda *_: embedding_validator
    )
    monkeypatch.setattr(
        flow_module, "EmbeddingExportStage", lambda *_: embedding_exporter
    )

    assert flow_module.load_chunks_task.fn(embedding_config) == chunks
    assert (
        flow_module.calculate_embeddings_task.fn(
            embedding_config, chunks, "embedding-run"
        )
        == embedded
    )
    assert (
        flow_module.validate_embeddings_task.fn(embedding_config, chunks, embedded)
        is embedding_validation
    )
    assert (
        flow_module.export_embeddings_task.fn(
            embedding_config, embedded, "embedding-run", counts, diagnostics
        )
        is embedding_export
    )
    flow_module.log_embedding_mlflow_task.fn(embedding_config, counts, embedding_export)


def test_vector_store_prefect_tasks_share_client(monkeypatch) -> None:
    """Проверяет отдельные и объединённый Qdrant task без открытия реального индекса."""
    logger = Mock()
    monkeypatch.setattr(flow_module, "get_run_logger", lambda: logger)
    config = SimpleNamespace(
        paths=SimpleNamespace(input_jsonl=Path("embeddings.jsonl")),
        vector_store=object(),
    )
    embedded = [object()]
    index = SimpleNamespace(
        points_upserted=1, collection_name="test", collection_points_count=1
    )
    validation = VectorStoreValidationResult(
        embeddings_count=1, collection_points_count=1
    )
    searches = []
    loader = _stage(embedded)
    indexer = _stage(index)
    validator = _stage(validation)
    searcher = _stage(searches)
    export = object()
    exporter = _stage(export)
    tracker = Mock()
    client = object()
    monkeypatch.setattr(flow_module, "EmbeddingLoadingStage", lambda: loader)
    monkeypatch.setattr(flow_module, "QdrantIndexingStage", lambda *_: indexer)
    monkeypatch.setattr(flow_module, "QdrantValidationStage", lambda *_: validator)
    monkeypatch.setattr(flow_module, "QdrantSearchStage", lambda *_: searcher)
    monkeypatch.setattr(flow_module, "VectorStoreExportStage", lambda *_: exporter)
    monkeypatch.setattr(flow_module, "MLflowTracker", lambda *_: tracker)
    monkeypatch.setattr(
        flow_module, "qdrant_client_context", lambda *_: nullcontext(client)
    )

    assert flow_module.load_embedding_records_task.fn(config) == embedded
    assert flow_module.index_qdrant_task.fn(config, embedded) is index
    assert flow_module.validate_qdrant_task.fn(config, embedded) is validation
    assert flow_module.search_qdrant_task.fn(config, embedded) == searches
    assert flow_module.index_validate_search_qdrant_task.fn(config, embedded) == (
        index,
        validation,
        searches,
    )
    assert indexer.run.call_args.kwargs["client"] is client
    assert validator.run.call_args.kwargs["client"] is client
    assert searcher.run.call_args.kwargs["client"] is client
    counts = {"points_count": 1}
    diagnostics = {"validation": {}}
    assert (
        flow_module.export_vector_store_report_task.fn(
            config,
            index,
            validation,
            searches,
            "vector-run",
            counts,
            diagnostics,
        )
        is export
    )
    flow_module.log_vector_store_mlflow_task.fn(config, counts, export)
    tracker.log_run.assert_called_with(counts, export)
