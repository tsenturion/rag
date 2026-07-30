"""Проверки связей между стадиями в четырёх ООП-фасадах RAG-конвейера."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import rag_prep.pipeline as pipeline_module
from rag_prep.models import (
    ChunkingExportResult,
    ChunkingValidationResult,
    EmbeddingExportResult,
    EmbeddingValidationResult,
    ExportResult,
    VectorStoreExportResult,
    VectorStoreIndexResult,
    VectorStoreValidationResult,
)
from rag_prep.pipeline import (
    RagChunkingPipeline,
    RagEmbeddingPipeline,
    RagPreparationPipeline,
    RagVectorStorePipeline,
)


def _path(root: Path, name: str) -> Path:
    """Формирует фиктивный путь артефакта без ненужной записи на диск."""
    return root / name


def test_preparation_facade_connects_all_stages(tmp_path: Path) -> None:
    """Проверяет порядок подготовки и передачу полной статистики экспорту и MLflow."""
    pipeline = RagPreparationPipeline.__new__(RagPreparationPipeline)
    pipeline.config = SimpleNamespace(
        run=SimpleNamespace(seed=42), paths=SimpleNamespace(input_dir=tmp_path)
    )
    sources = [object(), object()]
    raw = [object(), object(), object()]
    cleaned = [object(), object()]
    normalized = [object(), object()]
    deduplicated = [object()]
    documents = [object()]
    pipeline.loader = Mock()
    pipeline.loader.run.return_value = sources
    pipeline.parser = Mock()
    pipeline.parser.run.return_value = SimpleNamespace(
        elements=raw,
        failures=[SimpleNamespace(model_dump=lambda **_kwargs: {"source": "bad"})],
    )
    pipeline.cleaner = Mock()
    pipeline.cleaner.run.return_value = cleaned
    pipeline.normalizer = Mock()
    pipeline.normalizer.run.return_value = normalized
    pipeline.deduplicator = Mock()
    pipeline.deduplicator.run.return_value = SimpleNamespace(
        elements=deduplicated,
        duplicates_removed=1,
        exact_duplicates_removed=1,
        near_duplicates_removed=0,
    )
    pipeline.structurer = Mock()
    pipeline.structurer.run.return_value = documents
    pipeline.structurer.to_llama_documents.return_value = [object(), object()]
    export = ExportResult(
        json_path=_path(tmp_path, "documents.json"),
        jsonl_path=_path(tmp_path, "documents.jsonl"),
        manifest_path=_path(tmp_path, "manifest.json"),
        documents_count=1,
        duplicates_removed=1,
        run_id="export-run",
    )
    pipeline.exporter = Mock()
    pipeline.exporter.run.return_value = export
    pipeline.tracker = Mock()

    result = pipeline.run()

    assert result.sources_count == 2
    assert result.raw_elements_count == 3
    assert result.parse_failed_sources_count == 1
    assert result.prepared_documents_count == 1
    counts = pipeline.exporter.run.call_args.kwargs["counts"]
    assert counts["llama_index_documents_count"] == 2
    assert counts["exact_duplicates_removed"] == 1
    assert pipeline.tracker.log_run.call_args.args == (counts, export)


def test_chunking_facade_computes_structural_metrics(tmp_path: Path) -> None:
    """Проверяет сводные token/structure-метрики и связь run_id с splitter."""
    pipeline = RagChunkingPipeline.__new__(RagChunkingPipeline)
    pipeline.config = SimpleNamespace(
        run=SimpleNamespace(seed=7),
        paths=SimpleNamespace(input_jsonl=tmp_path / "documents.jsonl"),
    )
    chunks = [
        SimpleNamespace(
            metadata=SimpleNamespace(
                chunk_token_count=10,
                quality={"structure_score": 0.8},
                semantic_block_ids=["a", "b"],
            )
        ),
        SimpleNamespace(
            metadata=SimpleNamespace(
                chunk_token_count=20,
                quality={"structure_score": 1.0},
                semantic_block_ids=["b"],
            )
        ),
    ]
    validation = ChunkingValidationResult()
    pipeline.loader = Mock()
    pipeline.loader.run.return_value = [object()]
    pipeline.splitter = Mock()
    pipeline.splitter.run.return_value = chunks
    pipeline.validator = Mock()
    pipeline.validator.run.return_value = validation
    export = ChunkingExportResult(
        json_path=_path(tmp_path, "chunks.json"),
        jsonl_path=_path(tmp_path, "chunks.jsonl"),
        manifest_path=_path(tmp_path, "manifest.json"),
        chunks_count=2,
        run_id="export-run",
    )
    pipeline.exporter = Mock()
    pipeline.exporter.run.return_value = export
    pipeline.tracker = Mock()

    result = pipeline.run()

    assert result.chunks_count == 2
    counts = pipeline.exporter.run.call_args.kwargs["counts"]
    assert counts["semantic_blocks_count"] == 2
    assert counts["multi_block_chunks_count"] == 1
    assert counts["avg_chunk_tokens"] == 15
    assert counts["avg_chunk_structure_score"] == 0.9
    assert pipeline.splitter.run.call_args.kwargs["run_id"] == result.run_id


def test_embedding_facade_uses_shared_counts_builder(
    monkeypatch, tmp_path: Path
) -> None:
    """Проверяет единый расчёт diagnostics/counts между embedding facade и flow."""
    pipeline = RagEmbeddingPipeline.__new__(RagEmbeddingPipeline)
    pipeline.config = SimpleNamespace(
        run=SimpleNamespace(seed=8),
        paths=SimpleNamespace(input_jsonl=tmp_path / "chunks.jsonl"),
    )
    chunks = [object(), object()]
    embedded = [object(), object()]
    validation = EmbeddingValidationResult()
    pipeline.loader = Mock()
    pipeline.loader.run.return_value = chunks
    pipeline.embedder = Mock()
    pipeline.embedder.run.return_value = embedded
    pipeline.validator = Mock()
    pipeline.validator.run.return_value = validation
    export = EmbeddingExportResult(
        json_path=_path(tmp_path, "embeddings.json"),
        jsonl_path=_path(tmp_path, "embeddings.jsonl"),
        manifest_path=_path(tmp_path, "manifest.json"),
        embeddings_count=2,
        run_id="export-run",
    )
    pipeline.exporter = Mock()
    pipeline.exporter.run.return_value = export
    pipeline.tracker = Mock()
    counts = {"embeddings_count": 2}
    monkeypatch.setattr(pipeline_module, "build_embedding_counts", lambda *_: counts)
    monkeypatch.setattr(
        pipeline_module,
        "build_embedding_diagnostics",
        lambda _validation: {"validation": "ok"},
    )

    result = pipeline.run()

    assert result.embeddings_count == 2
    assert pipeline.validator.run.call_args.args == (chunks, embedded)
    assert pipeline.exporter.run.call_args.kwargs["counts"] is counts
    assert pipeline.tracker.log_run.call_args.args == (counts, export)


def test_vector_store_facade_reuses_one_qdrant_client(
    monkeypatch, tmp_path: Path
) -> None:
    """Проверяет последовательные index/validate/search через один клиент и экспорт."""
    pipeline = RagVectorStorePipeline.__new__(RagVectorStorePipeline)
    pipeline.config = SimpleNamespace(
        run=SimpleNamespace(seed=9),
        paths=SimpleNamespace(input_jsonl=tmp_path / "embeddings.jsonl"),
        vector_store=object(),
    )
    embedded = [object()]
    client = object()
    index = VectorStoreIndexResult(
        collection_name="test",
        provider="qdrant",
        mode="embedded",
        points_upserted=1,
        collection_points_count=1,
        vector_size=3,
        distance="cosine",
    )
    validation = VectorStoreValidationResult(
        embeddings_count=1, collection_points_count=1
    )
    search_results = []
    pipeline.loader = Mock()
    pipeline.loader.run.return_value = embedded
    pipeline.indexer = Mock()
    pipeline.indexer.run.return_value = index
    pipeline.validator = Mock()
    pipeline.validator.run.return_value = validation
    pipeline.searcher = Mock()
    pipeline.searcher.run.return_value = search_results
    export = VectorStoreExportResult(
        manifest_path=_path(tmp_path, "manifest.json"),
        validation_path=_path(tmp_path, "validation.json"),
        search_results_path=_path(tmp_path, "search.json"),
        run_id="export-run",
    )
    pipeline.exporter = Mock()
    pipeline.exporter.run.return_value = export
    pipeline.tracker = Mock()
    counts = {"points_count": 1}
    monkeypatch.setattr(
        pipeline_module, "qdrant_client_context", lambda _config: nullcontext(client)
    )
    monkeypatch.setattr(pipeline_module, "build_vector_store_counts", lambda *_: counts)
    monkeypatch.setattr(
        pipeline_module,
        "build_vector_store_diagnostics",
        lambda *_: {"validation": "ok"},
    )

    result = pipeline.run()

    assert result.points_count == 1
    assert pipeline.indexer.run.call_args.kwargs["client"] is client
    assert pipeline.validator.run.call_args.kwargs["client"] is client
    assert pipeline.searcher.run.call_args.kwargs["client"] is client
    assert pipeline.tracker.log_run.call_args.args == (counts, export)
