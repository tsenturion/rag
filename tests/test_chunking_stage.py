"""Полные unit-тесты семантического и сквозного чанкинга."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rag_prep.chunking_stages.splitting import ChunkSplittingStage, SemanticBlock
from rag_prep.config import ChunkingConfig
from rag_prep.models import DocumentMetadata, PreparedDocument
from rag_prep.utils import text_sha256


def _document(text: str, *, origins: list[str] | None = None) -> PreparedDocument:
    """Создаёт документ с полным lineage для проверки metadata готовых чанков."""
    return PreparedDocument(
        text=text,
        metadata=DocumentMetadata(
            id="document-1",
            source="data/raw/source.txt",
            source_key="source.txt",
            section="Регламент",
            file_name="source.txt",
            file_type="txt",
            source_hash="source-hash",
            text_hash=text_sha256(text),
            parent_ids=["source-1"],
            origin_element_ids=origins or ["element-1"],
            lineage={"source_id": "source-1"},
            hierarchy={"section_path": ["Регламент"]},
            element_start=0,
            element_end=max(0, len(origins or ["element-1"]) - 1),
            element_types=["NarrativeText"],
            char_count=len(text),
            word_count=len(text.split()),
            sentence_count=text.count("."),
            pipeline_run_id="prep-run",
            parsed_at=datetime.now(timezone.utc),
            extra={"quality": {"source_quality": 0.9}},
        ),
    )


def _stage(**overrides: object) -> ChunkSplittingStage:
    """Создаёт stage с малым, но валидным token budget для тестовых текстов."""
    values = {
        "chunk_size": 32,
        "chunk_overlap": 8,
        "tokenizer_model": "cl100k_base",
        "embedding_model": "text-embedding-3-small",
        "min_chunk_tokens": 4,
    }
    values.update(overrides)
    return ChunkSplittingStage(ChunkingConfig(**values))


def test_semantic_chunking_preserves_lineage_offsets_and_quality() -> None:
    """Проверяет сборку чанков из абзацев и происхождение каждого результата."""
    paragraphs = [
        "Первый абзац описывает приём инженерной заявки и первичную диагностику.",
        "Второй абзац фиксирует владельца, приоритет и ожидаемый срок решения.",
        "Третий абзац требует записать результат проверки в журнал инцидента.",
    ]
    text = "\n\n".join(paragraphs)
    stage = _stage()

    chunks = stage.run(
        [_document(text, origins=["element-1", "element-2", "element-3"])],
        run_id="chunk-run",
    )

    assert chunks
    assert [chunk.metadata.position for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        metadata = chunk.metadata
        assert text[metadata.chunk_start_char : metadata.chunk_end_char] == chunk.text
        assert metadata.parent_ids == ["document-1"]
        assert metadata.chunking_run_id == "chunk-run"
        assert metadata.lineage["pipeline_stage"] == "chunk"
        assert metadata.quality["source_quality"] == 0.9
        assert metadata.quality["semantic_block_count"] >= 1


def test_oversized_block_is_split_without_cross_block_lineage() -> None:
    """Проверяет дробление одного большого абзаца внутри его точных source offsets."""
    text = " ".join(
        f"Предложение {index} содержит диагностические данные." for index in range(45)
    )
    stage = _stage(chunk_size=40, chunk_overlap=6)

    chunks = stage.run([_document(text)], run_id="oversized-run")

    assert len(chunks) > 1
    assert all(chunk.metadata.semantic_block_start == 0 for chunk in chunks)
    assert all(chunk.metadata.semantic_block_end == 0 for chunk in chunks)
    assert all(chunk.metadata.chunk_token_count <= 40 for chunk in chunks)
    assert all(
        chunk.metadata.offset_strategy.startswith("semantic_block_") for chunk in chunks
    )


def test_whole_document_mode_marks_exact_and_estimated_offsets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Проверяет cursor search и явную маркировку неразрешимого изменения текста."""
    text = "Повторяемый фрагмент.\n\nПовторяемый фрагмент.\n\nФинальная часть."
    stage = _stage(
        preserve_section_boundaries=False,
        preserve_block_boundaries=False,
    )

    class _Splitter:
        """Имитирует splitter, который один фрагмент нормализовал необратимо."""

        def split_text(self, _text: str) -> list[str]:
            """Возвращает точный и отсутствующий в исходнике фрагменты."""
            return ["Повторяемый фрагмент.", "Нормализованный отсутствующий текст"]

    stage.splitter = _Splitter()
    chunks = stage.run([_document(text)], run_id="whole-run")

    assert chunks[0].metadata.chunk_start_char == 0
    assert chunks[0].metadata.offset_strategy == "bounded_cursor_search"
    assert chunks[1].metadata.offset_strategy == "estimated_cursor_fallback"
    assert "оценочные offsets" in caplog.text


def test_span_overlap_and_quality_helpers_cover_boundary_cases() -> None:
    """Проверяет trimming, пересечение spans, overlap и признаки шумного текста."""
    stage = _stage(chunk_overlap=12)
    spans = stage._paragraph_spans("  один\nдва  \n\n\tтри\t")
    assert spans == [("один\nдва", 2, 10), ("три", 15, 18)]
    assert stage._paragraph_spans(" \n\n ") == []

    first = SemanticBlock("a", "короткий блок", 0, 2, 16, 3, ["one"])
    second = SemanticBlock("b", "следующий блок", 1, 20, 35, 3, ["two"])
    assert stage._blocks_for_span([first, second], 10, 22) == [first, second]
    assert stage._overlap_blocks([first], next_block=second) == [first]
    assert _stage(chunk_overlap=0)._overlap_blocks([first], next_block=second) == []

    quality = stage._quality("Текст с шумом �\x01.", 6, None, [first])
    assert quality["ocr_noise_score"] > 0
    assert quality["language_confidence"] > 0
    assert stage._ordered_unique(["a", "b", "a"]) == ["a", "b"]
    assert stage._is_supported_letter("Ё")
    assert not stage._is_supported_letter("中")
    assert stage._is_noise_char("�")


def test_unknown_tokenizer_falls_back_to_cl100k_base() -> None:
    """Проверяет воспроизводимый fallback при неизвестном tokenizer_model."""
    stage = _stage(tokenizer_model="unknown-tokenizer-for-test")
    assert stage._tokenize("проверка")
