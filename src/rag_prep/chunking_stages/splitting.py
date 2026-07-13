from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

import tiktoken
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

from rag_prep.config import ChunkingConfig
from rag_prep.models import ChunkMetadata, PreparedChunk, PreparedDocument
from rag_prep.utils import stable_id, text_sha256

LOGGER = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")
_COMMON_PUNCTUATION = set(".,;:!?()[]{}<>«»\"'`%-–—/\\№+=_*&@#$|")
_PARAGRAPH_SEPARATOR_RE = re.compile(r"(?:\r?\n[ \t]*){2,}")


@dataclass(frozen=True)
class SemanticBlock:
    id: str
    text: str
    position: int
    start_char: int
    end_char: int
    token_count: int
    origin_element_ids: list[str]


@dataclass(frozen=True)
class LocatedSpan:
    start_char: int
    end_char: int
    offset_strategy: str


class ChunkSplittingStage:
    """Делит подготовленные документы на чанки для embeddings без расчёта embeddings."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        try:
            self.encoding = tiktoken.encoding_for_model(config.tokenizer_model)
        except KeyError:
            try:
                self.encoding = tiktoken.get_encoding(config.tokenizer_model)
            except ValueError:
                LOGGER.warning(
                    "Неизвестная модель токенизатора %s; используется cl100k_base",
                    config.tokenizer_model,
                )
                self.encoding = tiktoken.get_encoding("cl100k_base")
        self.splitter = self._build_splitter()

    def run(self, documents: list[PreparedDocument]) -> list[PreparedChunk]:
        chunks: list[PreparedChunk] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        LOGGER.info(
            "Разделено документов: %d; получено чанков: %d", len(documents), len(chunks)
        )
        return chunks

    def _build_splitter(self):
        kwargs = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "tokenizer": self._tokenize,
        }
        if self.config.strategy == "token":
            return TokenTextSplitter(**kwargs)
        return SentenceSplitter(**kwargs)

    def _split_document(self, document: PreparedDocument) -> list[PreparedChunk]:
        if not document.text.strip():
            return []

        blocks = self._semantic_blocks(document)
        if (
            self.config.preserve_section_boundaries
            and self.config.preserve_block_boundaries
        ):
            return self._split_semantic_blocks(document, blocks)
        return self._split_whole_document(document, blocks)

    def _split_semantic_blocks(
        self, document: PreparedDocument, blocks: list[SemanticBlock]
    ) -> list[PreparedChunk]:
        chunks: list[PreparedChunk] = []
        current: list[SemanticBlock] = []

        for block in blocks:
            if block.token_count > self.config.chunk_size:
                if current:
                    chunks.append(
                        self._chunk_from_blocks(document, current, len(chunks))
                    )
                    current = []
                chunks.extend(self._split_oversized_block(document, block, len(chunks)))
                continue

            candidate = [*current, block]
            if current and self._joined_token_count(candidate) > self.config.chunk_size:
                chunks.append(self._chunk_from_blocks(document, current, len(chunks)))
                overlap = self._overlap_blocks(current, next_block=block)
                current = [*overlap, block]
            else:
                current = candidate

        if current:
            chunks.append(self._chunk_from_blocks(document, current, len(chunks)))
        return chunks

    def _split_whole_document(
        self, document: PreparedDocument, blocks: list[SemanticBlock]
    ) -> list[PreparedChunk]:
        source_text = document.text
        split_texts = self.splitter.split_text(source_text)
        chunks: list[PreparedChunk] = []
        cursor = 0
        for chunk_text in split_texts:
            normalized_chunk_text = chunk_text.strip()
            if not normalized_chunk_text:
                continue
            span = self._locate_split_with_cursor(
                source_text=source_text,
                chunk_text=normalized_chunk_text,
                cursor=cursor,
                exact_strategy="bounded_cursor_search",
                fallback_strategy="estimated_cursor_fallback",
            )
            cursor = span.end_char
            overlapping_blocks = self._blocks_for_span(
                blocks, span.start_char, span.end_char
            )
            chunks.append(
                self._chunk_from_text(
                    document=document,
                    chunk_text=normalized_chunk_text,
                    position=len(chunks),
                    start_char=span.start_char,
                    end_char=span.end_char,
                    blocks=overlapping_blocks,
                    offset_strategy=span.offset_strategy,
                )
            )
        return chunks

    def _split_oversized_block(
        self, document: PreparedDocument, block: SemanticBlock, start_position: int
    ) -> list[PreparedChunk]:
        chunks: list[PreparedChunk] = []
        cursor = 0
        for chunk_text in self.splitter.split_text(block.text):
            normalized_chunk_text = chunk_text.strip()
            if not normalized_chunk_text:
                continue
            span = self._locate_split_with_cursor(
                source_text=block.text,
                chunk_text=normalized_chunk_text,
                cursor=cursor,
                exact_strategy="semantic_block_bounded_cursor_search",
                fallback_strategy="semantic_block_estimated_cursor_fallback",
            )
            cursor = span.end_char
            chunks.append(
                self._chunk_from_text(
                    document=document,
                    chunk_text=normalized_chunk_text,
                    position=start_position + len(chunks),
                    start_char=block.start_char + span.start_char,
                    end_char=block.start_char + span.end_char,
                    blocks=[block],
                    offset_strategy=span.offset_strategy,
                )
            )
        return chunks

    def _chunk_from_blocks(
        self, document: PreparedDocument, blocks: list[SemanticBlock], position: int
    ) -> PreparedChunk:
        chunk_text = "\n\n".join(block.text for block in blocks).strip()
        return self._chunk_from_text(
            document=document,
            chunk_text=chunk_text,
            position=position,
            start_char=blocks[0].start_char,
            end_char=blocks[-1].end_char,
            blocks=blocks,
            offset_strategy="semantic_block_span",
        )

    def _chunk_from_text(
        self,
        *,
        document: PreparedDocument,
        chunk_text: str,
        position: int,
        start_char: int,
        end_char: int,
        blocks: list[SemanticBlock],
        offset_strategy: str,
    ) -> PreparedChunk:
        return PreparedChunk(
            text=chunk_text,
            metadata=self._metadata(
                document=document,
                chunk_text=chunk_text,
                position=position,
                start_char=start_char,
                end_char=end_char,
                blocks=blocks,
                offset_strategy=offset_strategy,
            ),
        )

    def _metadata(
        self,
        *,
        document: PreparedDocument,
        chunk_text: str,
        position: int,
        start_char: int,
        end_char: int,
        blocks: list[SemanticBlock],
        offset_strategy: str,
    ) -> ChunkMetadata:
        doc_meta = document.metadata
        token_count = len(self._tokenize(chunk_text))
        text_hash = text_sha256(chunk_text)
        semantic_block_ids = [block.id for block in blocks]
        semantic_positions = [block.position for block in blocks]
        origin_element_ids = self._ordered_unique(
            origin_id for block in blocks for origin_id in block.origin_element_ids
        )
        if not origin_element_ids:
            origin_element_ids = doc_meta.origin_element_ids

        chunk_id = stable_id(doc_meta.id, position, start_char, end_char, text_hash)
        lineage = dict(doc_meta.lineage)
        lineage.update(
            {
                "document_id": doc_meta.id,
                "chunk_id": chunk_id,
                "chunk_position": position,
                "semantic_block_ids": semantic_block_ids,
                "semantic_block_range": [
                    min(semantic_positions) if semantic_positions else None,
                    max(semantic_positions) if semantic_positions else None,
                ],
                "pipeline_stage": "chunk",
            }
        )
        hierarchy = dict(doc_meta.hierarchy)
        hierarchy.update(
            {
                "chunk_position": position,
                "semantic_block_count": len(blocks),
                "semantic_block_range": [
                    min(semantic_positions) if semantic_positions else None,
                    max(semantic_positions) if semantic_positions else None,
                ],
            }
        )

        return ChunkMetadata(
            id=chunk_id,
            document_id=doc_meta.id,
            source=doc_meta.source,
            section=doc_meta.section,
            position=position,
            chunk_start_char=start_char,
            chunk_end_char=end_char,
            chunk_token_count=token_count,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunking_strategy=self.config.strategy,
            tokenizer_model=self.config.tokenizer_model,
            embedding_model=self.config.embedding_model,
            semantic_block_ids=semantic_block_ids,
            semantic_block_start=min(semantic_positions)
            if semantic_positions
            else None,
            semantic_block_end=max(semantic_positions) if semantic_positions else None,
            offset_strategy=offset_strategy,
            parent_ids=[doc_meta.id],
            origin_element_ids=origin_element_ids,
            lineage=lineage,
            hierarchy=hierarchy,
            source_hash=doc_meta.source_hash,
            document_text_hash=doc_meta.text_hash,
            text_hash=text_hash,
            file_name=doc_meta.file_name,
            file_type=doc_meta.file_type,
            quality=self._quality(
                chunk_text, token_count, doc_meta.extra.get("quality"), blocks
            ),
        )

    def _semantic_blocks(self, document: PreparedDocument) -> list[SemanticBlock]:
        spans = self._paragraph_spans(document.text)
        if not spans:
            return []

        origin_ids = document.metadata.origin_element_ids
        one_origin_per_block = len(origin_ids) == len(spans)
        blocks: list[SemanticBlock] = []
        for position, (text, start_char, end_char) in enumerate(spans):
            block_origin_ids = (
                [origin_ids[position]] if one_origin_per_block else origin_ids
            )
            block_id = stable_id(
                document.metadata.id,
                "semantic_block",
                position,
                start_char,
                end_char,
                text_sha256(text),
            )
            blocks.append(
                SemanticBlock(
                    id=block_id,
                    text=text,
                    position=position,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=len(self._tokenize(text)),
                    origin_element_ids=block_origin_ids,
                )
            )
        return blocks

    @staticmethod
    def _paragraph_spans(text: str) -> list[tuple[str, int, int]]:
        blocks: list[tuple[str, int, int]] = []
        cursor = 0
        for separator in _PARAGRAPH_SEPARATOR_RE.finditer(text):
            ChunkSplittingStage._append_trimmed_span(
                blocks,
                text=text,
                start=cursor,
                end=separator.start(),
            )
            cursor = separator.end()
        ChunkSplittingStage._append_trimmed_span(
            blocks,
            text=text,
            start=cursor,
            end=len(text),
        )
        return blocks

    @staticmethod
    def _append_trimmed_span(
        blocks: list[tuple[str, int, int]], *, text: str, start: int, end: int
    ) -> None:
        segment = text[start:end]
        if not segment.strip():
            return

        leading = len(segment) - len(segment.lstrip())
        trailing = len(segment.rstrip())
        span_start = start + leading
        span_end = start + trailing
        block_text = text[span_start:span_end]
        if block_text:
            blocks.append((block_text, span_start, span_end))

    def _overlap_blocks(
        self, blocks: list[SemanticBlock], *, next_block: SemanticBlock
    ) -> list[SemanticBlock]:
        if self.config.chunk_overlap <= 0:
            return []

        overlap: list[SemanticBlock] = []
        for block in reversed(blocks):
            candidate = [block, *overlap]
            candidate_tokens = self._joined_token_count(candidate)
            if candidate_tokens <= self.config.chunk_overlap:
                overlap = candidate
                continue
            break
        if not overlap:
            return []

        if self._joined_token_count([*overlap, next_block]) <= self.config.chunk_size:
            return overlap

        LOGGER.debug(
            (
                "Block-level overlap отброшен, потому что overlap вместе со следующим "
                "блоком превышает chunk_size: overlap_blocks=%d next_block=%s"
            ),
            len(overlap),
            next_block.id,
        )
        return []

    def _joined_token_count(self, blocks: list[SemanticBlock]) -> int:
        return len(self._tokenize("\n\n".join(block.text for block in blocks)))

    @staticmethod
    def _blocks_for_span(
        blocks: list[SemanticBlock], start_char: int, end_char: int
    ) -> list[SemanticBlock]:
        return [
            block
            for block in blocks
            if block.start_char < end_char and block.end_char > start_char
        ]

    def _locate_split_with_cursor(
        self,
        *,
        source_text: str,
        chunk_text: str,
        cursor: int,
        exact_strategy: str,
        fallback_strategy: str,
    ) -> LocatedSpan:
        if source_text[cursor:].startswith(chunk_text):
            return LocatedSpan(cursor, cursor + len(chunk_text), exact_strategy)

        overlap_window = max(len(chunk_text), self.config.chunk_overlap * 8)
        search_from = max(0, cursor - overlap_window)
        start = source_text.find(chunk_text, search_from)
        if start == -1:
            start = min(cursor, len(source_text))
            LOGGER.warning(
                (
                    "Не удалось найти результат splitter в исходном тексте; используются "
                    "оценочные offsets. cursor=%d chunk_chars=%d source_chars=%d strategy=%s"
                ),
                cursor,
                len(chunk_text),
                len(source_text),
                fallback_strategy,
            )
            return LocatedSpan(
                start,
                min(start + len(chunk_text), len(source_text)),
                fallback_strategy,
            )
        return LocatedSpan(
            start,
            min(start + len(chunk_text), len(source_text)),
            exact_strategy,
        )

    def _quality(
        self,
        chunk_text: str,
        token_count: int,
        document_quality: object,
        blocks: list[SemanticBlock],
    ) -> dict[str, object]:
        quality = dict(document_quality) if isinstance(document_quality, dict) else {}
        words = [word.lower() for word in _WORD_RE.findall(chunk_text)]
        char_count = len(chunk_text)
        alpha_count = sum(1 for char in chunk_text if char.isalpha())
        known_script_count = sum(
            1 for char in chunk_text if self._is_supported_letter(char)
        )
        noisy_chars = sum(1 for char in chunk_text if self._is_noise_char(char))
        sentence_count = len(_SENTENCE_END_RE.findall(chunk_text))
        unique_token_ratio = len(set(words)) / len(words) if words else 0.0
        language_confidence = known_script_count / alpha_count if alpha_count else 0.0
        token_density = token_count / char_count if char_count else 0.0
        ocr_noise_score = min(1.0, noisy_chars / char_count) if char_count else 0.0
        length_score = min(1.0, token_count / self.config.min_chunk_tokens)
        sentence_score = 1.0 if sentence_count > 0 or len(blocks) > 1 else 0.5
        block_score = min(1.0, len(blocks) / 2) if blocks else 0.0
        structure_score = max(
            0.0,
            min(
                1.0,
                (0.35 * length_score)
                + (0.25 * sentence_score)
                + (0.20 * block_score)
                + (0.20 * unique_token_ratio)
                - (0.35 * ocr_noise_score),
            ),
        )

        quality.update(
            {
                "token_density": round(token_density, 4),
                "language_confidence": round(language_confidence, 3),
                "ocr_noise_score": round(ocr_noise_score, 3),
                "structure_score": round(structure_score, 3),
                "unique_token_ratio": round(unique_token_ratio, 3),
                "semantic_block_count": len(blocks),
                "sentence_count": sentence_count,
                "char_count": char_count,
                "word_count": len(words),
                "is_low_quality_chunk": structure_score < self.config.min_quality_score,
            }
        )
        return quality

    @staticmethod
    def _is_supported_letter(char: str) -> bool:
        lower = char.lower()
        return ("а" <= lower <= "я") or lower == "ё" or ("a" <= lower <= "z")

    @staticmethod
    def _is_noise_char(char: str) -> bool:
        if char == "\ufffd" or (ord(char) < 32 and char not in "\n\r\t"):
            return True
        return (
            not char.isalnum()
            and not char.isspace()
            and char not in _COMMON_PUNCTUATION
        )

    @staticmethod
    def _ordered_unique(values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _tokenize(self, text: str) -> list[int]:
        return self.encoding.encode(text)
