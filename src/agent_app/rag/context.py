"""Сборка ограниченного контекста для онлайн-RAG."""

from __future__ import annotations

import tiktoken

from agent_app.rag.models import RagCitation, RagRetrievedChunk


class RagContextBuilder:
    """Формирует текстовый контекст с цитатами для RAG, гарантируя укладывание в заданный лимит токенов и корректную трассировку источников."""

    def __init__(
        self,
        *,
        max_tokens: int,
        excerpt_chars: int,
        tokenizer_model: str,
    ):
        """Готовит экземпляр к построению контекста с учётом лимита токенов и выбранной модели токенизации, обеспечивая устойчивость к неизвестным моделям."""
        self.max_tokens = max_tokens
        self.excerpt_chars = excerpt_chars
        try:
            self.encoding = tiktoken.encoding_for_model(tokenizer_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def build(
        self,
        chunks: list[RagRetrievedChunk],
    ) -> tuple[str, list[RagCitation], int]:
        """Гарантирует построение контекста с цитатами, не превышающего лимит токенов, и возвращает трассируемые ссылки на использованные фрагменты."""
        blocks: list[str] = []
        citations: list[RagCitation] = []
        used_tokens = 0
        for chunk in chunks:
            index = len(citations) + 1
            reference = f"[Источник {index}]"
            header = self._header(reference, chunk)
            text = chunk.text.strip()
            block = f"{header}\n{text}"
            candidate_context = "\n\n".join([*blocks, block])
            if len(self.encoding.encode(candidate_context)) > self.max_tokens:
                text = self._fit_body(blocks, header, text)
                if not text:
                    break
                block = f"{header}\n{text}"
                candidate_context = "\n\n".join([*blocks, block])

            blocks.append(block)
            # BPE-токены не аддитивны на границах строк. Считаем собранный
            # контекст целиком, включая два перевода строки между источниками.
            used_tokens = len(self.encoding.encode(candidate_context))
            citations.append(
                RagCitation(
                    reference=reference,
                    point_id=chunk.point_id,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    section=chunk.section,
                    position=chunk.position,
                    score=round(chunk.score, 8),
                    excerpt=text[: self.excerpt_chars],
                )
            )
        return "\n\n".join(blocks), citations, used_tokens

    def _fit_body(self, blocks: list[str], header: str, text: str) -> str:
        """Бинарным поиском выбирает максимальный body в общем token budget."""
        body_tokens = self.encoding.encode(text)
        low = 1
        high = len(body_tokens)
        best = ""
        while low <= high:
            middle = (low + high) // 2
            candidate_body = self.encoding.decode(body_tokens[:middle]).strip()
            block = f"{header}\n{candidate_body}"
            candidate_context = "\n\n".join([*blocks, block])
            if (
                candidate_body
                and len(self.encoding.encode(candidate_context)) <= self.max_tokens
            ):
                best = candidate_body
                low = middle + 1
            else:
                high = middle - 1
        return best

    @staticmethod
    def _header(reference: str, chunk: RagRetrievedChunk) -> str:
        """Формирует заголовок блока с уникальной ссылкой на источник и раздел, обеспечивая однозначную идентификацию цитаты."""
        source = chunk.source or "неизвестный источник"
        section = chunk.section or "без раздела"
        return (
            f"{reference} source={source}; section={section}; chunk_id={chunk.chunk_id}"
        )
