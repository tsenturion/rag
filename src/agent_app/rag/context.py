from __future__ import annotations

import tiktoken

from agent_app.rag.models import RagCitation, RagRetrievedChunk


class RagContextBuilder:
    def __init__(
        self,
        *,
        max_tokens: int,
        excerpt_chars: int,
        tokenizer_model: str,
    ):
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
        blocks: list[str] = []
        citations: list[RagCitation] = []
        used_tokens = 0
        for index, chunk in enumerate(chunks, start=1):
            reference = f"[Источник {index}]"
            header = self._header(reference, chunk)
            available = self.max_tokens - used_tokens
            if available <= 0:
                break
            text = chunk.text.strip()
            block = f"{header}\n{text}"
            tokens = self.encoding.encode(block)
            if len(tokens) > available:
                header_tokens = self.encoding.encode(f"{header}\n")
                body_budget = available - len(header_tokens)
                if body_budget <= 0:
                    break
                text = self.encoding.decode(
                    self.encoding.encode(text)[:body_budget]
                ).strip()
                if not text:
                    break
                block = f"{header}\n{text}"
                tokens = self.encoding.encode(block)

            blocks.append(block)
            used_tokens += len(tokens)
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

    @staticmethod
    def _header(reference: str, chunk: RagRetrievedChunk) -> str:
        source = chunk.source or "неизвестный источник"
        section = chunk.section or "без раздела"
        return (
            f"{reference} source={source}; section={section}; chunk_id={chunk.chunk_id}"
        )
