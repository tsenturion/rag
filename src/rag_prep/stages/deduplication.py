"""Дедупликация текстовых элементов для подготовки документов."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from datasketch import MinHash, MinHashLSH

from rag_prep.config import DeduplicationConfig
from rag_prep.models import ProcessedElement
from rag_prep.utils import text_sha256

LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class DeduplicationResult:
    """Гарантирует прозрачный контракт о составе уникальных элементов и количестве удалённых дубликатов после дедупликации."""

    elements: list[ProcessedElement]
    duplicates_removed: int
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0


class DeduplicationStage:
    """Удаляет exact и near-дубли через datasketch MinHash LSH."""

    def __init__(self, config: DeduplicationConfig):
        """Гарантирует готовность экземпляра к дедупликации с заданной политикой поиска дубликатов."""
        self.config = config

    def run(self, elements: list[ProcessedElement]) -> DeduplicationResult:
        """Удаляет точные и близкие дубликаты, гарантируя уникальность возвращаемых элементов согласно политике конфигурации."""
        if not self.config.enabled:
            return DeduplicationResult(elements=elements, duplicates_removed=0)

        seen_hashes: set[str] = set()
        lsh = MinHashLSH(threshold=self.config.threshold, num_perm=self.config.num_perm)
        kept: list[ProcessedElement] = []
        inserted_keys: set[str] = set()
        kept_short_texts: list[str] = []
        exact_duplicates = 0
        near_duplicates = 0

        for position, element in enumerate(elements):
            digest = text_sha256(element.text)
            if digest in seen_hashes:
                exact_duplicates += 1
                continue

            tokens = self._tokens(element.text)
            if len(tokens) < self.config.min_tokens:
                normalized_short_text = self._normalized_short_text(tokens)
                if self._is_near_short_duplicate(
                    normalized_short_text, kept_short_texts
                ):
                    seen_hashes.add(digest)
                    near_duplicates += 1
                    continue
                kept_short_texts.append(normalized_short_text)
            else:
                minhash = self._minhash(tokens)
                matches = lsh.query(minhash)
                if matches:
                    seen_hashes.add(digest)
                    near_duplicates += 1
                    continue
                key = self._lsh_key(position, digest, inserted_keys)
                lsh.insert(key, minhash)
                inserted_keys.add(key)

            seen_hashes.add(digest)
            kept.append(element)

        duplicates = exact_duplicates + near_duplicates
        LOGGER.info(
            "Дедупликация элементов: %d -> %d; exact=%d near=%d",
            len(elements),
            len(kept),
            exact_duplicates,
            near_duplicates,
        )
        return DeduplicationResult(
            elements=kept,
            duplicates_removed=duplicates,
            exact_duplicates_removed=exact_duplicates,
            near_duplicates_removed=near_duplicates,
        )

    @staticmethod
    def _lsh_key(position: int, digest: str, inserted_keys: set[str]) -> str:
        """Гарантирует уникальность ключа для вставки в LSH, предотвращая коллизии при дедупликации."""
        key = f"{position}:{digest}"
        if key not in inserted_keys:
            return key

        suffix = 1
        while f"{key}:{suffix}" in inserted_keys:
            suffix += 1
        return f"{key}:{suffix}"

    def _tokens(self, text: str) -> list[str]:
        """Гарантирует получение нормализованного списка токенов для корректного сравнения текстов при дедупликации."""
        return [token.lower() for token in TOKEN_RE.findall(text)]

    @staticmethod
    def _normalized_short_text(tokens: list[str]) -> str:
        """Гарантирует воспроизводимую строку для сравнения коротких текстов вне зависимости от исходного форматирования."""
        return " ".join(tokens)

    def _is_near_short_duplicate(self, text: str, candidates: list[str]) -> bool:
        """Гарантирует, что короткие тексты с высокой степенью схожести не будут повторно включены в итоговую выборку."""
        if not text:
            return False
        return any(
            SequenceMatcher(None, text, candidate).ratio() >= self.config.threshold
            for candidate in candidates
        )

    def _minhash(self, tokens: list[str]) -> MinHash:
        """Обеспечивает воспроизводимое хеширование токенов для эффективного поиска дубликатов по схожести."""
        minhash = MinHash(num_perm=self.config.num_perm)
        shingles = self._shingles(tokens)
        for shingle in shingles:
            minhash.update(" ".join(shingle).encode("utf-8"))
        return minhash

    def _shingles(self, tokens: list[str]) -> list[tuple[str, ...]]:
        """Гарантирует разбиение последовательности токенов на перекрывающиеся фрагменты фиксированной длины для дальнейшего сравнения."""
        size = self.config.shingle_size
        if len(tokens) < size:
            return [tuple(tokens)]
        return [
            tuple(tokens[index : index + size])
            for index in range(len(tokens) - size + 1)
        ]
