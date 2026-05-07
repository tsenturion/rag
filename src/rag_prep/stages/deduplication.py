from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from rag_prep.config import DeduplicationConfig
from rag_prep.models import ProcessedElement
from rag_prep.utils import stable_id, text_sha256

LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class DeduplicationResult:
    elements: list[ProcessedElement]
    duplicates_removed: int


class DeduplicationStage:
    """Remove exact and near-duplicate text with datasketch MinHash LSH."""

    def __init__(self, config: DeduplicationConfig):
        self.config = config

    def run(self, elements: list[ProcessedElement]) -> DeduplicationResult:
        if not self.config.enabled:
            return DeduplicationResult(elements=elements, duplicates_removed=0)

        seen_hashes: set[str] = set()
        lsh = MinHashLSH(threshold=self.config.threshold, num_perm=self.config.num_perm)
        kept: list[ProcessedElement] = []
        duplicates = 0

        for element in elements:
            digest = text_sha256(element.text)
            if digest in seen_hashes:
                duplicates += 1
                continue

            tokens = self._tokens(element.text)
            key = stable_id(element.source_file.source, element.element_index, digest)
            if len(tokens) >= self.config.min_tokens:
                minhash = self._minhash(tokens)
                matches = lsh.query(minhash)
                if matches:
                    duplicates += 1
                    continue
                lsh.insert(key, minhash)

            seen_hashes.add(digest)
            kept.append(element)

        LOGGER.info("Deduplicated %d elements into %d elements", len(elements), len(kept))
        return DeduplicationResult(elements=kept, duplicates_removed=duplicates)

    def _tokens(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def _minhash(self, tokens: list[str]) -> MinHash:
        minhash = MinHash(num_perm=self.config.num_perm)
        shingles = self._shingles(tokens)
        for shingle in shingles:
            minhash.update(" ".join(shingle).encode("utf-8"))
        return minhash

    def _shingles(self, tokens: list[str]) -> list[tuple[str, ...]]:
        size = self.config.shingle_size
        if len(tokens) < size:
            return [tuple(tokens)]
        return [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]

