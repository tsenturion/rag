from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_prep.chunking_stages.splitting import (  # noqa: E402
    ChunkSplittingStage,
    SemanticBlock,
)
from rag_prep.config import ChunkingConfig  # noqa: E402


class ChunkOverlapTest(unittest.TestCase):
    def test_large_semantic_block_is_not_used_as_oversized_overlap(self) -> None:
        stage = ChunkSplittingStage(
            ChunkingConfig(
                chunk_size=100,
                chunk_overlap=10,
                tokenizer_model="cl100k_base",
                embedding_model="test-embedding",
            )
        )
        previous = self._block(stage, "данные " * 30, position=0)
        next_block = self._block(stage, "результат " * 20, position=1)
        self.assertGreater(previous.token_count, stage.config.chunk_overlap)

        overlap = stage._overlap_blocks([previous], next_block=next_block)

        self.assertEqual(overlap, [])
        self.assertLessEqual(
            stage._joined_token_count(overlap),
            stage.config.chunk_overlap,
        )

    @staticmethod
    def _block(
        stage: ChunkSplittingStage,
        text: str,
        *,
        position: int,
    ) -> SemanticBlock:
        normalized = text.strip()
        return SemanticBlock(
            id=f"block-{position}",
            text=normalized,
            position=position,
            start_char=0,
            end_char=len(normalized),
            token_count=len(stage._tokenize(normalized)),
            origin_element_ids=[f"element-{position}"],
        )


if __name__ == "__main__":
    unittest.main()
