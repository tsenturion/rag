from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_prep.config import LoaderConfig, StructuringConfig  # noqa: E402
from rag_prep.models import ProcessedElement  # noqa: E402
from rag_prep.stages.loading import LlamaIndexLoadingStage  # noqa: E402
from rag_prep.stages.parsing import UnstructuredParsingStage  # noqa: E402
from rag_prep.stages.structuring import LlamaIndexStructuringStage  # noqa: E402


class StableSourceIdsTest(unittest.TestCase):
    def test_ids_do_not_depend_on_source_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first_root = root / "first"
            second_root = root / "second"
            first_path = first_root / "one" / "document.txt"
            second_path = second_root / "other" / "renamed.txt"
            first_path.parent.mkdir(parents=True)
            second_path.parent.mkdir(parents=True)
            text = "Одинаковый документ с воспроизводимым идентификатором."
            first_path.write_text(text, encoding="utf-8")
            second_path.write_text(text, encoding="utf-8")

            loader = LlamaIndexLoadingStage(LoaderConfig())
            first = loader._to_source_file(first_path, input_dir=first_root)
            second = loader._to_source_file(second_path, input_dir=second_root)

            self.assertEqual(first.source_hash, second.source_hash)
            self.assertNotEqual(first.source, second.source)
            self.assertNotEqual(first.source_key, second.source_key)
            self.assertEqual(
                UnstructuredParsingStage._element_id(first, 0),
                UnstructuredParsingStage._element_id(second, 0),
            )

            first_document = self._structured_document(first, text)
            second_document = self._structured_document(second, text)
            self.assertEqual(first_document.metadata.id, second_document.metadata.id)
            self.assertEqual(
                first_document.metadata.parent_ids,
                second_document.metadata.parent_ids,
            )

    @staticmethod
    def _structured_document(source, text: str):
        element = ProcessedElement(
            source_file=source,
            element_id=UnstructuredParsingStage._element_id(source, 0),
            element_index=0,
            text=text,
            element_type="NarrativeText",
            section="Основной раздел",
            section_path=["Основной раздел"],
            metadata={"sentence_count": 1, "token_count": 8},
        )
        stage = LlamaIndexStructuringStage(StructuringConfig())
        return stage.run([element], run_id="stable-run")[0]


if __name__ == "__main__":
    unittest.main()
