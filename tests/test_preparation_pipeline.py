"""Регрессионные тесты для подсистемы preparation_pipeline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_prep.config import (  # noqa: E402
    CleaningConfig,
    DeduplicationConfig,
    LoaderConfig,
    NormalizationConfig,
    ParserConfig,
    StructuringConfig,
)
from rag_prep.stages.cleaning import TextCleaningStage  # noqa: E402
from rag_prep.stages.deduplication import DeduplicationStage  # noqa: E402
from rag_prep.stages.loading import LlamaIndexLoadingStage  # noqa: E402
from rag_prep.stages.normalization import TextNormalizationStage  # noqa: E402
from rag_prep.stages.parsing import UnstructuredParsingStage  # noqa: E402
from rag_prep.stages.structuring import LlamaIndexStructuringStage  # noqa: E402


class PreparationPipelineTest(unittest.TestCase):
    """Проверяет корректность обработки данных в подготовительном пайплайне, включая загрузку, очистку, нормализацию, дедупликацию и структурирование документов."""

    def test_empty_input_corpus_is_rejected(self) -> None:
        """Не публикует успешный запуск подготовки без исходных документов."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir) / "raw"
            input_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "Входной корпус пуст"):
                LlamaIndexLoadingStage(LoaderConfig()).run(input_dir)

    def test_csv_passes_loading_cleaning_normalization_dedup_and_structuring(
        self,
    ) -> None:
        """Проверяет, что CSV-файл корректно проходит этапы загрузки, очистки, нормализации, удаления дубликатов и структурирования, обеспечивая целостность и качество итоговых документов."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir) / "raw"
            input_dir.mkdir()
            csv_path = input_dir / "knowledge.csv"
            csv_path.write_text(
                "Раздел,Название,Описание\n"
                "Поддержка,Сброс пароля,Пользователь подтверждает личность.\n"
                "Поддержка,Сброс пароля,Пользователь подтверждает личность.\n",
                encoding="utf-8",
            )

            source = LlamaIndexLoadingStage(LoaderConfig())._to_source_file(
                csv_path,
                input_dir=input_dir,
            )
            parser = UnstructuredParsingStage(
                ParserConfig(),
                default_section="Полный документ",
            )
            raw = parser._parse_csv_source(source)
            cleaned = TextCleaningStage(CleaningConfig(min_chars=5)).run(raw)
            normalized = TextNormalizationStage(
                NormalizationConfig(spacy_language="ru")
            ).run(cleaned)
            deduplicated = DeduplicationStage(
                DeduplicationConfig(
                    threshold=0.9,
                    num_perm=32,
                    shingle_size=2,
                    min_tokens=3,
                )
            ).run(normalized)
            documents = LlamaIndexStructuringStage(StructuringConfig()).run(
                deduplicated.elements,
                run_id="preparation-test",
            )

            self.assertEqual(len(raw), 2)
            self.assertEqual(len(cleaned), 2)
            self.assertEqual(deduplicated.exact_duplicates_removed, 1)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].metadata.source_hash, source.source_hash)
            self.assertEqual(
                documents[0].metadata.origin_element_ids, [raw[0].element_id]
            )
            self.assertGreater(documents[0].metadata.sentence_count or 0, 0)

    def test_semicolon_csv_is_detected_without_flattening_columns(self) -> None:
        """Стандартный CSV Sniffer распознаёт распространённый русский delimiter ';'."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir) / "raw"
            input_dir.mkdir()
            csv_path = input_dir / "knowledge.csv"
            csv_path.write_text(
                "Раздел;Название;Описание\n"
                "Поддержка;Сброс пароля;Инженер проверяет личность.\n",
                encoding="utf-8",
            )
            source = LlamaIndexLoadingStage(LoaderConfig())._to_source_file(
                csv_path,
                input_dir=input_dir,
            )

            rows = UnstructuredParsingStage(
                ParserConfig(), default_section="Полный документ"
            )._parse_csv_source(source)

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0].metadata["csv_columns"],
            ["Раздел", "Название", "Описание"],
        )
        self.assertIn("Описание: Инженер проверяет личность.", rows[0].text)


if __name__ == "__main__":
    unittest.main()
