"""Регрессионные тесты для подсистемы artifact_transaction."""

from __future__ import annotations

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_prep.config import PathConfig, PipelineConfig  # noqa: E402
from rag_prep.stages.exporting import ExportStage  # noqa: E402
from rag_prep.utils import (  # noqa: E402
    artifact_set_transaction,
    recover_artifact_transactions,
)


class ArtifactTransactionTest(unittest.TestCase):
    """Проверяет корректность транзакционного поведения при экспорте артефактов, гарантируя сохранность предыдущего набора при ошибках записи."""

    def test_export_failure_keeps_previous_artifact_set(self) -> None:
        """Проверяет, что при ошибке записи в процессе экспорта сохраняется предыдущий набор артефактов без изменений и не создаются временные файлы."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            paths = self._targets(output_dir)
            self._write_old_set(paths)
            config = PipelineConfig(paths=PathConfig(output_dir=output_dir))
            stage = ExportStage(config)

            with (
                patch.object(
                    stage, "_write_jsonl", side_effect=OSError("write failed")
                ),
                self.assertRaises(OSError),
            ):
                stage.run([], run_id="new-run", counts={})

            self._assert_old_set(paths)
            self.assertEqual(list(output_dir.glob(".artifact-set-*")), [])

    def test_commit_failure_rolls_back_every_target(self) -> None:
        """Проверяет, что при сбое переименования в транзакции коммита все изменения откатываются, и предыдущий набор артефактов восстанавливается."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            targets = self._targets(output_dir)
            self._write_old_set(targets)
            real_replace = os.replace
            failed = False

            def failing_replace(source, destination):
                """Имитирует сбой на втором rename транзакционного commit."""
                nonlocal failed
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    not failed
                    and source_path.name.startswith("001-")
                    and destination_path.resolve() == targets[1].resolve()
                ):
                    failed = True
                    raise OSError("commit failed")
                return real_replace(source, destination)

            with (
                patch("rag_prep.utils.os.replace", side_effect=failing_replace),
                self.assertRaises(OSError),
            ):
                with artifact_set_transaction(targets) as staged:
                    for target, staged_path in staged.items():
                        staged_path.write_text(f"new:{target.name}", encoding="utf-8")

            self._assert_old_set(targets)
            self.assertEqual(list(output_dir.glob(".artifact-set-*")), [])

    def test_recovery_rolls_back_process_crash_during_commit(self) -> None:
        """Восстанавливает весь старый набор по durable prepared journal."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            targets = self._targets(output_dir)
            self._write_old_set(targets)
            staging = output_dir / ".artifact-set-crashed"
            staging.mkdir()
            entries = []
            for index, target in enumerate(targets):
                backup = staging / f".backup-{index:03d}-{target.name}"
                backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
                entries.append(
                    {
                        "target": str(target.resolve()),
                        "backup": str(backup.resolve()),
                        "existed": True,
                    }
                )
                target.write_text(f"new:{target.name}", encoding="utf-8")
            (staging / "journal.json").write_text(
                json.dumps({"version": 1, "state": "prepared", "entries": entries}),
                encoding="utf-8",
            )

            recover_artifact_transactions(output_dir)

            self._assert_old_set(targets)
            self.assertFalse(staging.exists())

    @staticmethod
    def _targets(output_dir: Path) -> list[Path]:
        """Проверяет, что список целевых файлов для транзакции воспроизводим и соответствует ожидаемой структуре выходных артефактов."""
        return [
            output_dir / "documents.json",
            output_dir / "documents.jsonl",
            output_dir / "manifest.json",
        ]

    @staticmethod
    def _write_old_set(paths: list[Path]) -> None:
        """Обеспечивает воспроизводимость тестов, записывая фиксированные данные в указанные пути для проверки корректности транзакций с артефактами."""
        for path in paths:
            path.write_text(f"old:{path.name}", encoding="utf-8")

    def _assert_old_set(self, paths: list[Path]) -> None:
        """Гарантирует целостность данных, проверяя, что содержимое файлов соответствует ожидаемому состоянию после операций с артефактами."""
        for path in paths:
            self.assertEqual(path.read_text(encoding="utf-8"), f"old:{path.name}")


if __name__ == "__main__":
    unittest.main()
