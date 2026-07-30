"""Экспорт воспроизводимых артефактов для оценки качества агентной системы."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_app.evaluation.models import EvaluationReport
from rag_prep.utils import artifact_set_transaction


class EvaluationExporter:
    """Инкапсулирует экспорт результатов оценки качества агента с гарантией целостности и атомарности артефактов."""

    def __init__(self, output_dir: Path):
        """Готовит экземпляр к атомарному экспорту результатов в указанный каталог."""
        self.output_dir = output_dir

    def export(self, report: EvaluationReport) -> EvaluationReport:
        """Гарантирует атомарную запись результатов оценки и контроль целостности артефактов для последующего аудита и автоматизации CI/CD."""
        run_dir = (self.output_dir / report.run_id).resolve()
        report = report.model_copy(update={"run_dir": str(run_dir)})
        report_path = run_dir / "report.json"
        results_path = run_dir / "results.jsonl"
        manifest_path = run_dir / "manifest.json"
        with artifact_set_transaction(
            [report_path, results_path, manifest_path]
        ) as staged:
            report_text = (
                json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
                + "\n"
            )
            staged[report_path].write_text(report_text, encoding="utf-8")
            lines = "".join(
                json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n"
                for item in report.results
            )
            staged[results_path].write_text(lines, encoding="utf-8")
            manifest = {
                "run_id": report.run_id,
                "suite_name": report.suite_name,
                "quality_gate_passed": report.quality_gate.passed,
                "artifacts": {
                    "report.json": _sha256(report_text),
                    "results.jsonl": _sha256(lines),
                },
            }
            staged[manifest_path].write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return report


def _sha256(text: str) -> str:
    """Гарантирует однозначную идентификацию содержимого артефакта по его SHA-256 хешу."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
