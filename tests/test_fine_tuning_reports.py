"""Регрессионные тесты для подсистемы fine_tuning_reports."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_tuning.comparison import FineTuningComparisonStage  # noqa: E402
from llm_tuning.config import (  # noqa: E402
    FineTuningPathConfig,
    FineTuningPipelineConfig,
)
from llm_tuning.exporting import FineTuningExportStage  # noqa: E402
from llm_tuning.models import (  # noqa: E402
    ComparisonResult,
    DatasetStats,
    DatasetValidationResult,
    DeviceReport,
    EvaluationReport,
    GeneratedAnswer,
)


class FineTuningReportsTest(unittest.TestCase):
    """Проверяет изоляцию и корректность экспорта отчётов тонкой настройки по уникальным идентификаторам запусков, обеспечивая независимость данных."""

    def test_reports_are_isolated_by_run_id(self) -> None:
        """Проверяет, что отчёты и манифесты для разных run_id сохраняются в отдельные директории, обеспечивая изоляцию данных между запусками."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            reports_dir = Path(temporary_dir) / "reports"
            config = FineTuningPipelineConfig(
                paths=FineTuningPathConfig(reports_dir=reports_dir)
            )
            exporter = FineTuningExportStage(config)
            first = self._report("run-one", [("example", False)])
            second = self._report("run-two", [("example", True)])

            first_report = exporter.write_evaluation(
                first,
                filename=config.paths.baseline_report_filename,
            )
            second_report = exporter.write_evaluation(
                second,
                filename=config.paths.baseline_report_filename,
            )
            first_comparison = exporter.write_comparison(
                self._comparison("run-one", first_report)
            )
            second_comparison = exporter.write_comparison(
                self._comparison("run-two", second_report)
            )
            first_export = exporter.write_manifest(
                run_id="run-one",
                dataset_validation=self._dataset_validation(),
                device=self._device(),
                baseline_report_path=first_report,
                comparison_report_path=first_comparison,
            )
            second_export = exporter.write_manifest(
                run_id="run-two",
                dataset_validation=self._dataset_validation(),
                device=self._device(),
                baseline_report_path=second_report,
                comparison_report_path=second_comparison,
            )

            self.assertEqual(first_report.parent, reports_dir / "run-one")
            self.assertEqual(second_report.parent, reports_dir / "run-two")
            self.assertNotEqual(first_report, second_report)
            self.assertTrue(first_report.exists())
            self.assertTrue(second_report.exists())
            self.assertEqual(first_export.manifest_path.parent, reports_dir / "run-one")
            self.assertEqual(
                second_export.manifest_path.parent, reports_dir / "run-two"
            )

    def test_comparison_rates_use_only_common_example_ids(self) -> None:
        """Проверяет, что при сравнении учитываются только примеры с общими example_id, корректно вычисляя показатели прохождения и улучшения."""
        baseline = self._report(
            "baseline-run",
            [("common-failed", False), ("baseline-only", True)],
        )
        tuned = self._report(
            "tuned-run",
            [("common-failed", True), ("tuned-only", False)],
        )

        result = FineTuningComparisonStage().compare(
            run_id="comparison-run",
            baseline=baseline,
            tuned=tuned,
            baseline_report_path=Path("baseline.json"),
            tuned_report_path=Path("tuned.json"),
        )

        self.assertEqual(result.examples_count, 1)
        self.assertEqual(result.baseline_pass_rate, 0.0)
        self.assertEqual(result.tuned_pass_rate, 1.0)
        self.assertEqual(result.pass_rate_delta, 1.0)
        self.assertEqual(result.improved_examples, ["common-failed"])

    def test_comparison_rejects_reports_without_common_ids(self) -> None:
        """Проверяет, что сравнение отчётов без общих example_id вызывает ошибку, предотвращая некорректный анализ."""
        with self.assertRaisesRegex(ValueError, "нет общих example_id"):
            FineTuningComparisonStage().compare(
                run_id="comparison-run",
                baseline=self._report("baseline", [("one", True)]),
                tuned=self._report("tuned", [("two", True)]),
                baseline_report_path=Path("baseline.json"),
                tuned_report_path=Path("tuned.json"),
            )

    @staticmethod
    def _report(run_id: str, answers: list[tuple[str, bool]]) -> EvaluationReport:
        """Формирует отчёт об оценке качества модели на основе заданных ответов, обеспечивая проверку метрик и статистики."""
        generated = [
            GeneratedAnswer(
                example_id=example_id,
                prompt="Запрос",
                expected_answer="Ответ",
                generated_answer="Ответ",
                passed=passed,
            )
            for example_id, passed in answers
        ]
        passed_count = sum(1 for answer in generated if answer.passed)
        return EvaluationReport(
            run_id=run_id,
            model_id="test-model",
            examples_count=len(generated),
            passed_count=passed_count,
            failed_count=len(generated) - passed_count,
            pass_rate=passed_count / len(generated),
            answers=generated,
        )

    @staticmethod
    def _comparison(run_id: str, report_path: Path) -> ComparisonResult:
        """Создаёт результат сравнения отчётов с фиксированными метриками для проверки логики анализа улучшений качества модели."""
        return ComparisonResult(
            run_id=run_id,
            baseline_report_path=report_path,
            tuned_report_path=report_path,
            examples_count=1,
            baseline_pass_rate=0.0,
            tuned_pass_rate=1.0,
            pass_rate_delta=1.0,
            conclusion="Качество улучшилось.",
        )

    @staticmethod
    def _dataset_validation() -> DatasetValidationResult:
        """Проверяет, что результат валидации датасета соответствует ожидаемому контракту fine_tuning_reports и пригоден для регрессионного тестирования."""
        train = DatasetStats(
            path=Path("train.jsonl"),
            examples_count=1,
            max_prompt_chars=10,
            max_answer_chars=10,
            avg_prompt_chars=10.0,
            avg_answer_chars=10.0,
            ids_are_unique=True,
        )
        evaluation = train.model_copy(update={"path": Path("eval.jsonl")})
        return DatasetValidationResult(
            train=train,
            eval=evaluation,
            train_eval_id_overlap_count=0,
            max_seq_length=128,
            estimated_train_tokens_max=10,
            estimated_eval_tokens_max=10,
        )

    @staticmethod
    def _device() -> DeviceReport:
        """Проверяет, что отчёт об устройстве соответствует ожидаемым характеристикам окружения для fine_tuning_reports."""
        return DeviceReport(
            requested_device="cpu",
            selected_device="cpu",
            torch_version="test",
            xpu_available=False,
            cuda_available=False,
            selected_dtype="fp32",
        )


if __name__ == "__main__":
    unittest.main()
