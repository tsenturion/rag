from __future__ import annotations

from pathlib import Path

from llm_tuning.models import ComparisonResult, EvaluationReport


class FineTuningComparisonStage:
    def compare(
        self,
        *,
        run_id: str,
        baseline: EvaluationReport,
        tuned: EvaluationReport,
        baseline_report_path: Path,
        tuned_report_path: Path,
    ) -> ComparisonResult:
        baseline_by_id = {answer.example_id: answer for answer in baseline.answers}
        tuned_by_id = {answer.example_id: answer for answer in tuned.answers}
        common_ids = sorted(set(baseline_by_id) & set(tuned_by_id))
        if not common_ids:
            raise ValueError("В baseline и tuned отчётах нет общих example_id")

        improved = []
        regressed = []
        unchanged_failed = []
        for example_id in common_ids:
            before = baseline_by_id[example_id].passed
            after = tuned_by_id[example_id].passed
            if not before and after:
                improved.append(example_id)
            elif before and not after:
                regressed.append(example_id)
            elif not before and not after:
                unchanged_failed.append(example_id)

        baseline_passed = sum(
            1 for example_id in common_ids if baseline_by_id[example_id].passed
        )
        tuned_passed = sum(
            1 for example_id in common_ids if tuned_by_id[example_id].passed
        )
        baseline_pass_rate = round(baseline_passed / len(common_ids), 6)
        tuned_pass_rate = round(tuned_passed / len(common_ids), 6)
        delta = round(tuned_pass_rate - baseline_pass_rate, 6)
        return ComparisonResult(
            run_id=run_id,
            baseline_report_path=baseline_report_path,
            tuned_report_path=tuned_report_path,
            examples_count=len(common_ids),
            baseline_pass_rate=baseline_pass_rate,
            tuned_pass_rate=tuned_pass_rate,
            pass_rate_delta=delta,
            improved_examples=improved,
            regressed_examples=regressed,
            unchanged_failed_examples=unchanged_failed,
            conclusion=self._conclusion(delta, regressed, unchanged_failed),
        )

    @staticmethod
    def _conclusion(
        delta: float,
        regressed: list[str],
        unchanged_failed: list[str],
    ) -> str:
        if delta > 0 and not regressed:
            return "Качество по проверочным критериям улучшилось без регрессий."
        if delta > 0 and regressed:
            return "Качество улучшилось, но есть регрессии на отдельных примерах."
        if delta == 0 and unchanged_failed:
            return "Метрика не изменилась; часть ошибок сохранилась после обучения."
        if delta == 0:
            return "Метрика не изменилась."
        return "Качество по проверочным критериям ухудшилось, нужно пересмотреть данные или параметры."
