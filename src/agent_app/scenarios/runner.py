from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agent_app.config import AgentAppConfig
from agent_app.graph import AgentRunner
from agent_app.memory import SQLiteMemoryStore
from agent_app.models import AgentResponse, utc_now
from agent_app.scenarios.evaluator import ScenarioEvaluator
from agent_app.scenarios.models import (
    AgentScenario,
    ScenarioCheck,
    ScenarioResult,
    ScenarioRunReport,
    ScenarioStepResult,
    ScenarioSuite,
)

LOGGER = logging.getLogger(__name__)


class ScenarioRunner:
    def __init__(
        self,
        config: AgentAppConfig,
        suite: ScenarioSuite,
        *,
        config_path: str,
    ):
        self.config = config
        self.suite = suite
        self.config_path = config_path
        self.store = SQLiteMemoryStore(config.memory.sqlite_path)
        self.evaluator = ScenarioEvaluator()
        self._shared_llm: Any | None = None

    def run_all(self) -> ScenarioRunReport:
        return self._report(self.suite.scenarios)

    def run_one(self, scenario_id: str) -> ScenarioRunReport:
        scenarios = [
            scenario for scenario in self.suite.scenarios if scenario.id == scenario_id
        ]
        if not scenarios:
            known = ", ".join(scenario.id for scenario in self.suite.scenarios)
            raise ValueError(
                f"Сценарий не найден: {scenario_id}. Доступные сценарии: {known}"
            )
        return self._report(scenarios)

    def write_report(self, report: ScenarioRunReport, path: Path | None = None) -> Path:
        report_path = path or self.suite.report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Сохранён отчёт сценариев: %s", report_path)
        return report_path

    def _report(self, scenarios: list[AgentScenario]) -> ScenarioRunReport:
        results = [self._run_scenario(scenario) for scenario in scenarios]
        return ScenarioRunReport(
            config_path=self.config_path,
            user_id=self.suite.default_user_id,
            passed=bool(results) and all(result.passed for result in results),
            results=results,
        )

    def _run_scenario(self, scenario: AgentScenario) -> ScenarioResult:
        started_at = utc_now()
        user_id = f"{self.suite.default_user_id}_{scenario.id}"
        session_id = f"{self.suite.session_prefix}_{scenario.id}"
        if scenario.reset_memory:
            self.store.clear_user(user_id=user_id)
        self._seed_memory(scenario, user_id=user_id, session_id=session_id)

        runner = AgentRunner(
            self.config,
            user_id=user_id,
            session_id=session_id,
            llm=self._shared_llm,
        )
        if self._shared_llm is None:
            self._shared_llm = runner.llm
        step_results: list[ScenarioStepResult] = []
        scenario_checks: list[ScenarioCheck] = []
        all_tool_calls: list[str] = []

        for step in scenario.steps:
            try:
                response = runner.ask(step.user_request)
            except Exception as exc:
                LOGGER.exception("Ошибка выполнения шага %s/%s", scenario.id, step.id)
                response = AgentResponse(
                    answer=f"Ошибка выполнения шага: {exc}",
                    user_id=user_id,
                    session_id=session_id,
                    tool_calls=[],
                )
                checks = [
                    ScenarioCheck(
                        name="step_exception",
                        passed=False,
                        details=str(exc),
                    )
                ]
                step_results.append(
                    ScenarioStepResult(
                        scenario_id=scenario.id,
                        step_id=step.id,
                        test_case_id=step.test_case_id,
                        title=step.title,
                        passed=False,
                        checks=checks,
                        response=response,
                    )
                )
                continue
            all_tool_calls.extend(response.tool_calls)
            memory_after_step = self.store.list_memories(user_id=user_id, limit=200)
            checks = self.evaluator.evaluate(
                criteria=step.criteria,
                response=response,
                memory_records=memory_after_step,
            )
            step_results.append(
                ScenarioStepResult(
                    scenario_id=scenario.id,
                    step_id=step.id,
                    test_case_id=step.test_case_id,
                    title=step.title,
                    passed=all(check.passed for check in checks),
                    checks=checks,
                    response=response,
                )
            )

        memory_after = self.store.list_memories(user_id=user_id, limit=200)
        if step_results:
            scenario_checks.extend(
                self.evaluator.evaluate(
                    criteria=scenario.pass_criteria,
                    response=step_results[-1].response,
                    memory_records=memory_after,
                )
            )
        for tool in scenario.required_tools:
            scenario_checks.append(
                ScenarioCheck(
                    name=f"scenario_required_tool:{tool}",
                    passed=tool in all_tool_calls,
                    details=f"Все вызванные tools: {all_tool_calls}",
                )
            )
        for tool in scenario.forbidden_tools:
            scenario_checks.append(
                ScenarioCheck(
                    name=f"scenario_forbidden_tool:{tool}",
                    passed=tool not in all_tool_calls,
                    details=f"Все вызванные tools: {all_tool_calls}",
                )
            )

        passed = all(result.passed for result in step_results) and all(
            check.passed for check in scenario_checks
        )
        LOGGER.info(
            "Сценарий %s завершён: %s",
            scenario.id,
            "passed" if passed else "failed",
        )
        return ScenarioResult(
            id=scenario.id,
            test_case_id=scenario.test_case_id,
            title=scenario.title,
            type=scenario.type,
            goal=scenario.goal,
            user_request=scenario.user_request,
            expected_result=scenario.expected_result,
            llm_role=scenario.llm_role,
            tools_role=scenario.tools_role,
            memory_role=scenario.memory_role,
            action_chain=scenario.action_chain,
            decision_points=scenario.decision_points,
            transition_rules=scenario.transition_rules,
            passed=passed,
            started_at=started_at,
            finished_at=utc_now(),
            step_results=step_results,
            checks=scenario_checks,
            memory_after=[record.model_dump(mode="json") for record in memory_after],
        )

    def _seed_memory(
        self,
        scenario: AgentScenario,
        *,
        user_id: str,
        session_id: str,
    ) -> None:
        for item in scenario.initial_memory:
            self.store.save(
                user_id=user_id,
                session_id=session_id,
                memory_type=item.memory_type,
                key=item.key,
                value=item.value,
                tags=item.tags,
                importance=item.importance,
                source="system",
                metadata=item.metadata,
            )
