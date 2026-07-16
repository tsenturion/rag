from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import yaml
from langchain_core.tools import BaseTool

from agent_app.config import AgentAppConfig
from agent_app.graph import AgentRunner
from agent_app.multi_agent.evaluation import assess_answer, assess_multi_response
from agent_app.multi_agent.exporting import MultiAgentExporter
from agent_app.multi_agent.graph import MultiAgentRunner
from agent_app.multi_agent.models import (
    AgentModeResult,
    ComparisonCaseResult,
    ComparisonScenarioSuite,
    MultiAgentComparisonReport,
    MultiAgentRunResult,
)
from agent_app.multi_agent.llm_routing import MultiAgentLLMRegistry
from agent_app.multi_agent.tracking import MultiAgentTracker
from agent_app.multi_agent.usage import estimate_mode_usage
from agent_app.rag.runtime import OnlineRagRuntime
from agent_app.support.incidents import IncidentStore
from agent_app.tools.mcp_external import ExternalMCPToolManager


class MultiAgentRuntime:
    def __init__(
        self,
        config: AgentAppConfig,
        *,
        llm: Any | None = None,
        rag_runtime: OnlineRagRuntime | None = None,
        incident_store: IncidentStore | None = None,
        external_tools: list[BaseTool] | None = None,
        llm_registry: MultiAgentLLMRegistry | None = None,
        role_llms: dict[str, Any] | None = None,
    ):
        if not config.multi_agent.enabled:
            raise ValueError("Multi-agent runtime отключён в конфигурации")
        self.config = config
        self._owns_llm_registry = llm_registry is None
        self.llm_registry = llm_registry or MultiAgentLLMRegistry(
            config,
            default_llm=llm,
            role_llms=role_llms,
        )
        self.llm = self.llm_registry.default_llm
        self._owns_rag = rag_runtime is None
        self.rag_runtime = rag_runtime or OnlineRagRuntime(config.rag)
        self.incident_store = incident_store or IncidentStore(
            config.tools.incident_sqlite_path
        )
        self._external_mcp_manager: ExternalMCPToolManager | None = None
        if external_tools is None and config.tools.mcp_servers:
            self._external_mcp_manager = ExternalMCPToolManager(
                config.tools.mcp_servers
            )
            external_tools = self._external_mcp_manager.start()
        self.external_tools = external_tools or []
        self.exporter = MultiAgentExporter(config.multi_agent.output_dir)
        self.tracker = MultiAgentTracker(config.multi_agent)

    def ask(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
    ) -> MultiAgentRunResult:
        runner = self._runner(user_id=user_id, session_id=session_id)
        try:
            return runner.run(message)
        finally:
            runner.close()

    def compare(
        self,
        suite: ComparisonScenarioSuite,
        *,
        user_id: str,
        session_prefix: str = "comparison",
    ) -> MultiAgentComparisonReport:
        cases: list[ComparisonCaseResult] = []
        for scenario in suite.scenarios:
            session_id = f"{session_prefix}-{scenario.id}"
            single_runner = AgentRunner(
                self.config,
                user_id=user_id,
                session_id=session_id + "-single",
                llm=self.llm,
                rag_runtime=self.rag_runtime,
                incident_store=self.incident_store,
                external_tools=self.external_tools,
            )
            started = perf_counter()
            try:
                single_response = single_runner.ask(scenario.request)
            finally:
                single_runner.close()
            single_duration = (perf_counter() - started) * 1000
            single_quality = assess_answer(
                single_response.answer,
                citations_count=len(single_response.citations),
                expected_terms=scenario.expected_terms,
                require_citations=scenario.require_citations,
            )
            single_usage = estimate_mode_usage(
                request=scenario.request,
                answer=single_response.answer,
                model=self.config.agent.model,
                llm_calls=max(1, 1 + len(single_response.tool_calls)),
                tool_calls=len(single_response.tool_calls),
                duration_ms=single_duration,
                input_cost_per_million=(
                    self.config.multi_agent.cost.input_cost_per_million
                ),
                output_cost_per_million=(
                    self.config.multi_agent.cost.output_cost_per_million
                ),
            )
            multi_result = self.ask(
                user_id=user_id,
                session_id=session_id + "-multi",
                message=scenario.request,
            )
            multi_response = multi_result.response
            multi_quality = assess_multi_response(multi_response, scenario)
            multi_usage = multi_response.usage
            if scenario.max_agents is not None:
                within_budget = (
                    len(multi_response.selected_agents) <= scenario.max_agents
                )
                multi_quality.checks["max_agents"] = within_budget
                if not within_budget:
                    multi_quality.notes.append("max_agents")
                multi_quality.score = round(
                    sum(multi_quality.checks.values()) / len(multi_quality.checks),
                    4,
                )
            multi_response = multi_response.model_copy(
                update={"quality": multi_quality}
            )
            cases.append(
                ComparisonCaseResult(
                    id=scenario.id,
                    title=scenario.title,
                    request=scenario.request,
                    single=AgentModeResult(
                        mode="single",
                        answer=single_response.answer,
                        citations_count=len(single_response.citations),
                        tool_calls=single_response.tool_calls,
                        quality=single_quality,
                        usage=single_usage,
                    ),
                    multi=AgentModeResult(
                        mode="multi",
                        answer=multi_response.answer,
                        citations_count=len(multi_response.citations),
                        tool_calls=[
                            tool
                            for result in multi_response.task_results
                            for tool in result.tool_calls
                        ],
                        selected_agents=multi_response.selected_agents,
                        quality=multi_quality,
                        usage=multi_usage,
                        run_id=multi_response.run_id,
                    ),
                    quality_delta=round(
                        multi_quality.score - single_quality.score,
                        4,
                    ),
                    duration_delta_ms=round(
                        multi_usage.duration_ms - single_usage.duration_ms,
                        3,
                    ),
                    token_delta=(multi_usage.total_tokens - single_usage.total_tokens),
                    cost_delta=round(
                        multi_usage.estimated_cost - single_usage.estimated_cost,
                        8,
                    ),
                )
            )

        single_quality_average = sum(case.single.quality.score for case in cases) / len(
            cases
        )
        multi_quality_average = sum(case.multi.quality.score for case in cases) / len(
            cases
        )
        single_cost = sum(case.single.usage.estimated_cost for case in cases)
        multi_cost = sum(case.multi.usage.estimated_cost for case in cases)
        report = MultiAgentComparisonReport(
            run_id=str(uuid4()),
            provider=self.llm_registry.provider_summary,
            model=self.llm_registry.model_summary,
            llm_routes=self.llm_registry.route_info(),
            cases=cases,
            average_single_quality=round(single_quality_average, 4),
            average_multi_quality=round(multi_quality_average, 4),
            quality_delta=round(multi_quality_average - single_quality_average, 4),
            total_single_cost=round(single_cost, 8),
            total_multi_cost=round(multi_cost, 8),
            total_cost_delta=round(multi_cost - single_cost, 8),
        )
        run_dir = self.exporter.export_comparison(report)
        report = report.model_copy(update={"run_dir": str(run_dir)})
        self.tracker.log_comparison(report)
        return report

    def load_run(self, run_id: str) -> dict[str, object] | None:
        return self.exporter.load_result(run_id)

    def close(self) -> None:
        if self._external_mcp_manager is not None:
            self._external_mcp_manager.close()
            self._external_mcp_manager = None
        if self._owns_llm_registry:
            self.llm_registry.close()
        if self._owns_rag:
            self.rag_runtime.close()

    def _runner(self, *, user_id: str, session_id: str) -> MultiAgentRunner:
        return MultiAgentRunner(
            self.config,
            user_id=user_id,
            session_id=session_id,
            llm=self.llm,
            rag_runtime=self.rag_runtime,
            incident_store=self.incident_store,
            external_tools=self.external_tools,
            llm_registry=self.llm_registry,
        )


def load_comparison_suite(path: str | Path) -> ComparisonScenarioSuite:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ComparisonScenarioSuite.model_validate(payload)


def comparison_report_json(report: MultiAgentComparisonReport) -> str:
    return json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
