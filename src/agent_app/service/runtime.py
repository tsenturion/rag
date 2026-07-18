from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from time import perf_counter
from typing import Any

from agent_app.config import AgentAppConfig
from agent_app.graph import AgentRunner
from agent_app.llm import build_llm
from agent_app.memory import SQLiteMemoryStore
from agent_app.models import AgentResponse
from agent_app.multi_agent.models import (
    ComparisonScenario,
    ComparisonScenarioSuite,
    MultiAgentComparisonReport,
    MultiAgentRunResult,
)
from agent_app.multi_agent.runtime import MultiAgentRuntime
from agent_app.orchestration.service import OrchestrationService, callback_executor
from agent_app.rag.runtime import OnlineRagRuntime
from agent_app.service.schemas import DeleteSessionResponse, SessionResponse
from agent_app.support.incidents import IncidentStore
from agent_app.support.security import redact_secrets
from agent_app.tools.mcp_external import ExternalMCPToolManager
from agent_app.tools.code_runner import code_runner_status

LOGGER = logging.getLogger(__name__)


class SupportApplicationRuntime:
    def __init__(
        self,
        config: AgentAppConfig,
        *,
        llm: Any | None = None,
        rag_runtime: OnlineRagRuntime | None = None,
    ):
        self.config = config
        self._llm = llm
        self._llm_error: str | None = None
        self._multi_agent_error: str | None = None
        self._owns_rag = rag_runtime is None
        self.rag_runtime = rag_runtime or OnlineRagRuntime(config.rag)
        self.incident_store = IncidentStore(config.tools.incident_sqlite_path)
        self.memory_store = SQLiteMemoryStore(config.memory.sqlite_path)
        self.external_mcp_manager = ExternalMCPToolManager(config.tools.mcp_servers)
        self.external_tools = self.external_mcp_manager.start()
        self._runners: OrderedDict[tuple[str, str], AgentRunner] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._execution_lock = threading.RLock()
        self.multi_agent_runtime: MultiAgentRuntime | None = None
        self.orchestration_service: OrchestrationService | None = None
        if self._llm is None:
            self._initialize_llm()
        self._initialize_multi_agent()
        self._initialize_orchestration()

    def ask(
        self, *, user_id: str, session_id: str, message: str
    ) -> tuple[AgentResponse, float]:
        started = perf_counter()
        with self._execution_lock:
            runner = self._runner(user_id=user_id, session_id=session_id)
            response = runner.ask(message)
        return response, round((perf_counter() - started) * 1000, 3)

    def ask_multi(
        self, *, user_id: str, session_id: str, message: str
    ) -> tuple[MultiAgentRunResult, float]:
        started = perf_counter()
        with self._execution_lock:
            runtime = self._require_multi_agent()
            result = runtime.ask(
                user_id=user_id,
                session_id=session_id,
                message=message,
            )
        return result, round((perf_counter() - started) * 1000, 3)

    def compare_multi(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        expected_terms: list[str],
        expected_tools: list[str],
        require_citations: bool,
    ) -> tuple[MultiAgentComparisonReport, float]:
        started = perf_counter()
        suite = ComparisonScenarioSuite(
            scenarios=[
                ComparisonScenario(
                    id="api-comparison",
                    title="Сравнение по API-запросу",
                    request=message,
                    expected_terms=expected_terms,
                    expected_tools=expected_tools,
                    require_citations=require_citations,
                )
            ]
        )
        with self._execution_lock:
            report = self._require_multi_agent().compare(
                suite,
                user_id=user_id,
                session_prefix=session_id,
            )
        return report, round((perf_counter() - started) * 1000, 3)

    def load_multi_run(self, run_id: str) -> dict[str, object] | None:
        return self._require_multi_agent().load_run(run_id)

    def session(self, *, user_id: str, session_id: str) -> SessionResponse:
        memory = self.memory_store.list_memories(user_id=user_id, limit=200)
        incidents = self.incident_store.list(
            user_id=user_id,
            session_id=session_id,
            limit=100,
        )
        return SessionResponse(
            user_id=user_id,
            session_id=session_id,
            memory=[record.model_dump(mode="json") for record in memory],
            incidents=[record.model_dump(mode="json") for record in incidents],
            multi_agent_history=(
                self.multi_agent_runtime.session_history(
                    user_id=user_id,
                    session_id=session_id,
                )
                if self.multi_agent_runtime is not None
                else []
            ),
        )

    def delete_session(self, *, user_id: str, session_id: str) -> DeleteSessionResponse:
        key = (user_id, session_id)
        with self._cache_lock:
            runner = self._runners.pop(key, None)
        deleted = self.memory_store.clear_session(
            user_id=user_id, session_id=session_id
        )
        if runner is not None:
            runner.short_term.clear()
            runner.close()
        checkpoint_deleted = (
            self.multi_agent_runtime.clear_session(
                user_id=user_id,
                session_id=session_id,
            )
            if self.multi_agent_runtime is not None
            else False
        )
        return DeleteSessionResponse(
            user_id=user_id,
            session_id=session_id,
            deleted_memory_count=deleted,
            runner_removed=runner is not None,
            multi_agent_checkpoint_deleted=checkpoint_deleted,
        )

    def readiness(self) -> dict[str, Any]:
        if self._llm is None:
            self._initialize_llm()
        self._initialize_multi_agent()
        self._initialize_orchestration()
        rag = self.rag_runtime.start().model_dump(mode="json")
        llm_ready = self._llm is not None
        api_key_ready = self._service_api_key_ready()
        multi_agent_ready = (
            not self.config.multi_agent.enabled or self.multi_agent_runtime is not None
        )
        runner = code_runner_status(self.config.code_runner)
        orchestration = (
            self.orchestration_service.status().model_dump(mode="json")
            if self.orchestration_service is not None
            else {
                "backend": self.config.orchestration.backend,
                "ready": not self.config.orchestration.enabled,
                "status_counts": {},
                "workers": {},
                "error": (
                    None
                    if not self.config.orchestration.enabled
                    else "Orchestration service не инициализирован"
                ),
            }
        )
        ready = (
            llm_ready
            and bool(rag["ready"])
            and api_key_ready
            and multi_agent_ready
            and bool(runner["ready"])
            and bool(orchestration["ready"])
        )
        return {
            "ready": ready,
            "llm": {
                "ready": llm_ready,
                "provider": self.config.agent.provider,
                "model": self.config.agent.model,
                "error": self._llm_error,
            },
            "rag": rag,
            "security": {
                "api_key_required": self.config.security.require_api_key,
                "api_key_configured": api_key_ready,
            },
            "sessions_cached": len(self._runners),
            "multi_agent": {
                "enabled": self.config.multi_agent.enabled,
                "ready": multi_agent_ready,
                "error": self._multi_agent_error,
                "execution_mode": self.config.multi_agent.execution_mode,
                "a2a_enabled": self.config.multi_agent.protocols.a2a_enabled,
                "mcp_enabled": self.config.multi_agent.protocols.mcp_enabled,
                "llm_routes": (
                    [
                        route.model_dump(mode="json")
                        for route in self.multi_agent_runtime.llm_registry.route_info()
                    ]
                    if self.multi_agent_runtime is not None
                    else []
                ),
            },
            "external_mcp": self.external_mcp_manager.status(),
            "code_runner": runner,
            "orchestration": orchestration,
        }

    def close(self) -> None:
        with self._cache_lock:
            runners = list(self._runners.values())
            self._runners.clear()
        for runner in runners:
            runner.close()
        if self.multi_agent_runtime is not None:
            self.multi_agent_runtime.close()
            self.multi_agent_runtime = None
        if self.orchestration_service is not None:
            self.orchestration_service.close()
            self.orchestration_service = None
        self.external_mcp_manager.close()
        if self._owns_rag:
            self.rag_runtime.close()

    def _runner(self, *, user_id: str, session_id: str) -> AgentRunner:
        if self._llm is None:
            self._initialize_llm()
        if self._llm is None:
            raise RuntimeError(f"LLM недоступна: {self._llm_error}")

        key = (user_id, session_id)
        with self._cache_lock:
            existing = self._runners.get(key)
            if existing is not None:
                self._runners.move_to_end(key)
                return existing
            runner = AgentRunner(
                self.config,
                user_id=user_id,
                session_id=session_id,
                llm=self._llm,
                rag_runtime=self.rag_runtime,
                incident_store=self.incident_store,
                external_tools=self.external_tools,
            )
            self._runners[key] = runner
            self._evict_runners()
            return runner

    def _evict_runners(self) -> None:
        while len(self._runners) > self.config.service.session_cache_size:
            _, runner = self._runners.popitem(last=False)
            runner.close()

    def _initialize_llm(self) -> None:
        try:
            self._llm = build_llm(self.config.agent)
            self._llm_error = None
        except Exception as exc:
            self._llm = None
            self._llm_error = redact_secrets(str(exc))[:500]
            LOGGER.exception("Не удалось инициализировать LLM")

    def _initialize_multi_agent(self) -> None:
        if (
            not self.config.multi_agent.enabled
            or self._llm is None
            or self.multi_agent_runtime is not None
        ):
            return
        try:
            self.multi_agent_runtime = MultiAgentRuntime(
                self.config,
                llm=self._llm,
                rag_runtime=self.rag_runtime,
                incident_store=self.incident_store,
                external_tools=self.external_tools,
            )
            self._multi_agent_error = None
        except Exception as exc:
            self.multi_agent_runtime = None
            self._multi_agent_error = redact_secrets(str(exc))[:500]
            LOGGER.exception("Не удалось инициализировать LLM-профили multi-agent")

    def _initialize_orchestration(self) -> None:
        if (
            not self.config.orchestration.enabled
            or self.orchestration_service is not None
        ):
            return
        if self.config.orchestration.backend == "inline":
            self._initialize_multi_agent()
            if self.multi_agent_runtime is None:
                return
        try:
            self.orchestration_service = OrchestrationService(
                self.config,
                executor_factory=callback_executor(self._orchestration_ask),
            )
        except Exception:
            self.orchestration_service = None
            LOGGER.exception("Не удалось инициализировать orchestration service")

    def _orchestration_ask(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> MultiAgentRunResult:
        with self._execution_lock:
            return self._require_multi_agent().ask(
                user_id=user_id,
                session_id=session_id,
                message=message,
            )

    def _require_multi_agent(self) -> MultiAgentRuntime:
        if self.multi_agent_runtime is None:
            self._initialize_multi_agent()
        if self.multi_agent_runtime is None:
            raise RuntimeError("Multi-agent runtime отключён или LLM недоступна")
        return self.multi_agent_runtime

    def _service_api_key_ready(self) -> bool:
        if not self.config.security.require_api_key:
            return True
        return bool(os.getenv(self.config.security.api_key_env))
