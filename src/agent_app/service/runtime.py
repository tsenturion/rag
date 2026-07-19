"""Жизненный цикл runtime и общих ресурсов для HTTP-сервиса поддержки."""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from time import perf_counter
from typing import Any

from agent_app.config import AgentAppConfig
from agent_app.currency import CBRCurrencyConverter
from agent_app.graph import AgentRunner
from agent_app.guardrails import GuardrailPipeline, HumanReviewStore, SecurityAuditStore
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
from agent_app.observability import traced
from agent_app.rag.runtime import OnlineRagRuntime
from agent_app.service.schemas import DeleteSessionResponse, SessionResponse
from agent_app.support.incidents import IncidentStore
from agent_app.support.security import redact_secrets
from agent_app.tools.mcp_external import ExternalMCPToolManager
from agent_app.tools.code_runner import code_runner_status

LOGGER = logging.getLogger(__name__)


class SupportApplicationRuntime:
    """Координирует работу подсистем поддержки, гарантируя согласованное взаимодействие агентов, хранилищ и инструментов."""

    def __init__(
        self,
        config: AgentAppConfig,
        *,
        llm: Any | None = None,
        rag_runtime: OnlineRagRuntime | None = None,
    ):
        """Гарантирует готовность экземпляра к обслуживанию запросов, включая владение всеми необходимыми ресурсами и инициализацию зависимостей."""
        self.config = config
        self.currency_converter = CBRCurrencyConverter(config.currency_conversion)
        self._llm = llm
        self._llm_error: str | None = None
        self._multi_agent_error: str | None = None
        self._owns_rag = rag_runtime is None
        self.guardrails = GuardrailPipeline(config.guardrails)
        self.security_audit = SecurityAuditStore(config.guardrails.audit_sqlite_path)
        self.review_store = HumanReviewStore(config.guardrails.review_sqlite_path)
        self.rag_runtime = rag_runtime or OnlineRagRuntime(
            config.rag, context_guardrail=self.guardrails
        )
        if rag_runtime is not None:
            rag_runtime.context_guardrail = self.guardrails
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
        """Гарантирует выполнение запроса пользователя к агенту с учётом синхронизации и возвратом времени отклика."""
        started = perf_counter()
        with traced("agent.ask", mode="single", user_id=user_id, session_id=session_id):
            with self._execution_lock:
                runner = self._runner(user_id=user_id, session_id=session_id)
                response = runner.ask(message)
        return response, round((perf_counter() - started) * 1000, 3)

    def ask_multi(
        self, *, user_id: str, session_id: str, message: str
    ) -> tuple[MultiAgentRunResult, float]:
        """Гарантирует параллельную обработку запроса несколькими агентами с контролем синхронизации и возвратом времени выполнения."""
        started = perf_counter()
        with traced("agent.ask", mode="multi", user_id=user_id, session_id=session_id):
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
        """Гарантирует проведение сравнения ответов агентов по заданным критериям с возвратом отчёта и времени выполнения."""
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
        """Гарантирует воспроизводимое извлечение истории многократного запуска агентов по идентификатору."""
        return self._require_multi_agent().load_run(run_id)

    def session(self, *, user_id: str, session_id: str) -> SessionResponse:
        """Гарантирует получение полной истории пользовательской сессии, включая память, инциденты и мультиагентную активность."""
        memory = self.memory_store.list_memories(
            user_id=user_id,
            session_id=session_id,
            limit=200,
        )
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
        """Гарантирует полное удаление пользовательской сессии и связанных с ней ресурсов из всех подсистем, предотвращая утечки памяти и конфликт идентификаторов."""
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
        """Собирает состояние критичных подсистем, по которому HTTP-сервис сообщает о готовности принимать запросы."""
        if self._llm is None:
            self._initialize_llm()
        self._initialize_multi_agent()
        self._initialize_orchestration()
        rag = self.rag_runtime.start().model_dump(mode="json")
        llm_ready = self._llm is not None
        api_key_ready = self._service_api_key_ready()
        jwt_secret_ready = self._jwt_secret_ready()
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
            and jwt_secret_ready
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
                "jwt_enabled": self.config.security.jwt_enabled,
                "jwt_secret_configured": jwt_secret_ready,
                "guardrails_enabled": self.config.guardrails.enabled,
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
        """Дожидается активного запроса и затем закрывает общие LLM/SQLite-ресурсы."""
        # Отмена asyncio.to_thread не останавливает уже запущенный sync LLM-вызов.
        # Тот же lock используется в ask/ask_multi: shutdown не может очистить
        # registry или закрыть checkpointer, пока фоновый поток обращается к ним.
        with self._execution_lock:
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
            self.security_audit.close()
            self.review_store.close()
            if self._owns_rag:
                self.rag_runtime.close()

    def _runner(self, *, user_id: str, session_id: str) -> AgentRunner:
        """Гарантирует наличие и актуальность AgentRunner для указанной сессии, автоматически инициализируя LLM и управляя кэшированием."""
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
        """Поддерживает ограничение размера кэша сессий, гарантируя своевременное освобождение памяти при превышении лимита."""
        while len(self._runners) > self.config.service.session_cache_size:
            _, runner = self._runners.popitem(last=False)
            runner.close()

    def _initialize_llm(self) -> None:
        """Гарантирует попытку инициализации LLM с сохранением диагностической информации об ошибках для последующего анализа."""
        try:
            self._llm = build_llm(self.config.agent)
            self._llm_error = None
        except Exception as exc:
            self._llm = None
            self._llm_error = redact_secrets(str(exc))[:500]
            LOGGER.exception("Не удалось инициализировать LLM")

    def _initialize_multi_agent(self) -> None:
        """Гарантирует корректную инициализацию multi-agent среды только при доступности LLM и необходимости по конфигурации."""
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
                currency_converter=self.currency_converter,
            )
            self._multi_agent_error = None
        except Exception as exc:
            self.multi_agent_runtime = None
            self._multi_agent_error = redact_secrets(str(exc))[:500]
            LOGGER.exception("Не удалось инициализировать LLM-профили multi-agent")

    def _initialize_orchestration(self) -> None:
        """Гарантирует инициализацию сервиса оркестрации только при выполнении всех зависимостей и корректной конфигурации."""
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
        role: str,
    ) -> MultiAgentRunResult:
        """Запускает назначенную роль и сериализует только локальный LLM-маршрут."""
        runtime = self._require_multi_agent()
        profile_name = self.config.multi_agent.role_llm_profiles.get(role)
        profile = (
            self.config.multi_agent.llm_profiles.get(profile_name)
            if profile_name is not None
            else self.config.agent
        )
        if profile is not None and profile.provider == "local":
            with self._execution_lock:
                return runtime.ask_role(
                    user_id=user_id,
                    session_id=session_id,
                    message=message,
                    role=role,
                )
        return runtime.ask_role(
            user_id=user_id,
            session_id=session_id,
            message=message,
            role=role,
        )

    def _require_multi_agent(self) -> MultiAgentRuntime:
        """Гарантирует наличие и готовность multi-agent среды, выбрасывая исключение при невозможности её использования."""
        if self.multi_agent_runtime is None:
            self._initialize_multi_agent()
        if self.multi_agent_runtime is None:
            raise RuntimeError("Multi-agent runtime отключён или LLM недоступна")
        return self.multi_agent_runtime

    def _service_api_key_ready(self) -> bool:
        """Гарантирует, что HTTP-сервис поддержки не стартует без валидного API-ключа, если он требуется политикой безопасности."""
        if not self.config.security.require_api_key:
            return True
        return bool(os.getenv(self.config.security.api_key_env))

    def _jwt_secret_ready(self) -> bool:
        """Не допускает готовность JWT-профиля без HMAC-ключа, необходимого для проверки подписи пользовательских токенов."""
        if not self.config.security.jwt_enabled:
            return True
        return bool(os.getenv(self.config.security.jwt_secret_env))
