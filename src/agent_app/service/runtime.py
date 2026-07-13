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
from agent_app.rag.runtime import OnlineRagRuntime
from agent_app.service.schemas import DeleteSessionResponse, SessionResponse
from agent_app.support.incidents import IncidentStore
from agent_app.support.security import redact_secrets

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
        self._owns_rag = rag_runtime is None
        self.rag_runtime = rag_runtime or OnlineRagRuntime(config.rag)
        self.incident_store = IncidentStore(config.tools.incident_sqlite_path)
        self.memory_store = SQLiteMemoryStore(config.memory.sqlite_path)
        self._runners: OrderedDict[tuple[str, str], AgentRunner] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._execution_lock = threading.RLock()
        if self._llm is None:
            self._initialize_llm()

    def ask(
        self, *, user_id: str, session_id: str, message: str
    ) -> tuple[AgentResponse, float]:
        started = perf_counter()
        with self._execution_lock:
            runner = self._runner(user_id=user_id, session_id=session_id)
            response = runner.ask(message)
        return response, round((perf_counter() - started) * 1000, 3)

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
        return DeleteSessionResponse(
            user_id=user_id,
            session_id=session_id,
            deleted_memory_count=deleted,
            runner_removed=runner is not None,
        )

    def readiness(self) -> dict[str, Any]:
        if self._llm is None:
            self._initialize_llm()
        rag = self.rag_runtime.start().model_dump(mode="json")
        llm_ready = self._llm is not None
        api_key_ready = self._service_api_key_ready()
        ready = llm_ready and bool(rag["ready"]) and api_key_ready
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
        }

    def close(self) -> None:
        with self._cache_lock:
            runners = list(self._runners.values())
            self._runners.clear()
        for runner in runners:
            runner.close()
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

    def _service_api_key_ready(self) -> bool:
        if not self.config.security.require_api_key:
            return True
        return bool(os.getenv(self.config.security.api_key_env))
