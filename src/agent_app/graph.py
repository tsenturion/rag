from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent_app.config import AgentAppConfig
from agent_app.llm import build_llm
from agent_app.memory import SQLiteMemoryStore, ShortTermMemory, SummaryMemory
from agent_app.models import (
    AgentResponse,
    AgentRetrievalInfo,
    AgentState,
    AgentToolResult,
    AgentTrace,
    AgentTraceState,
    MemoryRecord,
)
from agent_app.prompts import system_prompt
from agent_app.rag.models import RagCitation, RagRetrievalResult
from agent_app.rag.runtime import OnlineRagRuntime
from agent_app.support.incidents import IncidentStore
from agent_app.tools import build_tools
from agent_app.tools.mcp_external import ExternalMCPToolManager

LOGGER = logging.getLogger(__name__)
SECRET_STORAGE_RE = re.compile(
    r"(?is)\b(запомни|сохрани|remember|store)\b.{0,80}"
    r"\b(api[_ -]?key|token|password|secret|пароль|ключ)\b\s*[:=]",
)


class AgentRunner:
    """LangGraph-агент с tools и полноценной SQLite-памятью."""

    def __init__(
        self,
        config: AgentAppConfig,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        llm: Any | None = None,
        rag_runtime: OnlineRagRuntime | None = None,
        incident_store: IncidentStore | None = None,
        external_tools: list[BaseTool] | None = None,
    ):
        self.config = config
        self.user_id = user_id or config.memory.default_user_id
        self.session_id = session_id or config.memory.default_session_id
        self.store = SQLiteMemoryStore(config.memory.sqlite_path)
        self.short_term = ShortTermMemory(config.agent.max_history_messages)
        self.summary = SummaryMemory(
            self.store,
            user_id=self.user_id,
            session_id=self.session_id,
            max_chars=config.agent.max_summary_chars,
        )
        self.llm = llm if llm is not None else build_llm(config.agent)
        self._owns_rag_runtime = rag_runtime is None and config.rag.enabled
        self.rag_runtime = (
            rag_runtime
            if rag_runtime is not None
            else OnlineRagRuntime(config.rag)
            if config.rag.enabled
            else None
        )
        support_tool_names = {
            "search_knowledge_base",
            "find_runbook",
            "analyze_log_fragment",
            "create_incident",
            "get_incident",
            "update_incident_status",
            "list_incidents",
            "build_diagnostic_checklist",
        }
        needs_support_tools = config.rag.enabled or bool(
            set(config.tools.enabled).intersection(support_tool_names)
        )
        self.incident_store = (
            incident_store
            if incident_store is not None
            else IncidentStore(config.tools.incident_sqlite_path)
            if needs_support_tools
            else None
        )
        self._external_mcp_manager: ExternalMCPToolManager | None = None
        if external_tools is None and config.tools.mcp_servers:
            self._external_mcp_manager = ExternalMCPToolManager(
                config.tools.mcp_servers
            )
            external_tools = self._external_mcp_manager.start()
        self.tools = build_tools(
            config,
            self.store,
            user_id=self.user_id,
            session_id=self.session_id,
            rag_runtime=self.rag_runtime,
            incident_store=self.incident_store,
            external_tools=external_tools,
        )
        self.graph = self._build_graph()

    def ask(self, message: str) -> AgentResponse:
        memory_before = self.store.list_memories(user_id=self.user_id, limit=200)
        if self._is_secret_storage_request(message):
            answer = (
                "Нельзя хранить API-ключи, пароли, токены и другие секреты в памяти. "
                "Я не сохраняю это значение; используйте .env, secret manager или "
                "переменные окружения."
            )
            trace = self._trace(
                message=message,
                input_messages=[HumanMessage(content=message)],
                result_messages=[AIMessage(content=answer)],
                answer=answer,
                tool_calls=[],
                tool_results=[],
                memory_before=memory_before,
                memory_after=memory_before,
            )
            return AgentResponse(
                answer=answer,
                user_id=self.user_id,
                session_id=self.session_id,
                tool_calls=[],
                trace=trace,
            )
        memories = self.store.list_memories(
            user_id=self.user_id,
            limit=self.config.memory.search_limit,
        )
        system = SystemMessage(
            content=system_prompt(
                summary=self.summary.get(),
                memories=memories,
                rag_enabled=self.config.rag.enabled,
            )
        )
        input_messages: list[BaseMessage] = [
            system,
            *self.short_term.snapshot(),
            HumanMessage(content=message),
        ]
        try:
            result = self.graph.invoke(
                {
                    "messages": input_messages,
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                },
                config={"recursion_limit": self.config.agent.recursion_limit},
            )
        except Exception as exc:
            if self._is_recoverable_agent_error(exc):
                return self._error_response(message, input_messages, memory_before, exc)
            raise
        messages: list[BaseMessage] = result["messages"]
        loop_guard_triggered = bool(result.get("loop_guard_triggered", False))
        missing_tools = self._missing_required_tools(message, messages)
        if missing_tools:
            retry_message = HumanMessage(
                content=(
                    "Предыдущий ответ не вызвал обязательные tools: "
                    f"{', '.join(missing_tools)}. Верни только tool calls для этих "
                    "инструментов и не описывай результат обычным текстом до их выполнения."
                )
            )
            try:
                result = self.graph.invoke(
                    {
                        "messages": [*messages, retry_message],
                        "user_id": self.user_id,
                        "session_id": self.session_id,
                    },
                    config={"recursion_limit": self.config.agent.recursion_limit},
                )
            except Exception as exc:
                if self._is_recoverable_agent_error(exc):
                    return self._error_response(
                        message, input_messages, memory_before, exc
                    )
                raise
            messages = result["messages"]
            loop_guard_triggered = loop_guard_triggered or bool(
                result.get("loop_guard_triggered", False)
            )
        answer_message = self._last_ai_message(messages)
        answer = str(answer_message.content if answer_message else "")
        tool_calls = self._tool_call_names(messages)
        tool_results = self._tool_results(messages)
        citations, retrieval = self._retrieval_from_messages(messages)
        if self._looks_like_serialized_tool_call(answer):
            answer = self._repair_serialized_tool_answer(message, messages)
        if citations:
            answer = self._append_citations(answer, citations)
        elif (
            self.config.rag.enabled
            and self.config.rag.require_citations
            and self._requires_rag(message)
        ):
            reason = (
                retrieval.error
                if retrieval is not None and retrieval.error
                else "релевантные источники не найдены"
            )
            if self._has_deterministic_support_evidence(messages):
                answer = (
                    f"{answer.rstrip()}\n\n"
                    "Дополнительные подтверждённые сведения в базе знаний не "
                    f"найдены: {reason}."
                ).strip()
            else:
                answer = (
                    "Не удалось получить подтверждённые данные из базы знаний, "
                    "поэтому я не формирую технический ответ без источников. "
                    f"Причина: {reason}."
                )

        self.short_term.add(HumanMessage(content=message), AIMessage(content=answer))
        try:
            summarized_messages = self.summary.summarize_if_needed(
                llm=self.llm,
                messages=self.short_term.snapshot(),
                max_history_messages=self.config.agent.max_history_messages,
            )
        except Exception:
            LOGGER.exception(
                "Не удалось обновить summary memory; возвращается готовый ответ, "
                "а short-term history обрезается до последних полных ходов"
            )
            self.short_term.messages = self.summary.recent_messages(
                self.short_term.snapshot(),
                self.config.agent.max_history_messages,
            )
        else:
            if len(summarized_messages) != len(self.short_term.messages):
                self.short_term.messages = summarized_messages

        memory_after = self.store.list_memories(user_id=self.user_id, limit=200)
        trace = self._trace(
            message=message,
            input_messages=input_messages,
            result_messages=messages,
            answer=answer,
            tool_calls=tool_calls,
            tool_results=tool_results,
            memory_before=memory_before,
            memory_after=memory_after,
            loop_guard_triggered=loop_guard_triggered,
        )
        LOGGER.info("Агент ответил; tool calls: %d", len(tool_calls))
        return AgentResponse(
            answer=answer,
            user_id=self.user_id,
            session_id=self.session_id,
            tool_calls=tool_calls,
            citations=citations,
            retrieval=retrieval,
            trace=trace,
        )

    def list_memory(self) -> list[dict[str, object]]:
        return [
            record.model_dump(mode="json")
            for record in self.store.list_memories(user_id=self.user_id, limit=100)
        ]

    def clear_session_memory(self) -> int:
        self.short_term.clear()
        return self.store.clear_session(
            user_id=self.user_id, session_id=self.session_id
        )

    def close(self) -> None:
        if self._external_mcp_manager is not None:
            self._external_mcp_manager.close()
            self._external_mcp_manager = None
        if self._owns_rag_runtime and self.rag_runtime is not None:
            self.rag_runtime.close()

    def _build_graph(self):
        if not getattr(self.llm, "supports_tool_calling", True):

            def local_agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
                response = self.llm.invoke(state["messages"])
                return {"messages": [response]}

            workflow = StateGraph(AgentState)
            workflow.add_node("agent", local_agent_node)
            workflow.add_edge(START, "agent")
            return workflow.compile()

        llm_with_tools = self.llm.bind_tools(self.tools)

        def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        def finalize_node(state: AgentState) -> dict[str, object]:
            return {
                "messages": [
                    *self._cancel_pending_tool_calls(state["messages"]),
                    AIMessage(
                        content=self._fallback_answer_from_tools(state["messages"])
                    ),
                ],
                "loop_guard_triggered": True,
            }

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("finalize", finalize_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "finalize": "finalize", END: END},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    @staticmethod
    def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(
                message, "tool_calls", None
            ):
                return message
        return None

    @staticmethod
    def _tool_call_names(messages: list[BaseMessage]) -> list[str]:
        names: list[str] = []
        for message in messages:
            for call in getattr(message, "tool_calls", []) or []:
                name = call.get("name")
                if name:
                    names.append(str(name))
        return names

    def _missing_required_tools(
        self,
        user_message: str,
        messages: list[BaseMessage],
    ) -> list[str]:
        requested = self._requested_tools(user_message)
        if not requested:
            return []
        called = set(self._tool_call_names(messages))
        return [tool for tool in requested if tool not in called]

    def _requested_tools(self, user_message: str) -> list[str]:
        lower = user_message.lower()
        tool_names = [tool.name for tool in self.tools]
        requested = [name for name in tool_names if name.lower() in lower]
        if "обнови статус" in lower and "задач" in lower:
            requested.append("update_task_status")
        if "создай проект" in lower:
            requested.append("create_project")
        if "создай" in lower and "задач" in lower:
            requested.append("create_task")
        if "сохрани" in lower and "памят" in lower:
            requested.append("save_memory")
        if (
            self.config.rag.enabled
            and self._requires_rag(user_message)
            and not {"search_knowledge_base", "find_runbook"}.intersection(requested)
        ):
            requested.append("search_knowledge_base")

        deduplicated: list[str] = []
        known = set(tool_names)
        for name in requested:
            if name in known and name not in deduplicated:
                deduplicated.append(name)
        return deduplicated

    @staticmethod
    def _requires_rag(user_message: str) -> bool:
        lower = user_message.lower()
        markers = (
            "ошиб",
            "сбой",
            "проблем",
            "диагност",
            "инструк",
            "документ",
            "настро",
            "runbook",
            "лог",
            "парол",
            "архив",
            "service desk",
            "как ",
            "какие ",
            "что делать",
            "обязательн",
            "заявк",
            "процедур",
            "регламент",
        )
        return any(marker in lower for marker in markers)

    def _route_after_agent(self, state: AgentState) -> str:
        messages = state["messages"]
        if not messages:
            return END
        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []
        if not tool_calls:
            return END
        if self._should_finalize_tool_loop(messages):
            LOGGER.warning("Остановлен повторяющийся цикл tool-вызовов")
            return "finalize"
        return "tools"

    def _should_finalize_tool_loop(self, messages: list[BaseMessage]) -> bool:
        current = messages[-1]
        for current_call in getattr(current, "tool_calls", []) or []:
            signature = self._tool_call_signature(current_call)
            previous_calls = [
                call
                for message in messages[:-1]
                for call in getattr(message, "tool_calls", []) or []
                if self._tool_call_signature(call) == signature
            ]
            if previous_calls and not self._can_retry_failed_tool_call(
                messages,
                previous_calls,
            ):
                return True

        max_tool_calls = max(1, (self.config.agent.recursion_limit - 4) // 2)
        return len(self._tool_call_names(messages)) > max_tool_calls

    def _can_retry_failed_tool_call(
        self,
        messages: list[BaseMessage],
        previous_calls: list[dict[str, object]],
    ) -> bool:
        if len(previous_calls) > self.config.agent.tool_error_retries:
            return False
        latest_call_id = previous_calls[-1].get("id")
        if not latest_call_id:
            return False
        latest_result = next(
            (
                message
                for message in reversed(messages[:-1])
                if isinstance(message, ToolMessage)
                and message.tool_call_id == str(latest_call_id)
            ),
            None,
        )
        return latest_result is not None and self._tool_message_is_error(latest_result)

    @staticmethod
    def _tool_call_signature(call: dict[str, object]) -> str:
        payload = {
            "name": call.get("name"),
            "args": call.get("args", {}),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def _fallback_answer_from_tools(self, messages: list[BaseMessage]) -> str:
        tool_results = self._tool_results(messages)
        if not tool_results:
            return "Tool-вызовы остановлены: модель начала повторять цикл действий."

        lines = ["Собрал результаты tools и остановил повторный цикл вызовов."]
        for result in tool_results[-8:]:
            lines.append(self._format_tool_result_for_answer(result))
        return "\n".join(lines)

    def _repair_serialized_tool_answer(
        self,
        user_message: str,
        messages: list[BaseMessage],
    ) -> str:
        evidence = self._tool_evidence(messages)
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Сформируй окончательный ответ пользователю на русском "
                            "языке. Tools уже выполнены: не вызывай их повторно, не "
                            "возвращай JSON, XML или описание function call. Используй "
                            "только приведённые результаты и явно сообщи, если данных "
                            "недостаточно."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Исходный запрос:\n{user_message}\n\n"
                            f"Результаты tools:\n{evidence}"
                        )
                    ),
                ]
            )
            repaired = str(response.content).strip()
            if (
                repaired
                and not getattr(response, "tool_calls", None)
                and not self._looks_like_serialized_tool_call(repaired)
            ):
                return repaired
        except Exception:
            LOGGER.exception(
                "Не удалось восстановить финальный ответ после protocol leakage"
            )
        return self._fallback_answer_from_tools(messages)

    @staticmethod
    def _tool_evidence(messages: list[BaseMessage]) -> str:
        evidence: list[str] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            content = str(message.content)
            try:
                payload = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                evidence.append(content)
                continue
            if payload.get("status") == "cancelled":
                continue
            if getattr(message, "name", None) in {
                "search_knowledge_base",
                "find_runbook",
            }:
                evidence.append(str(payload.get("context") or payload))
            else:
                evidence.append(json.dumps(payload, ensure_ascii=False))
        return "\n\n".join(evidence)[:16000]

    @staticmethod
    def _looks_like_serialized_tool_call(answer: str) -> bool:
        lowered = answer.lower()
        return (
            ("recipient_name" in lowered and "parameters" in lowered)
            or "functions." in lowered
            or "<tool_call" in lowered
            or '"tool_calls"' in lowered
        )

    @staticmethod
    def _has_deterministic_support_evidence(messages: list[BaseMessage]) -> bool:
        grounded_tools = {"analyze_log_fragment", "build_diagnostic_checklist"}
        return any(
            isinstance(message, ToolMessage)
            and getattr(message, "name", None) in grounded_tools
            and not AgentRunner._tool_message_is_error(message)
            for message in messages
        )

    @staticmethod
    def _cancel_pending_tool_calls(messages: list[BaseMessage]) -> list[ToolMessage]:
        if not messages:
            return []
        pending_calls = getattr(messages[-1], "tool_calls", None) or []
        cancelled: list[ToolMessage] = []
        for index, call in enumerate(pending_calls, start=1):
            tool_name = str(call.get("name") or "tool")
            tool_call_id = str(call.get("id") or f"loop_guard_{index}")
            cancelled.append(
                ToolMessage(
                    content=json.dumps(
                        {
                            "status": "cancelled",
                            "reason": "loop_guard",
                            "message": "Повторный tool-вызов остановлен защитой от циклов.",
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )
        return cancelled

    @staticmethod
    def _format_tool_result_for_answer(result: AgentToolResult) -> str:
        name = result.name or "tool"
        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            return f"- {name}: {result.content[:300]}"

        if name == "get_weather":
            if "error" in payload:
                return f"- get_weather: ошибка для города {payload.get('city')}: {payload.get('message')}"
            return (
                "- get_weather: "
                f"{payload.get('city')}, {payload.get('temperature')}°C, "
                f"{payload.get('description')}"
            )
        if name == "calculate_travel_budget":
            return (
                "- calculate_travel_budget: "
                f"{payload.get('city')}, бюджет {payload.get('total')} "
                f"{payload.get('currency')}"
            )
        if name == "advise_packing":
            items = payload.get("items", [])
            if isinstance(items, list):
                return f"- advise_packing: {', '.join(map(str, items[:8]))}"
        if name == "save_memory":
            record = payload.get("record", {})
            if isinstance(record, dict):
                return f"- save_memory: сохранено в память, ключ {record.get('key')}"
        if name == "search_memory":
            records = payload.get("records", [])
            if isinstance(records, list):
                values = []
                for record in records[:3]:
                    if isinstance(record, dict):
                        values.append(f"{record.get('key')}: {record.get('value')}")
                return f"- search_memory: {'; '.join(values)}"
        if name == "summarize_project_state":
            return f"- summarize_project_state: {json.dumps(payload, ensure_ascii=False)[:500]}"
        if name in {"search_knowledge_base", "find_runbook"}:
            if payload.get("status") != "ok":
                return f"- {name}: база знаний недоступна: {payload.get('error')}"
            citations = payload.get("citations", [])
            return (
                f"- {name}: найдено {payload.get('retrieved_count', 0)} фрагментов, "
                f"использовано источников: {len(citations) if isinstance(citations, list) else 0}"
            )
        return f"- {name}: {json.dumps(payload, ensure_ascii=False)[:300]}"

    def _error_response(
        self,
        message: str,
        input_messages: list[BaseMessage],
        memory_before: list[MemoryRecord],
        exc: Exception,
    ) -> AgentResponse:
        answer = f"Ошибка выполнения агента: {self._sanitize_agent_error(exc)}"
        error_message = AIMessage(content=answer)
        trace = self._trace(
            message=message,
            input_messages=input_messages,
            result_messages=[error_message],
            answer=answer,
            tool_calls=[],
            tool_results=[
                AgentToolResult(
                    name="agent_runtime",
                    content=answer,
                    is_error=True,
                )
            ],
            memory_before=memory_before,
            memory_after=memory_before,
        )
        return AgentResponse(
            answer=answer,
            user_id=self.user_id,
            session_id=self.session_id,
            tool_calls=[],
            trace=trace,
        )

    def _retrieval_from_messages(
        self,
        messages: list[BaseMessage],
    ) -> tuple[list[RagCitation], AgentRetrievalInfo | None]:
        results: list[RagRetrievalResult] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            if getattr(message, "name", None) not in {
                "search_knowledge_base",
                "find_runbook",
            }:
                continue
            try:
                payload = json.loads(str(message.content))
                if payload.get("status") == "cancelled":
                    continue
                results.append(RagRetrievalResult.model_validate(payload))
            except (json.JSONDecodeError, ValueError, TypeError):
                LOGGER.warning("Не удалось разобрать результат RAG tool")
        if not results:
            return [], None

        citations: list[RagCitation] = []
        seen_chunks: set[str] = set()
        for result in results:
            for citation in result.citations:
                if citation.chunk_id in seen_chunks:
                    continue
                seen_chunks.add(citation.chunk_id)
                citations.append(
                    citation.model_copy(
                        update={"reference": f"[Источник {len(citations) + 1}]"}
                    )
                )
        latest = results[-1]
        status = "ok" if citations else latest.status
        error = next(
            (result.error for result in reversed(results) if result.error),
            None,
        )
        return citations, AgentRetrievalInfo(
            status=status,
            retrieved_count=sum(result.retrieved_count for result in results),
            used_count=len(citations),
            context_tokens=sum(result.context_tokens for result in results),
            provider=latest.provider,
            model=latest.model,
            collection_name=latest.collection_name,
            error=error,
        )

    @staticmethod
    def _append_citations(answer: str, citations: list[RagCitation]) -> str:
        lines = []
        for citation in citations:
            location = citation.source or "неизвестный источник"
            if citation.section:
                location = f"{location}, раздел: {citation.section}"
            lines.append(
                f"- {citation.reference} {location}; chunk_id={citation.chunk_id}; "
                f"score={citation.score:.4f}"
            )
        footer = "Источники:\n" + "\n".join(lines)
        if "Источники:" in answer:
            return answer
        return f"{answer.rstrip()}\n\n{footer}".strip()

    @staticmethod
    def _is_recoverable_agent_error(exc: Exception) -> bool:
        module = exc.__class__.__module__
        return (
            isinstance(exc, GraphRecursionError)
            or module.startswith("gigachat")
            or module.startswith("openai")
        )

    @staticmethod
    def _sanitize_agent_error(exc: Exception) -> str:
        text = str(exc)
        text = re.sub(
            r"Authorization':\s*'[^']+'", "Authorization': '<redacted>'", text
        )
        text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer <redacted>", text)
        if len(text) > 800:
            return f"{text[:800]}..."
        return text

    @staticmethod
    def _is_secret_storage_request(message: str) -> bool:
        return bool(SECRET_STORAGE_RE.search(message))

    @staticmethod
    def _tool_results(messages: list[BaseMessage]) -> list[AgentToolResult]:
        results: list[AgentToolResult] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                content = str(message.content)
                results.append(
                    AgentToolResult(
                        name=getattr(message, "name", None),
                        content=content[:2000],
                        is_error=AgentRunner._tool_message_is_error(message),
                    )
                )
        return results

    @staticmethod
    def _tool_message_is_error(message: ToolMessage) -> bool:
        if getattr(message, "status", None) == "error":
            return True
        content = str(message.content)
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            status = str(payload.get("status") or "").lower()
            if status in {"error", "failed", "unavailable"}:
                return True
            error = payload.get("error")
            if error not in {None, "", False}:
                return True
            return False
        lowered = content.lower()
        return (
            '"status": "error"' in lowered
            or "error invoking tool" in lowered
            or "ошибка" in lowered
        )

    def _trace(
        self,
        *,
        message: str,
        input_messages: list[BaseMessage],
        result_messages: list[BaseMessage],
        answer: str,
        tool_calls: list[str],
        tool_results: list[AgentToolResult],
        memory_before: list[MemoryRecord],
        memory_after: list[MemoryRecord],
        loop_guard_triggered: bool = False,
    ) -> AgentTrace:
        before_by_id = {record.id: record for record in memory_before}
        after_by_id = {record.id: record for record in memory_after}
        created_ids = [
            record_id for record_id in after_by_id if record_id not in before_by_id
        ]
        deleted_ids = [
            record_id for record_id in before_by_id if record_id not in after_by_id
        ]
        updated_ids = [
            record_id
            for record_id, record in after_by_id.items()
            if record_id in before_by_id
            and record.updated_at != before_by_id[record_id].updated_at
        ]
        intermediate_states: list[AgentTraceState] = []
        for index, name in enumerate(tool_calls, start=1):
            intermediate_states.append(
                AgentTraceState(
                    name="tool_call",
                    data={"position": index, "tool": name},
                )
            )
        for index, result in enumerate(tool_results, start=1):
            intermediate_states.append(
                AgentTraceState(
                    name="tool_result",
                    data={
                        "position": index,
                        "tool": result.name,
                        "is_error": result.is_error,
                        "content_preview": result.content[:400],
                    },
                )
            )
        if loop_guard_triggered:
            intermediate_states.append(
                AgentTraceState(
                    name="loop_guard",
                    data={"reason": "модель начала повторять tool-вызовы"},
                )
            )
        if not intermediate_states:
            intermediate_states.append(
                AgentTraceState(
                    name="llm_answer",
                    data={"reason": "tool не потребовался"},
                )
            )

        return AgentTrace(
            user_request=message,
            start_state=AgentTraceState(
                name="start",
                data={
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "memory_count": len(memory_before),
                    "memory_keys": [record.key for record in memory_before],
                    "input_messages_count": len(input_messages),
                },
            ),
            intermediate_states=intermediate_states,
            final_state=AgentTraceState(
                name="final",
                data={
                    "answer": answer,
                    "answer_chars": len(answer),
                    "tool_call_count": len(tool_calls),
                    "tool_error_count": sum(
                        1 for result in tool_results if result.is_error
                    ),
                    "loop_guard_triggered": loop_guard_triggered,
                    "result_messages_count": len(result_messages),
                    "memory_count": len(memory_after),
                    "memory_keys": [record.key for record in memory_after],
                },
            ),
            transition_rules=[
                "START -> agent: сбор system prompt, short-term memory и пользовательского сообщения",
                "agent -> tools: если LLM вернула tool_calls",
                "agent -> loop_guard: если LLM повторяет один и тот же tool-вызов",
                "tools -> agent: после выполнения tools результат возвращается в LLM",
                "agent -> final: если LLM вернула ответ без tool_calls",
                "secret_guardrail -> final: если запрос просит сохранить секрет",
            ],
            decision_points=[
                "LLM выбирает, нужен ли tool для текущего запроса",
                "route_after_agent маршрутизирует граф в tools, loop_guard или финальный ответ",
                "memory diff показывает, созданы, обновлены или удалены записи",
                f"recursion_limit={self.config.agent.recursion_limit} предотвращает зацикливание",
            ],
            tool_calls=tool_calls,
            tool_results=tool_results,
            memory_created_ids=created_ids,
            memory_updated_ids=updated_ids,
            memory_deleted_ids=deleted_ids,
            loop_guard_triggered=loop_guard_triggered,
            recursion_limit=self.config.agent.recursion_limit,
        )
