from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_app.config import AgentAppConfig
from agent_app.llm import build_llm
from agent_app.memory import SQLiteMemoryStore, ShortTermMemory, SummaryMemory
from agent_app.models import (
    AgentResponse,
    AgentState,
    AgentToolResult,
    AgentTrace,
    AgentTraceState,
    MemoryRecord,
)
from agent_app.prompts import system_prompt
from agent_app.tools import build_tools

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
        self.llm = build_llm(config.agent)
        self.tools = build_tools(
            config,
            self.store,
            user_id=self.user_id,
            session_id=self.session_id,
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
            content=system_prompt(summary=self.summary.get(), memories=memories)
        )
        input_messages: list[BaseMessage] = [
            system,
            *self.short_term.snapshot(),
            HumanMessage(content=message),
        ]
        result = self.graph.invoke(
            {
                "messages": input_messages,
                "user_id": self.user_id,
                "session_id": self.session_id,
            },
            config={"recursion_limit": self.config.agent.recursion_limit},
        )
        messages: list[BaseMessage] = result["messages"]
        answer_message = self._last_ai_message(messages)
        answer = str(answer_message.content if answer_message else "")
        tool_calls = self._tool_call_names(messages)
        tool_results = self._tool_results(messages)

        self.short_term.add(HumanMessage(content=message), AIMessage(content=answer))
        summarized_messages = self.summary.summarize_if_needed(
            llm=self.llm,
            messages=self.short_term.snapshot(),
            max_history_messages=self.config.agent.max_history_messages,
        )
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
        )
        LOGGER.info("Агент ответил; tool calls: %d", len(tool_calls))
        return AgentResponse(
            answer=answer,
            user_id=self.user_id,
            session_id=self.session_id,
            tool_calls=tool_calls,
            trace=trace,
        )

    def list_memory(self) -> list[dict[str, object]]:
        return [
            record.model_dump(mode="json")
            for record in self.store.list_memories(user_id=self.user_id, limit=100)
        ]

    def clear_session_memory(self) -> int:
        self.short_term.clear()
        return self.store.clear_session(user_id=self.user_id, session_id=self.session_id)

    def _build_graph(self):
        llm_with_tools = self.llm.bind_tools(self.tools)

        def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    @staticmethod
    def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
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

    @staticmethod
    def _is_secret_storage_request(message: str) -> bool:
        return bool(SECRET_STORAGE_RE.search(message))

    @staticmethod
    def _tool_results(messages: list[BaseMessage]) -> list[AgentToolResult]:
        results: list[AgentToolResult] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                content = str(message.content)
                lowered = content.lower()
                results.append(
                    AgentToolResult(
                        name=getattr(message, "name", None),
                        content=content[:2000],
                        is_error='"status": "error"' in lowered
                        or '"error":' in lowered
                        or "ошибка" in lowered,
                    )
                )
        return results

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
    ) -> AgentTrace:
        before_by_id = {record.id: record for record in memory_before}
        after_by_id = {record.id: record for record in memory_after}
        created_ids = [record_id for record_id in after_by_id if record_id not in before_by_id]
        deleted_ids = [record_id for record_id in before_by_id if record_id not in after_by_id]
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
                    "tool_error_count": sum(1 for result in tool_results if result.is_error),
                    "result_messages_count": len(result_messages),
                    "memory_count": len(memory_after),
                    "memory_keys": [record.key for record in memory_after],
                },
            ),
            transition_rules=[
                "START -> agent: сбор system prompt, short-term memory и пользовательского сообщения",
                "agent -> tools: если LLM вернула tool_calls",
                "tools -> agent: после выполнения tools результат возвращается в LLM",
                "agent -> final: если LLM вернула ответ без tool_calls",
                "secret_guardrail -> final: если запрос просит сохранить секрет",
            ],
            decision_points=[
                "LLM выбирает, нужен ли tool для текущего запроса",
                "tools_condition маршрутизирует граф в tools или финальный ответ",
                "memory diff показывает, созданы, обновлены или удалены записи",
                f"recursion_limit={self.config.agent.recursion_limit} предотвращает зацикливание",
            ],
            tool_calls=tool_calls,
            tool_results=tool_results,
            memory_created_ids=created_ids,
            memory_updated_ids=updated_ids,
            memory_deleted_ids=deleted_ids,
            recursion_limit=self.config.agent.recursion_limit,
        )
