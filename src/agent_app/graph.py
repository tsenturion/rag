from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_app.config import AgentAppConfig
from agent_app.llm import build_llm
from agent_app.memory import SQLiteMemoryStore, ShortTermMemory, SummaryMemory
from agent_app.models import AgentResponse, AgentState
from agent_app.prompts import system_prompt
from agent_app.tools import build_tools

LOGGER = logging.getLogger(__name__)


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
            config={"recursion_limit": 12},
        )
        messages: list[BaseMessage] = result["messages"]
        answer_message = self._last_ai_message(messages)
        answer = str(answer_message.content if answer_message else "")
        tool_calls = self._tool_call_names(messages)

        self.short_term.add(HumanMessage(content=message), AIMessage(content=answer))
        summarized_messages = self.summary.summarize_if_needed(
            llm=self.llm,
            messages=self.short_term.snapshot(),
            max_history_messages=self.config.agent.max_history_messages,
        )
        if len(summarized_messages) != len(self.short_term.messages):
            self.short_term.messages = summarized_messages

        LOGGER.info("Агент ответил; tool calls: %d", len(tool_calls))
        return AgentResponse(
            answer=answer,
            user_id=self.user_id,
            session_id=self.session_id,
            tool_calls=tool_calls,
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
