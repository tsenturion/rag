from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


def session_thread_id(user_id: str, session_id: str) -> str:
    """Возвращает непрозрачный и воспроизводимый thread_id для пары user/session."""
    return str(uuid5(NAMESPACE_URL, f"rag-multi-agent:{user_id}:{session_id}"))


class MultiAgentCheckpointStore:
    """Владеет SQLite checkpointer и операциями над multi-agent сессиями."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=30,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("agent_app.multi_agent.models", "AgentTask"),
                ("agent_app.multi_agent.models", "AgentTaskResult"),
                ("agent_app.multi_agent.models", "TaskExecutionState"),
                ("agent_app.rag.models", "RagCitation"),
            ]
        )
        self._saver = SqliteSaver(self._connection, serde=serializer)
        self._saver.setup()
        self._lock = threading.RLock()

    @property
    def saver(self) -> SqliteSaver:
        return self._saver

    @staticmethod
    def runnable_config(user_id: str, session_id: str) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": session_thread_id(user_id, session_id),
            }
        }

    def history(self, *, user_id: str, session_id: str) -> list[BaseMessage]:
        config = self.runnable_config(user_id, session_id)
        with self._lock:
            checkpoint = self._saver.get_tuple(config)
        if checkpoint is None:
            return []
        values = checkpoint.checkpoint.get("channel_values", {})
        history = values.get("history", [])
        return [item for item in history if isinstance(item, BaseMessage)]

    def clear(self, *, user_id: str, session_id: str) -> bool:
        thread_id = session_thread_id(user_id, session_id)
        config = self.runnable_config(user_id, session_id)
        with self._lock:
            existed = self._saver.get_tuple(config) is not None
            self._saver.delete_thread(thread_id)
        return existed

    def close(self) -> None:
        with self._lock:
            self._connection.close()
