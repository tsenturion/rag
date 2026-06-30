from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent_app.memory.store import SQLiteMemoryStore


class SummaryMemory:
    """Резюме сессии, сохранённое в долговременной памяти с type=summary."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        user_id: str,
        session_id: str,
        max_chars: int,
    ):
        self.store = store
        self.user_id = user_id
        self.session_id = session_id
        self.max_chars = max_chars

    @property
    def key(self) -> str:
        return f"session_summary:{self.session_id}"

    def get(self) -> str:
        record = self.store.find_by_key(
            user_id=self.user_id,
            key=self.key,
            memory_type="summary",
            session_id=self.session_id,
        )
        return record.value if record else ""

    def save(self, value: str) -> None:
        if not value.strip():
            return
        self.store.save(
            user_id=self.user_id,
            session_id=self.session_id,
            memory_type="summary",
            key=self.key,
            value=value[-self.max_chars :],
            tags=["summary", "session"],
            importance=4,
            source="system",
        )

    def summarize_if_needed(
        self,
        *,
        llm,
        messages: list[BaseMessage],
        max_history_messages: int,
    ) -> list[BaseMessage]:
        if len(messages) <= max_history_messages:
            return messages

        overflow = messages[: -max_history_messages]
        kept = messages[-max_history_messages:]
        previous_summary = self.get()
        transcript = "\n".join(
            f"{message.type}: {getattr(message, 'content', '')}" for message in overflow
        )
        prompt = [
            SystemMessage(
                content=(
                    "Сожми историю диалога в короткую память для будущего контекста. "
                    "Сохраняй только факты, предпочтения, задачи и важные договорённости. "
                    "Не добавляй новые сведения."
                )
            ),
            HumanMessage(
                content=(
                    f"Предыдущее резюме:\n{previous_summary or 'нет'}\n\n"
                    f"Новые сообщения:\n{transcript}\n\n"
                    "Обновлённое резюме:"
                )
            ),
        ]
        summary = llm.invoke(prompt).content
        if isinstance(summary, list):
            summary = " ".join(str(item) for item in summary)
        self.save(str(summary))
        return kept
