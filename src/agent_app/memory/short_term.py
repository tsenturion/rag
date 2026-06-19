from __future__ import annotations

from langchain_core.messages import BaseMessage


class ShortTermMemory:
    """In-session message buffer with fixed maximum length."""

    def __init__(self, max_messages: int):
        self.max_messages = max_messages
        self.messages: list[BaseMessage] = []

    def add(self, *messages: BaseMessage) -> None:
        self.messages.extend(messages)

    def snapshot(self) -> list[BaseMessage]:
        return list(self.messages)

    def overflow(self) -> list[BaseMessage]:
        if len(self.messages) <= self.max_messages:
            return []
        return self.messages[: -self.max_messages]

    def clear(self) -> None:
        self.messages.clear()
