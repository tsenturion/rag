"""Краткосрочная память диалога для памяти агента."""

from __future__ import annotations

from langchain_core.messages import BaseMessage


class ShortTermMemory:
    """Буфер сообщений текущей сессии с фиксированной максимальной длиной."""

    def __init__(self, max_messages: int):
        """Готовит экземпляр к хранению ограниченного числа сообщений с гарантией пустого состояния при создании."""
        self.max_messages = max_messages
        self.messages: list[BaseMessage] = []

    def add(self, *messages: BaseMessage) -> None:
        """Гарантирует добавление новых сообщений в краткосрочную память без потери порядка."""
        self.messages.extend(messages)

    def snapshot(self) -> list[BaseMessage]:
        """Гарантирует получение полной копии текущего состояния краткосрочной памяти без побочных эффектов."""
        return list(self.messages)

    def overflow(self) -> list[BaseMessage]:
        """Гарантирует возврат сообщений, превышающих лимит краткосрочной памяти, для последующей обработки или удаления."""
        if len(self.messages) <= self.max_messages:
            return []
        return self.messages[: -self.max_messages]

    def clear(self) -> None:
        """Гарантирует полное удаление всех сообщений из краткосрочной памяти агента для предотвращения утечек контекста между сессиями."""
        self.messages.clear()
