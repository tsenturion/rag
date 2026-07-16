from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from agent_app.multi_agent.models import (
    AgentEnvelope,
    MessageDeliveryState,
    MessageKind,
)

MessageHandler = Callable[[AgentEnvelope], Awaitable[AgentEnvelope]]
EventHandler = Callable[[AgentEnvelope], Awaitable[None]]


class AsyncMessageBus:
    """In-memory request-response/pub-sub transport с дедупликацией и timeout."""

    def __init__(self):
        self._agents: dict[str, MessageHandler] = {}
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._responses: dict[str, AgentEnvelope] = {}
        self._seen: set[str] = set()
        self._journal: list[AgentEnvelope] = []
        self._dead_letters: list[AgentEnvelope] = []
        self._lock = asyncio.Lock()

    def register_agent(self, name: str, handler: MessageHandler) -> None:
        if name in self._agents:
            raise ValueError(f"Обработчик агента уже зарегистрирован: {name}")
        self._agents[name] = handler

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        self._subscribers[topic].append(handler)

    async def request(self, envelope: AgentEnvelope) -> AgentEnvelope:
        if envelope.kind != MessageKind.REQUEST:
            raise ValueError("request() принимает только сообщения kind=request")
        if not envelope.recipient:
            raise ValueError("Request-сообщению нужен recipient")

        async with self._lock:
            cached = self._responses.get(envelope.message_id)
            if cached is not None:
                self._record(
                    envelope.model_copy(
                        update={"delivery_state": MessageDeliveryState.DUPLICATE}
                    )
                )
                return cached.model_copy(deep=True)
            if envelope.message_id in self._seen:
                raise RuntimeError(
                    f"Повторное незавершённое сообщение: {envelope.message_id}"
                )
            self._seen.add(envelope.message_id)
            self._record(
                envelope.model_copy(
                    update={"delivery_state": MessageDeliveryState.SENT}
                )
            )

        if self._expired(envelope):
            expired = envelope.model_copy(
                update={
                    "delivery_state": MessageDeliveryState.EXPIRED,
                    "error": "Истёк TTL сообщения",
                }
            )
            self._dead_letters.append(expired)
            self._record(expired)
            raise TimeoutError(expired.error)

        handler = self._agents.get(envelope.recipient)
        if handler is None:
            failed = envelope.model_copy(
                update={
                    "delivery_state": MessageDeliveryState.FAILED,
                    "error": f"Получатель не зарегистрирован: {envelope.recipient}",
                }
            )
            self._dead_letters.append(failed)
            self._record(failed)
            raise LookupError(failed.error)

        self._record(
            envelope.model_copy(
                update={"delivery_state": MessageDeliveryState.DELIVERED}
            )
        )
        try:
            response = await asyncio.wait_for(
                handler(envelope),
                timeout=envelope.ttl_seconds,
            )
        except TimeoutError as exc:
            error = f"Превышен timeout доставки сообщения: {envelope.ttl_seconds} с"
            failed = envelope.model_copy(
                update={
                    "delivery_state": MessageDeliveryState.FAILED,
                    "error": error,
                }
            )
            self._dead_letters.append(failed)
            self._record(failed)
            raise TimeoutError(error) from exc
        except Exception as exc:
            failed = envelope.model_copy(
                update={
                    "delivery_state": MessageDeliveryState.FAILED,
                    "error": str(exc)[:500],
                }
            )
            self._dead_letters.append(failed)
            self._record(failed)
            raise

        completed = response.model_copy(
            update={"delivery_state": MessageDeliveryState.COMPLETED}
        )
        async with self._lock:
            self._responses[envelope.message_id] = completed
            self._record(completed)
        return completed.model_copy(deep=True)

    async def publish(self, envelope: AgentEnvelope) -> int:
        if envelope.kind != MessageKind.EVENT:
            raise ValueError("publish() принимает только сообщения kind=event")
        if not envelope.topic:
            raise ValueError("Event-сообщению нужен topic")
        self._record(
            envelope.model_copy(update={"delivery_state": MessageDeliveryState.SENT})
        )
        subscribers = list(self._subscribers.get(envelope.topic, []))
        if subscribers:
            await asyncio.gather(*(handler(envelope) for handler in subscribers))
        self._record(
            envelope.model_copy(
                update={"delivery_state": MessageDeliveryState.COMPLETED}
            )
        )
        return len(subscribers)

    def journal(self) -> list[AgentEnvelope]:
        return [message.model_copy(deep=True) for message in self._journal]

    def dead_letters(self) -> list[AgentEnvelope]:
        return [message.model_copy(deep=True) for message in self._dead_letters]

    def _record(self, envelope: AgentEnvelope) -> None:
        self._journal.append(envelope.model_copy(deep=True))

    @staticmethod
    def _expired(envelope: AgentEnvelope) -> bool:
        elapsed = datetime.now(timezone.utc) - envelope.created_at
        return elapsed.total_seconds() > envelope.ttl_seconds
