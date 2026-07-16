from __future__ import annotations

import asyncio

from agent_app.multi_agent.bus import AsyncMessageBus
from agent_app.multi_agent.models import AgentEnvelope, MessageKind
from agent_app.multi_agent.protocols.acp import ACPMessage, ACPProtocolAdapter


def run_protocol_simulation() -> dict[str, object]:
    return asyncio.run(_simulate())


async def _simulate() -> dict[str, object]:
    bus = AsyncMessageBus()
    events: list[str] = []

    async def specialist(envelope: AgentEnvelope) -> AgentEnvelope:
        return AgentEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.message_id,
            sender="diagnostics_agent",
            recipient=envelope.sender,
            kind=MessageKind.RESPONSE,
            payload={"status": "ok", "result": "Проверить timeout budget."},
        )

    async def event_handler(envelope: AgentEnvelope) -> None:
        events.append(str(envelope.payload.get("status")))

    bus.register_agent("diagnostics_agent", specialist)
    bus.subscribe("task.completed", event_handler)
    request = AgentEnvelope(
        correlation_id="simulation-run",
        sender="coordinator",
        recipient="diagnostics_agent",
        kind=MessageKind.REQUEST,
        payload={"task": "Диагностика timeout"},
    )
    response = await bus.request(request)
    subscribers = await bus.publish(
        AgentEnvelope(
            correlation_id="simulation-run",
            sender="coordinator",
            topic="task.completed",
            kind=MessageKind.EVENT,
            payload={"status": "completed"},
        )
    )
    acp = ACPMessage(
        agent_name="legacy-client",
        role="user",
        parts=[{"content": "Диагностика timeout"}],
    )
    return {
        "request_response": response.payload,
        "published_subscribers": subscribers,
        "received_events": events,
        "acp_legacy": acp.model_dump(mode="json"),
        "a2a_migration": ACPProtocolAdapter.to_a2a_message(acp),
        "journal_size": len(bus.journal()),
        "dead_letters": len(bus.dead_letters()),
    }
