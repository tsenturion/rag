from __future__ import annotations

from threading import RLock

from agent_app.multi_agent.models import AgentRunState, LifecycleEvent


class LifecycleTracker:
    """Проверяет допустимые переходы и сохраняет воспроизводимый журнал запуска."""

    _ALLOWED: dict[AgentRunState, set[AgentRunState]] = {
        AgentRunState.RECEIVED: {
            AgentRunState.DECOMPOSED,
            AgentRunState.FAILED,
        },
        AgentRunState.DECOMPOSED: {
            AgentRunState.DELEGATED,
            AgentRunState.FAILED,
        },
        AgentRunState.DELEGATED: {
            AgentRunState.RUNNING,
            AgentRunState.FAILED,
        },
        AgentRunState.RUNNING: {
            AgentRunState.REVIEWING,
            AgentRunState.FAILED,
        },
        AgentRunState.REVIEWING: {
            AgentRunState.DELEGATED,
            AgentRunState.COMPLETED,
            AgentRunState.FAILED,
        },
        AgentRunState.COMPLETED: set(),
        AgentRunState.FAILED: set(),
    }

    def __init__(self, *, details: dict[str, object] | None = None):
        self._lock = RLock()
        self._events = [
            LifecycleEvent(
                state=AgentRunState.RECEIVED,
                details=dict(details or {}),
            )
        ]

    @property
    def state(self) -> AgentRunState:
        return self._events[-1].state

    def transition(
        self,
        state: AgentRunState,
        *,
        details: dict[str, object] | None = None,
    ) -> LifecycleEvent:
        with self._lock:
            if state not in self._ALLOWED[self.state]:
                raise ValueError(
                    f"Недопустимый lifecycle-переход: {self.state} -> {state}"
                )
            event = LifecycleEvent(state=state, details=dict(details or {}))
            self._events.append(event)
            return event

    def fail(self, error: str) -> LifecycleEvent:
        if self.state in {AgentRunState.COMPLETED, AgentRunState.FAILED}:
            return self._events[-1]
        return self.transition(AgentRunState.FAILED, details={"error": error[:500]})

    def snapshot(self) -> list[LifecycleEvent]:
        with self._lock:
            return [event.model_copy(deep=True) for event in self._events]
