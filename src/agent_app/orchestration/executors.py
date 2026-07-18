from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_app.multi_agent.models import MultiAgentRunResult
from agent_app.orchestration.models import (
    OrchestrationJob,
    PlanStep,
    StepResult,
    StepStatus,
    utc_now,
)

AskMulti = Callable[[str, str, str], MultiAgentRunResult]


class MultiAgentStepExecutor:
    """Адаптирует существующий supervisor-граф к шагу orchestration-плана."""

    def __init__(self, ask: AskMulti, *, max_context_chars: int = 6000):
        self.ask = ask
        self.max_context_chars = max_context_chars

    def execute(
        self,
        step: PlanStep,
        job: OrchestrationJob,
        context: Mapping[str, StepResult],
    ) -> StepResult:
        prior = "\n\n".join(
            f"{step_id}: {result.output}"
            for step_id, result in context.items()
            if result.output and result.assigned_role is not None
        )[-self.max_context_chars :]
        role = step.assigned_role or "coordinator"
        prompt = (
            f"Оркестратор назначил текущему шагу роль: {role}.\n"
            f"Цель шага: {step.prompt}\n\n"
            f"Исходный запрос:\n{job.message}"
        )
        if prior:
            prompt += f"\n\nРезультаты предыдущих шагов:\n{prior}"
        result = self.ask(
            job.user_id,
            f"{job.session_id}:orchestration:{job.id}:{step.id}",
            prompt,
        )
        response = result.response
        return StepResult(
            step_id=step.id,
            status=StepStatus.COMPLETED,
            output=response.answer,
            assigned_role=role,
            started_at=utc_now(),
            finished_at=utc_now(),
            metadata={
                "multi_agent_run_id": response.run_id,
                "selected_agents": response.selected_agents,
                "citations": [
                    citation.model_dump(mode="json") for citation in response.citations
                ],
                "degraded": response.degraded,
            },
        )


def runtime_executor(config: Any) -> tuple[MultiAgentStepExecutor, Any]:
    from agent_app.multi_agent.runtime import MultiAgentRuntime

    runtime = MultiAgentRuntime(config)

    def ask(user_id: str, session_id: str, message: str) -> MultiAgentRunResult:
        return runtime.ask(user_id=user_id, session_id=session_id, message=message)

    return MultiAgentStepExecutor(ask), runtime
