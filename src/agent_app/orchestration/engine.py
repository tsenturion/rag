from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from time import perf_counter
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_app.orchestration.errors import TransientOrchestrationError
from agent_app.orchestration.models import (
    ExecutionPlan,
    JobStatus,
    OrchestrationJob,
    OrchestrationPattern,
    OrchestrationResult,
    PlanRevision,
    PlanStep,
    StepResult,
    StepStatus,
    SynchronizationResult,
    utc_now,
)
from agent_app.orchestration.planning import OrchestrationPlanBuilder
from agent_app.orchestration.synchronization import QuorumCoordinator


class StepExecutor(Protocol):
    def execute(
        self,
        step: PlanStep,
        job: OrchestrationJob,
        context: Mapping[str, StepResult],
    ) -> StepResult: ...


class OrchestrationGraphState(TypedDict):
    job: OrchestrationJob
    plan: ExecutionPlan
    step_results: list[StepResult]
    revisions: list[PlanRevision]
    synchronization: SynchronizationResult


class OrchestrationEngine:
    """LangGraph-контур для паттернов, barrier/quorum и replanning."""

    def __init__(
        self,
        executor: StepExecutor,
        *,
        plan_builder: OrchestrationPlanBuilder | None = None,
        max_parallelism: int = 3,
        allow_parallel: bool = True,
    ):
        self.executor = executor
        self.plan_builder = plan_builder or OrchestrationPlanBuilder()
        self.max_parallelism = max(1, max_parallelism)
        self.allow_parallel = allow_parallel
        self.quorum = QuorumCoordinator()
        self.graph = self._build_graph()

    def run(self, job: OrchestrationJob) -> OrchestrationResult:
        started = perf_counter()
        plan = self.plan_builder.build(job)
        if job.expired:
            return OrchestrationResult(
                job_id=job.id,
                status=JobStatus.EXPIRED,
                plan=plan,
                error="Deadline задания истёк до начала выполнения",
            )
        state = self.graph.invoke(
            {
                "job": job,
                "plan": plan,
                "step_results": [],
                "revisions": [],
                "synchronization": SynchronizationResult(),
            }
        )
        results = state["step_results"]
        required_failures = self._required_failures(state["plan"], results)
        synchronization = state["synchronization"]
        quorum_failed = job.pattern == OrchestrationPattern.QUORUM and (
            not synchronization.quorum_reached
            or synchronization.consensus == "undetermined"
        )
        status = (
            JobStatus.EXPIRED
            if job.expired
            else JobStatus.FAILED
            if required_failures or quorum_failed
            else JobStatus.COMPLETED
        )
        answer = self._answer(results)
        errors = [result.error for result in required_failures if result.error]
        if quorum_failed:
            if not synchronization.quorum_reached:
                errors.append(
                    "Кворум не достигнут: "
                    f"{synchronization.successful}/{synchronization.required}"
                )
            else:
                errors.append("Кворум собран, но согласованное решение не получено")
        return OrchestrationResult(
            job_id=job.id,
            status=status,
            answer=answer,
            plan=state["plan"],
            step_results=results,
            revisions=state["revisions"],
            synchronization=synchronization,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            error="; ".join(errors) if errors else None,
        )

    def _build_graph(self):
        workflow = StateGraph(OrchestrationGraphState)

        def execute(state: OrchestrationGraphState) -> dict[str, object]:
            return {
                "step_results": self._execute_plan(
                    state["job"],
                    state["plan"],
                    state["step_results"],
                )
            }

        def synchronize(state: OrchestrationGraphState) -> dict[str, object]:
            agent_steps = [
                step
                for step in state["plan"].steps
                if step.kind == "agent" and self._condition_matches(step, state["job"])
            ]
            required = (
                state["job"].quorum_size
                if state["job"].pattern == OrchestrationPattern.QUORUM
                else len([step for step in agent_steps if step.required])
            )
            return {
                "synchronization": self.quorum.evaluate(
                    state["step_results"],
                    required=required,
                )
            }

        def route(state: OrchestrationGraphState) -> str:
            if state["job"].pattern != OrchestrationPattern.DYNAMIC:
                return "finish"
            retryable_failures = [
                result
                for result in self._required_failures(
                    state["plan"], state["step_results"]
                )
                if result.retryable
            ]
            can_replan = (
                bool(retryable_failures)
                and len(state["revisions"]) < state["job"].max_plan_revisions
            )
            return "replan" if can_replan else "finish"

        def replan(state: OrchestrationGraphState) -> dict[str, object]:
            plan, revision = self.plan_builder.replan(
                state["plan"], state["step_results"]
            )
            retained = [
                result
                for result in state["step_results"]
                if result.status in {StepStatus.COMPLETED, StepStatus.SKIPPED}
                and result.step_id != "aggregate"
            ]
            return {
                "plan": plan,
                "revisions": [*state["revisions"], revision],
                "step_results": retained,
            }

        workflow.add_node("execute", execute)
        workflow.add_node("synchronize", synchronize)
        workflow.add_node("replan", replan)
        workflow.add_edge(START, "execute")
        workflow.add_edge("execute", "synchronize")
        workflow.add_conditional_edges(
            "synchronize",
            route,
            {"replan": "replan", "finish": END},
        )
        workflow.add_edge("replan", "execute")
        return workflow.compile()

    def _execute_plan(
        self,
        job: OrchestrationJob,
        plan: ExecutionPlan,
        prior_results: list[StepResult],
    ) -> list[StepResult]:
        result_by_id = {result.step_id: result for result in prior_results}
        pending = [step for step in plan.steps if step.id not in result_by_id]
        while pending:
            if job.expired:
                for step in pending:
                    result_by_id[step.id] = self._failure(
                        step,
                        "Deadline задания истёк",
                        timed_out=True,
                        retryable=False,
                    )
                break

            for step in list(pending):
                if not self._condition_matches(step, job):
                    result_by_id[step.id] = StepResult(
                        step_id=step.id,
                        status=StepStatus.SKIPPED,
                        output="Ветка не выбрана условием",
                        assigned_role=step.assigned_role,
                    )
                    pending.remove(step)

            ready = [
                step
                for step in pending
                if all(
                    dependency in result_by_id and result_by_id[dependency].successful
                    for dependency in step.depends_on
                )
            ]
            if not ready:
                for step in pending:
                    failed_dependencies = [
                        dependency
                        for dependency in step.depends_on
                        if dependency in result_by_id
                        and not result_by_id[dependency].successful
                    ]
                    result_by_id[step.id] = self._failure(
                        step,
                        "Не выполнены зависимости: "
                        + ", ".join(failed_dependencies or step.depends_on),
                        retryable=False,
                    )
                break

            run_parallel = (
                self.allow_parallel
                and plan.pattern
                in {OrchestrationPattern.PARALLEL, OrchestrationPattern.QUORUM}
                and len(ready) > 1
            )
            batch = ready if run_parallel else ready[:1]
            batch_results = self._execute_batch(batch, job, result_by_id)
            for result in batch_results:
                result_by_id[result.step_id] = result
            pending = [step for step in pending if step.id not in result_by_id]

        return [result_by_id[step.id] for step in plan.steps if step.id in result_by_id]

    def _execute_batch(
        self,
        steps: list[PlanStep],
        job: OrchestrationJob,
        context: Mapping[str, StepResult],
    ) -> list[StepResult]:
        if len(steps) == 1:
            return [self._execute_step(steps[0], job, context)]
        results: list[StepResult] = []
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallelism, len(steps)),
            thread_name_prefix="orchestration-step",
        ) as pool:
            futures = {
                step.id: (step, pool.submit(self._execute_step, step, job, context))
                for step in steps
            }
            for step, future in futures.values():
                try:
                    results.append(future.result(timeout=step.timeout_seconds))
                except FutureTimeoutError:
                    future.cancel()
                    results.append(
                        self._failure(
                            step,
                            f"Timeout шага: {step.timeout_seconds} с",
                            timed_out=True,
                            retryable=True,
                        )
                    )
        return results

    def _execute_step(
        self,
        step: PlanStep,
        job: OrchestrationJob,
        context: Mapping[str, StepResult],
    ) -> StepResult:
        started = utc_now()
        if step.kind == "validate":
            return StepResult(
                step_id=step.id,
                status=StepStatus.COMPLETED,
                output="Входное задание прошло проверку",
                started_at=started,
                finished_at=utc_now(),
            )
        if step.kind == "decision":
            return StepResult(
                step_id=step.id,
                status=StepStatus.COMPLETED,
                output=job.risk_level,
                started_at=started,
                finished_at=utc_now(),
                metadata={"risk_level": job.risk_level},
            )
        if step.kind == "aggregate":
            completed = [
                context[dependency]
                for dependency in step.depends_on
                if dependency in context
                and context[dependency].status == StepStatus.COMPLETED
                and context[dependency].output
            ]
            if not completed:
                return self._failure(
                    step,
                    "Нет успешных результатов для агрегации",
                    retryable=False,
                )
            output = (
                completed[0].output
                if len(completed) == 1
                else "\n\n".join(
                    f"### {result.assigned_role or result.step_id}\n{result.output}"
                    for result in completed
                )
            )
            return StepResult(
                step_id=step.id,
                status=StepStatus.COMPLETED,
                output=output,
                started_at=started,
                finished_at=utc_now(),
                metadata={"aggregated_steps": [item.step_id for item in completed]},
            )
        try:
            result = self.executor.execute(step, job, context)
            return result.model_copy(
                update={
                    "step_id": step.id,
                    "assigned_role": step.assigned_role,
                    "started_at": started,
                    "finished_at": utc_now(),
                }
            )
        except (TransientOrchestrationError, TimeoutError, ConnectionError) as exc:
            return self._failure(step, str(exc), retryable=True)
        except Exception as exc:
            return self._failure(step, str(exc), retryable=False)

    def _required_failures(
        self,
        plan: ExecutionPlan,
        results: list[StepResult],
    ) -> list[StepResult]:
        required = {step.id for step in plan.steps if step.required}
        return [
            result
            for result in results
            if result.step_id in required
            and result.status in {StepStatus.FAILED, StepStatus.TIMED_OUT}
        ]

    @staticmethod
    def _condition_matches(step: PlanStep, job: OrchestrationJob) -> bool:
        if step.condition == "always":
            return True
        if step.condition == "high_risk":
            return job.risk_level == "high"
        return job.risk_level in {"low", "medium"}

    @staticmethod
    def _failure(
        step: PlanStep,
        error: str,
        *,
        retryable: bool,
        timed_out: bool = False,
    ) -> StepResult:
        return StepResult(
            step_id=step.id,
            status=StepStatus.TIMED_OUT if timed_out else StepStatus.FAILED,
            error=error[:1000],
            assigned_role=step.assigned_role,
            retryable=retryable,
            finished_at=utc_now(),
        )

    @staticmethod
    def _answer(results: list[StepResult]) -> str:
        aggregate = next(
            (
                result.output
                for result in reversed(results)
                if result.step_id == "aggregate"
                and result.status == StepStatus.COMPLETED
            ),
            "",
        )
        if aggregate:
            return aggregate
        return "\n\n".join(
            result.output
            for result in results
            if result.assigned_role is not None
            and result.status == StepStatus.COMPLETED
            and result.output
        )
