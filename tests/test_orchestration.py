from __future__ import annotations

import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent_app.config import (
    AgentAppConfig,
    AgentConfig,
    AgentSecurityConfig,
    MemoryConfig,
    MultiAgentConfig,
    OrchestrationConfig,
)
from agent_app.orchestration.camunda import CamundaAgentWorker
from agent_app.orchestration.engine import OrchestrationEngine
from agent_app.orchestration.errors import QueueCapacityError
from agent_app.orchestration.models import (
    JobPriority,
    JobRecord,
    JobStatus,
    OrchestrationJob,
    OrchestrationPattern,
    PlanStep,
    StepResult,
    StepStatus,
)
from agent_app.orchestration.queue import CeleryJobDispatcher, create_celery_app
from agent_app.orchestration.service import OrchestrationService
from agent_app.orchestration.store import InMemoryJobStore
from agent_app.service.app import create_app


class StaticExecutor:
    def __init__(
        self,
        *,
        barrier: threading.Barrier | None = None,
        transient_role: str | None = None,
        votes: dict[str, str] | None = None,
    ):
        self.barrier = barrier
        self.transient_role = transient_role
        self.votes = votes or {
            "diagnostics_agent": "approve",
            "knowledge_agent": "approve",
            "critic_agent": "reject",
        }
        self.calls: list[str] = []
        self._failed_once = False
        self._lock = threading.Lock()

    def execute(
        self,
        step: PlanStep,
        job: OrchestrationJob,
        context: Mapping[str, StepResult],
    ) -> StepResult:
        del job, context
        role = step.assigned_role or "unknown"
        with self._lock:
            self.calls.append(role)
            should_fail = role == self.transient_role and not self._failed_once
            if should_fail:
                self._failed_once = True
        if should_fail:
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                assigned_role=role,
                error="Временная ошибка профиля",
                retryable=True,
            )
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        vote = self.votes.get(role)
        marker = f" [{vote}]" if vote else ""
        return StepResult(
            step_id=step.id,
            status=StepStatus.COMPLETED,
            assigned_role=role,
            output=f"Результат роли {role}{marker}",
            vote=vote,
        )


def _job(pattern: OrchestrationPattern, **updates) -> OrchestrationJob:
    return OrchestrationJob(
        user_id="engineer",
        session_id="incident-42",
        message="Диагностируй временную недоступность API",
        pattern=pattern,
        **updates,
    )


def _config(root: Path, *, max_pending_jobs: int = 20) -> AgentAppConfig:
    return AgentAppConfig(
        agent=AgentConfig(provider="local", model="test-model"),
        memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
        security=AgentSecurityConfig(require_api_key=False),
        multi_agent=MultiAgentConfig(
            enabled=False,
            output_dir=root / "runs",
            checkpoint_path=root / "checkpoints.sqlite",
            mlflow_enabled=False,
        ),
        orchestration=OrchestrationConfig(
            enabled=True,
            backend="inline",
            max_pending_jobs=max_pending_jobs,
        ),
    )


class OrchestrationEngineTest(unittest.TestCase):
    def test_sequential_pattern(self) -> None:
        executor = StaticExecutor()
        result = OrchestrationEngine(executor).run(
            _job(OrchestrationPattern.SEQUENTIAL)
        )

        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertEqual(executor.calls, ["diagnostics_agent"])
        self.assertIn("Результат роли", result.answer)

    def test_parallel_pattern_uses_barrier(self) -> None:
        executor = StaticExecutor(barrier=threading.Barrier(3))
        result = OrchestrationEngine(executor, max_parallelism=3).run(
            _job(OrchestrationPattern.PARALLEL)
        )

        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertCountEqual(
            executor.calls,
            ["diagnostics_agent", "knowledge_agent", "critic_agent"],
        )

    def test_conditional_pattern_executes_only_selected_branch(self) -> None:
        executor = StaticExecutor()
        result = OrchestrationEngine(executor).run(
            _job(OrchestrationPattern.CONDITIONAL, risk_level="high")
        )

        statuses = {item.step_id: item.status for item in result.step_results}
        self.assertEqual(statuses["standard-analysis"], StepStatus.SKIPPED)
        self.assertEqual(statuses["high-risk-analysis"], StepStatus.COMPLETED)
        self.assertEqual(executor.calls, ["critic_agent"])

    def test_quorum_requires_consensus(self) -> None:
        accepted = OrchestrationEngine(StaticExecutor()).run(
            _job(OrchestrationPattern.QUORUM, quorum_size=2)
        )
        undecided = OrchestrationEngine(
            StaticExecutor(
                votes={
                    "diagnostics_agent": "approve",
                    "knowledge_agent": "reject",
                    "critic_agent": "abstain",
                }
            )
        ).run(_job(OrchestrationPattern.QUORUM, quorum_size=2))

        self.assertEqual(accepted.status, JobStatus.COMPLETED)
        self.assertEqual(accepted.synchronization.consensus, "approve")
        self.assertEqual(undecided.status, JobStatus.FAILED)
        self.assertIn("согласованное решение", undecided.error or "")

    def test_dynamic_pattern_reassigns_failed_role(self) -> None:
        executor = StaticExecutor(transient_role="diagnostics_agent")
        result = OrchestrationEngine(executor).run(
            _job(OrchestrationPattern.DYNAMIC, max_plan_revisions=2)
        )

        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertEqual(
            executor.calls,
            ["diagnostics_agent", "knowledge_agent"],
        )
        self.assertEqual(len(result.revisions), 1)
        self.assertEqual(
            result.revisions[0].changed_roles,
            {"adaptive-analysis": "knowledge_agent"},
        )


class OrchestrationServiceTest(unittest.TestCase):
    def test_inline_service_is_idempotent_and_exports_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config(Path(temporary_dir))
            service = OrchestrationService(
                config,
                executor_factory=lambda: StaticExecutor(),
            )
            try:
                first = service.submit(
                    _job(
                        OrchestrationPattern.SEQUENTIAL,
                        idempotency_key="incident-42-v1",
                    )
                )
                second = service.submit(
                    _job(
                        OrchestrationPattern.SEQUENTIAL,
                        idempotency_key="incident-42-v1",
                    )
                )
                events = service.events(first.record.job.id)
            finally:
                service.close()

        self.assertFalse(first.deduplicated)
        self.assertTrue(second.deduplicated)
        self.assertEqual(second.record.job.id, first.record.job.id)
        self.assertEqual(events[0].kind, "submitted")
        self.assertEqual(events[-1].kind, "completed")

    def test_backpressure_rejects_full_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config(Path(temporary_dir), max_pending_jobs=1)
            store = InMemoryJobStore()
            store.save(JobRecord(job=_job(OrchestrationPattern.SEQUENTIAL)))
            service = OrchestrationService(
                config,
                store=store,
                executor_factory=lambda: StaticExecutor(),
            )
            try:
                with self.assertRaises(QueueCapacityError):
                    service.submit(_job(OrchestrationPattern.SEQUENTIAL))
            finally:
                service.close()

    def test_http_contract_submits_and_reads_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config(Path(temporary_dir))
            service = OrchestrationService(
                config,
                executor_factory=lambda: StaticExecutor(),
            )
            runtime = SimpleNamespace(
                config=config,
                orchestration_service=service,
                close=lambda: None,
            )
            app = create_app(runtime=runtime)
            try:
                with TestClient(app) as client:
                    submitted = client.post(
                        "/v1/orchestration/jobs",
                        json={
                            "message": "Проверь недоступность API",
                            "user_id": "engineer",
                            "session_id": "incident-42",
                            "pattern": "sequential",
                        },
                    )
                    job_id = submitted.json()["record"]["job"]["id"]
                    loaded = client.get(f"/v1/orchestration/jobs/{job_id}")
            finally:
                service.close()

        self.assertEqual(submitted.status_code, 202)
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.json()["status"], "completed")


class InfrastructureContractTest(unittest.IsolatedAsyncioTestCase):
    def test_celery_has_priority_and_dead_letter_queues(self) -> None:
        config = OrchestrationConfig(enabled=True, backend="celery")
        app = create_celery_app(config)
        queues = {queue.name: queue for queue in app.conf.task_queues}
        dispatcher = CeleryJobDispatcher(config, app=app)

        self.assertEqual(
            queues[config.queue_default].queue_arguments["x-max-priority"],
            config.max_priority,
        )
        self.assertEqual(
            queues[config.queue_default].queue_arguments["x-dead-letter-exchange"],
            "agent.dead_letter",
        )
        self.assertEqual(
            dispatcher._route(JobPriority.HIGH),
            (config.queue_high, "high"),
        )
        self.assertEqual(app.conf.worker_prefetch_multiplier, 1)
        self.assertTrue(app.conf.control_queue_exclusive)
        self.assertFalse(app.conf.control_queue_durable)
        self.assertTrue(app.conf.event_queue_exclusive)
        self.assertFalse(app.conf.event_queue_durable)

    async def test_camunda_deterministic_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config(Path(temporary_dir))
            service = OrchestrationService(
                config,
                executor_factory=lambda: StaticExecutor(),
            )
            worker = CamundaAgentWorker(config, service=service)
            variables = SimpleNamespace(
                to_dict=lambda: {
                    "message": "Проверь API",
                    "userId": "engineer",
                    "sessionId": "incident-42",
                    "riskLevel": "high",
                    "agentStatus": "completed",
                    "agentAnswer": "Достаточно длинный проверенный инженерный ответ.",
                    "priority": "normal",
                }
            )
            job = SimpleNamespace(
                variables=variables,
                process_instance_key="2251799813685249",
                element_id="agent",
                retries=3,
            )
            try:
                validated = await worker.validate_request(job)
                classified = await worker.classify_risk(job)
                verified = await worker.verify_result(job)
                first_agent_run = await worker.run_agent(job)
                deduplicated_agent_run = await worker.run_agent(job)
            finally:
                service.close()

        self.assertTrue(validated["requestValid"])
        self.assertTrue(classified["requiresApproval"])
        self.assertTrue(verified["verificationPassed"])
        self.assertEqual(
            first_agent_run["orchestrationJobId"],
            deduplicated_agent_run["orchestrationJobId"],
        )

    def test_bpmn_is_valid_xml_with_agent_task(self) -> None:
        root = ET.parse("bpmn/engineer_support.bpmn").getroot()
        xml = ET.tostring(root, encoding="unicode")
        self.assertIn("engineer-support-process", xml)
        self.assertIn("run-support-agent", xml)
        self.assertIn("verify-support-result", xml)


if __name__ == "__main__":
    unittest.main()
