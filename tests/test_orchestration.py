"""Регрессионные тесты для подсистемы orchestration."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

import jwt
from fastapi.testclient import TestClient

from agent_app.config import (
    AgentAppConfig,
    AgentConfig,
    AgentSecurityConfig,
    MemoryConfig,
    MultiAgentConfig,
    MultiAgentLLMProfileConfig,
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
from agent_app.orchestration.queue import (
    CeleryJobDispatcher,
    create_celery_app,
    declare_celery_topology,
)
from agent_app.orchestration.service import JobRunner, OrchestrationService
from agent_app.orchestration.store import InMemoryJobStore
from agent_app.service.app import create_app


class StaticExecutor:
    """Обеспечивает имитацию выполнения шагов оркестрации с контролируемыми ошибками и голосами для тестирования устойчивости и логики повторных попыток."""

    def __init__(
        self,
        *,
        barrier: threading.Barrier | None = None,
        transient_role: str | None = None,
        votes: dict[str, str] | None = None,
    ):
        """Инициализирует executor с предопределёнными голосами и синхронизацией для контролируемого тестирования оркестрации."""
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
        """Выполняет шаг оркестрации с имитацией временной ошибки и учётом голосов, обеспечивая проверку устойчивости и логики retry."""
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


class SlowExecutor:
    """Имитирует внешний синхронный вызов, который нельзя остановить как thread."""

    def execute(self, step, job, context):
        """Занимает ресурс дольше step timeout для проверки lease и latency."""
        del job, context
        time.sleep(0.5)
        return StepResult(
            step_id=step.id,
            status=StepStatus.COMPLETED,
            assigned_role=step.assigned_role,
            output="Поздний результат",
        )


def _job(pattern: OrchestrationPattern, **updates) -> OrchestrationJob:
    """Создаёт предсказуемое задание оркестрации с базовыми параметрами для проверки логики обработки сценариев."""
    values = {
        "user_id": "engineer",
        "session_id": "incident-42",
        "message": "Диагностируй временную недоступность API",
        "pattern": pattern,
    }
    values.update(updates)
    return OrchestrationJob.model_validate(values)


def _config(root: Path, *, max_pending_jobs: int = 20) -> AgentAppConfig:
    """Обеспечивает воспроизводимую конфигурацию приложения с включённой оркестрацией и ограничением на количество ожидающих заданий."""
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
    """Проверяет корректность работы движка оркестрации, включая последовательные, параллельные, условные и кворумные паттерны исполнения заданий."""

    def test_sequential_pattern(self) -> None:
        """Проверяет, что при последовательном паттерне оркестрации выполняется только первый шаг с ожидаемым результатом и статусом."""
        executor = StaticExecutor()
        result = OrchestrationEngine(executor).run(
            _job(OrchestrationPattern.SEQUENTIAL)
        )

        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertEqual(executor.calls, ["diagnostics_agent"])
        self.assertIn("Результат роли", result.answer)

    def test_parallel_pattern_uses_barrier(self) -> None:
        """Проверяет, что при параллельном паттерне оркестрации используется барьер для синхронизации выполнения всех шагов одновременно."""
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
        """Проверяет, что при использовании условного паттерна выполняется только ветка, соответствующая заданному условию, а остальные шаги пропускаются."""
        executor = StaticExecutor()
        result = OrchestrationEngine(executor).run(
            _job(OrchestrationPattern.CONDITIONAL, risk_level="high")
        )

        statuses = {item.step_id: item.status for item in result.step_results}
        self.assertEqual(statuses["standard-analysis"], StepStatus.SKIPPED)
        self.assertEqual(statuses["high-risk-analysis"], StepStatus.COMPLETED)
        self.assertEqual(executor.calls, ["critic_agent"])

    def test_quorum_requires_consensus(self) -> None:
        """Проверяет, что паттерн кворума завершается успешно только при достижении согласованного решения, иначе возвращает ошибку с указанием отсутствия консенсуса."""
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
        """Проверяет, что при динамическом паттерне роль, не выполнившая задачу, переназначается другому агенту, и план пересматривается с учётом изменений."""
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
    """Тестирует сервис оркестрации на идемпотентность, обработку событий, управление нагрузкой и корректность HTTP-интерфейса."""

    def test_inline_service_is_idempotent_and_exports_events(self) -> None:
        """Проверяет, что сервис обеспечивает идемпотентность при повторной отправке задачи с одинаковым ключом и корректно экспортирует события жизненного цикла задачи."""
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

    def test_inline_idempotency_key_expires_after_ttl(self) -> None:
        """Проверяет повторное закрепление ключа за новым заданием после его срока действия."""
        store = InMemoryJobStore()
        with patch(
            "agent_app.orchestration.store.time.monotonic",
            side_effect=[100.0, 100.5, 101.1],
        ):
            first = store.claim_idempotency("incident-key", "job-1", 1)
            before_expiry = store.claim_idempotency("incident-key", "job-2", 1)
            after_expiry = store.claim_idempotency("incident-key", "job-3", 1)

        self.assertEqual(first, "job-1")
        self.assertEqual(before_expiry, "job-1")
        self.assertEqual(after_expiry, "job-3")

    def test_idempotency_key_is_isolated_between_users(self) -> None:
        """Проверяет, что одинаковый внешний ключ разных пользователей не объединяет их задания."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = OrchestrationService(
                _config(Path(temporary_dir)),
                executor_factory=lambda: StaticExecutor(),
            )
            try:
                alice = service.submit(
                    _job(
                        OrchestrationPattern.SEQUENTIAL,
                        user_id="alice",
                        idempotency_key="shared-key",
                    )
                )
                bob = service.submit(
                    _job(
                        OrchestrationPattern.SEQUENTIAL,
                        user_id="bob",
                        idempotency_key="shared-key",
                    )
                )
            finally:
                service.close()

        self.assertFalse(alice.deduplicated)
        self.assertFalse(bob.deduplicated)
        self.assertNotEqual(alice.record.job.id, bob.record.job.id)

    def test_stale_idempotency_binding_is_rebound_once(self) -> None:
        """Восстанавливает key, указывающий на уже отсутствующее состояние job."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config(Path(temporary_dir))
            store = InMemoryJobStore()
            service = OrchestrationService(
                config,
                store=store,
                executor_factory=lambda: StaticExecutor(),
            )
            first_job = _job(
                OrchestrationPattern.SEQUENTIAL,
                idempotency_key="stale-key",
            )
            scoped_key = service._scoped_idempotency_key(first_job)
            store.claim_idempotency(
                scoped_key,
                "missing-job-id",
                config.orchestration.idempotency_ttl_seconds,
            )
            try:
                first = service.submit(first_job)
                second = service.submit(
                    _job(
                        OrchestrationPattern.SEQUENTIAL,
                        idempotency_key="stale-key",
                    )
                )
            finally:
                service.close()

        self.assertFalse(first.deduplicated)
        self.assertTrue(second.deduplicated)
        self.assertEqual(second.record.job.id, first.record.job.id)

    def test_step_timeout_returns_promptly_but_holds_provider_slot(self) -> None:
        """Timeout ограничивает latency, не освобождая ресурс работающего thread."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config(Path(temporary_dir))
            config = config.model_copy(
                update={
                    "multi_agent": config.multi_agent.model_copy(
                        update={"task_timeout_seconds": 0.1}
                    ),
                    "orchestration": config.orchestration.model_copy(
                        update={"slot_lease_seconds": 1}
                    ),
                }
            )
            store = InMemoryJobStore()
            started = time.perf_counter()
            result = JobRunner(config, store, SlowExecutor()).run(
                _job(OrchestrationPattern.SEQUENTIAL)
            )
            elapsed = time.perf_counter() - started
            occupied = store.acquire_slot(
                "provider:local",
                "second-job",
                limit=1,
                lease_seconds=1,
            )
            time.sleep(0.55)
            released = store.acquire_slot(
                "provider:local",
                "second-job",
                limit=1,
                lease_seconds=1,
            )

        self.assertEqual(result.status, JobStatus.FAILED)
        self.assertLess(elapsed, 0.4)
        self.assertFalse(occupied)
        self.assertTrue(released)

    def test_provider_slots_follow_roles_in_selected_plan(self) -> None:
        """Не резервирует provider профилей, роли которых отсутствуют в плане."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config(Path(temporary_dir))
            config = config.model_copy(
                update={
                    "multi_agent": config.multi_agent.model_copy(
                        update={
                            "llm_profiles": {
                                "diagnostics": MultiAgentLLMProfileConfig(
                                    provider="openai",
                                    model="gpt-test",
                                ),
                                "unused_local": MultiAgentLLMProfileConfig(
                                    provider="local",
                                    model="unused-model",
                                ),
                            },
                            "role_llm_profiles": {
                                "diagnostics_agent": "diagnostics",
                            },
                        }
                    )
                }
            )
            runner = JobRunner(config, InMemoryJobStore(), StaticExecutor())

            sequential = runner._required_providers(
                _job(OrchestrationPattern.SEQUENTIAL)
            )
            dynamic = runner._required_providers(_job(OrchestrationPattern.DYNAMIC))

        self.assertEqual(sequential, ["openai"])
        # Fallback-роли dynamic plan не имеют отдельных профилей и используют base.
        self.assertEqual(dynamic, ["local", "openai"])

    def test_terminal_job_is_not_executed_twice(self) -> None:
        """Проверяет, что повторная доставка завершённого задания не повторяет внешние действия шагов."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config(Path(temporary_dir))
            store = InMemoryJobStore()
            executor = StaticExecutor()
            runner = JobRunner(config, store, executor)
            job = _job(OrchestrationPattern.SEQUENTIAL)

            first = runner.run(job)
            second = runner.run(job)

        self.assertEqual(first.status, JobStatus.COMPLETED)
        self.assertEqual(second.status, JobStatus.COMPLETED)
        self.assertEqual(executor.calls, ["diagnostics_agent"])

    def test_backpressure_rejects_full_store(self) -> None:
        """Проверяет, что сервис отклоняет новые задачи с ошибкой при достижении максимального количества ожидающих заданий, обеспечивая защиту от перегрузки."""
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
        """Проверяет, что HTTP API корректно принимает новую задачу, возвращает её идентификатор и позволяет получить статус выполненной задачи."""
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

    def test_http_object_access_is_restricted_to_job_owner(self) -> None:
        """JWT пользователя не позволяет читать или отменять чужой job_id."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config(Path(temporary_dir)).model_copy(
                update={
                    "security": AgentSecurityConfig(
                        jwt_enabled=True,
                        jwt_secret_env="TEST_ORCHESTRATION_JWT_SECRET",
                    )
                }
            )
            service = OrchestrationService(
                config,
                executor_factory=lambda: StaticExecutor(),
            )
            bob = service.submit(_job(OrchestrationPattern.SEQUENTIAL, user_id="bob"))
            runtime = SimpleNamespace(
                config=config,
                orchestration_service=service,
                close=lambda: None,
            )
            secret = "d" * 40
            now = datetime.now(timezone.utc)
            token = jwt.encode(
                {
                    "sub": "alice",
                    "roles": ["engineer"],
                    "iat": now,
                    "exp": now + timedelta(minutes=5),
                    "iss": "rag-support",
                    "aud": "rag-support-api",
                },
                secret,
                algorithm="HS256",
            )
            headers = {"Authorization": f"Bearer {token}"}
            with patch.dict(
                os.environ,
                {"TEST_ORCHESTRATION_JWT_SECRET": secret},
            ):
                with TestClient(create_app(runtime=runtime)) as client:
                    loaded = client.get(
                        f"/v1/orchestration/jobs/{bob.record.job.id}",
                        headers=headers,
                    )
                    cancelled = client.delete(
                        f"/v1/orchestration/jobs/{bob.record.job.id}",
                        headers=headers,
                    )
            service.close()

        self.assertEqual(loaded.status_code, 403)
        self.assertEqual(cancelled.status_code, 403)


class InfrastructureContractTest(unittest.IsolatedAsyncioTestCase):
    """Проверяет инфраструктурные аспекты оркестрации, включая конфигурацию Celery и поведение детерминированных Camunda-воркеров."""

    def test_celery_has_priority_and_dead_letter_queues(self) -> None:
        """Проверяет, что конфигурация Celery содержит очереди с приоритетами и мёртвые очереди для обработки неуспешных задач, а также корректные параметры воркера."""
        config = OrchestrationConfig(enabled=True, backend="celery")
        app = create_celery_app(config)
        queues = {queue.name: queue for queue in app.conf.task_queues}
        dispatcher = CeleryJobDispatcher(config, app=app)

        self.assertEqual(
            queues[config.queue_default].queue_arguments["x-queue-type"],
            "quorum",
        )
        self.assertEqual(
            queues[config.queue_default].queue_arguments["x-dead-letter-exchange"],
            "agent.dead_letter",
        )
        self.assertTrue(app.conf.worker_detect_quorum_queues)
        self.assertEqual(
            dispatcher._route(JobPriority.HIGH),
            (config.queue_high, "high"),
        )
        self.assertEqual(app.conf.worker_prefetch_multiplier, 1)
        self.assertTrue(app.conf.control_queue_exclusive)
        self.assertFalse(app.conf.control_queue_durable)
        self.assertTrue(app.conf.event_queue_exclusive)
        self.assertFalse(app.conf.event_queue_durable)

    def test_celery_topology_declares_dead_letter_queue(self) -> None:
        """Объявляет DLQ до запуска worker, хотя она отсутствует в consume list."""

        class BoundQueue:
            """Фиксирует объявление одной тестовой очереди."""

            def __init__(self, name: str, declared: list[str]):
                """Сохраняет имя и общий журнал объявлений."""
                self.name = name
                self.declared = declared

            def declare(self) -> None:
                """Добавляет очередь в журнал фактически объявленной topology."""
                self.declared.append(self.name)

        class QueueStub:
            """Имитирует привязку Kombu Queue к открытому AMQP channel."""

            def __init__(self, name: str, declared: list[str]):
                """Сохраняет имя очереди и журнал объявлений."""
                self.name = name
                self.declared = declared

            def bind(self, _channel) -> BoundQueue:
                """Возвращает объект, способный объявить очередь на channel."""
                return BoundQueue(self.name, self.declared)

        class ChannelStub:
            """Фиксирует корректное закрытие AMQP channel после деклараций."""

            closed = False

            def close(self) -> None:
                """Отмечает освобождение channel."""
                self.closed = True

        class ConnectionStub:
            """Предоставляет context manager и channel для topology setup."""

            def __init__(self, channel: ChannelStub):
                """Сохраняет единственный тестовый channel."""
                self._channel = channel

            def __enter__(self):
                """Возвращает соединение при входе в context manager."""
                return self

            def __exit__(self, *_args) -> None:
                """Завершает context manager без подавления исключений."""

            def channel(self) -> ChannelStub:
                """Предоставляет AMQP-соединение для объявления exchanges и queues."""
                return self._channel

        declared: list[str] = []
        channel = ChannelStub()
        queues = [
            QueueStub("agent.default.quorum", declared),
            QueueStub("agent.dead_letter.quorum", declared),
        ]
        app = SimpleNamespace(
            conf=SimpleNamespace(task_queues=queues),
            connection_for_write=lambda: ConnectionStub(channel),
        )

        declare_celery_topology(app)

        self.assertEqual(
            declared,
            ["agent.default.quorum", "agent.dead_letter.quorum"],
        )
        self.assertTrue(channel.closed)

    async def test_camunda_deterministic_workers(self) -> None:
        """Проверяет, что Camunda-воркеры последовательно валидируют, классифицируют, проверяют и выполняют задачи, обеспечивая детерминированность и идемпотентность."""
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
        """Проверяет, что BPMN-файл содержит корректный XML с определёнными элементами процесса и задачами агентов, необходимыми для работы оркестрации."""
        root = ET.parse("bpmn/engineer_support.bpmn").getroot()
        xml = ET.tostring(root, encoding="unicode")
        self.assertIn("engineer-support-process", xml)
        self.assertIn("run-support-agent", xml)
        self.assertIn("verify-support-result", xml)


if __name__ == "__main__":
    unittest.main()
