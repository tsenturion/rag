from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from agent_app.config import AgentAppConfig
from agent_app.orchestration.engine import OrchestrationEngine, StepExecutor
from agent_app.orchestration.errors import (
    JobNotFoundError,
    QueueCapacityError,
    TransientOrchestrationError,
)
from agent_app.orchestration.executors import MultiAgentStepExecutor, runtime_executor
from agent_app.orchestration.models import (
    JobEvent,
    JobRecord,
    JobStatus,
    JobSubmission,
    OrchestrationJob,
    QueueStatus,
    StepStatus,
    utc_now,
)
from agent_app.orchestration.planning import OrchestrationPlanBuilder
from agent_app.orchestration.queue import CeleryJobDispatcher
from agent_app.orchestration.store import InMemoryJobStore, JobStore, RedisJobStore

ExecutorFactory = Callable[[], StepExecutor]


class JobRunner:
    def __init__(
        self,
        config: AgentAppConfig,
        store: JobStore,
        executor: StepExecutor,
    ):
        self.config = config
        self.store = store
        self.executor = executor

    def run(self, job: OrchestrationJob) -> JobRecord:
        record = self.store.get(job.id) or JobRecord(job=job)
        if record.status == JobStatus.CANCELLED:
            return record
        if job.expired:
            return self._finish(record, JobStatus.EXPIRED, "Deadline задания истёк")

        provider = self.config.agent.provider
        limit = self.config.orchestration.provider_concurrency_limits.get(provider, 1)
        slot_name = f"provider:{provider}"
        if not self.store.acquire_slot(
            slot_name,
            job.id,
            limit=limit,
            lease_seconds=self.config.orchestration.slot_lease_seconds,
        ):
            raise TransientOrchestrationError(
                f"Достигнут лимит параллелизма provider={provider}: {limit}"
            )

        record.status = JobStatus.RUNNING
        record.attempts += 1
        record.started_at = record.started_at or utc_now()
        record.updated_at = utc_now()
        record.error = None
        self.store.save(record)
        self.store.append_event(
            JobEvent(
                job_id=job.id,
                kind="started",
                status=JobStatus.RUNNING,
                message=f"Начата попытка {record.attempts}",
                payload={"attempt": record.attempts, "provider": provider},
            )
        )
        try:
            engine = OrchestrationEngine(
                self.executor,
                plan_builder=OrchestrationPlanBuilder(
                    step_timeout_seconds=(self.config.multi_agent.task_timeout_seconds)
                ),
                max_parallelism=self.config.orchestration.max_parallelism,
                allow_parallel=provider != "local",
            )
            result = engine.run(job)
            latest = self.store.get(job.id)
            if latest is not None and latest.status == JobStatus.CANCELLED:
                return latest
            record.result = result
            record.status = result.status
            record.error = result.error
            record.finished_at = utc_now() if result.status.terminal else None
            record.updated_at = utc_now()
            self.store.save(record)
            self._record_result_events(record)
            return record
        finally:
            self.store.release_slot(slot_name, job.id)

    def mark_retry(self, job_id: str, error: str, *, countdown: int) -> JobRecord:
        record = self._require(job_id)
        record.status = JobStatus.RETRYING
        record.error = error[:1000]
        record.updated_at = utc_now()
        self.store.save(record)
        self.store.append_event(
            JobEvent(
                job_id=job_id,
                kind="retry",
                status=JobStatus.RETRYING,
                message=f"Повтор через {countdown} с",
                payload={"countdown": countdown, "error": error[:500]},
            )
        )
        return record

    def mark_failed(self, job_id: str, error: str) -> JobRecord:
        return self._finish(self._require(job_id), JobStatus.FAILED, error)

    def _record_result_events(self, record: JobRecord) -> None:
        result = record.result
        if result is None:
            return
        for step in result.step_results:
            self.store.append_event(
                JobEvent(
                    job_id=record.job.id,
                    kind="step",
                    status=record.status,
                    message=f"Шаг {step.step_id}: {step.status.value}",
                    payload={
                        "step_id": step.step_id,
                        "step_status": step.status.value,
                        "assigned_role": step.assigned_role,
                        "retryable": step.retryable,
                        "error": step.error,
                    },
                )
            )
        for revision in result.revisions:
            self.store.append_event(
                JobEvent(
                    job_id=record.job.id,
                    kind="replanned",
                    status=record.status,
                    message=revision.reason,
                    payload=revision.model_dump(mode="json"),
                )
            )
        kind = (
            "completed"
            if record.status == JobStatus.COMPLETED
            else "expired"
            if record.status == JobStatus.EXPIRED
            else "failed"
        )
        self.store.append_event(
            JobEvent(
                job_id=record.job.id,
                kind=kind,
                status=record.status,
                message=(
                    "Задание завершено"
                    if record.status == JobStatus.COMPLETED
                    else record.error or "Задание завершилось ошибкой"
                ),
                payload={
                    "duration_ms": result.duration_ms,
                    "plan_version": result.plan.version,
                    "quorum_reached": result.synchronization.quorum_reached,
                },
            )
        )

    def _finish(self, record: JobRecord, status: JobStatus, error: str) -> JobRecord:
        record.status = status
        record.error = error[:1000]
        record.updated_at = utc_now()
        record.finished_at = utc_now()
        self.store.save(record)
        kind = "expired" if status == JobStatus.EXPIRED else "failed"
        self.store.append_event(
            JobEvent(
                job_id=record.job.id,
                kind=kind,
                status=status,
                message=record.error,
            )
        )
        return record

    def _require(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        if record is None:
            raise JobNotFoundError(f"Задание не найдено: {job_id}")
        return record


class OrchestrationService:
    def __init__(
        self,
        config: AgentAppConfig,
        *,
        store: JobStore | None = None,
        executor_factory: ExecutorFactory | None = None,
        dispatcher: CeleryJobDispatcher | None = None,
    ):
        self.config = config
        self.backend = config.orchestration.backend
        self.store = store or self._build_store()
        self.executor_factory = executor_factory
        self.dispatcher = (
            dispatcher
            if dispatcher is not None
            else CeleryJobDispatcher(config.orchestration)
            if self.backend == "celery"
            else None
        )
        self._owned_runtime: Any | None = None

    def submit(self, job: OrchestrationJob) -> JobSubmission:
        if not self.config.orchestration.enabled:
            raise RuntimeError("Оркестрация отключена в конфигурации")
        if job.idempotency_key:
            claimed_id = self.store.claim_idempotency(
                job.idempotency_key,
                job.id,
                self.config.orchestration.idempotency_ttl_seconds,
            )
            if claimed_id != job.id:
                existing = self.store.get(claimed_id)
                if existing is not None:
                    return JobSubmission(record=existing, deduplicated=True)

        counts = self.store.counts()
        pending = sum(
            counts.get(status.value, 0)
            for status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRYING}
        )
        if pending >= self.config.orchestration.max_pending_jobs:
            raise QueueCapacityError(
                "Очередь достигла max_pending_jobs="
                f"{self.config.orchestration.max_pending_jobs}"
            )

        record = JobRecord(job=job)
        self.store.save(record)
        self.store.append_event(
            JobEvent(
                job_id=job.id,
                kind="submitted",
                status=JobStatus.QUEUED,
                message="Задание принято оркестратором",
                payload={
                    "pattern": job.pattern.value,
                    "priority": job.priority.value,
                },
            )
        )
        if self.backend == "celery":
            if self.dispatcher is None:
                raise RuntimeError("Celery dispatcher не настроен")
            try:
                task_id = self.dispatcher.dispatch(
                    job,
                    self.config.model_dump(mode="json"),
                )
            except Exception as exc:
                self._mark_submission_failed(
                    record,
                    f"Не удалось отправить задание в broker: {exc}",
                )
                raise
            record.task_id = task_id
            record.updated_at = utc_now()
            self.store.save(record)
        else:
            record = JobRunner(
                self.config,
                self.store,
                self._executor(),
            ).run(job)
        return JobSubmission(record=record)

    def get(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        if record is None:
            raise JobNotFoundError(f"Задание не найдено: {job_id}")
        return record

    def events(self, job_id: str) -> list[JobEvent]:
        self.get(job_id)
        return self.store.events(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record.status.terminal:
            return record
        if self.dispatcher is not None and record.task_id:
            self.dispatcher.cancel(record.task_id)
        record.status = JobStatus.CANCELLED
        record.finished_at = utc_now()
        record.updated_at = utc_now()
        record.error = "Задание отменено пользователем"
        self.store.save(record)
        self.store.append_event(
            JobEvent(
                job_id=job_id,
                kind="cancelled",
                status=JobStatus.CANCELLED,
                message=record.error,
            )
        )
        return record

    def wait(self, job_id: str, *, timeout_seconds: float) -> JobRecord:
        deadline = time.monotonic() + timeout_seconds
        while True:
            record = self.get(job_id)
            if record.status.terminal:
                return record
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Задание {job_id} не завершилось за {timeout_seconds} с"
                )
            time.sleep(0.5)

    def status(self) -> QueueStatus:
        try:
            ready = self.store.ping()
            workers = self.dispatcher.workers() if self.dispatcher else {"inline": True}
            if self.dispatcher is not None:
                ready = ready and bool(workers.get("ping"))
            return QueueStatus(
                backend=self.backend,
                ready=ready,
                status_counts=self.store.counts(),
                workers=workers,
            )
        except Exception as exc:
            return QueueStatus(
                backend=self.backend,
                ready=False,
                error=str(exc)[:500],
            )

    def close(self) -> None:
        if self._owned_runtime is not None:
            self._owned_runtime.close()
            self._owned_runtime = None
        self.store.close()

    def _build_store(self) -> JobStore:
        if self.backend == "inline":
            return InMemoryJobStore()
        url = os.getenv(self.config.orchestration.state_store_url_env)
        if not url:
            raise RuntimeError(
                "Для Celery backend задайте "
                f"{self.config.orchestration.state_store_url_env}"
            )
        return RedisJobStore(
            url,
            state_ttl_seconds=self.config.orchestration.state_ttl_seconds,
            event_limit=self.config.orchestration.event_limit,
        )

    def _executor(self) -> StepExecutor:
        if self.executor_factory is not None:
            return self.executor_factory()
        executor, runtime = runtime_executor(self.config)
        self._owned_runtime = runtime
        return executor

    def _mark_submission_failed(self, record: JobRecord, error: str) -> None:
        record.status = JobStatus.FAILED
        record.error = error[:1000]
        record.updated_at = utc_now()
        record.finished_at = utc_now()
        self.store.save(record)
        self.store.append_event(
            JobEvent(
                job_id=record.job.id,
                kind="failed",
                status=JobStatus.FAILED,
                message=record.error,
            )
        )


def callback_executor(
    callback: Callable[[str, str, str], Any],
) -> ExecutorFactory:
    return lambda: MultiAgentStepExecutor(callback)


def result_has_retryable_failure(record: JobRecord) -> bool:
    return bool(
        record.result
        and any(
            step.retryable and step.status in {StepStatus.FAILED, StepStatus.TIMED_OUT}
            for step in record.result.step_results
        )
    )
