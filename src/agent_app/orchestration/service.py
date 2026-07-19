"""Сервис управления заданиями для распределённой оркестрации."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

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
    """Обеспечивает выполнение и контроль жизненного цикла задания с учётом лимитов параллелизма и устойчивости к сбоям."""

    def __init__(
        self,
        config: AgentAppConfig,
        store: JobStore,
        executor: StepExecutor,
    ):
        """Готовит экземпляр к запуску заданий, обеспечивая владение конфигурацией, хранилищем и исполнителем шагов."""
        self.config = config
        self.store = store
        self.executor = executor
        self._claim_tokens: dict[str, str] = {}

    def run(self, job: OrchestrationJob) -> JobRecord:
        """Проводит задание по плану с учётом дедлайна, отмены, лимитов параллелизма и политики ошибок."""
        record = self.store.get(job.id)
        if record is None:
            candidate = JobRecord(job=job)
            self.store.create_if_capacity(candidate, max_pending=2**31 - 1)
            record = self.store.get(job.id) or candidate
        if record.status.terminal:
            return record
        if job.expired:
            if record.status == JobStatus.RUNNING:
                # Активная попытка сама завершит CAS. Duplicate delivery не имеет
                # права отзывать claim другого worker только из-за deadline.
                return record
            return self._finish(record, JobStatus.EXPIRED, "Deadline задания истёк")

        claim_token = uuid4().hex
        claimed = self.store.claim_run(
            job.id,
            claim_token,
            lease_seconds=self.config.orchestration.slot_lease_seconds,
        )
        if claimed is None:
            # Повторная delivery не должна выполнять side effects второй раз. Пока
            # другой worker владеет claim, возвращаем наблюдаемое состояние.
            return self.store.get(job.id) or record
        record = claimed
        self._claim_tokens[job.id] = claim_token

        providers = self._required_providers(job)
        acquired_slots: list[str] = []
        for provider in providers:
            limit = self.config.orchestration.provider_concurrency_limits.get(
                provider, 1
            )
            slot_name = f"provider:{provider}"
            if self.store.acquire_slot(
                slot_name,
                claim_token,
                limit=limit,
                lease_seconds=self.config.orchestration.slot_lease_seconds,
            ):
                acquired_slots.append(slot_name)
                continue
            for acquired in reversed(acquired_slots):
                self.store.release_slot(acquired, claim_token)
            raise TransientOrchestrationError(
                f"Достигнут лимит параллелизма provider={provider}: {limit}"
            )

        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._renew_slots,
            args=(acquired_slots, job.id, claim_token, heartbeat_stop),
            name=f"orchestration-lease-{job.id}",
            daemon=True,
        )
        heartbeat.start()

        engine: OrchestrationEngine | None = None
        try:
            self.store.append_event(
                JobEvent(
                    job_id=job.id,
                    kind="started",
                    status=JobStatus.RUNNING,
                    message=f"Начата попытка {record.attempts}",
                    payload={"attempt": record.attempts, "providers": providers},
                )
            )
            engine = OrchestrationEngine(
                self.executor,
                plan_builder=OrchestrationPlanBuilder(
                    step_timeout_seconds=(self.config.multi_agent.task_timeout_seconds)
                ),
                max_parallelism=self.config.orchestration.max_parallelism,
                allow_parallel="local" not in providers,
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
            saved = self.store.compare_and_save(
                record,
                expected_statuses={JobStatus.RUNNING},
                expected_attempt=record.attempts,
                claim_token=claim_token,
                release_claim=True,
            )
            self._claim_tokens.pop(job.id, None)
            if not saved:
                # Отмена или новый владелец stale-attempt имеют приоритет над
                # результатом worker, утратившего право записи.
                return self.store.get(job.id) or record
            self._record_result_events(record)
            return record
        finally:
            if engine is not None and engine.has_inflight_steps:
                threading.Thread(
                    target=self._release_after_inflight,
                    args=(
                        engine,
                        acquired_slots,
                        claim_token,
                        heartbeat_stop,
                        heartbeat,
                    ),
                    name=f"orchestration-release-{job.id}",
                    daemon=True,
                ).start()
            else:
                self._release_slots(
                    acquired_slots,
                    claim_token,
                    heartbeat_stop,
                    heartbeat,
                )

    def _required_providers(self, job: OrchestrationJob) -> list[str]:
        """Возвращает провайдеров ролей фактического плана и его fallback-ветвей."""
        plan = OrchestrationPlanBuilder(
            step_timeout_seconds=self.config.multi_agent.task_timeout_seconds
        ).build(job)
        roles = {
            role
            for step in plan.steps
            for role in [step.assigned_role, *step.fallback_roles]
            if role
        }
        providers: set[str] = set()
        for role in roles:
            profile_name = self.config.multi_agent.role_llm_profiles.get(role)
            profile = (
                self.config.multi_agent.llm_profiles.get(profile_name)
                if profile_name
                else None
            )
            providers.add(
                profile.provider if profile is not None else self.config.agent.provider
            )
        # Планы без LLM-шагов не должны захватывать provider-слоты.
        return sorted(providers)

    def _renew_slots(
        self,
        slot_names: list[str],
        job_id: str,
        claim_token: str,
        stop: threading.Event,
    ) -> None:
        """Продлевает run claim и provider lease до окончания попытки."""
        lease_seconds = self.config.orchestration.slot_lease_seconds
        interval = max(0.2, lease_seconds / 3)
        while not stop.wait(interval):
            if not self.store.renew_run_claim(
                job_id,
                claim_token,
                lease_seconds=lease_seconds,
            ):
                current = self.store.get(job_id)
                if current is None or not current.status.terminal:
                    self.store.append_event(
                        JobEvent(
                            job_id=job_id,
                            kind="lease_lost",
                            status=JobStatus.RUNNING,
                            message="Потерян lease текущей попытки выполнения",
                        )
                    )
                    return
                # Engine мог вернуть TIMED_OUT раньше синхронного внешнего вызова.
                # Run claim уже не нужен для terminal record, но provider-slot
                # удерживается до фактического завершения фонового thread.
            for slot_name in slot_names:
                if not self.store.renew_slot(
                    slot_name,
                    claim_token,
                    lease_seconds=lease_seconds,
                ):
                    # Потеря lease означает, что лимит параллелизма больше нельзя
                    # гарантировать. Прерывание LLM-потока небезопасно, поэтому
                    # фиксируем проблему и не захватываем слот заново в обход очереди.
                    self.store.append_event(
                        JobEvent(
                            job_id=job_id,
                            kind="lease_lost",
                            status=JobStatus.RUNNING,
                            message=f"Потерян lease ресурса {slot_name}",
                        )
                    )
                    return

    def _release_after_inflight(
        self,
        engine: OrchestrationEngine,
        slot_names: list[str],
        token: str,
        stop: threading.Event,
        heartbeat: threading.Thread,
    ) -> None:
        """Удерживает provider lease, пока просроченный вызов реально работает."""
        engine.wait_for_inflight_steps()
        self._release_slots(slot_names, token, stop, heartbeat)

    def _release_slots(
        self,
        slot_names: list[str],
        token: str,
        stop: threading.Event,
        heartbeat: threading.Thread,
    ) -> None:
        """Останавливает heartbeat и освобождает все захваченные ресурсы."""
        stop.set()
        if heartbeat is not threading.current_thread():
            heartbeat.join(timeout=1)
        for slot_name in reversed(slot_names):
            self.store.release_slot(slot_name, token)

    def mark_retry(self, job_id: str, error: str, *, countdown: int) -> JobRecord:
        """Гарантирует перевод задания в состояние повторной попытки с фиксацией причины и времени следующего запуска."""
        record = self._require(job_id)
        claim_token = self._claim_tokens.pop(job_id, None)
        expected_statuses = {JobStatus.RUNNING} if claim_token else {JobStatus.FAILED}
        record.status = JobStatus.RETRYING
        record.error = error[:1000]
        record.updated_at = utc_now()
        saved = self.store.compare_and_save(
            record,
            expected_statuses=expected_statuses,
            expected_attempt=record.attempts,
            claim_token=claim_token,
            release_claim=True,
        )
        if not saved:
            return self._require(job_id)
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
        """Гарантирует перевод задания в финальное состояние ошибки с фиксацией причины сбоя."""
        record = self._require(job_id)
        return self._finish(
            record,
            JobStatus.FAILED,
            error,
            claim_token=self._claim_tokens.pop(job_id, None),
        )

    def _record_result_events(self, record: JobRecord) -> None:
        """Фиксирует в хранилище полную историю событий выполнения задания, чтобы обеспечить аудит и трассировку всех этапов и пересчётов результата."""
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

    def _finish(
        self,
        record: JobRecord,
        status: JobStatus,
        error: str,
        *,
        claim_token: str | None = None,
    ) -> JobRecord:
        """Сохраняет финальный статус задания и добавляет соответствующее событие в журнал выполнения."""
        original_status = record.status
        original_attempt = record.attempts
        record.status = status
        record.error = error[:1000]
        record.updated_at = utc_now()
        record.finished_at = utc_now()
        saved = self.store.compare_and_save(
            record,
            expected_statuses={original_status},
            expected_attempt=original_attempt,
            claim_token=claim_token,
            release_claim=True,
        )
        if not saved:
            return self._require(record.job.id)
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
        """Гарантирует, что задание с указанным идентификатором существует, иначе выбрасывает исключение для явного контроля потока."""
        record = self.store.get(job_id)
        if record is None:
            raise JobNotFoundError(f"Задание не найдено: {job_id}")
        return record


class OrchestrationService:
    """Обеспечивает единый интерфейс для постановки, мониторинга и управления заданиями в распределённой оркестрации."""

    def __init__(
        self,
        config: AgentAppConfig,
        *,
        store: JobStore | None = None,
        executor_factory: ExecutorFactory | None = None,
        dispatcher: CeleryJobDispatcher | None = None,
    ):
        """Готовит экземпляр к приёму и обработке заданий, настраивая все зависимости и владение ресурсами согласно конфигурации."""
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
        """Гарантирует однократную постановку задания в очередь с учётом идемпотентности и лимитов, либо возвращает существующую запись."""
        if not self.config.orchestration.enabled:
            raise RuntimeError("Оркестрация отключена в конфигурации")
        if job.idempotency_key:
            existing = self._claim_idempotency(job)
            if existing is not None:
                return JobSubmission(record=existing, deduplicated=True)

        record = JobRecord(job=job)
        if not self.store.create_if_capacity(
            record,
            max_pending=self.config.orchestration.max_pending_jobs,
        ):
            raise QueueCapacityError(
                "Очередь достигла max_pending_jobs="
                f"{self.config.orchestration.max_pending_jobs}"
            )
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
            record = self.store.set_task_id(job.id, task_id) or record
        else:
            record = JobRunner(
                self.config,
                self.store,
                self._executor(),
            ).run(job)
        return JobSubmission(record=record)

    def _claim_idempotency(self, job: OrchestrationJob) -> JobRecord | None:
        """Закрепляет ключ, восстанавливая stale binding без race condition."""
        key = self._scoped_idempotency_key(job)
        ttl = self.config.orchestration.idempotency_ttl_seconds
        for _ in range(4):
            claimed_id = self.store.claim_idempotency(key, job.id, ttl)
            if claimed_id == job.id:
                return None
            existing = self.store.get(claimed_id)
            if existing is not None:
                return existing
            if self.store.rebind_idempotency(
                key,
                expected_job_id=claimed_id,
                new_job_id=job.id,
                ttl_seconds=ttl,
            ):
                return None
        raise RuntimeError("Не удалось атомарно восстановить idempotency binding")

    @staticmethod
    def _scoped_idempotency_key(job: OrchestrationJob) -> str:
        """Изолирует пользовательские ключи идемпотентности без раскрытия user_id в backend."""
        raw = f"{job.user_id}\0{job.idempotency_key}".encode("utf-8")
        return "user:" + hashlib.sha256(raw).hexdigest()

    def get(self, job_id: str) -> JobRecord:
        """Гарантирует получение актуального состояния задания по идентификатору или явную ошибку при отсутствии."""
        record = self.store.get(job_id)
        if record is None:
            raise JobNotFoundError(f"Задание не найдено: {job_id}")
        return record

    def events(self, job_id: str) -> list[JobEvent]:
        """Возвращает полную хронологию событий по заданию, гарантируя целостность истории для аудита и отладки."""
        self.get(job_id)
        return self.store.events(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        """Гарантирует корректную отмену задания с фиксацией статуса и событий, либо возвращает финальное состояние, если отмена невозможна."""
        record = self.get(job_id)
        if record.status.terminal:
            return record
        if self.dispatcher is not None and record.task_id:
            self.dispatcher.cancel(record.task_id)
        record.status = JobStatus.CANCELLED
        record.finished_at = utc_now()
        record.updated_at = utc_now()
        record.error = "Задание отменено пользователем"
        saved = self.store.compare_and_save(
            record,
            expected_statuses={
                JobStatus.QUEUED,
                JobStatus.RUNNING,
                JobStatus.RETRYING,
            },
            expected_attempt=record.attempts,
            release_claim=True,
        )
        if not saved:
            return self.get(job_id)
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
        """Блокирует выполнение до завершения задания или истечения таймаута, гарантируя возврат финального состояния или ошибку ожидания."""
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
        """Гарантирует получение актуального статуса очереди и доступности воркеров, либо диагностическую ошибку при сбое."""
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
        """Гарантирует корректное освобождение всех ресурсов оркестрации и завершение работы с хранилищем состояния."""
        if self._owned_runtime is not None:
            self._owned_runtime.close()
            self._owned_runtime = None
        self.store.close()

    def _build_store(self) -> JobStore:
        """Обеспечивает создание хранилища заданий, подходящего для выбранного backend, и выбрасывает ошибку при отсутствии необходимой конфигурации."""
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
        """Гарантирует получение готового к запуску исполнителя шагов с корректно инициализированным окружением."""
        if self.executor_factory is not None:
            return self.executor_factory()
        executor, runtime = runtime_executor(self.config)
        self._owned_runtime = runtime
        return executor

    def _mark_submission_failed(self, record: JobRecord, error: str) -> None:
        """Фиксирует неудачную отправку задания с сохранением причины ошибки и генерацией события для аудита."""
        record.status = JobStatus.FAILED
        record.error = error[:1000]
        record.updated_at = utc_now()
        record.finished_at = utc_now()
        saved = self.store.compare_and_save(
            record,
            expected_statuses={JobStatus.QUEUED},
            expected_attempt=record.attempts,
            release_claim=True,
        )
        if not saved:
            return
        self.store.append_event(
            JobEvent(
                job_id=record.job.id,
                kind="failed",
                status=JobStatus.FAILED,
                message=record.error,
            )
        )


def callback_executor(
    callback: Callable[[str, str, str, str], Any],
) -> ExecutorFactory:
    """Гарантирует создание фабрики исполнителя шагов, вызывающей пользовательский callback для интеграции с внешними системами."""
    return lambda: MultiAgentStepExecutor(callback)


def result_has_retryable_failure(record: JobRecord) -> bool:
    """Гарантирует определение наличия в результате задания шагов, которые завершились ошибкой и допускают повторную попытку."""
    return bool(
        record.result
        and any(
            step.retryable and step.status in {StepStatus.FAILED, StepStatus.TIMED_OUT}
            for step in record.result.step_results
        )
    )
