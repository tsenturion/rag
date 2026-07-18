from __future__ import annotations

import random
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from billiard.exceptions import SoftTimeLimitExceeded
from celery import shared_task
from celery.exceptions import Reject, Retry

from agent_app.config import AgentAppConfig, load_agent_config
from agent_app.orchestration.errors import TransientOrchestrationError
from agent_app.orchestration.executors import runtime_executor
from agent_app.orchestration.models import (
    JobStatus,
    OrchestrationJob,
    PlanStep,
    StepResult,
)
from agent_app.orchestration.queue import EXECUTE_JOB_TASK
from agent_app.orchestration.service import JobRunner, result_has_retryable_failure
from agent_app.orchestration.store import RedisJobStore

_EXECUTORS: dict[str, tuple[Any, Any]] = {}
_EXECUTOR_LOCK = threading.RLock()


@shared_task(bind=True, name=EXECUTE_JOB_TASK, acks_late=True)
def execute_job_task(
    self,
    *,
    job_payload: dict[str, Any],
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    config = _load_worker_config(config_payload)
    job = OrchestrationJob.model_validate(job_payload)
    state_url = _required_state_url(config)
    store = RedisJobStore(
        state_url,
        state_ttl_seconds=config.orchestration.state_ttl_seconds,
        event_limit=config.orchestration.event_limit,
    )
    existing = store.get(job.id)
    if existing is not None and existing.status == JobStatus.CANCELLED:
        store.close()
        return existing.model_dump(mode="json")
    if job.expired:
        runner = JobRunner(config, store, _UnreachableExecutor())
        try:
            return runner.run(job).model_dump(mode="json")
        finally:
            store.close()
    executor, _runtime = _cached_executor(config)
    runner = JobRunner(config, store, executor)
    try:
        record = runner.run(job)
        if record.status == JobStatus.FAILED and result_has_retryable_failure(record):
            if self.request.retries < config.orchestration.max_retries:
                countdown = _countdown(config, self.request.retries)
                runner.mark_retry(
                    job.id,
                    record.error or "Временная ошибка",
                    countdown=countdown,
                )
                raise self.retry(
                    exc=TransientOrchestrationError(record.error or "Временная ошибка"),
                    countdown=countdown,
                    max_retries=config.orchestration.max_retries,
                )
            raise Reject(record.error or "Лимит повторов исчерпан", requeue=False)
        return record.model_dump(mode="json")
    except (Retry, Reject):
        raise
    except SoftTimeLimitExceeded as exc:
        error = "Превышен soft time limit Celery task"
        if self.request.retries < config.orchestration.max_retries:
            countdown = _countdown(config, self.request.retries)
            runner.mark_retry(job.id, error, countdown=countdown)
            raise self.retry(
                exc=exc,
                countdown=countdown,
                max_retries=config.orchestration.max_retries,
            ) from exc
        runner.mark_failed(job.id, error)
        raise Reject(error, requeue=False) from exc
    except TransientOrchestrationError as exc:
        if self.request.retries < config.orchestration.max_retries:
            countdown = _countdown(config, self.request.retries)
            runner.mark_retry(job.id, str(exc), countdown=countdown)
            raise self.retry(
                exc=exc,
                countdown=countdown,
                max_retries=config.orchestration.max_retries,
            ) from exc
        runner.mark_failed(job.id, str(exc))
        raise Reject(str(exc), requeue=False) from exc
    except Exception as exc:
        return runner.mark_failed(job.id, str(exc)).model_dump(mode="json")
    finally:
        store.close()


def _cached_executor(config: AgentAppConfig) -> tuple[Any, Any]:
    key = (
        f"{config.agent.provider}:{config.agent.model}:"
        f"{config.multi_agent.checkpoint_path}"
    )
    with _EXECUTOR_LOCK:
        existing = _EXECUTORS.get(key)
        if existing is not None:
            return existing
        created = runtime_executor(config)
        _EXECUTORS[key] = created
        return created


def _required_state_url(config: AgentAppConfig) -> str:
    import os

    value = os.getenv(config.orchestration.state_store_url_env)
    if not value:
        raise RuntimeError(
            f"Не задана переменная {config.orchestration.state_store_url_env}"
        )
    return value


def _load_worker_config(config_payload: dict[str, Any]) -> AgentAppConfig:
    import os

    config_path = os.getenv("SUPPORT_AGENT_CONFIG")
    if config_path:
        return load_agent_config(Path(config_path))
    return AgentAppConfig.model_validate(config_payload)


def _countdown(config: AgentAppConfig, retries: int) -> int:
    base = min(
        config.orchestration.retry_backoff_seconds * (2**retries),
        config.orchestration.retry_backoff_max_seconds,
    )
    if not config.orchestration.retry_jitter:
        return base
    return max(1, int(base * random.uniform(0.75, 1.25)))


class _UnreachableExecutor:
    def execute(
        self,
        step: PlanStep,
        job: OrchestrationJob,
        context: Mapping[str, StepResult],
    ) -> StepResult:
        del step, job, context
        raise RuntimeError("Executor не должен вызываться для завершённого задания")
