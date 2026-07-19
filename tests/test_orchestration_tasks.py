"""Проверки Celery-task orchestration без подключения к Redis и broker."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agent_app.config import AgentAppConfig, AgentConfig, OrchestrationConfig
from agent_app.orchestration.models import (
    JobRecord,
    JobStatus,
    OrchestrationJob,
    utc_now,
)
from agent_app.orchestration import tasks


def _config() -> AgentAppConfig:
    """Создаёт минимальную конфигурацию распределённого worker."""
    return AgentAppConfig(
        agent=AgentConfig(provider="local", model="test-model"),
        orchestration=OrchestrationConfig(
            enabled=True,
            backend="celery",
            max_retries=2,
            retry_backoff_seconds=3,
            retry_backoff_max_seconds=10,
            retry_jitter=False,
        ),
    )


def _job(**updates) -> OrchestrationJob:
    """Создаёт переносимый payload фонового задания."""
    payload = {
        "user_id": "engineer",
        "session_id": "incident",
        "message": "Проверь сервис",
        "pattern": "sequential",
    }
    payload.update(updates)
    return OrchestrationJob.model_validate(payload)


def _store(existing: JobRecord | None = None) -> Mock:
    """Возвращает RedisJobStore-совместимый mock с контролируемым состоянием."""
    store = Mock()
    store.get.return_value = existing
    return store


def test_execute_job_task_returns_completed_record_and_closes_store() -> None:
    """Проверяет основной worker path и гарантированное закрытие Redis-клиента."""
    config = _config()
    job = _job()
    record = JobRecord(job=job, status=JobStatus.COMPLETED)
    store = _store()
    runner = Mock()
    runner.run.return_value = record
    with (
        patch.object(tasks, "_load_worker_config", return_value=config),
        patch.object(tasks, "_required_state_url", return_value="redis://state"),
        patch.object(tasks, "RedisJobStore", return_value=store),
        patch.object(tasks, "_cached_executor", return_value=(object(), object())),
        patch.object(tasks, "JobRunner", return_value=runner),
    ):
        result = tasks.execute_job_task.run(
            job_payload=job.model_dump(mode="json"), config_payload={}
        )

    assert result["status"] == "completed"
    runner.run.assert_called_once_with(job)
    store.close.assert_called_once()


def test_execute_job_task_skips_cancelled_record() -> None:
    """Повторная delivery не выполняет уже отменённое задание."""
    config = _config()
    job = _job()
    cancelled = JobRecord(job=job, status=JobStatus.CANCELLED)
    store = _store(cancelled)
    with (
        patch.object(tasks, "_load_worker_config", return_value=config),
        patch.object(tasks, "_required_state_url", return_value="redis://state"),
        patch.object(tasks, "RedisJobStore", return_value=store),
        patch.object(tasks, "_cached_executor") as cached,
    ):
        result = tasks.execute_job_task.run(
            job_payload=job.model_dump(mode="json"), config_payload={}
        )

    assert result["status"] == "cancelled"
    cached.assert_not_called()
    store.close.assert_called_once()


def test_execute_job_task_marks_unexpected_failure() -> None:
    """Неповторяемая ошибка преобразуется в FAILED record, а не теряет состояние."""
    config = _config()
    job = _job()
    store = _store()
    failed = JobRecord(job=job, status=JobStatus.FAILED, error="provider failure")
    runner = Mock()
    runner.run.side_effect = RuntimeError("provider failure")
    runner.mark_failed.return_value = failed
    with (
        patch.object(tasks, "_load_worker_config", return_value=config),
        patch.object(tasks, "_required_state_url", return_value="redis://state"),
        patch.object(tasks, "RedisJobStore", return_value=store),
        patch.object(tasks, "_cached_executor", return_value=(object(), object())),
        patch.object(tasks, "JobRunner", return_value=runner),
    ):
        result = tasks.execute_job_task.run(
            job_payload=job.model_dump(mode="json"), config_payload={}
        )

    assert result["status"] == "failed"
    runner.mark_failed.assert_called_once_with(job.id, "provider failure")
    store.close.assert_called_once()


def test_worker_helpers_cache_runtime_and_validate_environment(monkeypatch) -> None:
    """Проверяет runtime cache, server config precedence и bounded backoff."""
    config = _config()
    tasks._EXECUTORS.clear()
    created = (object(), object())
    with patch.object(tasks, "runtime_executor", return_value=created) as factory:
        assert tasks._cached_executor(config) is created
        assert tasks._cached_executor(config) is created
    factory.assert_called_once()

    env_name = config.orchestration.state_store_url_env
    monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(RuntimeError, match=env_name):
        tasks._required_state_url(config)
    monkeypatch.setenv(env_name, "redis://localhost/0")
    assert tasks._required_state_url(config) == "redis://localhost/0"

    monkeypatch.delenv("SUPPORT_AGENT_CONFIG", raising=False)
    payload_config = tasks._load_worker_config(config.model_dump(mode="json"))
    assert payload_config.agent.model == "test-model"
    monkeypatch.setenv("SUPPORT_AGENT_CONFIG", "config/support_agent_openai.yaml")
    with patch.object(tasks, "load_agent_config", return_value=config) as loader:
        assert tasks._load_worker_config({}) is config
    loader.assert_called_once_with(Path("config/support_agent_openai.yaml"))

    assert tasks._countdown(config, 0) == 3
    assert tasks._countdown(config, 4) == 10
    jittered = config.model_copy(
        update={
            "orchestration": config.orchestration.model_copy(
                update={"retry_jitter": True}
            )
        }
    )
    with patch.object(tasks.random, "uniform", return_value=0.75):
        assert tasks._countdown(jittered, 0) == 2


def test_expired_job_uses_terminal_runner_without_runtime() -> None:
    """Просроченное задание завершается без создания тяжёлой LLM runtime."""
    config = _config()
    job = _job(deadline_at=utc_now() - timedelta(seconds=1))
    store = _store()
    expired = JobRecord(job=job, status=JobStatus.EXPIRED)
    runner = Mock()
    runner.run.return_value = expired
    with (
        patch.object(tasks, "_load_worker_config", return_value=config),
        patch.object(tasks, "_required_state_url", return_value="redis://state"),
        patch.object(tasks, "RedisJobStore", return_value=store),
        patch.object(tasks, "JobRunner", return_value=runner),
        patch.object(tasks, "_cached_executor") as cached,
    ):
        result = tasks.execute_job_task.run(
            job_payload=job.model_dump(mode="json"), config_payload={}
        )

    assert result["status"] == "expired"
    cached.assert_not_called()
    store.close.assert_called_once()
