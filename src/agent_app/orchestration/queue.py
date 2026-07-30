"""Настройка очередей задач для распределённой оркестрации."""

from __future__ import annotations

import os
from typing import Any

from celery import Celery
from kombu import Exchange, Queue

from agent_app.config import OrchestrationConfig
from agent_app.orchestration.models import JobPriority, OrchestrationJob

EXECUTE_JOB_TASK = "agent_app.orchestration.execute_job"


def create_celery_app(config: OrchestrationConfig | None = None) -> Celery:
    """Инициализирует и конфигурирует Celery-приложение с очередями и параметрами для надёжной и приоритетной обработки задач оркестрации."""
    broker_url = _env(
        config.broker_url_env if config else "ORCHESTRATION_BROKER_URL",
        "amqp://rag:rag-dev@127.0.0.1:5672//",
    )
    backend_url = _env(
        config.result_backend_url_env if config else "ORCHESTRATION_RESULT_BACKEND_URL",
        "redis://127.0.0.1:6379/1",
    )
    queue_high = config.queue_high if config else "agent.high.quorum"
    queue_default = config.queue_default if config else "agent.default.quorum"
    queue_low = config.queue_low if config else "agent.low.quorum"
    queue_dead = config.queue_dead_letter if config else "agent.dead_letter.quorum"
    prefetch = config.worker_prefetch_multiplier if config else 1
    soft_limit = config.task_soft_time_limit_seconds if config else 300
    hard_limit = config.task_time_limit_seconds if config else 330

    app = Celery(
        "rag-orchestration",
        broker=broker_url,
        backend=backend_url,
        include=["agent_app.orchestration.tasks"],
    )
    task_exchange = Exchange("agent.tasks", type="direct", durable=True)
    dead_exchange = Exchange("agent.dead_letter", type="direct", durable=True)
    queue_arguments = {
        # Quorum queues заставляют Celery использовать per-consumer QoS и не
        # зависят от удаляемого RabbitMQ global_qos. Отдельные high/default/low
        # очереди сохраняют трёхуровневую маршрутизацию задания.
        "x-queue-type": "quorum",
        "x-dead-letter-exchange": dead_exchange.name,
        "x-dead-letter-routing-key": "dead",
    }
    app.conf.update(
        task_queues=(
            Queue(
                queue_high,
                exchange=task_exchange,
                routing_key="high",
                durable=True,
                queue_arguments=queue_arguments,
            ),
            Queue(
                queue_default,
                exchange=task_exchange,
                routing_key="default",
                durable=True,
                queue_arguments=queue_arguments,
            ),
            Queue(
                queue_low,
                exchange=task_exchange,
                routing_key="low",
                durable=True,
                queue_arguments=queue_arguments,
            ),
            Queue(
                queue_dead,
                exchange=dead_exchange,
                routing_key="dead",
                durable=True,
                queue_arguments={"x-queue-type": "quorum"},
            ),
        ),
        task_default_queue=queue_default,
        task_default_exchange=task_exchange.name,
        task_default_exchange_type="direct",
        task_default_routing_key="default",
        task_default_priority=JobPriority.NORMAL.broker_value,
        task_inherit_parent_priority=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        enable_utc=True,
        timezone="UTC",
        task_acks_late=True,
        task_acks_on_failure_or_timeout=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=prefetch,
        worker_detect_quorum_queues=True,
        worker_enable_prefetch_count_reduction=True,
        worker_cancel_long_running_tasks_on_connection_loss=True,
        task_track_started=True,
        task_soft_time_limit=soft_limit,
        task_time_limit=hard_limit,
        broker_connection_retry_on_startup=True,
        broker_transport_options={"confirm_publish": True},
        control_queue_durable=False,
        control_queue_exclusive=True,
        event_queue_durable=False,
        event_queue_exclusive=True,
        result_expires=(config.state_ttl_seconds if config else 86_400),
    )
    return app


def declare_celery_topology(app: Celery) -> None:
    """Объявляет exchanges и все очереди, включая не потребляемую worker DLQ."""
    with app.connection_for_write() as connection:
        channel = connection.channel()
        try:
            for queue in app.conf.task_queues:
                queue.bind(channel).declare()
        finally:
            channel.close()


class CeleryJobDispatcher:
    """Обеспечивает управление жизненным циклом задач оркестрации через Celery, гарантируя корректную отправку, отмену и мониторинг заданий."""

    def __init__(self, config: OrchestrationConfig, app: Celery | None = None):
        """Готовит диспетчер задач с конфигурацией и Celery-приложением, обеспечивая готовность к отправке и управлению задачами."""
        self.config = config
        self.app = app or create_celery_app(config)

    def dispatch(self, job: OrchestrationJob, config_payload: dict[str, Any]) -> str:
        """Отправляет задачу в очередь Celery с учётом приоритета и ограничений времени, гарантируя её выполнение в распределённой системе."""
        queue, routing_key = self._route(job.priority)
        result = self.app.send_task(
            EXECUTE_JOB_TASK,
            kwargs={
                "job_payload": job.model_dump(mode="json"),
                "config_payload": config_payload,
            },
            task_id=job.id,
            queue=queue,
            routing_key=routing_key,
            priority=min(job.priority.broker_value, self.config.max_priority),
            soft_time_limit=self.config.task_soft_time_limit_seconds,
            time_limit=self.config.task_time_limit_seconds,
        )
        return result.id

    def cancel(self, task_id: str) -> None:
        """Отменяет выполнение задачи в Celery без принудительного завершения, позволяя корректно освободить ресурсы."""
        self.app.control.revoke(task_id, terminate=False)

    def workers(self) -> dict[str, Any]:
        """Гарантирует получение актуального состояния всех воркеров Celery для мониторинга и диагностики распределённой очереди."""
        inspector = self.app.control.inspect(timeout=1.0)
        return {
            "ping": inspector.ping() or {},
            "active": inspector.active() or {},
            "reserved": inspector.reserved() or {},
            "scheduled": inspector.scheduled() or {},
        }

    def _route(self, priority: JobPriority) -> tuple[str, str]:
        """Гарантирует маршрутизацию задания в очередь с нужным приоритетом согласно политике оркестрации."""
        if priority == JobPriority.HIGH:
            return self.config.queue_high, "high"
        if priority == JobPriority.LOW:
            return self.config.queue_low, "low"
        return self.config.queue_default, "default"


def _env(name: str, default: str) -> str:
    """Возвращает значение переменной окружения с запасным значением, обеспечивая устойчивость конфигурации к отсутствию переменных."""
    return os.getenv(name) or default


celery_app = create_celery_app()
