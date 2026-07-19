"""Интеграция с Camunda для распределённой оркестрации."""

from __future__ import annotations

import asyncio
from typing import Any

from camunda_orchestration_sdk import (
    CamundaAsyncClient,
    CamundaClient,
    ConnectedJobContext,
    JobError,
    JobFailure,
    ProcessCreationById,
    ProcessInstanceCreationInstructionByIdVariables,
    WorkerConfig,
)

from agent_app.config import AgentAppConfig
from agent_app.orchestration.models import (
    JobStatus,
    OrchestrationJob,
    OrchestrationPattern,
)
from agent_app.orchestration.service import OrchestrationService


def deploy_process(config: AgentAppConfig) -> dict[str, Any]:
    """Гарантирует регистрацию BPMN-процесса в Camunda и сообщает вызывающему коду идентификаторы развёрнутых ресурсов или ошибку при отсутствии файла."""
    process_path = config.orchestration.camunda.process_path
    if not process_path.is_file():
        raise FileNotFoundError(f"BPMN-файл не найден: {process_path}")
    with CamundaClient() as client:
        result = client.deploy_resources_from_files([process_path])
    return _to_dict(result)


def start_process(
    config: AgentAppConfig,
    *,
    user_id: str,
    session_id: str,
    message: str,
    risk_level: str = "medium",
    priority: str = "normal",
) -> dict[str, Any]:
    """Запускает экземпляр процесса в Camunda с заданными переменными и возвращает идентификатор созданного процесса для отслеживания."""
    variables = ProcessInstanceCreationInstructionByIdVariables.from_dict(
        {
            "userId": user_id,
            "sessionId": session_id,
            "message": message,
            "riskLevel": risk_level,
            "priority": priority,
        }
    )
    request = ProcessCreationById(
        process_definition_id=config.orchestration.camunda.process_id,
        variables=variables,
    )
    with CamundaClient() as client:
        result = client.create_process_instance(data=request)
    return _to_dict(result)


class CamundaAgentWorker:
    """Обеспечивает асинхронную обработку задач Camunda с гарантией корректного распределения ролей и управления жизненным циклом воркера."""

    def __init__(
        self,
        config: AgentAppConfig,
        *,
        service: OrchestrationService | None = None,
    ):
        """Готовит экземпляр к запуску воркера, обеспечивая владение сервисом оркестрации и доступ к конфигурации."""
        self.config = config
        self.service = service or OrchestrationService(config)
        self._owns_service = service is None

    async def run(self) -> None:
        """Гарантирует регистрацию всех обработчиков задач Camunda и запуск асинхронного цикла обработки до остановки."""
        worker_config = self.config.orchestration.camunda
        timeout_ms = worker_config.worker_timeout_seconds * 1000
        poll_timeout_ms = worker_config.poll_request_timeout_seconds * 1000
        async with CamundaAsyncClient() as client:
            registrations = (
                (worker_config.job_type_validate, self.validate_request),
                (worker_config.job_type_classify, self.classify_risk),
                (worker_config.job_type_agent, self.run_agent),
                (worker_config.job_type_verify, self.verify_result),
            )
            for job_type, callback in registrations:
                client.create_job_worker(
                    WorkerConfig(
                        job_type=job_type,
                        job_timeout_milliseconds=timeout_ms,
                        request_timeout_milliseconds=poll_timeout_ms,
                        max_concurrent_jobs=1,
                        worker_name=f"rag-{job_type}",
                    ),
                    callback,
                    execution_strategy="async",
                )
            await client.run_workers()

    def close(self) -> None:
        """Гарантирует корректное освобождение ресурсов сервиса оркестрации при владении им."""
        if self._owns_service:
            self.service.close()

    async def validate_request(self, job: ConnectedJobContext) -> dict[str, Any]:
        """Проверяет, что входные переменные задачи содержат обязательные поля и сообщает об ошибке при их отсутствии."""
        variables = job.variables.to_dict()
        missing = [
            name
            for name in ("message", "userId", "sessionId")
            if not str(variables.get(name, "")).strip()
        ]
        if missing:
            raise JobError(
                "INVALID_SUPPORT_REQUEST",
                "Не заполнены обязательные поля: " + ", ".join(missing),
            )
        return {"requestValid": True}

    async def classify_risk(self, job: ConnectedJobContext) -> dict[str, Any]:
        """Определяет уровень риска заявки и необходимость согласования, гарантируя корректную классификацию для дальнейшей маршрутизации."""
        variables = job.variables.to_dict()
        supplied = str(variables.get("riskLevel", "")).lower()
        if supplied in {"low", "medium", "high"}:
            risk = supplied
        else:
            message = str(variables.get("message", "")).lower()
            high_markers = ("удал", "production", "прод", "секрет", "доступ")
            risk = "high" if any(item in message for item in high_markers) else "medium"
        return {
            "riskLevel": risk,
            "requiresApproval": risk == "high",
        }

    async def run_agent(self, job: ConnectedJobContext) -> dict[str, Any]:
        """Гарантирует выполнение агентской задачи с ожиданием результата и сообщает вызывающему коду итоговый статус и ответ агента."""
        variables = job.variables.to_dict()
        orchestration_job = OrchestrationJob(
            user_id=str(variables["userId"]),
            session_id=str(variables["sessionId"]),
            message=str(variables["message"]),
            pattern=OrchestrationPattern.DYNAMIC,
            priority=str(variables.get("priority", "normal")),
            risk_level=str(variables.get("riskLevel", "medium")),
            idempotency_key=f"camunda-{job.process_instance_key}",
            max_plan_revisions=2,
            metadata={
                "camunda_process_instance_key": str(job.process_instance_key),
                "camunda_element_id": str(job.element_id),
            },
        )
        submission = await asyncio.to_thread(self.service.submit, orchestration_job)
        record = submission.record
        effective_job_id = record.job.id
        if not record.status.terminal:
            record = await asyncio.to_thread(
                self.service.wait,
                effective_job_id,
                timeout_seconds=(
                    self.config.orchestration.camunda.worker_timeout_seconds
                ),
            )
        if record.status != JobStatus.COMPLETED or record.result is None:
            raise JobFailure(
                record.error or f"Агент завершился со статусом {record.status.value}",
                retries=max(job.retries - 1, 0),
                retry_back_off=5_000,
            )
        return {
            "orchestrationJobId": effective_job_id,
            "agentStatus": record.status.value,
            "agentAnswer": record.result.answer,
            "planVersion": record.result.plan.version,
            "planRevisions": len(record.result.revisions),
        }

    async def verify_result(self, job: ConnectedJobContext) -> dict[str, Any]:
        """Обеспечивает проверку корректности и полноты ответа агента для принятия решения о завершении задания в распределённой оркестрации."""
        variables = job.variables.to_dict()
        answer = str(variables.get("agentAnswer", "")).strip()
        passed = (
            variables.get("agentStatus") == JobStatus.COMPLETED.value
            and len(answer) >= 20
        )
        return {
            "verificationPassed": passed,
            "verificationReason": (
                "Ответ агента получен и прошёл минимальную проверку"
                if passed
                else "Ответ отсутствует, слишком короткий или задание не завершено"
            ),
        }


def _to_dict(value: Any) -> dict[str, Any]:
    """Гарантирует сериализацию результата Camunda в словарь для унифицированной передачи между подсистемами."""
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        payload = converter()
        if isinstance(payload, dict):
            return payload
    return {"result": str(value)}
