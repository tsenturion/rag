"""Интеграция с Camunda для распределённой оркестрации."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

from camunda_orchestration_sdk import (
    CamundaAsyncClient,
    CamundaClient,
    ConnectedJobContext,
    JobError,
    JobFailure,
    ProcessCreationById,
    ProcessInstanceCreationInstructionByIdVariables,
    UserTaskCompletionRequest,
    UserTaskCompletionRequestVariables,
    WorkerConfig,
)
from camunda_orchestration_sdk.errors import NotFoundError

from agent_app.config import AgentAppConfig
from agent_app.orchestration.models import (
    JobStatus,
    OrchestrationJob,
    OrchestrationPattern,
)

if TYPE_CHECKING:
    from agent_app.orchestration.service import OrchestrationService

LOGGER = logging.getLogger(__name__)
_BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
_ZEEBE_NS = "http://camunda.org/schema/zeebe/1.0"
_TERMINAL_PROCESS_STATES = {"COMPLETED", "TERMINATED", "CANCELED"}


def validate_process_model(config: AgentAppConfig) -> dict[str, Any]:
    """Проверяет, что BPMN и связанные формы существуют, а все service task имеют зарегистрированный обработчик."""
    camunda = config.orchestration.camunda
    if not camunda.process_path.is_file():
        raise FileNotFoundError(f"BPMN-файл не найден: {camunda.process_path}")
    missing_forms = [str(path) for path in camunda.form_paths if not path.is_file()]
    if missing_forms:
        raise FileNotFoundError(
            "Не найдены связанные Camunda Forms: " + ", ".join(missing_forms)
        )

    root = ET.parse(camunda.process_path).getroot()
    process = root.find(f".//{{{_BPMN_NS}}}process")
    if process is None:
        raise ValueError("BPMN не содержит исполняемый process")
    process_id = str(process.get("id", ""))
    if process_id != camunda.process_id:
        raise ValueError(
            f"BPMN process id {process_id!r} не совпадает с {camunda.process_id!r}"
        )
    if process.get("isExecutable") != "true":
        raise ValueError(f"BPMN process {process_id!r} не помечен isExecutable=true")

    form_ids: set[str] = set()
    for path in camunda.form_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        form_id = str(payload.get("id", "")).strip()
        if not form_id:
            raise ValueError(f"Camunda Form не содержит id: {path}")
        form_ids.add(form_id)
    referenced_form_ids = {
        str(node.get("formId"))
        for node in process.findall(f".//{{{_ZEEBE_NS}}}formDefinition")
        if node.get("formId")
    }
    unresolved_forms = sorted(referenced_form_ids - form_ids)
    if unresolved_forms:
        raise ValueError(
            "BPMN ссылается на отсутствующие Camunda Forms: "
            + ", ".join(unresolved_forms)
        )

    model_job_types = {
        str(node.get("type"))
        for node in process.findall(f".//{{{_ZEEBE_NS}}}taskDefinition")
        if node.get("type")
    }
    registered_job_types = set(_configured_job_types(config).values())
    missing_workers = sorted(model_job_types - registered_job_types)
    missing_tasks = sorted(registered_job_types - model_job_types)
    if missing_workers:
        raise ValueError(
            "В BPMN есть service task без worker: " + ", ".join(missing_workers)
        )
    if missing_tasks:
        raise ValueError(
            "В конфигурации есть worker без service task: " + ", ".join(missing_tasks)
        )
    return {
        "process_id": process_id,
        "job_types": sorted(model_job_types),
        "form_ids": sorted(form_ids),
        "resources": [
            str(camunda.process_path),
            *(str(path) for path in camunda.form_paths),
        ],
    }


def deploy_process(
    config: AgentAppConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Разворачивает BPMN и формы только при изменении модели, не создавая лишние версии процесса."""
    model = validate_process_model(config)
    camunda = config.orchestration.camunda
    resources = [camunda.process_path, *camunda.form_paths]
    with _sync_client(config) as client:
        latest = _latest_process_definition(client, camunda.process_id)
        if latest is not None and not force:
            deployed_xml = client.get_process_definition_xml(
                latest["processDefinitionKey"]
            )
            local_xml = camunda.process_path.read_text(encoding="utf-8")
            if _normalized_resource(deployed_xml) == _normalized_resource(local_xml):
                return {
                    "deployed": False,
                    "reason": "unchanged",
                    "process_definition": latest,
                    "model": model,
                }
        result = client.deploy_resources_from_files(resources)
    payload = _to_dict(result)
    payload.update({"deployed": True, "model": model})
    return payload


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
    with _sync_client(config) as client:
        result = client.create_process_instance(data=request)
    return _to_dict(result)


def process_status(config: AgentAppConfig, process_instance_key: str) -> dict[str, Any]:
    """Возвращает состояние, инциденты и активные ручные задачи конкретного экземпляра процесса."""
    with _sync_client(config) as client:
        process = _to_dict(client.get_process_instance(process_instance_key))
        incidents: list[dict[str, Any]] = []
        if process.get("hasIncident"):
            incidents = _to_dict(
                client.search_process_instance_incidents(process_instance_key)
            ).get("items", [])
        user_tasks: list[dict[str, Any]] = []
        if process.get("state") == "ACTIVE" and not incidents:
            user_tasks = [
                item
                for item in _to_dict(client.search_user_tasks()).get("items", [])
                if str(item.get("processInstanceKey")) == str(process_instance_key)
                and item.get("state") == "CREATED"
            ]
    return {
        "process": process,
        "incidents": incidents,
        "user_tasks": user_tasks,
    }


def wait_for_process(
    config: AgentAppConfig,
    process_instance_key: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Ожидает терминального состояния, ручной задачи или инцидента и возвращает наблюдаемый результат процесса."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            status = process_status(config, process_instance_key)
        except NotFoundError:
            # Команда создания подтверждается журналом Zeebe раньше, чем RDBMS
            # secondary storage начинает отдавать экземпляр через Search API.
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Процесс {process_instance_key} не появился в secondary storage "
                    f"за {timeout_seconds:g} с"
                ) from None
            time.sleep(config.orchestration.camunda.poll_interval_seconds)
            continue
        process = status["process"]
        if (
            str(process.get("state")) in _TERMINAL_PROCESS_STATES
            or bool(process.get("hasIncident"))
            or bool(status["user_tasks"])
        ):
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Процесс {process_instance_key} не достиг наблюдаемого состояния "
                f"за {timeout_seconds:g} с"
            )
        time.sleep(config.orchestration.camunda.poll_interval_seconds)


def complete_approval(
    config: AgentAppConfig,
    process_instance_key: str,
    *,
    approved: bool,
) -> dict[str, Any]:
    """Завершает ручное согласование и записывает решение, используемое следующим BPMN gateway."""
    with _sync_client(config) as client:
        tasks = [
            item
            for item in _to_dict(client.search_user_tasks()).get("items", [])
            if str(item.get("processInstanceKey")) == str(process_instance_key)
            and item.get("elementId") == "approval"
            and item.get("state") == "CREATED"
        ]
        if len(tasks) != 1:
            raise LookupError(
                f"Для процесса {process_instance_key} ожидалась одна активная задача "
                f"approval, найдено: {len(tasks)}"
            )
        task = tasks[0]
        variables = UserTaskCompletionRequestVariables.from_dict(
            {"approvalGranted": approved}
        )
        client.complete_user_task(
            task["userTaskKey"],
            data=UserTaskCompletionRequest(
                variables=variables,
                action="approve" if approved else "reject",
            ),
        )
    return {
        "process_instance_key": str(process_instance_key),
        "user_task_key": str(task["userTaskKey"]),
        "approved": approved,
    }


def diagnose_camunda(config: AgentAppConfig) -> dict[str, Any]:
    """Проверяет REST-соединение, локальные ресурсы и актуальность развёрнутой версии процесса."""
    model = validate_process_model(config)
    camunda = config.orchestration.camunda
    with _sync_client(config) as client:
        topology = _to_dict(client.get_topology())
        latest = _latest_process_definition(client, camunda.process_id)
        deployment_current = False
        if latest is not None:
            deployed_xml = client.get_process_definition_xml(
                latest["processDefinitionKey"]
            )
            deployment_current = _normalized_resource(
                deployed_xml
            ) == _normalized_resource(camunda.process_path.read_text(encoding="utf-8"))
    return {
        "connected": True,
        "rest_address": os.getenv("CAMUNDA_REST_ADDRESS", "http://localhost:8080/v2"),
        "gateway_version": topology.get("gatewayVersion"),
        "brokers": topology.get("brokers", []),
        "model": model,
        "process_deployed": latest is not None,
        "bpmn_current": deployment_current,
        "process_definition": latest,
    }


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
        owns_service = service is None
        if service is None:
            # Deploy/status CLI не должны загружать LangGraph, LLM и vector store.
            from agent_app.orchestration.service import OrchestrationService

            service = OrchestrationService(config)
        self.service = service
        self._owns_service = owns_service

    async def run(self) -> None:
        """Гарантирует регистрацию всех обработчиков задач Camunda и запуск асинхронного цикла обработки до остановки."""
        worker_config = self.config.orchestration.camunda
        timeout_ms = worker_config.worker_timeout_seconds * 1000
        poll_timeout_ms = worker_config.poll_request_timeout_seconds * 1000
        validate_process_model(self.config)
        async with _async_client(self.config) as client:
            registrations = (
                (worker_config.job_type_validate, self.validate_request),
                (worker_config.job_type_classify, self.classify_risk),
                (worker_config.job_type_agent, self.run_agent),
                (worker_config.job_type_verify, self.verify_result),
                (worker_config.job_type_notify, self.handle_invalid_request),
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
            heartbeat = asyncio.create_task(self._health_heartbeat())
            try:
                await client.run_workers()
            finally:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
                self._health_path().unlink(missing_ok=True)

    async def _health_heartbeat(self) -> None:
        """Обновляет признак живого event loop, используемый healthcheck контейнера."""
        path = self._health_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            path.write_text(
                datetime.now(timezone.utc).isoformat(),
                encoding="utf-8",
            )
            await asyncio.sleep(5)

    @staticmethod
    def _health_path() -> Path:
        """Разрешает путь heartbeat без привязки к рабочему каталогу процесса."""
        return Path(
            os.getenv("CAMUNDA_WORKER_HEALTH_FILE", "/tmp/camunda-worker.health")
        )

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

    async def handle_invalid_request(self, job: ConnectedJobContext) -> dict[str, Any]:
        """Фиксирует отклонение некорректного запроса и завершает BPMN error path без зависшей задачи."""
        variables = job.variables.to_dict()
        missing = [
            name
            for name in ("message", "userId", "sessionId")
            if not str(variables.get(name, "")).strip()
        ]
        reason = (
            "Не заполнены обязательные поля: " + ", ".join(missing)
            if missing
            else "Запрос отклонён обработчиком валидации"
        )
        LOGGER.warning(
            "Camunda отклонила запрос process_instance=%s: %s",
            job.process_instance_key,
            reason,
        )
        return {
            "requestStatus": "rejected",
            "rejectionReason": reason,
            "errorHandled": True,
            "errorHandledAt": datetime.now(timezone.utc).isoformat(),
        }


def _configured_job_types(config: AgentAppConfig) -> dict[str, str]:
    """Возвращает единственный источник истины для типов BPMN-задач, обслуживаемых Python worker."""
    camunda = config.orchestration.camunda
    return {
        "validate": camunda.job_type_validate,
        "classify": camunda.job_type_classify,
        "agent": camunda.job_type_agent,
        "verify": camunda.job_type_verify,
        "notify": camunda.job_type_notify,
    }


def _client_httpx_args(config: AgentAppConfig) -> dict[str, Any]:
    """Не позволяет системному proxy перехватывать локальный Camunda REST, сохраняя явную настройку для удалённого кластера."""
    return {
        "httpx_args": {
            "trust_env": config.orchestration.camunda.use_environment_proxy,
        }
    }


def _sync_client(config: AgentAppConfig) -> CamundaClient:
    """Создаёт синхронный SDK-клиент с сетевой политикой выбранного профиля."""
    return CamundaClient(**_client_httpx_args(config))


def _async_client(config: AgentAppConfig) -> CamundaAsyncClient:
    """Создаёт асинхронный SDK-клиент для long-poll job workers."""
    return CamundaAsyncClient(**_client_httpx_args(config))


def _latest_process_definition(
    client: CamundaClient,
    process_id: str,
) -> dict[str, Any] | None:
    """Находит последнюю развёрнутую версию процесса среди результатов Camunda Search API."""
    items = _to_dict(client.search_process_definitions()).get("items", [])
    matching = [item for item in items if item.get("processDefinitionId") == process_id]
    if not matching:
        return None
    return max(matching, key=lambda item: int(item.get("version", 0)))


def _normalized_resource(value: str) -> str:
    """Нормализует только окончания строк, сохраняя содержательное сравнение XML-ресурса."""
    return value.replace("\r\n", "\n").strip()


def _to_dict(value: Any) -> dict[str, Any]:
    """Гарантирует сериализацию результата Camunda в словарь для унифицированной передачи между подсистемами."""
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        payload = converter()
        if isinstance(payload, dict):
            return payload
    return {"result": str(value)}
