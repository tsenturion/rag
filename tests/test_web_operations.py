"""Очередь браузерных операций: воспроизводимость, ограничения и реальные CLI-вызовы."""

from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from uuid import uuid4
import json
import time

import pytest
import yaml
from fastapi import HTTPException
from pydantic import ValidationError

from agent_app.database import DatabaseRuntime
from agent_app.service.operation_store import OperationRequest, OperationStore
from agent_app.service.operations import OperationService, public_data
from agent_app.service.web_config import WebConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def operation_service(tmp_path):
    """Временная очередь использует настоящие SQL-транзакции и каталог проекта."""
    database = DatabaseRuntime(backend="sqlite")
    config = WebConfig(
        sqlite_path=tmp_path / "web.sqlite",
        data_dir=tmp_path / "web",
        operations_catalog=ROOT / "config/web_operations.yaml",
    )
    store = OperationStore(database, config)
    store.initialize()
    service = OperationService(store)
    store.heartbeat("worker", list(service.catalog()))
    yield service
    database.close()


def request(profile="prepare", **values):
    """Новый ключ моделирует намеренное создание независимой операции."""
    return OperationRequest(profile=profile, idempotency_key=str(uuid4()), **values)


def complete(service: OperationService, record, token: str | None = None):
    """Завершает запись через настоящий lease, не запуская внешнюю CLI-команду."""
    worker_token = token or str(uuid4())
    claimed = service.store.claim(worker_token, [record.profile])
    assert claimed is not None and claimed.id == record.id
    service.store.finish(record.id, worker_token, status="completed", exit_code=0)
    return service.store.get(record.id, record.user_id)


class FakeProcess:
    """Минимальный управляемый subprocess для проверки состояний worker без CLI."""

    def __init__(self, returncode: int | None = None) -> None:
        """Задаёт исходный exit code и пустой stdout для контролируемого завершения worker."""
        self.returncode = returncode
        self.stdout = StringIO("")

    def poll(self) -> int | None:
        """Возвращает заданное тестом состояние дочернего процесса."""
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        """Имитирует немедленное ожидание уже остановленного процесса."""
        return self.returncode


def test_submit_deduplicates_and_enforces_capacity_atomically(operation_service):
    """Конкурентные HTTP-повторы не расходуют несколько мест в очереди."""
    store = operation_service.store
    payload = request()
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda _: store.submit("alice", payload), range(8)))
    assert len({record.id for record in records}) == 1
    with pytest.raises(HTTPException) as conflict:
        store.submit("alice", payload.model_copy(update={"profile": "chunk-local"}))
    assert conflict.value.status_code == 409
    store.config.max_pending_operations = 1
    with pytest.raises(HTTPException) as full:
        store.submit("bob", request())
    assert full.value.status_code == 429


def test_claim_cancellation_and_lease_recovery(operation_service):
    """Два workers не запускают одну работу, завершение не перезаписывает отмену."""
    store = operation_service.store
    queued = store.submit("alice", request())
    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(
            executor.map(lambda token: store.claim(token, ["prepare"]), ["one", "two"])
        )
    assert sum(record is not None for record in records) == 1
    token = "one" if records[0] else "two"
    assert store.renew(token)
    assert not store.renew("not-owner")
    assert store.cancel(queued.id, "alice").status == "cancel_requested"
    store.finish(queued.id, token, status="completed", exit_code=0)
    assert store.get(queued.id).status == "cancelled"
    assert store.claim("next", ["prepare"]) is None
    second = store.submit("alice", request())
    store.claim("old", ["prepare"])
    with store.database.connection(store.path) as conn:
        conn.execute("UPDATE web_worker_lock SET expires_at = 0")
    assert store.claim("new", ["prepare"]) is None
    assert store.get(second.id).status == "interrupted"
    store.finish(second.id, "old", status="completed", exit_code=0)
    assert store.get(second.id).status == "interrupted"


def test_owner_pagination_and_queued_cancel(operation_service):
    """Чужой cursor или ID не раскрывает запись; отмена до claim не запускает CLI."""
    store = operation_service.store
    first = store.submit("alice", request())
    other = store.submit("bob", request())
    store.submit("alice", request())
    page = store.list("alice", limit=1)
    assert len(page.items) == 1 and page.next_cursor
    assert store.list("alice", limit=1, cursor=page.next_cursor).items[0].id == first.id
    for get in (
        lambda: store.get(other.id, "alice"),
        lambda: store.list("alice", cursor=other.id),
    ):
        with pytest.raises(HTTPException) as missing:
            get()
        assert missing.value.status_code == 404
    assert store.cancel(first.id, "alice").status == "cancelled"
    assert all(
        record.status == "cancelled"
        for record in store.list("alice", status="cancelled").items
    )


def test_profiles_validate_parameters_and_freeze_config(operation_service, tmp_path):
    """Меняющийся YAML после submit не меняет размер чанка в ожидающей операции."""
    service = operation_service
    catalog = tmp_path / "catalog.yaml"
    config_path = ROOT / "config/chunking_openai.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "chunks": {
                    "title": "Чанкинг",
                    "kind": "chunk",
                    "config": str(config_path),
                    "parameters": ["chunking.chunk_size"],
                }
            }
        ),
        encoding="utf-8",
    )
    service.config.operations_catalog = catalog
    service.store.heartbeat("worker", ["chunks"])
    job = service.submit(
        "alice", request("chunks", parameters={"chunking.chunk_size": 210})
    )
    catalog.write_text("{}", encoding="utf-8")
    module, arguments = service.command(job)
    assert module == "rag_prep.cli" and arguments[0] == "chunk"
    snapshot = yaml.safe_load(
        (service.job_dir(job.id) / "config.yaml").read_text(encoding="utf-8")
    )
    assert snapshot["chunking"]["chunk_size"] == 210
    assert Path(snapshot["paths"]["input_jsonl"]).is_absolute()


def test_all_registered_profiles_have_valid_parameters(operation_service):
    """Каталог frontend не предлагает отсутствующие train или chunking поля."""
    service = operation_service
    for profile in service.catalog().values():
        model = service._load_config(profile)
        data = model.model_dump()
        for parameter in profile.parameters:
            section, field = parameter.split(".")
            assert field in data[section]
    with pytest.raises(HTTPException) as invalid:
        service.submit(
            "alice", request("chunk-openai", parameters={"agent.model": "external"})
        )
    assert invalid.value.status_code == 422
    with pytest.raises(ValidationError):
        service.submit(
            "alice", request("chunk-openai", parameters={"chunking.chunk_size": -1})
        )
    with service.store.database.connection(service.store.path) as conn:
        conn.execute("DELETE FROM web_worker_status")
    assert not any(profile.worker_ready for profile in service.public_profiles())
    with pytest.raises(HTTPException) as unavailable:
        service.submit("alice", request())
    assert unavailable.value.status_code == 503


def test_commands_cover_every_catalog_kind_and_preserve_index_safety(operation_service):
    """Все виды каталога формируют фиксированные CLI-команды без запуска моделей."""
    service = operation_service
    expected_modules = {
        "prepare": "rag_prep.cli",
        "chunk": "rag_prep.cli",
        "embed": "rag_prep.cli",
        "index": "rag_prep.vector_store_cli",
        "evaluation": "agent_app.evaluation.cli",
        "scenarios": "agent_app.cli",
        "tuning-inspect": "llm_tuning.cli",
        "tuning-validate": "llm_tuning.cli",
        "tuning-baseline": "llm_tuning.cli",
        "tuning-train": "llm_tuning.cli",
        "tuning-evaluate": "llm_tuning.cli",
        "tuning-compare": "llm_tuning.cli",
    }
    prepared = complete(service, service.submit("alice", request("prepare")))
    chunked = complete(
        service,
        service.submit("alice", request("chunk-openai", upstream_id=prepared.id)),
    )
    embedded = complete(
        service,
        service.submit("alice", request("embed-openai", upstream_id=chunked.id)),
    )
    baseline = complete(service, service.submit("alice", request("tuning-baseline")))
    trained = service.submit("alice", request("tuning-train"))
    train_module, _ = service.command(trained)
    assert train_module == expected_modules["tuning-train"]
    adapter = service.job_dir(trained.id) / "artifacts" / "adapter" / "run"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    trained = complete(service, trained)
    evaluated = service.submit(
        "alice", request("tuning-evaluate", upstream_id=trained.id)
    )
    evaluation_module, evaluation_args = service.command(evaluated)
    assert evaluation_module == expected_modules["tuning-evaluate"]
    assert evaluation_args[-2:] == ["--adapter-path", str(adapter)]
    tuned_report = service.job_dir(evaluated.id) / "artifacts" / "tuned_report.json"
    tuned_report.write_text("{}", encoding="utf-8")
    evaluated = complete(service, evaluated)
    baseline_report = (
        service.job_dir(baseline.id) / "artifacts" / "baseline_report.json"
    )
    baseline_report.parent.mkdir(parents=True, exist_ok=True)
    baseline_report.write_text("{}", encoding="utf-8")

    records = {
        "prepare": prepared,
        "chunk": service.submit(
            "alice", request("chunk-local", upstream_id=prepared.id)
        ),
        "embed": service.submit(
            "alice", request("embed-local", upstream_id=chunked.id)
        ),
        "index": service.submit(
            "alice", request("index-local", upstream_id=embedded.id)
        ),
        "evaluation": service.submit("alice", request("evaluate-local")),
        "scenarios": service.submit("alice", request("scenarios-openai")),
        "tuning-inspect": service.submit("alice", request("tuning-inspect")),
        "tuning-validate": service.submit("alice", request("tuning-validate")),
        "tuning-baseline": baseline,
        "tuning-train": trained,
        "tuning-evaluate": evaluated,
        "tuning-compare": service.submit(
            "alice",
            request(
                "tuning-compare",
                baseline_id=baseline.id,
                upstream_id=evaluated.id,
            ),
        ),
    }
    for kind, record in records.items():
        module, arguments = service.command(record)
        assert module == expected_modules[kind]
        snapshot = service.job_dir(record.id) / "config.yaml"
        assert snapshot.is_file()
        if kind in {"prepare", "chunk", "embed"}:
            assert arguments[:1] == [kind]
            assert arguments[-1] == "--no-prefect"
        if kind == "index":
            data = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
            assert data["vector_store"]["recreate_collection"] is False
            assert data["vector_store"]["prune_stale_points"] is False
        if kind == "tuning-compare":
            assert arguments[-4:] == [
                "--baseline-report",
                str(baseline_report),
                "--tuned-report",
                str(tuned_report),
            ]
        if kind in {"evaluation", "scenarios"}:
            assert (service.job_dir(record.id) / "suite.yaml").is_file()
    assert set(records) == {profile.kind for profile in service.catalog().values()}


def test_submit_rejects_invalid_upstream_and_keeps_idempotent_retry_without_worker(
    operation_service,
):
    """Граф запуска отклоняет неверные входы, но повтор не зависит от heartbeat worker."""
    service = operation_service
    with pytest.raises(HTTPException) as absent:
        service.submit("alice", request("chunk-openai", upstream_id=str(uuid4())))
    assert absent.value.status_code == 404
    incomplete = service.submit("alice", request("prepare"))
    with pytest.raises(HTTPException) as unfinished:
        service.submit("alice", request("chunk-openai", upstream_id=incomplete.id))
    assert unfinished.value.status_code == 409
    wrong_stage = complete(service, service.submit("alice", request("tuning-baseline")))
    with pytest.raises(HTTPException) as wrong:
        service.submit("alice", request("chunk-openai", upstream_id=wrong_stage.id))
    assert wrong.value.status_code == 422

    payload = request("tuning-validate")
    submitted = service.submit("alice", payload)
    with service.store.database.connection(service.store.path) as conn:
        conn.execute("DELETE FROM web_worker_status")
    assert service.submit("alice", payload).id == submitted.id


def test_suite_snapshot_and_tuning_input_paths_are_immutable(
    operation_service, tmp_path
):
    """Suite и пути evaluate/compare берутся из снимка задания, а не из текущего каталога."""
    service = operation_service
    suite = tmp_path / "suite.yaml"
    original_suite = "cases:\n  - id: original\n"
    suite.write_text(original_suite, encoding="utf-8")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "title": "Проверка",
                    "kind": "evaluation",
                    "config": str(ROOT / "config" / "support_agent_local.yaml"),
                    "suite": str(suite),
                }
            }
        ),
        encoding="utf-8",
    )
    service.config.operations_catalog = catalog
    service.store.heartbeat("worker", ["evaluation"])
    record = service.submit("alice", request("evaluation"))
    suite.write_text("cases:\n  - id: changed\n", encoding="utf-8")
    _, arguments = service.command(record)
    snapshot_suite = service.job_dir(record.id) / "suite.yaml"
    assert snapshot_suite.read_text(encoding="utf-8") == original_suite
    assert arguments[-1] == str(snapshot_suite)


def test_execute_operation_marks_cancel_timeout_and_nonzero_exit(
    operation_service, monkeypatch
):
    """Worker останавливает управляемый процесс и сохраняет итоговые статусы без CLI."""
    from agent_app.service import web_cli

    service = operation_service

    def claimed_record():
        """Создаёт одну running-операцию для проверки конкретного завершения worker."""
        job = service.submit("alice", request("tuning-validate"))
        token = str(uuid4())
        claimed = service.store.claim(token, ["tuning-validate"])
        assert claimed is not None and claimed.id == job.id
        return claimed, token

    def stop(process):
        """Подменяет остановку дерева процессом с фиксированным кодом завершения."""
        process.returncode = -15

    cancelled, cancel_token = claimed_record()
    cancel_process = FakeProcess()
    original_get = service.store.get
    reads = 0

    def cancel_after_start(job_id: str, owner: str | None = None):
        """Запрашивает отмену на первой проверке состояния после старта процесса."""
        nonlocal reads
        result = original_get(job_id, owner)
        if job_id == cancelled.id:
            reads += 1
            if reads == 2:
                with service.store.database.connection(service.store.path) as conn:
                    conn.execute(
                        "UPDATE web_operations SET status = 'cancel_requested' WHERE id = ?",
                        (job_id,),
                    )
                result = original_get(job_id, owner)
        return result

    monkeypatch.setattr(
        web_cli.subprocess, "Popen", lambda *args, **kwargs: cancel_process
    )
    monkeypatch.setattr(web_cli, "stop_process", stop)
    monkeypatch.setattr(service.store, "get", cancel_after_start)
    web_cli.execute_operation(service, cancelled, cancel_token)
    assert original_get(cancelled.id).status == "cancelled"
    assert original_get(cancelled.id).exit_code == -15

    monkeypatch.setattr(service.store, "get", original_get)
    timed_out, timeout_token = claimed_record()
    timeout_process = FakeProcess()
    monotonic_values = iter([0.0, service.config.operation_timeout_seconds + 1.0])
    original_monotonic = web_cli.time.monotonic
    monkeypatch.setattr(
        web_cli.subprocess, "Popen", lambda *args, **kwargs: timeout_process
    )
    monkeypatch.setattr(web_cli.time, "monotonic", lambda: next(monotonic_values))
    web_cli.execute_operation(service, timed_out, timeout_token)
    timeout_result = original_get(timed_out.id)
    assert timeout_result.status == "failed"
    assert timeout_result.exit_code == -15
    assert timeout_result.error == "Превышено время выполнения операции."

    monkeypatch.setattr(web_cli.time, "monotonic", original_monotonic)
    completed, exit_token = claimed_record()
    failing_process = FakeProcess(returncode=7)
    monkeypatch.setattr(
        web_cli.subprocess, "Popen", lambda *args, **kwargs: failing_process
    )
    web_cli.execute_operation(service, completed, exit_token)
    failed_result = original_get(completed.id)
    assert failed_result.status == "failed"
    assert failed_result.exit_code == 7
    assert (
        failed_result.error
        == "CLI завершился с ошибкой; подробности в журнале операции."
    )


def test_sources_and_upstream_cannot_cross_owner(operation_service):
    """UUID источника и upstream проверяются до записи задания в очередь."""
    service = operation_service
    source = service.add_source(
        "alice", "регламент.csv", "Код;Результат\n200;Успех".encode()
    )
    assert service.sources("bob") == []
    with pytest.raises(HTTPException) as wrong_owner:
        service.submit("bob", request(source_ids=[source.id]))
    assert wrong_owner.value.status_code == 404
    prepared = service.submit("alice", request(source_ids=[source.id]))
    with pytest.raises(HTTPException) as active:
        service.delete_source(source.id, "alice")
    assert active.value.status_code == 409
    service.command(prepared)
    assert len(list((service.job_dir(prepared.id) / "inputs").glob("*.csv"))) == 1
    service.store.claim("worker", ["prepare"])
    service.store.finish(prepared.id, "worker", status="completed", exit_code=0)
    chunk = service.submit("alice", request("chunk-openai", upstream_id=prepared.id))
    service.command(chunk)
    snapshot = yaml.safe_load(
        (service.job_dir(chunk.id) / "config.yaml").read_text(encoding="utf-8")
    )
    assert prepared.id in snapshot["paths"]["input_jsonl"]
    with pytest.raises(HTTPException) as cross:
        service.submit("bob", request("chunk-openai", upstream_id=prepared.id))
    assert cross.value.status_code == 404
    service.delete_source(source.id, "alice")
    assert service.sources("alice") == []


@pytest.mark.parametrize(
    "name,body",
    [("../a.txt", b"text"), ("x.exe", b"bytes"), ("x.pdf", b"not-pdf"), ("x.txt", b"")],
)
def test_upload_rejects_unsafe_names_and_invalid_content(operation_service, name, body):
    """Заголовок и расширение не разрешают загрузку исполняемого файла или traversal."""
    with pytest.raises(HTTPException):
        operation_service.add_source("alice", name, body)


def test_artifact_paths_integrity_and_log_retention(operation_service):
    """Просмотр ограничен каталогом запуска; повреждённый источник не используется."""
    service = operation_service
    source = service.add_source("alice", "a.txt", b"example")
    _, path = service.source(source.id, "alice")
    path.write_bytes(b"changed")
    with pytest.raises(HTTPException) as corrupt:
        service.source(source.id, "alice")
    assert corrupt.value.status_code == 409
    with pytest.raises(HTTPException):
        service.job_dir("../outside")
    job = service.submit("alice", request("tuning-validate"))
    service.command(job)
    result = service.job_dir(job.id) / "artifacts" / "result.json"
    result.write_text('{"ok":true}', encoding="utf-8")
    artifact = service.artifacts(job.id)[0]
    assert service.artifact(job.id, artifact.id) == result
    with pytest.raises(HTTPException):
        service.artifact(job.id, "../config.yaml")
    import os

    log = service.job_dir(job.id) / "execution.log"
    log.write_text("Старый запуск", encoding="utf-8")
    os.utime(log, (time.time() - 31 * 86400,) * 2)
    service.cleanup_logs()
    assert not log.exists() and result.exists()


def test_json_redaction_preserves_structure():
    """Очистка DSN не повреждает кавычки и многомерные числовые embeddings."""
    data = {
        "dsn": "postgresql+psycopg://account:password@host/db",
        "embedding": [1.0, 2.0],
        "text": "Путь C:\\private\\alice\\token.json",
    }
    cleaned = public_data(data)
    assert "password" not in json.dumps(cleaned)
    assert cleaned["embedding"] == [1.0, 2.0]


def test_worker_runs_existing_validation_cli(operation_service):
    """Настоящий дочерний процесс проверяет датасет без модели и без API-запросов."""
    from agent_app.service.web_cli import execute_operation

    service = operation_service
    job = service.submit("alice", request("tuning-validate"))
    record = service.store.claim("worker", ["tuning-validate"])
    execute_operation(service, record, "worker")
    completed = service.store.get(job.id)
    assert completed.status == "completed", (
        service.job_dir(job.id) / "execution.log"
    ).read_text(encoding="utf-8")
    assert completed.exit_code == 0


def test_web_cli_bootstrap_and_worker_health(operation_service, monkeypatch, capsys):
    """Первичная настройка создаёт аккаунт в той же БД, а heartbeat отражает запуск worker."""
    import sys

    from agent_app.config import AgentAppConfig
    from agent_app.service.accounts import AccountStore, LoginRequest
    from agent_app.service.web_cli import main

    service = operation_service
    config = AgentAppConfig(
        agent={"provider": "local", "model": "schema-only"}, web=service.config
    )
    path = service.config.sqlite_path.parent / "service.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    prefix = ["rag-web", "--config", str(path)]
    monkeypatch.setenv("WEB_BOOTSTRAP_PASSWORD", "test-password-for-cli-only")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *prefix,
            "create-user",
            "cli-user",
            "--name",
            "Проверка CLI",
            "--role",
            "operator",
            "--password-env",
            "WEB_BOOTSTRAP_PASSWORD",
        ],
    )
    assert main() == 0
    account = json.loads(capsys.readouterr().out)
    assert account["username"] == "cli-user" and account["roles"] == ["operator"]
    assert "password" not in account
    accounts = AccountStore(service.store.database, service.config)
    accounts.initialize()
    _, session = accounts.login(
        LoginRequest(username="cli-user", password="test-password-for-cli-only"),
        client_id="cli-check",
    )
    assert session.user.roles == ["operator"]

    with service.store.database.connection(service.config.sqlite_path) as conn:
        conn.execute("DELETE FROM web_worker_status")
    monkeypatch.setattr(sys, "argv", [*prefix, "worker-health"])
    assert main() == 1
    assert json.loads(capsys.readouterr().out) == {"ready": False}
    monkeypatch.setattr(sys, "argv", [*prefix, "worker", "--once"])
    assert main() == 0
    monkeypatch.setattr(sys, "argv", [*prefix, "worker-health"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out) == {"ready": True}


def test_worker_does_not_spawn_previously_cancelled_job(operation_service, monkeypatch):
    """Отмена между claim и запуском CLI не должна запускать платную или долгую работу."""
    from agent_app.service.web_cli import execute_operation

    service = operation_service
    job = service.submit("alice", request("tuning-inspect"))
    record = service.store.claim("worker", ["tuning-inspect"])
    service.store.cancel(job.id, "alice")

    def forbidden_command(_record):
        """Маркер гарантирует, что отменённый запуск не доходит даже до сборки команды."""
        pytest.fail("Отменённая операция не должна создавать CLI-команду")

    monkeypatch.setattr(service, "command", forbidden_command)
    execute_operation(service, record, "worker")
    assert service.store.get(job.id).status == "cancelled"
