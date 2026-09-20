"""Создание первого пользователя и запуск постоянного worker web-операций."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

LOGGER = logging.getLogger(__name__)


def stop_process(process: subprocess.Popen) -> None:
    """Прерывает дерево дочернего CLI и ждёт освобождения его ресурсов."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def execute_operation(service, record, token: str) -> None:
    """Heartbeat контролирует lease, отмену и wall timeout во время реального CLI."""
    from agent_app.service.operations import public_text

    process, reader = None, None
    try:
        if not service.store.renew(token):
            return
        if service.store.get(record.id).status == "cancel_requested":
            service.store.finish(record.id, token, status="cancelled", exit_code=None)
            return
        module, arguments = service.command(record)
        directory = service.job_dir(record.id)
        environment = dict(
            os.environ,
            PYTHONUTF8="1",
            PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        process = subprocess.Popen(
            [sys.executable, "-m", module, *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

        def drain() -> None:
            """Пишет очищенный журнал; ограничение размера не блокирует stdout процесса."""
            size = 0
            with (directory / "execution.log").open("a", encoding="utf-8") as stream:
                while line := process.stdout.readline(65536):
                    if size < 20_000_000:
                        clean = public_text(line)
                        stream.write(clean)
                        stream.flush()
                        size += len(clean.encode("utf-8"))
            process.stdout.close()

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        started = time.monotonic()
        outcome, error = "completed", None
        while process.poll() is None:
            if not service.store.renew(token):
                stop_process(process)
                return
            current = service.store.get(record.id)
            if current.status == "cancel_requested":
                outcome = "cancelled"
                stop_process(process)
                break
            if time.monotonic() - started > service.config.operation_timeout_seconds:
                outcome, error = "failed", "Превышено время выполнения операции."
                stop_process(process)
                break
            service.store.heartbeat(token, service.available_profiles())
            time.sleep(1)
        reader.join(timeout=10)
        if process.returncode != 0 and outcome == "completed":
            outcome, error = (
                "failed",
                "CLI завершился с ошибкой; подробности в журнале операции.",
            )
        service.store.finish(
            record.id, token, status=outcome, exit_code=process.returncode, error=error
        )
        LOGGER.info(
            "Операция завершена", extra={"job_id": record.id, "status": outcome}
        )
    except BaseException as exc:
        if process is not None:
            stop_process(process)
        LOGGER.exception("Ошибка worker", extra={"job_id": record.id})
        service.store.finish(
            record.id,
            token,
            status="failed",
            exit_code=None,
            error="Не удалось выполнить операцию. Проверьте журнал worker.",
        )
        if not isinstance(exc, Exception):
            raise
    finally:
        if reader is not None:
            reader.join(timeout=10)


def main() -> int:
    """CLI не запускает API или LLM до выбора конкретного действия."""
    parser = argparse.ArgumentParser(
        description="Браузерные пользователи и worker операций."
    )
    parser.add_argument("--config", required=True, help="Конфигурация support-сервиса.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser(
        "create-user", help="Создать аккаунт; пароль вводится скрыто."
    )
    create.add_argument("username")
    create.add_argument("--name", required=True)
    create.add_argument(
        "--role",
        choices=["admin", "operator", "engineer", "viewer"],
        default="engineer",
    )
    create.add_argument(
        "--password-env", help="Имя переменной с паролем для автоматизации."
    )
    worker = sub.add_parser("worker", help="Выполнять постоянную очередь операций.")
    worker.add_argument(
        "--once",
        action="store_true",
        help="Обработать одно доступное задание и завершиться.",
    )
    sub.add_parser("worker-health", help="Проверить свежий heartbeat worker.")
    args = parser.parse_args()
    from agent_app.config import load_agent_config
    from agent_app.database import DatabaseRuntime
    from agent_app.observability import configure_service_logging
    from agent_app.service.accounts import AccountStore, UserCreate
    from agent_app.service.operation_store import OperationStore
    from agent_app.service.operations import OperationService

    config = load_agent_config(args.config)
    configure_service_logging(config.logging.level, json_format=True)
    database = DatabaseRuntime.from_config(config.persistence)
    try:
        accounts = AccountStore(database, config.web)
        accounts.initialize()
        if args.command == "create-user":
            password = (
                os.environ.get(args.password_env, "")
                if args.password_env
                else getpass.getpass("Пароль (не менее 12 символов): ")
            )
            user = accounts.create(
                UserCreate(
                    username=args.username,
                    password=password,
                    display_name=args.name,
                    roles=[args.role],
                )
            )
            print(user.model_dump_json())
            return 0
        store = OperationStore(database, config.web)
        store.initialize()
        service = OperationService(store)
        if args.command == "worker-health":
            ready = bool(store.capabilities())
            print(json.dumps({"ready": ready}))
            return 0 if ready else 1
        token = str(uuid4())

        def terminate(_signal, _frame):
            """SIGTERM проходит через finally и останавливает дерево дочернего CLI."""
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, terminate)
        while True:
            service.cleanup_logs()
            capabilities = service.available_profiles()
            store.heartbeat(token, capabilities)
            record = store.claim(token, capabilities)
            if record:
                LOGGER.info("Операция принята worker", extra={"job_id": record.id})
                execute_operation(service, record, token)
            if args.once:
                return 0
            time.sleep(2)
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
