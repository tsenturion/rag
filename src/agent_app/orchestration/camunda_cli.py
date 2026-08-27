"""Командный интерфейс Camunda для распределённой оркестрации."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agent_app.cli_formatting import RussianHelpFormatter, add_russian_help
from agent_app.config import load_agent_config
from agent_app.orchestration.camunda import (
    CamundaAgentWorker,
    complete_approval,
    deploy_process,
    diagnose_camunda,
    process_status,
    start_process,
    wait_for_process,
)


def build_parser() -> argparse.ArgumentParser:
    """Создаёт и настраивает parser аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Управление гибридным BPMN-процессом Camunda и агентами.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    add_russian_help(parser)
    parser.add_argument("--config", required=True, help="Provider-конфиг агента.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    deploy = subparsers.add_parser(
        "deploy",
        help="Развернуть BPMN-процесс в Camunda.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    add_russian_help(deploy, positionals_title="аргументы")
    deploy.add_argument(
        "--force",
        action="store_true",
        help="Создать новую версию даже при неизменном BPMN.",
    )
    start = subparsers.add_parser(
        "start",
        help="Запустить экземпляр процесса.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    add_russian_help(start, positionals_title="аргументы")
    start.add_argument("--message", required=True, help="Инженерное задание.")
    start.add_argument("--user-id", default="engineer-1")
    start.add_argument("--session-id", default="camunda-demo")
    start.add_argument(
        "--risk-level", choices=["low", "medium", "high"], default="medium"
    )
    start.add_argument(
        "--priority", choices=["low", "normal", "high"], default="normal"
    )
    start.add_argument(
        "--wait",
        action="store_true",
        help="Дождаться завершения, инцидента или ручного согласования.",
    )
    start.add_argument(
        "--wait-timeout",
        type=float,
        default=300.0,
        help="Максимальное время ожидания процесса в секундах.",
    )
    status = subparsers.add_parser(
        "status",
        help="Показать состояние экземпляра процесса.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    add_russian_help(status, positionals_title="аргументы")
    status.add_argument("--process-instance-key", required=True)
    approval = subparsers.add_parser(
        "approve",
        help="Завершить ручное согласование решением approve/reject.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    add_russian_help(approval, positionals_title="аргументы")
    approval.add_argument("--process-instance-key", required=True)
    approval.add_argument(
        "--decision",
        choices=["approve", "reject"],
        required=True,
    )
    doctor = subparsers.add_parser(
        "doctor",
        help="Проверить REST-соединение, BPMN и deployment.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    add_russian_help(doctor, positionals_title="аргументы")
    worker = subparsers.add_parser(
        "worker",
        help="Запустить Camunda job workers.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    add_russian_help(worker, positionals_title="аргументы")
    return parser


def main() -> int:
    """Запускает командный интерфейс и возвращает код завершения."""
    _configure_stdio()
    args = build_parser().parse_args()
    config = load_agent_config(Path(args.config))
    if not config.orchestration.camunda.enabled:
        raise ValueError("Camunda отключена в выбранной конфигурации")
    if args.command == "deploy":
        payload = deploy_process(config, force=args.force)
    elif args.command == "start":
        payload = start_process(
            config,
            user_id=args.user_id,
            session_id=args.session_id,
            message=args.message,
            risk_level=args.risk_level,
            priority=args.priority,
        )
        if args.wait:
            process_key = str(payload["processInstanceKey"])
            payload = {
                "started": payload,
                "status": wait_for_process(
                    config,
                    process_key,
                    timeout_seconds=args.wait_timeout,
                ),
            }
    elif args.command == "status":
        payload = process_status(config, args.process_instance_key)
    elif args.command == "approve":
        payload = complete_approval(
            config,
            args.process_instance_key,
            approved=args.decision == "approve",
        )
    elif args.command == "doctor":
        payload = diagnose_camunda(config)
    else:
        worker = CamundaAgentWorker(config)
        try:
            asyncio.run(worker.run())
        finally:
            worker.close()
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _configure_stdio() -> None:
    """Сохраняет Unicode в BPMN payload и ответах агента на Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
