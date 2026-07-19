"""Командный интерфейс Camunda для распределённой оркестрации."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent_app.cli_formatting import RussianHelpFormatter, add_russian_help
from agent_app.config import load_agent_config
from agent_app.orchestration.camunda import (
    CamundaAgentWorker,
    deploy_process,
    start_process,
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
    args = build_parser().parse_args()
    config = load_agent_config(Path(args.config))
    if not config.orchestration.camunda.enabled:
        raise ValueError("Camunda отключена в выбранной конфигурации")
    if args.command == "deploy":
        payload = deploy_process(config)
    elif args.command == "start":
        payload = start_process(
            config,
            user_id=args.user_id,
            session_id=args.session_id,
            message=args.message,
            risk_level=args.risk_level,
            priority=args.priority,
        )
    else:
        worker = CamundaAgentWorker(config)
        try:
            asyncio.run(worker.run())
        finally:
            worker.close()
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
