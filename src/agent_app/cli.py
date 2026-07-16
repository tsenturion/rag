from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from agent_app.config import load_agent_config
from agent_app.graph import AgentRunner
from agent_app.memory import SQLiteMemoryStore
from agent_app.multi_agent.exporting import MultiAgentExporter
from agent_app.multi_agent.persistence import MultiAgentCheckpointStore
from agent_app.multi_agent.protocols.simulation import run_protocol_simulation
from agent_app.multi_agent.runtime import (
    MultiAgentRuntime,
    load_comparison_suite,
)
from agent_app.scenarios import ScenarioRunner, load_scenario_suite
from agent_app.tools.mcp_external import ExternalMCPToolManager


class RussianHelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, *args, **kwargs) -> str:
        return (
            super()
            ._format_usage(*args, **kwargs)
            .replace("usage:", "использование:", 1)
        )


def _add_russian_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="показать это сообщение и выйти",
    )
    parser._optionals.title = "параметры"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Запуск LangGraph-агента с tools и памятью.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(parser)
    parser.add_argument(
        "--config",
        required=True,
        help="Путь к явному provider-конфигу агента.",
    )
    parser.add_argument(
        "--message", default=None, help="Одно сообщение для отправки агенту."
    )
    parser.add_argument(
        "--user-id", default=None, help="Идентификатор пользователя для памяти."
    )
    parser.add_argument(
        "--session-id", default=None, help="Идентификатор сессии для памяти."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Напечатать машиночитаемый JSON AgentResponse.",
    )
    parser.add_argument(
        "--list-memory",
        action="store_true",
        help="Показать долговременную память текущего пользователя и выйти.",
    )
    parser.add_argument(
        "--clear-session-memory",
        action="store_true",
        help="Очистить память текущей сессии и выйти.",
    )
    parser.add_argument(
        "--list-multi-agent-history",
        action="store_true",
        help="Показать persistent историю multi-agent сессии и выйти.",
    )
    parser.add_argument(
        "--scenarios-config",
        default="config/agent_scenarios.yaml",
        help="Путь к YAML-конфигу сценариев MVP-агента.",
    )
    parser.add_argument(
        "--run-scenario",
        default=None,
        help="Запустить один сценарий по id и выйти.",
    )
    parser.add_argument(
        "--run-scenarios",
        action="store_true",
        help="Запустить все сценарии MVP-агента и выйти.",
    )
    parser.add_argument(
        "--scenario-report",
        default=None,
        help="Путь для JSON-отчёта сценариев.",
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Выполнить --message через supervisor-граф нескольких агентов.",
    )
    parser.add_argument(
        "--compare-agents",
        action="store_true",
        help="Сравнить single-agent и multi-agent на общем наборе сценариев.",
    )
    parser.add_argument(
        "--multi-agent-scenarios",
        default="config/multi_agent_scenarios.yaml",
        help="YAML со сценариями сравнения single vs multi.",
    )
    parser.add_argument(
        "--simulate-protocols",
        action="store_true",
        help="Запустить локальную request-response/pub-sub и ACP→A2A симуляцию.",
    )
    parser.add_argument(
        "--get-multi-agent-run",
        default=None,
        help="Показать сохранённый multi-agent result.json по run_id.",
    )
    parser.add_argument(
        "--list-mcp-tools",
        action="store_true",
        help="Подключить внешние MCP-серверы, вывести доступные tools и выйти.",
    )
    parser.add_argument(
        "--call-mcp-tool",
        default=None,
        help="Вызвать внешний MCP tool по его имени с префиксом.",
    )
    parser.add_argument(
        "--mcp-arguments",
        default="{}",
        help="JSON-объект аргументов для --call-mcp-tool.",
    )
    return parser


def main() -> int:
    _configure_stdio()
    args = build_parser().parse_args()
    config = load_agent_config(Path(args.config))
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    user_id = args.user_id or config.memory.default_user_id
    session_id = args.session_id or config.memory.default_session_id

    if args.simulate_protocols:
        print(
            json.dumps(
                run_protocol_simulation(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.list_mcp_tools or args.call_mcp_tool:
        manager = ExternalMCPToolManager(config.tools.mcp_servers)
        try:
            tools = manager.start()
            if args.call_mcp_tool:
                arguments = json.loads(args.mcp_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("--mcp-arguments должен содержать JSON-объект")
                tool = next(
                    (item for item in tools if item.name == args.call_mcp_tool),
                    None,
                )
                if tool is None:
                    known = ", ".join(item.name for item in tools) or "нет"
                    raise ValueError(
                        f"Внешний MCP tool не найден: {args.call_mcp_tool}. "
                        f"Доступные: {known}"
                    )
                result = tool.invoke(arguments)
                print(result)
                return 0
            print(
                json.dumps(
                    manager.status(),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if tools or not config.tools.mcp_servers else 1
        finally:
            manager.close()

    if args.get_multi_agent_run:
        payload = MultiAgentExporter(config.multi_agent.output_dir).load_result(
            args.get_multi_agent_run
        )
        if payload is None:
            print(f"Multi-agent запуск не найден: {args.get_multi_agent_run}")
            return 1
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.list_multi_agent_history:
        checkpoints = MultiAgentCheckpointStore(config.multi_agent.checkpoint_path)
        try:
            history = [
                {
                    "id": message.id,
                    "type": message.type,
                    "content": str(message.content),
                }
                for message in checkpoints.history(
                    user_id=user_id,
                    session_id=session_id,
                )
            ]
        finally:
            checkpoints.close()
        print(json.dumps(history, ensure_ascii=False, indent=2))
        return 0

    if args.multi_agent or args.compare_agents:
        if not config.multi_agent.enabled:
            raise ValueError(
                "Multi-agent режим отключён. Используйте config/multi_agent_*.yaml."
            )
        runtime = MultiAgentRuntime(config)
        try:
            if args.compare_agents:
                suite = load_comparison_suite(Path(args.multi_agent_scenarios))
                report = runtime.compare(
                    suite,
                    user_id=user_id,
                    session_prefix=session_id,
                )
                print(
                    json.dumps(
                        report.model_dump(mode="json"),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            if not args.message:
                raise ValueError("Для --multi-agent обязательно задайте --message")
            result = runtime.ask(
                user_id=user_id,
                session_id=session_id,
                message=args.message,
            )
            payload = result.model_dump(mode="json")
            _print_response(payload["response"], as_json=args.json)
            if not args.json:
                print(f"Run ID: {result.response.run_id}")
                print(f"Артефакты: {result.run_dir}")
            return 0 if result.response.lifecycle[-1].state == "completed" else 1
        finally:
            runtime.close()

    if args.list_memory:
        store = SQLiteMemoryStore(config.memory.sqlite_path)
        records = [
            record.model_dump(mode="json")
            for record in store.list_memories(user_id=user_id, limit=100)
        ]
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0

    if args.clear_session_memory:
        store = SQLiteMemoryStore(config.memory.sqlite_path)
        deleted_count = store.clear_session(user_id=user_id, session_id=session_id)
        checkpoint_deleted = False
        if config.multi_agent.enabled:
            checkpoints = MultiAgentCheckpointStore(config.multi_agent.checkpoint_path)
            try:
                checkpoint_deleted = checkpoints.clear(
                    user_id=user_id,
                    session_id=session_id,
                )
            finally:
                checkpoints.close()
        print(
            json.dumps(
                {
                    "deleted_count": deleted_count,
                    "multi_agent_checkpoint_deleted": checkpoint_deleted,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.run_scenario or args.run_scenarios:
        suite = load_scenario_suite(Path(args.scenarios_config))
        scenario_runner = ScenarioRunner(
            config,
            suite,
            config_path=str(Path(args.scenarios_config)),
        )
        report = (
            scenario_runner.run_one(args.run_scenario)
            if args.run_scenario
            else scenario_runner.run_all()
        )
        report_path = scenario_runner.write_report(
            report,
            Path(args.scenario_report) if args.scenario_report else None,
        )
        payload = report.model_dump(mode="json")
        payload["report_path"] = str(report_path)
        _print_scenario_report(payload, as_json=args.json)
        scenario_runner.close()
        return 0 if report.passed else 1

    runner = AgentRunner(
        config,
        user_id=user_id,
        session_id=session_id,
    )

    if args.message:
        try:
            response = runner.ask(args.message)
            _print_response(response.model_dump(mode="json"), as_json=args.json)
            return 0
        finally:
            runner.close()

    print("Интерактивный режим. Введите exit, quit или выход для завершения.")
    while True:
        try:
            message = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"exit", "quit", "выход"}:
            break
        if not message:
            continue
        response = runner.ask(message)
        _print_response(response.model_dump(mode="json"), as_json=args.json)
    runner.close()
    return 0


def _print_response(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload.get("answer", ""))


def _print_scenario_report(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    results = payload.get("results", [])
    passed = bool(payload.get("passed"))
    print(f"Сценарии: {'passed' if passed else 'failed'}")
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, dict):
            continue
        print(f"- {result.get('id')}: {'passed' if result.get('passed') else 'failed'}")
    print(f"Отчёт: {payload.get('report_path')}")


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
