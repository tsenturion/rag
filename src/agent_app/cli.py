from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from agent_app.config import load_agent_config
from agent_app.graph import AgentRunner
from agent_app.memory import SQLiteMemoryStore
from agent_app.scenarios import ScenarioRunner, load_scenario_suite


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
        print(
            json.dumps({"deleted_count": deleted_count}, ensure_ascii=False, indent=2)
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
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
