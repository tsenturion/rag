from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from agent_app.config import load_agent_config
from agent_app.graph import AgentRunner
from agent_app.memory import SQLiteMemoryStore


class RussianHelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, *args, **kwargs) -> str:
        return super()._format_usage(*args, **kwargs).replace("usage:", "использование:", 1)


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
        default="config/agent.yaml",
        help="Путь к YAML-конфигу агента.",
    )
    parser.add_argument("--message", default=None, help="Одно сообщение для отправки агенту.")
    parser.add_argument("--user-id", default=None, help="Идентификатор пользователя для памяти.")
    parser.add_argument("--session-id", default=None, help="Идентификатор сессии для памяти.")
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
    return parser


def main() -> None:
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
        return

    if args.clear_session_memory:
        store = SQLiteMemoryStore(config.memory.sqlite_path)
        deleted_count = store.clear_session(user_id=user_id, session_id=session_id)
        print(json.dumps({"deleted_count": deleted_count}, ensure_ascii=False, indent=2))
        return

    runner = AgentRunner(
        config,
        user_id=user_id,
        session_id=session_id,
    )

    if args.message:
        response = runner.ask(args.message)
        _print_response(response.model_dump(mode="json"), as_json=args.json)
        return

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


def _print_response(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload.get("answer", ""))


if __name__ == "__main__":
    main()
