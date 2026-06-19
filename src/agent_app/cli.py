from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from agent_app.config import load_agent_config
from agent_app.graph import AgentRunner
from agent_app.memory import SQLiteMemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the tools and memory LangGraph agent."
    )
    parser.add_argument(
        "--config",
        default="config/agent.yaml",
        help="Path to agent YAML config.",
    )
    parser.add_argument("--message", default=None, help="Single message to send.")
    parser.add_argument("--user-id", default=None, help="Memory user id.")
    parser.add_argument("--session-id", default=None, help="Memory session id.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable AgentResponse JSON.",
    )
    parser.add_argument(
        "--list-memory",
        action="store_true",
        help="List current user's long-term memory and exit.",
    )
    parser.add_argument(
        "--clear-session-memory",
        action="store_true",
        help="Clear current session-scoped memory and exit.",
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

    print("Interactive mode. Type exit or quit to stop.")
    while True:
        try:
            message = input("You: ").strip()
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
