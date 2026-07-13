from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from agent_app.config import load_agent_config
from agent_app.service.app import create_app


class RussianHelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, *args, **kwargs) -> str:
        return (
            super()
            ._format_usage(*args, **kwargs)
            .replace("usage:", "использование:", 1)
        )


def build_parser() -> argparse.ArgumentParser:
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(
        description="Запуск HTTP API ИИ-агента поддержки инженера.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="показать это сообщение и выйти",
    )
    parser._optionals.title = "параметры"
    parser.add_argument(
        "--config",
        default=os.getenv("SUPPORT_AGENT_CONFIG"),
        help=(
            "Путь к явному provider-конфигу support-агента. "
            "Можно задать через SUPPORT_AGENT_CONFIG."
        ),
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Переопределить service.host из YAML-конфига.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Переопределить service.port из YAML-конфига.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.config:
        parser.error("задайте --config или переменную окружения SUPPORT_AGENT_CONFIG")

    config_path = Path(args.config).expanduser().resolve()
    config = load_agent_config(config_path)
    if config.service.workers != 1:
        raise ValueError(
            "SQLite memory/incident storage поддерживает один worker. "
            "Установите service.workers: 1."
        )
    uvicorn.run(
        create_app(config_path),
        host=args.host or config.service.host,
        port=args.port or config.service.port,
        workers=1,
        log_level=config.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
