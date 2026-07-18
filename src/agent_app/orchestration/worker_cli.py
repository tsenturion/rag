from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent_app.cli_formatting import RussianHelpFormatter, add_russian_help
from agent_app.config import load_agent_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Запуск Celery worker распределённой оркестрации.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    add_russian_help(parser, positionals_title="аргументы")
    parser.add_argument(
        "--config",
        default=os.getenv("SUPPORT_AGENT_CONFIG"),
        help="Provider-конфиг; также загружает .env до создания Celery app.",
    )
    parser.add_argument(
        "--queues",
        default="agent.high,agent.default,agent.low",
        help="Список очередей через запятую.",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument(
        "--pool",
        default="solo" if __import__("os").name == "nt" else "prefork",
    )
    args = parser.parse_args()
    if not args.config:
        parser.error("задайте --config или SUPPORT_AGENT_CONFIG")
    config = load_agent_config(Path(args.config))
    from agent_app.orchestration.queue import create_celery_app

    celery_app = create_celery_app(config.orchestration)
    celery_app.worker_main(
        [
            "worker",
            "--loglevel",
            args.loglevel,
            "--queues",
            args.queues,
            "--concurrency",
            str(args.concurrency),
            "--pool",
            args.pool,
        ]
    )


if __name__ == "__main__":
    main()
