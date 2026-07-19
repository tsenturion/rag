"""Командный интерфейс для изолированного выполнения Python."""

from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    """Описывает сетевые параметры сервиса и стандартную справку без запуска Uvicorn."""
    parser = argparse.ArgumentParser(
        description="Запуск изолированного сервиса выполнения Python-кода."
    )
    parser.add_argument(
        "--host",
        default=os.getenv("CODE_RUNNER_HOST", "127.0.0.1"),
        help="Адрес прослушивания; также читается из CODE_RUNNER_HOST.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=os.getenv("CODE_RUNNER_PORT", "8010"),
        help="Порт сервиса; также читается из CODE_RUNNER_PORT.",
    )
    return parser


def main() -> None:
    """Разбирает параметры командной строки и запускает один процесс HTTP-сервиса."""
    args = build_parser().parse_args()

    # Импорт откладывается до разбора аргументов, чтобы --help не загружал
    # ASGI-стек и гарантированно не пытался занять сетевой порт.
    import uvicorn

    uvicorn.run(
        "code_runner.app:app",
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
