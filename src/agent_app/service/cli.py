"""Командный интерфейс для HTTP-сервиса поддержки."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


class RussianHelpFormatter(argparse.HelpFormatter):
    """Гарантирует вывод справки CLI с русскоязычным заголовком для повышения удобства пользователей."""

    def _format_usage(self, *args, **kwargs) -> str:
        """Обеспечивает русскоязычный заголовок раздела использования в справке командной строки."""
        return (
            super()
            ._format_usage(*args, **kwargs)
            .replace("usage:", "использование:", 1)
        )


def build_parser() -> argparse.ArgumentParser:
    """Создаёт и настраивает parser аргументов командной строки."""
    from dotenv import load_dotenv

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
    """Запускает командный интерфейс и возвращает код завершения."""
    parser = build_parser()
    args = parser.parse_args()
    if not args.config:
        parser.error("задайте --config или переменную окружения SUPPORT_AGENT_CONFIG")

    # Серверные зависимости импортируются после --help и проверки обязательного
    # пути, поэтому получение справки не создаёт FastAPI runtime.
    import uvicorn

    from agent_app.config import load_agent_config

    config_path = Path(args.config).expanduser().resolve()
    config = load_agent_config(config_path)
    if config.service.workers != 1 and config.persistence.backend != "postgresql":
        raise ValueError(
            "Несколько Uvicorn workers требуют persistence.backend=postgresql."
        )
    # Factory импортируется отдельно в каждом worker. Абсолютный путь в env
    # обеспечивает одинаковую конфигурацию независимо от рабочего каталога.
    os.environ["SUPPORT_AGENT_CONFIG"] = str(config_path)
    uvicorn.run(
        "agent_app.service.app:create_app",
        factory=True,
        host=args.host or config.service.host,
        port=args.port or config.service.port,
        workers=config.service.workers,
        log_level=config.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
