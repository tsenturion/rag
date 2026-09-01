"""Экспорт OpenAPI-схемы support-сервиса для генерации frontend-клиента."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agent_app.service.cli import RussianHelpFormatter


def build_parser() -> argparse.ArgumentParser:
    """Описывает CLI экспорта схемы без импорта runtime до разбора аргументов."""
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(
        description=(
            "Экспорт OpenAPI JSON для типизации будущего Vite web-приложения."
        ),
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
            "Путь к provider-конфигу support-агента. "
            "Можно задать через SUPPORT_AGENT_CONFIG."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/openapi/support-api.json",
        help="Путь к результирующему OpenAPI JSON.",
    )
    return parser


def export_openapi(config_path: str | Path, output_path: str | Path) -> Path:
    """Строит схему из реального FastAPI-приложения и атомарно записывает JSON."""
    from agent_app.service.app import create_app

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    schema = create_app(Path(config_path)).openapi()
    temporary.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def main() -> None:
    """Проверяет обязательный конфиг и сообщает путь экспортированной схемы."""
    parser = build_parser()
    args = parser.parse_args()
    if not args.config:
        parser.error("задайте --config или переменную окружения SUPPORT_AGENT_CONFIG")
    output = export_openapi(args.config, args.output)
    print(output)


if __name__ == "__main__":
    main()
