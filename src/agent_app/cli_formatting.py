"""Форматирование справки командной строки для агентного приложения."""

from __future__ import annotations

import argparse


class RussianHelpFormatter(argparse.HelpFormatter):
    """Гарантирует отображение справки CLI на русском языке для повышения доступности."""

    def _format_usage(self, *args, **kwargs) -> str:
        """Гарантирует, что строка использования в справке CLI будет на русском языке."""
        return (
            super()
            ._format_usage(*args, **kwargs)
            .replace("usage:", "использование:", 1)
        )


def add_russian_help(
    parser: argparse.ArgumentParser,
    *,
    positionals_title: str = "команды",
) -> None:
    """Гарантирует наличие русскоязычной справки и заголовков в CLI для пользователя."""
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="показать это сообщение и выйти",
    )
    parser._positionals.title = positionals_title
    parser._optionals.title = "параметры"
