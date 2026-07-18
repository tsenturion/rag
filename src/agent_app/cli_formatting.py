from __future__ import annotations

import argparse


class RussianHelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, *args, **kwargs) -> str:
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
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="показать это сообщение и выйти",
    )
    parser._positionals.title = positionals_title
    parser._optionals.title = "параметры"
