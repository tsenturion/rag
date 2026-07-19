"""Командный интерфейс для RAG-конвейера."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


class RussianHelpFormatter(argparse.HelpFormatter):
    """Гарантирует вывод справки CLI с русскоязычными заголовками и терминологией для повышения доступности."""

    def _format_usage(self, *args, **kwargs) -> str:
        """Гарантирует отображение инструкции по использованию CLI на русском языке для повышения удобства пользователя."""
        return (
            super()
            ._format_usage(*args, **kwargs)
            .replace("usage:", "использование:", 1)
        )


def _add_russian_help(parser: argparse.ArgumentParser) -> None:
    """Гарантирует наличие русскоязычной справки и корректных заголовков для всех CLI-аргументов."""
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="показать это сообщение и выйти",
    )
    parser._positionals.title = "позиционные аргументы"
    parser._optionals.title = "параметры"


def build_parser() -> argparse.ArgumentParser:
    """Создаёт и настраивает parser аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Запуск пайплайнов подготовки данных, чанкинга, embeddings и vector store для RAG.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(parser)
    parser.add_argument(
        "--config",
        default=None,
        help="Legacy-путь к конфигу подготовки. По умолчанию config/default.yaml.",
    )
    parser.add_argument(
        "--no-prefect",
        action="store_true",
        help="Legacy-флаг: запустить подготовку напрямую без оркестрации Prefect.",
    )
    subparsers = parser.add_subparsers(dest="command", title="команды")

    prepare = subparsers.add_parser(
        "prepare",
        help="Запустить пайплайн подготовки данных.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(prepare)
    prepare.add_argument(
        "--config",
        default="config/default.yaml",
        help="Путь к YAML-конфигу подготовки данных.",
    )
    prepare.add_argument(
        "--no-prefect",
        action="store_true",
        help="Запустить подготовку напрямую без оркестрации Prefect.",
    )

    chunk = subparsers.add_parser(
        "chunk",
        help="Запустить пайплайн чанкинга.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(chunk)
    chunk.add_argument(
        "--config",
        required=True,
        help="Путь к явному конфигу чанкинга для выбранной embedding-модели.",
    )
    chunk.add_argument(
        "--no-prefect",
        action="store_true",
        help="Запустить чанкинг напрямую без оркестрации Prefect.",
    )

    embed = subparsers.add_parser(
        "embed",
        help="Запустить пайплайн embeddings.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(embed)
    embed.add_argument(
        "--config",
        required=True,
        help="Путь к явному provider-конфигу embeddings.",
    )
    embed.add_argument(
        "--no-prefect",
        action="store_true",
        help="Запустить embeddings напрямую без оркестрации Prefect.",
    )

    vector_store = subparsers.add_parser(
        "vector-store",
        help="Загрузить embeddings в локальный vector store и выполнить проверки поиска.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(vector_store)
    vector_store.add_argument(
        "--config",
        required=True,
        help="Путь к явному конфигу vector store для выбранных embeddings.",
    )
    vector_store.add_argument(
        "--no-prefect",
        action="store_true",
        help="Запустить индексацию vector store напрямую без оркестрации Prefect.",
    )
    return parser


def main() -> None:
    """Запускает командный интерфейс и возвращает код завершения."""
    _configure_stdio()
    args = build_parser().parse_args()
    command = args.command or "prepare"

    if command == "chunk":
        config_path = args.config
        if args.no_prefect:
            from rag_prep.config import load_chunking_config
            from rag_prep.pipeline import RagChunkingPipeline
            from rag_prep.utils import setup_logging

            config = load_chunking_config(Path(config_path))
            setup_logging(config.logging.level)
            result = RagChunkingPipeline(config).run()
        else:
            _configure_local_prefect_runtime()
            from rag_prep.flow import rag_chunking_flow

            result = rag_chunking_flow(config_path=config_path)
    elif command == "embed":
        config_path = args.config
        if args.no_prefect:
            from rag_prep.config import load_embedding_config
            from rag_prep.pipeline import RagEmbeddingPipeline
            from rag_prep.utils import setup_logging

            config = load_embedding_config(Path(config_path))
            setup_logging(config.logging.level)
            result = RagEmbeddingPipeline(config).run()
        else:
            _configure_local_prefect_runtime()
            from rag_prep.flow import rag_embeddings_flow

            result = rag_embeddings_flow(config_path=config_path)
    elif command == "vector-store":
        config_path = args.config
        if args.no_prefect:
            from rag_prep.config import load_vector_store_config
            from rag_prep.pipeline import RagVectorStorePipeline
            from rag_prep.utils import setup_logging

            config = load_vector_store_config(Path(config_path))
            setup_logging(config.logging.level)
            result = RagVectorStorePipeline(config).run()
        else:
            _configure_local_prefect_runtime()
            from rag_prep.flow import rag_vector_store_flow

            result = rag_vector_store_flow(config_path=config_path)
    else:
        config_path = args.config or "config/default.yaml"
        if args.no_prefect:
            from rag_prep.config import load_config
            from rag_prep.pipeline import RagPreparationPipeline
            from rag_prep.utils import setup_logging

            config = load_config(Path(config_path))
            setup_logging(config.logging.level)
            result = RagPreparationPipeline(config).run()
        else:
            _configure_local_prefect_runtime()
            from rag_prep.flow import rag_data_preparation_flow

            result = rag_data_preparation_flow(config_path=config_path)

    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def _configure_local_prefect_runtime() -> None:
    """Гарантирует корректную локальную работу Prefect без конфликтов с прокси и лишней телеметрии."""
    _ensure_local_prefect_no_proxy()
    _disable_prefect_events_worker()


def _configure_stdio() -> None:
    """Переключает Windows stdout/stderr на UTF-8 до инициализации Prefect."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _ensure_local_prefect_no_proxy() -> None:
    """Гарантирует, что обращения к локальным адресам не будут перенаправлены через прокси-серверы."""
    local_hosts = ["localhost", "127.0.0.1", "::1"]
    for name in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(name, "")
        if "*" in existing.split(","):
            continue
        values = [value.strip() for value in existing.split(",") if value.strip()]
        for host in local_hosts:
            if host not in values:
                values.append(host)
        os.environ[name] = ",".join(values)


def _disable_prefect_events_worker() -> None:
    """Гарантирует отключение отправки событий Prefect для предотвращения лишней нагрузки и ошибок в локальном окружении."""
    try:
        from prefect.events.clients import NullEventsClient
        from prefect.events.worker import EventsWorker

        EventsWorker.set_client_override(NullEventsClient)
    except Exception:
        return


if __name__ == "__main__":
    main()
