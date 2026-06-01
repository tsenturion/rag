from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag_prep.config import (
    load_chunking_config,
    load_config,
    load_embedding_config,
    load_vector_store_config,
)
from rag_prep.pipeline import (
    RagChunkingPipeline,
    RagEmbeddingPipeline,
    RagPreparationPipeline,
    RagVectorStorePipeline,
)
from rag_prep.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RAG preparation, chunking, embedding, and vector store pipelines."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Legacy prepare config path. Defaults to config/default.yaml.",
    )
    parser.add_argument(
        "--no-prefect",
        action="store_true",
        help="Legacy flag: run preparation directly without Prefect orchestration.",
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare", help="Run data preparation pipeline.")
    prepare.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to preparation YAML config.",
    )
    prepare.add_argument(
        "--no-prefect",
        action="store_true",
        help="Run preparation directly without Prefect orchestration.",
    )

    chunk = subparsers.add_parser("chunk", help="Run chunking pipeline.")
    chunk.add_argument(
        "--config",
        default="config/chunking.yaml",
        help="Path to chunking YAML config.",
    )
    chunk.add_argument(
        "--no-prefect",
        action="store_true",
        help="Run chunking directly without Prefect orchestration.",
    )

    embed = subparsers.add_parser("embed", help="Run embeddings pipeline.")
    embed.add_argument(
        "--config",
        default="config/embeddings.yaml",
        help="Path to embeddings YAML config.",
    )
    embed.add_argument(
        "--no-prefect",
        action="store_true",
        help="Run embeddings directly without Prefect orchestration.",
    )

    vector_store = subparsers.add_parser(
        "vector-store",
        help="Load embeddings into a local vector store and run search checks.",
    )
    vector_store.add_argument(
        "--config",
        default="config/vector_store.yaml",
        help="Path to vector store YAML config.",
    )
    vector_store.add_argument(
        "--no-prefect",
        action="store_true",
        help="Run vector store indexing directly without Prefect orchestration.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "prepare"

    if command == "chunk":
        config_path = args.config or "config/chunking.yaml"
        if args.no_prefect:
            config = load_chunking_config(Path(config_path))
            setup_logging(config.logging.level)
            result = RagChunkingPipeline(config).run()
        else:
            _configure_local_prefect_runtime()
            from rag_prep.flow import rag_chunking_flow

            result = rag_chunking_flow(config_path=config_path)
    elif command == "embed":
        config_path = args.config or "config/embeddings.yaml"
        if args.no_prefect:
            config = load_embedding_config(Path(config_path))
            setup_logging(config.logging.level)
            result = RagEmbeddingPipeline(config).run()
        else:
            _configure_local_prefect_runtime()
            from rag_prep.flow import rag_embeddings_flow

            result = rag_embeddings_flow(config_path=config_path)
    elif command == "vector-store":
        config_path = args.config or "config/vector_store.yaml"
        if args.no_prefect:
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
            config = load_config(Path(config_path))
            setup_logging(config.logging.level)
            result = RagPreparationPipeline(config).run()
        else:
            _configure_local_prefect_runtime()
            from rag_prep.flow import rag_data_preparation_flow

            result = rag_data_preparation_flow(config_path=config_path)

    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def _configure_local_prefect_runtime() -> None:
    _ensure_local_prefect_no_proxy()
    _disable_prefect_events_worker()


def _ensure_local_prefect_no_proxy() -> None:
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
    try:
        from prefect.events.clients import NullEventsClient
        from prefect.events.worker import EventsWorker

        EventsWorker.set_client_override(NullEventsClient)
    except Exception:
        return


if __name__ == "__main__":
    main()
