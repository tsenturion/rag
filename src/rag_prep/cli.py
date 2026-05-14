from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag_prep.config import load_chunking_config, load_config
from rag_prep.pipeline import RagChunkingPipeline, RagPreparationPipeline
from rag_prep.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAG preparation and chunking pipelines.")
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
            _ensure_local_prefect_no_proxy()
            from rag_prep.flow import rag_chunking_flow

            result = rag_chunking_flow(config_path=config_path)
    else:
        config_path = args.config or "config/default.yaml"
        if args.no_prefect:
            config = load_config(Path(config_path))
            setup_logging(config.logging.level)
            result = RagPreparationPipeline(config).run()
        else:
            _ensure_local_prefect_no_proxy()
            from rag_prep.flow import rag_data_preparation_flow

            result = rag_data_preparation_flow(config_path=config_path)

    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def _ensure_local_prefect_no_proxy() -> None:
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")


if __name__ == "__main__":
    main()
