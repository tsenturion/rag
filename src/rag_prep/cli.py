from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag_prep.config import load_config
from rag_prep.pipeline import RagPreparationPipeline
from rag_prep.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare PDF/TXT/HTML/CSV files for RAG.")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config. Defaults to config/default.yaml.",
    )
    parser.add_argument(
        "--no-prefect",
        action="store_true",
        help="Run the same OOP pipeline directly without Prefect orchestration.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.no_prefect:
        config = load_config(Path(args.config))
        setup_logging(config.logging.level)
        result = RagPreparationPipeline(config).run()
    else:
        _ensure_local_prefect_no_proxy()
        from rag_prep.flow import rag_data_preparation_flow

        result = rag_data_preparation_flow(config_path=args.config)

    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def _ensure_local_prefect_no_proxy() -> None:
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")


if __name__ == "__main__":
    main()
