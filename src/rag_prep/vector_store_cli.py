from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_prep.config import load_vector_store_config
from rag_prep.models import VectorStorePipelineResult
from rag_prep.utils import new_run_id, setup_logging
from rag_prep.vector_store_stages.client import qdrant_client_context
from rag_prep.vector_store_stages.exporting import VectorStoreExportStage
from rag_prep.vector_store_stages.indexing import QdrantIndexingStage
from rag_prep.vector_store_stages.loading import EmbeddingLoadingStage
from rag_prep.vector_store_stages.metrics import (
    build_vector_store_counts,
    build_vector_store_diagnostics,
)
from rag_prep.vector_store_stages.search import QdrantSearchStage
from rag_prep.vector_store_stages.validation import QdrantValidationStage


def run_index(config_path: str | Path) -> VectorStorePipelineResult:
    config = load_vector_store_config(config_path)
    setup_logging(config.logging.level)
    run_id = new_run_id()
    embedded_chunks = EmbeddingLoadingStage().run(config.paths.input_jsonl)
    with qdrant_client_context(config.vector_store) as client:
        index = QdrantIndexingStage(config.vector_store).run(
            embedded_chunks,
            client=client,
        )
        validation = QdrantValidationStage(config.vector_store).run(
            embedded_chunks,
            client=client,
        )
        search_results = QdrantSearchStage(config.vector_store).run(
            embedded_chunks,
            client=client,
        )
    counts = build_vector_store_counts(
        config,
        embedded_chunks,
        index,
        validation,
        search_results,
    )
    export = VectorStoreExportStage(config).run(
        index=index,
        validation=validation,
        search_results=search_results,
        run_id=run_id,
        counts=counts,
        diagnostics=build_vector_store_diagnostics(validation, search_results),
    )
    return VectorStorePipelineResult(
        run_id=run_id,
        embeddings_count=len(embedded_chunks),
        points_count=index.collection_points_count,
        search_results_count=len(search_results),
        validation=validation,
        export=export,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Загрузка готовых embeddings в Qdrant без offline preprocessing stack."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Путь к явному YAML-конфигу Qdrant для выбранных embeddings.",
    )
    args = parser.parse_args()
    result = run_index(args.config)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
