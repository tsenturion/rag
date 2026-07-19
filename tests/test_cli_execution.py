"""Проверки CLI-диспетчеризации без запуска сетевых сервисов и моделей."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from rag_prep.config import load_vector_store_config
from rag_prep.models import (
    VectorStoreExportResult,
    VectorStoreIndexResult,
    VectorStorePipelineResult,
    VectorStoreValidationResult,
)


class _Payload:
    """Предоставляет общий model_dump-контракт для CLI-результатов."""

    def __init__(self, value: str) -> None:
        """Сохраняет значение, которое должно попасть в JSON CLI."""
        self.value = value

    def model_dump(self, **_kwargs) -> dict[str, str]:
        """Возвращает JSON-совместимый результат выбранной команды."""
        return {"result": self.value}


def test_rag_index_runs_all_stages_with_one_qdrant_client(tmp_path: Path) -> None:
    """Проверяет порядок load/index/validate/search/export и общий client context."""
    from rag_prep import vector_store_cli

    config = load_vector_store_config("config/vector_store_openai.yaml")
    client = object()
    embedded = [object(), object()]
    index = VectorStoreIndexResult(
        collection_name="collection",
        provider="openai",
        mode="local",
        points_upserted=2,
        collection_points_count=2,
        vector_size=config.vector_store.vector_size,
        distance=config.vector_store.distance,
    )
    validation = VectorStoreValidationResult(
        embeddings_count=2,
        collection_points_count=2,
        verified_points_count=2,
    )
    export = VectorStoreExportResult(
        manifest_path=tmp_path / "manifest.json",
        validation_path=tmp_path / "validation.json",
        search_results_path=tmp_path / "search.json",
        run_id="index-run",
    )

    @contextmanager
    def client_context(_config):
        """Передаёт один и тот же тестовый клиент всем Qdrant stages."""
        yield client

    loading = Mock()
    loading.run.return_value = embedded
    indexing = Mock()
    indexing.run.return_value = index
    validating = Mock()
    validating.run.return_value = validation
    searching = Mock()
    searching.run.return_value = []
    exporting = Mock()
    exporting.run.return_value = export
    with (
        patch.object(vector_store_cli, "load_vector_store_config", return_value=config),
        patch.object(vector_store_cli, "new_run_id", return_value="index-run"),
        patch.object(vector_store_cli, "qdrant_client_context", client_context),
        patch.object(vector_store_cli, "EmbeddingLoadingStage", return_value=loading),
        patch.object(vector_store_cli, "QdrantIndexingStage", return_value=indexing),
        patch.object(vector_store_cli, "QdrantValidationStage", return_value=validating),
        patch.object(vector_store_cli, "QdrantSearchStage", return_value=searching),
        patch.object(vector_store_cli, "VectorStoreExportStage", return_value=exporting),
    ):
        result = vector_store_cli.run_index("vector.yaml")

    assert result.run_id == "index-run"
    assert result.embeddings_count == 2
    assert result.points_count == 2
    indexing.run.assert_called_once_with(embedded, client=client)
    validating.run.assert_called_once_with(embedded, client=client)
    searching.run.assert_called_once_with(embedded, client=client)
    assert exporting.run.call_args.kwargs["counts"]["embeddings_count"] == 2


def test_rag_index_main_prints_machine_readable_result(capsys) -> None:
    """Проверяет обязательный config и сериализацию результата rag-index."""
    from rag_prep import vector_store_cli

    result = VectorStorePipelineResult(
        run_id="run",
        embeddings_count=0,
        points_count=0,
        search_results_count=0,
        validation=VectorStoreValidationResult(),
        export=VectorStoreExportResult(
            manifest_path=Path("manifest.json"),
            validation_path=Path("validation.json"),
            search_results_path=Path("search.json"),
            run_id="run",
        ),
    )
    with (
        patch.object(sys, "argv", ["rag-index", "--config", "config.yaml"]),
        patch.object(vector_store_cli, "run_index", return_value=result) as run_index,
    ):
        vector_store_cli.main()

    assert json.loads(capsys.readouterr().out)["run_id"] == "run"
    run_index.assert_called_once_with("config.yaml")


@pytest.mark.parametrize(
    ("arguments", "method"),
    [
        (["inspect-env"], "validate"),
        (["validate-data"], "validate"),
        (["baseline"], "run_baseline"),
        (["train"], "run_training"),
        (["evaluate", "--adapter-path", "adapter"], "run_evaluation"),
        (
            [
                "compare",
                "--baseline-report",
                "baseline.json",
                "--tuned-report",
                "tuned.json",
            ],
            "compare_reports",
        ),
    ],
)
def test_llm_tune_cli_dispatches_pipeline_commands(
    arguments: list[str], method: str, capsys
) -> None:
    """Проверяет все pipeline-команды llm-tune и их JSON-контракт."""
    from llm_tuning import cli

    validation = _Payload("validation")
    device = _Payload("device")
    pipeline = SimpleNamespace(
        validate=Mock(return_value=(validation, device)),
        run_baseline=Mock(return_value=_Payload("baseline")),
        run_training=Mock(return_value=_Payload("training")),
        run_evaluation=Mock(return_value=_Payload("evaluation")),
        compare_reports=Mock(return_value=_Payload("comparison")),
    )
    config = SimpleNamespace(logging=SimpleNamespace(level="INFO"))
    with (
        patch.object(sys, "argv", ["llm-tune", "--config", "fine.yaml", *arguments]),
        patch.object(cli, "load_fine_tuning_config", return_value=config),
        patch.object(cli, "FineTuningPipeline", return_value=pipeline),
    ):
        cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert getattr(pipeline, method).called


def test_llm_tune_generate_dispatches_local_generation(capsys) -> None:
    """Проверяет передачу prompt, system, adapter и token limit в generation stage."""
    from llm_tuning import cli

    config = SimpleNamespace(logging=SimpleNamespace(level="INFO"))
    generation = Mock()
    generation.run.return_value = _Payload("generated")
    with (
        patch.object(
            sys,
            "argv",
            [
                "llm-tune",
                "--config",
                "fine.yaml",
                "generate",
                "--prompt",
                "запрос",
                "--system",
                "инструкция",
                "--adapter-path",
                "adapter",
                "--max-new-tokens",
                "17",
            ],
        ),
        patch.object(cli, "load_fine_tuning_config", return_value=config),
        patch.object(cli, "FineTuningPipeline"),
        patch.object(cli, "LocalGenerationStage", return_value=generation),
    ):
        cli.main()

    assert json.loads(capsys.readouterr().out) == {"result": "generated"}
    assert generation.run.call_args.kwargs["max_new_tokens"] == 17


def test_code_runner_cli_passes_network_options_to_uvicorn() -> None:
    """Проверяет, что service CLI запускает ровно один Uvicorn worker."""
    from code_runner import cli

    with (
        patch.object(
            sys,
            "argv",
            ["rag-code-runner", "--host", "127.0.0.2", "--port", "8123"],
        ),
        patch("uvicorn.run") as run,
    ):
        cli.main()

    run.assert_called_once_with(
        "code_runner.app:app", host="127.0.0.2", port=8123, workers=1
    )
