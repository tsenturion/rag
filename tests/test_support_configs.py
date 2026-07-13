from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.cli import build_parser as build_agent_parser  # noqa: E402
from agent_app.config import AgentConfig, load_agent_config  # noqa: E402
from agent_app.service.cli import build_parser  # noqa: E402
from rag_prep.cli import build_parser as build_rag_parser  # noqa: E402
from rag_prep.config import EmbeddingConfig, VectorStoreConfig  # noqa: E402
from rag_prep.config import (  # noqa: E402
    load_chunking_config,
    load_embedding_config,
    load_vector_store_config,
)


class SupportProviderConfigsTest(unittest.TestCase):
    def test_service_cli_reads_config_selector_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / ".env").write_text(
                "SUPPORT_AGENT_CONFIG=config/support_agent_local.yaml\n",
                encoding="utf-8",
            )
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("agent_app.service.cli.Path.cwd", return_value=root),
            ):
                args = build_parser().parse_args([])

        self.assertEqual(args.config, "config/support_agent_local.yaml")

    def test_rag_profile_is_shared_by_all_pipeline_stages(self) -> None:
        expected = {
            "openai": ("text-embedding-3-small", 1536, "rag_chunks_openai"),
            "local": (
                "multilingual-e5-small",
                384,
                "rag_chunks_local",
            ),
            "gigachat": ("Embeddings-2", 1024, "rag_chunks_gigachat"),
        }
        for provider, values in expected.items():
            with self.subTest(provider=provider):
                chunking = load_chunking_config(
                    PROJECT_ROOT / "config" / f"chunking_{provider}.yaml"
                )
                embedding = load_embedding_config(
                    PROJECT_ROOT / "config" / f"embeddings_{provider}.yaml"
                )
                vector_store = load_vector_store_config(
                    PROJECT_ROOT / "config" / f"vector_store_{provider}.yaml"
                )
                model, dimensions, collection = values
                self.assertTrue(chunking.chunking.embedding_model.endswith(model))
                self.assertTrue(embedding.embedding.model.endswith(model))
                self.assertEqual(embedding.embedding.dimensions, dimensions)
                self.assertEqual(vector_store.vector_store.vector_size, dimensions)
                self.assertEqual(
                    vector_store.vector_store.collection_name,
                    collection,
                )

    def test_provider_selection_has_no_implicit_openai_fallback(self) -> None:
        with self.assertRaises(ValidationError):
            AgentConfig.model_validate({})
        with self.assertRaises(ValidationError):
            EmbeddingConfig.model_validate({})
        with self.assertRaises(ValidationError):
            VectorStoreConfig.model_validate({})

        for command in ("chunk", "embed", "vector-store"):
            with self.subTest(command=command), self.assertRaises(SystemExit):
                build_rag_parser().parse_args([command])
        with self.assertRaises(SystemExit):
            build_agent_parser().parse_args([])

    def test_service_cli_uses_provider_config_from_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SUPPORT_AGENT_CONFIG": (
                    "config/support_agent_docker_gigachat_openai_embeddings.yaml"
                )
            },
        ):
            args = build_parser().parse_args([])

        self.assertEqual(
            args.config,
            "config/support_agent_docker_gigachat_openai_embeddings.yaml",
        )

    def test_provider_presets_are_dimensionally_compatible(self) -> None:
        expected = {
            "support_agent_openai.yaml": ("openai", "openai", 1536, "local"),
            "support_agent_docker_openai.yaml": (
                "openai",
                "openai",
                1536,
                "http",
            ),
            "support_agent_gigachat_openai_embeddings.yaml": (
                "gigachat",
                "openai",
                1536,
                "local",
            ),
            "support_agent_gigachat_local_embeddings.yaml": (
                "gigachat",
                "local",
                384,
                "local",
            ),
            "support_agent_local.yaml": ("local", "local", 384, "local"),
            "support_agent_docker_gigachat_openai_embeddings.yaml": (
                "gigachat",
                "openai",
                1536,
                "http",
            ),
            "support_agent_docker_gigachat_local_embeddings.yaml": (
                "gigachat",
                "local",
                384,
                "http",
            ),
            "support_agent_docker_local.yaml": (
                "local",
                "local",
                384,
                "http",
            ),
        }

        for filename, values in expected.items():
            with self.subTest(config=filename):
                config = load_agent_config(PROJECT_ROOT / "config" / filename)
                llm_provider, embedding_provider, dimensions, mode = values
                self.assertEqual(config.agent.provider, llm_provider)
                self.assertEqual(config.rag.embedding.provider, embedding_provider)
                self.assertEqual(config.rag.embedding.dimensions, dimensions)
                self.assertEqual(config.rag.vector_store.vector_size, dimensions)
                self.assertEqual(config.rag.vector_store.mode, mode)
                expected_collection = (
                    "rag_chunks_openai"
                    if embedding_provider == "openai"
                    else "rag_chunks_local"
                )
                self.assertEqual(
                    config.rag.vector_store.collection_name,
                    expected_collection,
                )
                self.assertIn("get_weather", config.tools.enabled)

    def test_local_presets_resolve_model_paths_from_project_root(self) -> None:
        for filename in (
            "support_agent_gigachat_local_embeddings.yaml",
            "support_agent_local.yaml",
            "support_agent_docker_gigachat_local_embeddings.yaml",
            "support_agent_docker_local.yaml",
        ):
            with self.subTest(config=filename):
                config = load_agent_config(PROJECT_ROOT / "config" / filename)
                embedding_model = Path(config.rag.embedding.model)
                self.assertTrue(embedding_model.is_absolute())
                self.assertEqual(embedding_model.name, "multilingual-e5-small")
                self.assertEqual(
                    config.rag.vector_store.collection_name,
                    "rag_chunks_local",
                )

        local = load_agent_config(PROJECT_ROOT / "config" / "support_agent_local.yaml")
        self.assertTrue(Path(local.agent.model).is_absolute())
        self.assertEqual(Path(local.agent.model).name, "Qwen2.5-1.5B-Instruct")

    def test_docker_presets_require_service_api_key(self) -> None:
        for filename in (
            "support_agent_docker_openai.yaml",
            "support_agent_docker_gigachat_openai_embeddings.yaml",
            "support_agent_docker_gigachat_local_embeddings.yaml",
            "support_agent_docker_local.yaml",
        ):
            with self.subTest(config=filename):
                config = load_agent_config(PROJECT_ROOT / "config" / filename)
                self.assertTrue(config.security.require_api_key)
                self.assertEqual(
                    config.security.api_key_env,
                    "SUPPORT_SERVICE_API_KEY",
                )


if __name__ == "__main__":
    unittest.main()
