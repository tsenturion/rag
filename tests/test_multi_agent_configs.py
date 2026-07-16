from __future__ import annotations

import unittest
from pathlib import Path

from agent_app.config import load_agent_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MultiAgentConfigTest(unittest.TestCase):
    def test_host_presets_have_explicit_provider_and_compatible_rag(self) -> None:
        expected = {
            "multi_agent_openai.yaml": ("openai", "parallel"),
            "multi_agent_gigachat.yaml": ("gigachat", "parallel"),
            "multi_agent_local.yaml": ("local", "sequential"),
        }

        for filename, (provider, execution_mode) in expected.items():
            with self.subTest(filename=filename):
                config = load_agent_config(PROJECT_ROOT / "config" / filename)
                self.assertTrue(config.multi_agent.enabled)
                self.assertEqual(config.agent.provider, provider)
                self.assertEqual(config.multi_agent.execution_mode, execution_mode)
                self.assertIsNotNone(config.rag.embedding)
                self.assertIsNotNone(config.rag.vector_store)
                self.assertEqual(
                    config.rag.embedding.dimensions,
                    config.rag.vector_store.vector_size,
                )

    def test_docker_presets_require_api_key_and_server_qdrant(self) -> None:
        filenames = (
            "multi_agent_docker_openai.yaml",
            "multi_agent_docker_gigachat_openai_embeddings.yaml",
            "multi_agent_docker_gigachat_local_embeddings.yaml",
            "multi_agent_docker_local.yaml",
            "multi_agent_docker_mixed.yaml",
        )

        for filename in filenames:
            with self.subTest(filename=filename):
                config = load_agent_config(PROJECT_ROOT / "config" / filename)
                self.assertTrue(config.multi_agent.enabled)
                self.assertTrue(config.security.require_api_key)
                self.assertEqual(config.service.host, "0.0.0.0")
                self.assertIsNotNone(config.rag.vector_store)
                self.assertEqual(config.rag.vector_store.mode, "http")

    def test_mixed_preset_assigns_provider_profiles_to_roles(self) -> None:
        config = load_agent_config(PROJECT_ROOT / "config" / "multi_agent_mixed.yaml")

        self.assertEqual(
            config.multi_agent.role_llm_profiles["coordinator"],
            "openai_coordination",
        )
        self.assertEqual(
            config.multi_agent.llm_profiles["openai_coordination"].provider,
            "openai",
        )
        self.assertEqual(
            config.multi_agent.llm_profiles["gigachat_review"].provider,
            "gigachat",
        )
        local_profile = config.multi_agent.llm_profiles["local_incidents"]
        self.assertEqual(local_profile.provider, "local")
        self.assertTrue(Path(local_profile.model).is_absolute())


if __name__ == "__main__":
    unittest.main()
