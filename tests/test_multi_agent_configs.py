"""Регрессионные тесты для подсистемы multi_agent_configs."""

from __future__ import annotations

import unittest
from pathlib import Path

from agent_app.config import load_agent_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MultiAgentConfigTest(unittest.TestCase):
    """Проверяет корректность загрузки и совместимости конфигураций multi-agent, включая провайдеров, режимы исполнения и параметры RAG, для обеспечения согласованной работы подсистемы."""

    def test_host_presets_have_explicit_provider_and_compatible_rag(self) -> None:
        """Проверяет, что хостовые пресеты содержат явные провайдеры и совместимые параметры RAG, включая абсолютные пути и согласованные размеры векторов и эмбеддингов."""
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
                self.assertTrue(config.multi_agent.checkpoint_path.is_absolute())
                self.assertTrue(config.file_tools.workspace_path.is_absolute())
                self.assertEqual(
                    config.code_runner.base_url,
                    "http://127.0.0.1:8010",
                )
                self.assertIsNotNone(config.rag.embedding)
                self.assertIsNotNone(config.rag.vector_store)
                self.assertEqual(
                    config.rag.embedding.dimensions,
                    config.rag.vector_store.vector_size,
                )

    def test_docker_presets_require_api_key_and_server_qdrant(self) -> None:
        """Проверяет, что докер-пресеты требуют обязательного API-ключа, используют сервер Qdrant и корректно настраивают параметры безопасности и сервисов для безопасного и корректного запуска."""
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
                self.assertEqual(
                    config.code_runner.base_url,
                    "http://code-runner:8010",
                )
                self.assertIsNotNone(config.rag.vector_store)
                self.assertEqual(config.rag.vector_store.mode, "http")

    def test_observability_presets_enable_jwt_security(self) -> None:
        """Проверяет единый JWT-контракт observability-пресетов без копирования security-настроек по provider-файлам."""
        filenames = (
            "multi_agent_docker_openai_observability.yaml",
            "multi_agent_docker_gigachat_openai_embeddings_observability.yaml",
            "multi_agent_docker_gigachat_local_embeddings_observability.yaml",
            "multi_agent_docker_local_observability.yaml",
            "multi_agent_docker_mixed_observability.yaml",
        )

        for filename in filenames:
            with self.subTest(filename=filename):
                config = load_agent_config(PROJECT_ROOT / "config" / filename)
                self.assertTrue(config.observability.enabled)
                self.assertTrue(config.security.jwt_enabled)
                self.assertTrue(config.security.enforce_user_scope)
                self.assertEqual(
                    config.security.jwt_secret_env,
                    "SUPPORT_JWT_SECRET",
                )

    def test_mixed_preset_assigns_provider_profiles_to_roles(self) -> None:
        """Проверяет, что смешанный пресет правильно сопоставляет профили провайдеров с ролями агентов, гарантируя корректное распределение моделей и абсолютные пути к локальным моделям."""
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
