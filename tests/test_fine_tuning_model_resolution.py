"""Регрессионные тесты для подсистемы fine_tuning_model_resolution."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_tuning.config import (  # noqa: E402
    FineTuningPipelineConfig,
    LocalModelConfig,
    load_fine_tuning_config,
)
from llm_tuning.modeling import LocalCausalModelLoader  # noqa: E402


class FineTuningModelResolutionTest(unittest.TestCase):
    """Проверяет корректность разрешения путей моделей и загрузки токенизаторов в процессе настройки тонкой настройки моделей."""

    def test_local_model_path_is_resolved_independently_from_cwd(self) -> None:
        """Проверяет, что локальный путь к модели разрешается относительно конфигурационного файла независимо от текущей рабочей директории."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            project = Path(temporary_dir) / "project"
            config_dir = project / "config"
            model_dir = project / "data" / "models" / "primary"
            foreign_cwd = Path(temporary_dir) / "foreign"
            config_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            foreign_cwd.mkdir()
            config_path = config_dir / "fine_tuning.yaml"
            config_path.write_text(
                "model:\n"
                "  model_id: data/models/primary\n"
                "  fallback_model_id: Qwen/Qwen2.5-0.5B-Instruct\n",
                encoding="utf-8",
            )

            original_cwd = Path.cwd()
            try:
                os.chdir(foreign_cwd)
                config = load_fine_tuning_config(config_path)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(config.model.model_id, str(model_dir.resolve()))
            self.assertEqual(
                config.model.fallback_model_id,
                "Qwen/Qwen2.5-0.5B-Instruct",
            )

    def test_unavailable_primary_tokenizer_uses_fallback(self) -> None:
        """Проверяет, что при недоступности основного токенизатора используется запасной, обеспечивая устойчивость загрузки моделей."""
        config = FineTuningPipelineConfig(
            model=LocalModelConfig(
                model_id="missing-primary",
                fallback_model_id="available-fallback",
            )
        )
        loader = LocalCausalModelLoader(config)
        tokenizer = SimpleNamespace(
            pad_token=None,
            eos_token="<eos>",
            padding_side="left",
        )

        def load_tokenizer(model_id: str, **_kwargs):
            """Проверяет обработку загрузки токенизатора, включая корректное исключение при отсутствии модели, для тестирования разрешения моделей."""
            if model_id == "missing-primary":
                raise OSError("model unavailable")
            return tokenizer

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=load_tokenizer,
        ) as mocked_loader:
            result = loader.load_tokenizer()

        self.assertIs(result, tokenizer)
        self.assertEqual(loader.active_model_id, "available-fallback")
        self.assertEqual(mocked_loader.call_count, 2)
        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertEqual(tokenizer.padding_side, "right")

    def test_model_fallback_overrides_explicit_primary_tokenizer(self) -> None:
        """Не смешивает fallback-веса с tokenizer исходной недоступной модели."""
        config = FineTuningPipelineConfig(
            model=LocalModelConfig(
                model_id="missing-primary",
                tokenizer_id="primary-tokenizer",
                fallback_model_id="available-fallback",
            )
        )
        loader = LocalCausalModelLoader(config)
        model = SimpleNamespace(
            config=SimpleNamespace(use_cache=True),
            to=lambda _device: model,
        )
        tokenizer = SimpleNamespace(
            pad_token="<pad>",
            eos_token="<eos>",
            padding_side="left",
        )

        def load_model(model_id: str, **_kwargs):
            """Имитирует недоступность primary-весов и успешный fallback."""
            if model_id == "missing-primary":
                raise OSError("model unavailable")
            return model

        with (
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                side_effect=load_model,
            ),
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as tokenizer_loader,
            patch(
                "llm_tuning.modeling.build_device_report",
                return_value=SimpleNamespace(
                    selected_dtype="float32",
                    selected_device="cpu",
                ),
            ),
            patch("llm_tuning.modeling.torch_dtype", return_value=None),
        ):
            loader.load_base_model()
            loader.load_tokenizer()

        tokenizer_loader.assert_called_once()
        self.assertEqual(tokenizer_loader.call_args.args[0], "available-fallback")


if __name__ == "__main__":
    unittest.main()
