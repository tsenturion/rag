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
    def test_local_model_path_is_resolved_independently_from_cwd(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
