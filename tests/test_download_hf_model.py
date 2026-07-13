from __future__ import annotations

import io
import importlib.util
import unittest
from contextlib import redirect_stderr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "download_hf_model.py"
SPEC = importlib.util.spec_from_file_location("download_hf_model_script", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Не удалось загрузить скрипт: {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_parser = MODULE.build_parser


class DownloadHfModelCliTest(unittest.TestCase):
    def test_model_and_destination_are_required(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_qwen_and_e5_use_the_same_cli_contract(self) -> None:
        cases = (
            (
                "Qwen/Qwen2.5-1.5B-Instruct",
                Path("data/models/hf/Qwen2.5-1.5B-Instruct"),
            ),
            (
                "intfloat/multilingual-e5-small",
                Path("data/models/hf/multilingual-e5-small"),
            ),
        )
        for model_id, local_dir in cases:
            with self.subTest(model_id=model_id):
                args = build_parser().parse_args(
                    [
                        "--model-id",
                        model_id,
                        "--local-dir",
                        str(local_dir),
                    ]
                )
                self.assertEqual(args.model_id, model_id)
                self.assertEqual(args.local_dir, local_dir)


if __name__ == "__main__":
    unittest.main()
