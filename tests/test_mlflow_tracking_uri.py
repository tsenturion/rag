from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_tuning.config import load_fine_tuning_config  # noqa: E402
from rag_prep.config import PipelineConfig, load_config  # noqa: E402
from rag_prep.tracking import MLflowTracker  # noqa: E402


class MlflowTrackingUriTest(unittest.TestCase):
    def test_tracker_fallback_does_not_depend_on_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            original_cwd = Path.cwd()
            try:
                os.chdir(temporary_dir)
                uri = MLflowTracker(PipelineConfig())._tracking_uri()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(uri, (PROJECT_ROOT / "mlruns").resolve().as_uri())

    def test_rag_tracking_uri_is_resolved_from_config_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            project = Path(temporary_dir) / "project"
            config_dir = project / "config"
            foreign_cwd = Path(temporary_dir) / "foreign"
            config_dir.mkdir(parents=True)
            foreign_cwd.mkdir()
            config_path = config_dir / "default.yaml"
            config_path.write_text(
                "logging:\n  mlflow_tracking_uri: mlruns\n",
                encoding="utf-8",
            )

            config = self._load_from_foreign_cwd(
                foreign_cwd,
                load_config,
                config_path,
            )

            expected = (project / "mlruns").resolve()
            self.assertEqual(config.logging.mlflow_tracking_uri, str(expected))
            self.assertEqual(MLflowTracker(config)._tracking_uri(), expected.as_uri())

    def test_fine_tuning_tracking_uri_uses_same_resolution_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            project = Path(temporary_dir) / "project"
            config_dir = project / "config"
            foreign_cwd = Path(temporary_dir) / "foreign"
            config_dir.mkdir(parents=True)
            foreign_cwd.mkdir()
            config_path = config_dir / "fine_tuning.yaml"
            config_path.write_text(
                "logging:\n  mlflow_tracking_uri: metrics/mlruns\n",
                encoding="utf-8",
            )

            config = self._load_from_foreign_cwd(
                foreign_cwd,
                load_fine_tuning_config,
                config_path,
            )

            self.assertEqual(
                config.logging.mlflow_tracking_uri,
                str((project / "metrics" / "mlruns").resolve()),
            )

    @staticmethod
    def _load_from_foreign_cwd(foreign_cwd: Path, loader, config_path: Path):
        original_cwd = Path.cwd()
        try:
            os.chdir(foreign_cwd)
            return loader(config_path)
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
