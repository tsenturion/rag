"""Регрессионные тесты для подсистемы mlflow_tracking_uri."""

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
from agent_app.config import MultiAgentConfig  # noqa: E402
from agent_app.multi_agent.tracking import MultiAgentTracker  # noqa: E402
from rag_prep.config import PipelineConfig, load_config  # noqa: E402
from rag_prep.tracking import MLflowTracker  # noqa: E402
from rag_prep.mlflow_uri import resolve_mlflow_tracking_uri  # noqa: E402


class MlflowTrackingUriTest(unittest.TestCase):
    """Проверяет корректность формирования URI для трекинга MLflow в разных конфигурациях и условиях окружения."""

    def test_multi_agent_tracker_uses_file_uri_for_windows_path(self) -> None:
        """Проверяет, что MultiAgentTracker корректно преобразует путь Windows в URI для MLflow трекинга."""
        path = (PROJECT_ROOT / "mlruns").resolve()
        config = MultiAgentConfig(mlflow_tracking_uri=str(path))

        uri = MultiAgentTracker(config)._tracking_uri()

        self.assertEqual(uri, path.as_uri())

    def test_tracker_fallback_does_not_depend_on_cwd(self) -> None:
        """Проверяет, что MLflowTracker формирует URI трекинга независимо от текущей рабочей директории процесса."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            original_cwd = Path.cwd()
            try:
                os.chdir(temporary_dir)
                uri = MLflowTracker(PipelineConfig())._tracking_uri()
            finally:
                os.chdir(original_cwd)

        expected = (PROJECT_ROOT / "mlruns" / "mlflow.db").resolve().as_posix()
        self.assertEqual(uri, f"sqlite:///{expected}")

    def test_sqlite_tracking_uri_is_resolved_from_project_root(self) -> None:
        """Проверяет, что относительный путь к SQLite базе данных корректно разрешается относительно корня проекта, обеспечивая правильное формирование абсолютного URI для MLflow."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "project"

            uri = resolve_mlflow_tracking_uri(
                "sqlite:///metrics/mlflow.db",
                base_dir=root,
            )

        self.assertEqual(
            uri,
            f"sqlite:///{(root / 'metrics' / 'mlflow.db').resolve().as_posix()}",
        )

    def test_rag_tracking_uri_is_resolved_from_config_root(self) -> None:
        """Проверяет, что MLflow tracking URI из конфигурации корректно разрешается относительно корня проекта, даже если загрузка происходит из другого рабочего каталога."""
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
        """Проверяет, что MLflow tracking URI для fine-tuning конфигурации разрешается по тем же правилам, что и для основной конфигурации, обеспечивая консистентность путей."""
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
        """Проверяет, что загрузка конфигурации из чужого рабочего каталога не зависит от текущего положения процесса."""
        original_cwd = Path.cwd()
        try:
            os.chdir(foreign_cwd)
            return loader(config_path)
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
