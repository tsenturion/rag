from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_app.scenarios.models import ScenarioSuite


def load_scenario_suite(path: str | Path = "config/agent_scenarios.yaml") -> ScenarioSuite:
    config_path = _resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    suite = ScenarioSuite.model_validate(raw)
    if not suite.report_path.is_absolute():
        suite = suite.model_copy(
            update={"report_path": (_config_base_dir(config_path) / suite.report_path).resolve()}
        )
    return suite


def _resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path).expanduser()
    if config_path.is_absolute() or config_path.exists():
        return config_path.resolve()

    project_root = Path(__file__).resolve().parents[3]
    project_config_path = project_root / config_path
    if project_config_path.exists():
        return project_config_path.resolve()

    return config_path.resolve()


def _config_base_dir(config_path: Path) -> Path:
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent
