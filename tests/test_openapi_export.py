"""Проверки воспроизводимого контракта для будущего frontend-клиента."""

from __future__ import annotations

import json
from pathlib import Path

from agent_app.service.openapi_cli import export_openapi

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_openapi_export_contains_frontend_endpoints(tmp_path: Path) -> None:
    """Экспортирует ту же схему, которую обслуживает FastAPI, без запуска LLM."""
    output = export_openapi(
        PROJECT_ROOT / "config" / "support_agent_local.yaml",
        tmp_path / "contract" / "support-api.json",
    )

    schema = json.loads(output.read_text(encoding="utf-8"))

    assert schema["info"]["version"] == "1.1.0"
    assert "/v1/app/config" in schema["paths"]
    assert "/v1/auth/me" in schema["paths"]
    assert "/v1/sessions" in schema["paths"]
    assert "ApiValidationDetail" in schema["components"]["schemas"]
    assert not output.with_name(f".{output.name}.tmp").exists()
