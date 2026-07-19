"""Статические регрессии локальной сетевой границы Docker Compose."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compose_publishes_only_loopback_ports() -> None:
    """Не позволяет учебной инфраструктуре открыться во внешнюю сеть хоста."""
    compose = (PROJECT_ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    published = re.findall(r'^\s+-\s+"([^"/]+:\d+:\d+)"\s*$', compose, re.MULTILINE)

    assert published
    assert all(binding.startswith("127.0.0.1:") for binding in published)


def test_compose_requires_non_default_infrastructure_passwords() -> None:
    """Запрещает тихий запуск RabbitMQ и Grafana с демонстрационным паролем."""
    compose = (PROJECT_ROOT / "docker-compose.yaml").read_text(encoding="utf-8")

    assert "${RABBITMQ_DEFAULT_PASS:?" in compose
    assert "${GRAFANA_ADMIN_PASSWORD:?" in compose
    assert "GRAFANA_ADMIN_PASSWORD:-admin" not in compose


def test_support_and_camunda_workers_have_readiness_healthchecks() -> None:
    """Проверяет readiness API и heartbeat event loop Camunda worker."""
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    )
    support_health = compose["services"]["support-agent"].get("healthcheck")
    camunda_health = compose["services"]["camunda-worker"]["healthcheck"]
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert support_health is None  # Healthcheck наследуется из Docker image.
    assert "/ready" in dockerfile
    assert camunda_health.get("disable") is not True
    assert "camunda-worker.health" in " ".join(camunda_health["test"])


def test_feature_branch_release_cannot_publish_latest() -> None:
    """Ограничивает mutable-тег latest основной веткой и release-тегами."""
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "container-release.yaml"
    ).read_text(encoding="utf-8")

    assert '"$GITHUB_REF_TYPE" == "tag"' in workflow
    assert '"$GITHUB_REF" == "refs/heads/main"' in workflow
    latest_line = 'echo "ghcr.io/${repository}/${{ matrix.image }}:latest"'
    assert workflow.index("if [[") < workflow.index(latest_line)


def test_docker_smoke_supplies_all_readiness_secrets() -> None:
    """Не позволяет чистому CI runner скрыто зависеть от локального файла .env."""
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "quality.yaml"
    ).read_text(encoding="utf-8")

    for name in (
        "SUPPORT_SERVICE_API_KEY",
        "SUPPORT_JWT_SECRET",
        "CODE_RUNNER_API_KEY",
        "RABBITMQ_DEFAULT_PASS",
    ):
        assert f"{name}:" in workflow
        assert f"printf '{name}=%s" in workflow
