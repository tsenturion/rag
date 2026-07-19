"""Статические регрессии локальной сетевой границы Docker Compose."""

from __future__ import annotations

import re
from pathlib import Path


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
