"""Регрессии воспроизводимой TLS-настройки реальных API-проверок."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIGACHAT_CA_RELATIVE_PATH = Path("data/certs/russian_trusted_root_ca_pem.crt")
GIGACHAT_CA_SHA256 = "936a43fea6e8e525bcc0f81acd9c3d21b4fc4b9b68acea7906d698005afc6504"


def test_live_api_uses_pinned_gigachat_root_ca() -> None:
    """Не допускает запуск GigaChat smoke без проверенного trust anchor."""
    certificate = PROJECT_ROOT / GIGACHAT_CA_RELATIVE_PATH
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "live-api.yaml").read_text(
        encoding="utf-8"
    )

    assert certificate.is_file()
    assert sha256(certificate.read_bytes()).hexdigest() == GIGACHAT_CA_SHA256
    assert str(GIGACHAT_CA_RELATIVE_PATH).replace("\\", "/") in workflow
    assert GIGACHAT_CA_SHA256 in workflow
    assert "sha256sum --check --strict" in workflow
    assert "openssl x509" in workflow
    assert "-checkend 86400" in workflow
