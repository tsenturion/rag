"""Регрессии воспроизводимой TLS-настройки реальных API-проверок."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import ssl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIGACHAT_CA_RELATIVE_PATH = Path("data/certs/russian_trusted_root_ca_pem.crt")
GIGACHAT_CA_FINGERPRINT = (
    "d26d2d0231b7c39f92cc738512ba54103519e4405d68b5bd703e9788ca8ecf31"
)


def test_live_api_uses_pinned_gigachat_root_ca() -> None:
    """Не допускает запуск GigaChat smoke без проверенного trust anchor."""
    certificate = PROJECT_ROOT / GIGACHAT_CA_RELATIVE_PATH
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "live-api.yaml").read_text(
        encoding="utf-8"
    )

    assert certificate.is_file()
    certificate_der = ssl.PEM_cert_to_DER_cert(certificate.read_text(encoding="ascii"))
    assert sha256(certificate_der).hexdigest() == GIGACHAT_CA_FINGERPRINT
    assert str(GIGACHAT_CA_RELATIVE_PATH).replace("\\", "/") in workflow
    assert GIGACHAT_CA_FINGERPRINT in workflow
    assert "-outform DER" in workflow
    assert "sha256sum" in workflow
    assert "openssl x509" in workflow
    assert "-checkend 86400" in workflow
