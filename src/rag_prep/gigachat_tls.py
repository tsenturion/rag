"""Разрешение доверенного CA bundle для клиентов GigaChat."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_gigachat_ca_bundle() -> str | None:
    """Возвращает существующий CA bundle независимо от рабочего каталога CLI.

    SDK также умеет читать ``GIGACHAT_CA_BUNDLE_FILE`` самостоятельно, но
    относительный путь тогда зависит от CWD. Проект проверяет рабочий каталог
    установленного сервиса и корень source checkout.
    """
    configured = os.getenv("GIGACHAT_CA_BUNDLE_FILE", "").strip().strip("\"'")
    if not configured:
        return None
    path = Path(configured).expanduser()
    candidates = (
        [path]
        if path.is_absolute()
        else [Path.cwd() / path, Path(__file__).resolve().parents[2] / path]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return str(resolved)
    checked = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise RuntimeError(
        "GIGACHAT_CA_BUNDLE_FILE указывает на отсутствующий сертификат; "
        f"проверены пути: {checked}"
    )
