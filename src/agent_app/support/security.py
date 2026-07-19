"""Редактирование чувствительных данных для инженерной поддержки."""

from __future__ import annotations

import re
import threading
from functools import lru_cache
from pathlib import PurePosixPath, PureWindowsPath

from detect_secrets.core.scan import scan_line
from detect_secrets.settings import transient_settings

SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(api[_ -]?key|authorization[_ -]?key|auth[_ -]?key|token|"
        r"password|passwd|secret|пароль|ключ|openai_api_key|hf_token|"
        r"gigachat_auth_key|openweather_api_key)\b"
        r"\s*[:=]\s*([^\s,;]+)"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]+"),
    re.compile(
        r"(?x)\b(?:"
        r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}|"
        r"hf_[A-Za-z0-9]{20,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|"
        r"github_pat_[A-Za-z0-9_]{20,}|"
        r"glpat-[A-Za-z0-9_-]{20,}|"
        r"xox[baprs]-[A-Za-z0-9-]{20,}|"
        r"AKIA[0-9A-Z]{16}|"
        r"AIza[0-9A-Za-z_-]{35}|"
        r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
        r")\b"
    ),
    # Authorization key GigaChat и некоторые service credentials поставляются
    # как standalone base64 без префикса/label. Длина 40+ и строгие границы
    # уменьшают риск принять обычное слово или часть текста за credential.
    re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/=])"),
)

_DETECT_SECRETS_SETTINGS = {
    "plugins": [
        {"name": "Base64HighEntropyString", "limit": 4.5},
        {"name": "HexHighEntropyString", "limit": 3.0},
        {"name": "KeywordDetector"},
    ]
}
_DETECT_SECRETS_LOCK = threading.Lock()

LOCAL_PATH_PATTERNS = (
    re.compile(r"(?<!\w)[A-Za-z]:\\[^\r\n,;]+"),
    re.compile(r"(?<![\w:])/(?:app|home|Users|var|tmp)/[^\s,;]+"),
)


def redact_secrets(text: str) -> str:
    """Маскирует известные токены и высокоэнтропийные значения detect-secrets."""
    redacted = text
    redacted = SECRET_PATTERNS[0].sub(r"\1=<redacted>", redacted)
    redacted = SECRET_PATTERNS[1].sub("Bearer <redacted>", redacted)
    redacted = SECRET_PATTERNS[2].sub("Basic <redacted>", redacted)
    redacted = SECRET_PATTERNS[3].sub("<secret:redacted>", redacted)
    redacted = SECRET_PATTERNS[4].sub("<secret:redacted>", redacted)
    for candidate in _entropy_secrets(redacted):
        redacted = redacted.replace(candidate, "<secret:redacted>")
    return redacted


def contains_secret(text: str) -> bool:
    """Определяет секреты по provider-prefix, заголовкам и энтропии значения."""
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        return True
    return bool(_entropy_secrets(text))


@lru_cache(maxsize=2048)
def _entropy_secrets(text: str) -> tuple[str, ...]:
    """Возвращает длинные high-entropy значения через проверенный scanner."""
    found: set[str] = set()
    with _DETECT_SECRETS_LOCK, transient_settings(_DETECT_SECRETS_SETTINGS):
        for line in text.splitlines() or [text]:
            for secret in scan_line(line):
                value = secret.secret_value.strip()
                # Короткие слова KeywordDetector и полная строка не являются
                # безопасной единицей редактирования; labelled pairs уже закрыты regex.
                if len(value) >= 20 and value != line.strip():
                    found.add(value)
    return tuple(sorted(found, key=len, reverse=True))


def redact_local_paths(text: str) -> str:
    """Гарантирует, что локальные пути в тексте заменяются на безопасные псевдонимы для защиты приватной информации пользователя."""

    def windows_name(match: re.Match[str]) -> str:
        """Возвращает безопасный псевдоним для локального Windows-пути, скрывая структуру файловой системы пользователя."""
        return f"<local-path:{PureWindowsPath(match.group(0)).name}>"

    def posix_name(match: re.Match[str]) -> str:
        """Возвращает безопасный псевдоним для локального POSIX-пути, скрывая структуру файловой системы пользователя."""
        return f"<local-path:{PurePosixPath(match.group(0)).name}>"

    redacted = LOCAL_PATH_PATTERNS[0].sub(windows_name, text)
    return LOCAL_PATH_PATTERNS[1].sub(posix_name, redacted)
