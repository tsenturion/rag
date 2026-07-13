from __future__ import annotations

import re

SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(api[_-]?key|token|password|passwd|secret|пароль|ключ)\b"
        r"\s*[:=]\s*([^\s,;]+)"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]+"),
)


def redact_secrets(text: str) -> str:
    redacted = text
    redacted = SECRET_PATTERNS[0].sub(r"\1=<redacted>", redacted)
    redacted = SECRET_PATTERNS[1].sub("Bearer <redacted>", redacted)
    return SECRET_PATTERNS[2].sub("Basic <redacted>", redacted)
