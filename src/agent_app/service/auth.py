"""Реализация компонентов для HTTP-сервиса поддержки."""

from __future__ import annotations

import hmac
import os
from enum import StrEnum

import jwt
from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from agent_app.config import AgentSecurityConfig


class Permission(StrEnum):
    """Определяет уровни доступа в HTTP-сервисе поддержки, обеспечивая контроль и разграничение прав пользователей и компонентов."""

    CHAT = "chat:write"
    SESSION_READ = "session:read"
    SESSION_DELETE = "session:delete"
    RUN_READ = "run:read"
    ORCHESTRATION_READ = "orchestration:read"
    ORCHESTRATION_WRITE = "orchestration:write"
    REVIEW_READ = "review:read"
    REVIEW_WRITE = "review:write"
    AUDIT_READ = "audit:read"
    METRICS_READ = "metrics:read"


class Principal(BaseModel):
    """Гарантирует неизменяемое описание субъекта доступа с ролями и способом аутентификации для всех подсистем авторизации."""

    subject: str
    roles: list[str] = Field(default_factory=list)
    auth_method: str


ROLE_PERMISSIONS: dict[str, set[Permission]] = {
    "viewer": {
        Permission.SESSION_READ,
        Permission.RUN_READ,
        Permission.ORCHESTRATION_READ,
        Permission.METRICS_READ,
    },
    "engineer": {
        Permission.CHAT,
        Permission.SESSION_READ,
        Permission.SESSION_DELETE,
        Permission.RUN_READ,
        Permission.ORCHESTRATION_READ,
        Permission.ORCHESTRATION_WRITE,
    },
    "operator": set(Permission) - {Permission.AUDIT_READ},
    "admin": set(Permission),
    "service": set(Permission),
}


class AuthManager:
    """Гарантирует централизованное управление политиками аутентификации и авторизации для всех HTTP-запросов сервиса."""

    def __init__(self, config: AgentSecurityConfig):
        """Гарантирует, что экземпляр готов к проверке прав и использует заданную конфигурацию безопасности."""
        self.config = config

    def authenticate(
        self, *, api_key: str | None = None, bearer_token: str | None = None
    ) -> Principal:
        """Гарантирует получение валидного субъекта доступа или выдаёт ошибку 401 при отсутствии корректных учётных данных."""
        if bearer_token:
            return self._from_jwt(bearer_token)
        if api_key and self.config.require_api_key:
            return self._from_api_key(api_key)
        if not self.config.require_api_key and not self.config.jwt_enabled:
            return Principal(
                subject="local-development", roles=["admin"], auth_method="none"
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Передайте X-API-Key или Bearer JWT.",
        )

    @staticmethod
    def authorize(principal: Principal, permission: Permission) -> None:
        """Гарантирует отказ в доступе с ошибкой 403, если у субъекта нет требуемого разрешения."""
        permissions = {
            value
            for role in principal.roles
            for value in ROLE_PERMISSIONS.get(role, set())
        }
        if permission not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Недостаточно прав: требуется {permission.value}.",
            )

    def enforce_user_scope(self, principal: Principal, user_id: str) -> None:
        """Гарантирует, что пользователь может работать только со своими данными, если политика безопасности это требует."""
        if not self.config.enforce_user_scope:
            return
        if {"admin", "service"}.intersection(principal.roles):
            return
        if principal.subject != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Пользователь может работать только со своим user_id.",
            )

    def _from_api_key(self, supplied: str) -> Principal:
        """Гарантирует создание субъекта доступа только при точном совпадении API-ключа с конфигурацией окружения, иначе выдаёт ошибку."""
        expected = os.getenv(self.config.api_key_env)
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Сервисный API key не настроен.",
            )
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Некорректный API key.",
            )
        return Principal(
            subject="service-api-key",
            roles=[self.config.api_key_role],
            auth_method="api_key",
        )

    def _from_jwt(self, token: str) -> Principal:
        """Гарантирует, что вызывающий получит валидированного пользователя с поддерживаемой ролью или получит корректный HTTP-ответ об ошибке аутентификации."""
        if not self.config.jwt_enabled:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWT-аутентификация отключена.",
            )
        secret = os.getenv(self.config.jwt_secret_env)
        if not secret:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="JWT secret не настроен.",
            )
        try:
            payload = jwt.decode(
                token,
                secret,
                algorithms=[self.config.jwt_algorithm],
                issuer=self.config.jwt_issuer,
                audience=self.config.jwt_audience,
                options={"require": ["exp", "iat", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWT недействителен или просрочен.",
            ) from exc
        roles = payload.get("roles", [])
        if isinstance(roles, str):
            roles = [roles]
        known_roles = [role for role in roles if role in ROLE_PERMISSIONS]
        if not known_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="JWT не содержит поддерживаемую роль.",
            )
        return Principal(
            subject=str(payload["sub"]), roles=known_roles, auth_method="jwt"
        )
