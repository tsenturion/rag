"""Cookie-аутентификация интегрируется с существующим RBAC и API-контрактом."""

from __future__ import annotations

import secrets

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from agent_app.service.accounts import (
    AccountStore,
    BrowserSession,
    LoginRequest,
    UserAccount,
    UserCreate,
    UserUpdate,
)
from agent_app.service.auth import Permission, Principal


def cookie_principal(request: Request, accounts: AccountStore) -> Principal:
    """Изменяющие запросы требуют привязанный к сессии CSRF-токен."""
    token = request.cookies.get(accounts.config.cookie_name, "")
    session = accounts.session(token)
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        supplied = request.headers.get("X-CSRF-Token", "")
        if not secrets.compare_digest(supplied, session.csrf_token):
            raise HTTPException(403, "Некорректный CSRF-токен.")
    return accounts.principal(session)


def install_browser_auth(
    app: FastAPI, config, accounts: AccountStore, authenticate, require_permission
) -> None:
    """Регистрирует вход, отзыв сессии и административное управление аккаунтами."""

    def enabled() -> None:
        """Явно отключённый web-профиль не открывает альтернативный способ входа."""
        if not config.web.enabled:
            raise HTTPException(503, "Браузерный вход отключён.")

    @app.post(
        "/v1/auth/login",
        response_model=BrowserSession,
        tags=["Приложение"],
        dependencies=[Depends(enabled)],
    )
    def login(
        payload: LoginRequest, request: Request, response: Response
    ) -> BrowserSession:
        """Проверяет origin до выдачи host-only HttpOnly cookie с фиксированным сроком."""
        origin = request.headers.get("Origin")
        allowed = {str(request.base_url).rstrip("/"), *config.service.cors_origins}
        if config.service.public_base_url:
            allowed.add(config.service.public_base_url.rstrip("/"))
        if (origin and origin not in allowed) or request.headers.get(
            "Sec-Fetch-Site"
        ) == "cross-site":
            raise HTTPException(403, "Источник запроса не разрешён.")
        token, session = accounts.login(
            payload, client_id=request.client.host if request.client else "unknown"
        )
        previous = request.cookies.get(config.web.cookie_name)
        if previous:
            accounts.logout(previous)
        response.set_cookie(
            config.web.cookie_name,
            token,
            httponly=True,
            secure=config.web.cookie_secure,
            samesite="lax",
            max_age=config.web.session_ttl_seconds,
            path="/",
        )
        response.headers["Cache-Control"] = "no-store"
        return session

    @app.get(
        "/v1/auth/session",
        response_model=BrowserSession,
        tags=["Приложение"],
        dependencies=[Depends(enabled)],
    )
    def session(request: Request, response: Response) -> BrowserSession:
        """Восстанавливает identity и CSRF после перезагрузки страницы без продления TTL."""
        response.headers["Cache-Control"] = "no-store"
        return accounts.session(request.cookies.get(config.web.cookie_name, ""))

    @app.post(
        "/v1/auth/logout",
        status_code=204,
        tags=["Приложение"],
        dependencies=[Depends(enabled), Depends(authenticate)],
    )
    def logout(request: Request, response: Response) -> None:
        """Отзывает cookie в БД и очищает её в браузере."""
        accounts.logout(request.cookies.get(config.web.cookie_name, ""))
        response.delete_cookie(
            config.web.cookie_name,
            path="/",
            secure=config.web.cookie_secure,
            httponly=True,
            samesite="lax",
        )

    @app.get(
        "/v1/admin/users",
        response_model=list[UserAccount],
        tags=["Безопасность"],
        dependencies=[
            Depends(enabled),
            Depends(require_permission(Permission.ADMIN_WRITE)),
        ],
    )
    def users(after: str = "") -> list[UserAccount]:
        """Возвращает следующую страницу пользователей без хешей паролей."""
        return accounts.list(after=after)

    @app.post(
        "/v1/admin/users",
        response_model=UserAccount,
        status_code=201,
        tags=["Безопасность"],
        dependencies=[
            Depends(enabled),
            Depends(require_permission(Permission.ADMIN_WRITE)),
        ],
    )
    def create_user(payload: UserCreate) -> UserAccount:
        """Саморегистрация отсутствует: аккаунт создаёт проверенный администратор."""
        return accounts.create(payload)

    @app.patch(
        "/v1/admin/users/{username}",
        response_model=UserAccount,
        tags=["Безопасность"],
        dependencies=[
            Depends(enabled),
            Depends(require_permission(Permission.ADMIN_WRITE)),
        ],
    )
    def update_user(
        username: str, payload: UserUpdate, request: Request
    ) -> UserAccount:
        """Блокировка и сброс пароля немедленно отзывают активные сессии."""
        if username == request.state.principal.subject and (
            payload.active is False
            or (payload.roles is not None and "admin" not in payload.roles)
        ):
            raise HTTPException(
                409, "Нельзя отозвать собственный административный доступ."
            )
        return accounts.update(username, payload)
