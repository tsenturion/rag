"""Учётные записи Argon2id и отзываемые браузерные сессии в общей БД."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from agent_app.database import DatabaseRuntime
from agent_app.service.auth import Principal
from agent_app.service.web_config import WebConfig

ROLES = {"viewer", "engineer", "operator", "admin"}
HASHER = PasswordHasher()
LOGGER = logging.getLogger(__name__)


class UserAccount(BaseModel):
    """Публичная часть учётной записи без пароля и токенов сессии."""

    username: str
    display_name: str
    roles: list[str]
    active: bool


class LoginRequest(BaseModel):
    """Ограничивает размер credentials до дорогостоящей проверки Argon2."""

    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=128, pattern=r"^[\w.@+-]+$")
    password: SecretStr = Field(min_length=1, max_length=1024)


class UserCreate(LoginRequest):
    """Создание пользователя доступно администратору, роли задаются явно."""

    display_name: str = Field(min_length=1, max_length=200)
    roles: list[str] = Field(default_factory=lambda: ["engineer"], min_length=1)


class UserUpdate(BaseModel):
    """Изменение ролей или пароля отзывает все ранее выданные сессии."""

    model_config = ConfigDict(extra="forbid")
    roles: list[str] | None = None
    active: bool | None = None
    password: SecretStr | None = Field(default=None, min_length=12, max_length=1024)


class BrowserSession(BaseModel):
    """CSRF-токен связывается с HttpOnly-сессией, доступной только серверу."""

    user: UserAccount
    csrf_token: str
    expires_at: float


def digest(value: str) -> str:
    """Индексирует случайный токен, не сохраняя credential в открытом виде."""
    return hashlib.sha256(value.encode()).hexdigest()


class AccountStore:
    """Одинаковые транзакции для локального SQLite и нескольких PG workers."""

    def __init__(self, database: DatabaseRuntime, config: WebConfig):
        """Сохраняет общую БД; schema создаётся только при фактическом запуске."""
        self.database, self.config = database, config
        self.path: Path = config.sqlite_path
        self._dummy_hash: str | None = None

    def initialize(self) -> None:
        """Создаёт индексы сессий и общие счётчики неуспешных попыток входа."""
        self._dummy_hash = HASHER.hash(secrets.token_urlsafe(32))
        with self.database.connection(self.path) as conn:
            if self.database.is_postgresql:
                # IF NOT EXISTS не сериализует DDL двух впервые запущенных workers.
                conn.execute("SELECT pg_advisory_xact_lock(?)", (6384201,))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS web_users (username TEXT PRIMARY KEY, display_name TEXT NOT NULL, password_hash TEXT NOT NULL, roles TEXT NOT NULL, active INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS web_sessions (token_hash TEXT PRIMARY KEY, username TEXT NOT NULL REFERENCES web_users(username) ON DELETE CASCADE, csrf TEXT NOT NULL, expires_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS web_sessions_expiry ON web_sessions(expires_at)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS web_login_limits (key TEXT PRIMARY KEY, attempts INTEGER NOT NULL, expires_at REAL NOT NULL)"
            )
            conn.execute(
                "DELETE FROM web_sessions WHERE expires_at <= ?", (time.time(),)
            )
            conn.execute(
                "DELETE FROM web_login_limits WHERE expires_at <= ?", (time.time(),)
            )

    def create(self, data: UserCreate) -> UserAccount:
        """Сохраняет только Argon2id-хеш; существующие имена не перезаписывает."""
        self._roles(data.roles)
        password = data.password.get_secret_value()
        if len(password) < 12:
            raise HTTPException(422, "Пароль должен содержать не менее 12 символов.")
        hashed = HASHER.hash(password)
        with self.database.connection(self.path) as conn:
            row = conn.execute(
                "INSERT INTO web_users VALUES (?, ?, ?, ?, 1) ON CONFLICT(username) DO NOTHING RETURNING username",
                (data.username, data.display_name, hashed, json.dumps(data.roles)),
            ).fetchone()
            if row is None:
                raise HTTPException(409, "Имя пользователя уже занято.")
        LOGGER.info("Создан web-аккаунт", extra={"user_id": data.username})
        return UserAccount(
            username=data.username,
            display_name=data.display_name,
            roles=data.roles,
            active=True,
        )

    def list(self, *, after: str = "", limit: int = 50) -> list[UserAccount]:
        """Листает пользователей по уникальному стабильному имени."""
        with self.database.connection(self.path) as conn:
            rows = conn.execute(
                "SELECT * FROM web_users WHERE username > ? ORDER BY username LIMIT ?",
                (after, limit),
            ).fetchall()
        return [self._user(row) for row in rows]

    def update(self, username: str, data: UserUpdate) -> UserAccount:
        """Обновляет аккаунт и отзывает сессии в одной транзакции."""
        if data.roles is not None:
            self._roles(data.roles)
        hashed = (
            HASHER.hash(data.password.get_secret_value()) if data.password else None
        )
        with self.database.connection(self.path) as conn:
            row = conn.execute(
                "UPDATE web_users SET roles = COALESCE(?, roles), active = COALESCE(?, active), password_hash = COALESCE(?, password_hash) WHERE username = ? RETURNING *",
                (
                    json.dumps(data.roles) if data.roles is not None else None,
                    int(data.active) if data.active is not None else None,
                    hashed,
                    username,
                ),
            ).fetchone()
            if row is None:
                raise HTTPException(404, "Пользователь не найден.")
            conn.execute("DELETE FROM web_sessions WHERE username = ?", (username,))
        LOGGER.info("Изменён web-аккаунт; сессии отозваны", extra={"user_id": username})
        return self._user(row)

    def login(
        self, data: LoginRequest, *, client_id: str
    ) -> tuple[str, BrowserSession]:
        """Ограничивает подбор одновременно по имени и IP через общую БД."""
        now = time.time()
        for key in ("user:" + data.username, "ip:" + client_id):
            with self.database.connection(self.path) as conn:
                conn.execute(
                    "DELETE FROM web_login_limits WHERE expires_at <= ?", (now,)
                )
                row = conn.execute(
                    "INSERT INTO web_login_limits VALUES (?, 1, ?) ON CONFLICT(key) DO UPDATE SET attempts = web_login_limits.attempts + 1 RETURNING attempts",
                    (digest(key), now + self.config.login_window_seconds),
                ).fetchone()
            if row["attempts"] > self.config.login_attempts:
                raise HTTPException(
                    429,
                    "Превышен лимит попыток входа.",
                    headers={"Retry-After": str(self.config.login_window_seconds)},
                )
        with self.database.connection(self.path) as conn:
            row = conn.execute(
                "SELECT * FROM web_users WHERE username = ?", (data.username,)
            ).fetchone()
        # Для неизвестного пользователя также выполняется дорогая проверка.
        stored = row["password_hash"] if row else self._dummy_hash
        if stored is None:
            raise RuntimeError("Хранилище аккаунтов не инициализировано.")
        try:
            HASHER.verify(stored, data.password.get_secret_value())
        except VerificationError:
            LOGGER.warning(
                "Отклонён вход в web-аккаунт", extra={"user_id": data.username}
            )
            raise HTTPException(401, "Неверные учётные данные.") from None
        if row is None or not row["active"]:
            raise HTTPException(401, "Неверные учётные данные.")
        token, csrf = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
        expires = now + self.config.session_ttl_seconds
        with self.database.connection(self.path) as conn:
            # Условие active предотвращает выдачу сессии при блокировке аккаунта.
            inserted = conn.execute(
                "INSERT INTO web_sessions SELECT ?, username, ?, ? FROM web_users WHERE username = ? AND active = 1 AND password_hash = ? RETURNING token_hash",
                (digest(token), csrf, expires, data.username, stored),
            ).fetchone()
            if inserted is None:
                raise HTTPException(401, "Учётная запись изменилась, повторите вход.")
            conn.execute(
                "DELETE FROM web_login_limits WHERE key = ?",
                (digest("user:" + data.username),),
            )
            conn.execute("DELETE FROM web_sessions WHERE expires_at <= ?", (now,))
        LOGGER.info("Выполнен браузерный вход", extra={"user_id": data.username})
        return token, BrowserSession(
            user=self._user(row), csrf_token=csrf, expires_at=expires
        )

    def session(self, token: str) -> BrowserSession:
        """Каждый запрос проверяет срок, отзыв и текущие роли аккаунта."""
        with self.database.connection(self.path) as conn:
            row = conn.execute(
                "SELECT u.*, s.csrf, s.expires_at FROM web_sessions s JOIN web_users u ON u.username = s.username WHERE s.token_hash = ? AND s.expires_at > ? AND u.active = 1",
                (digest(token), time.time()),
            ).fetchone()
        if row is None:
            raise HTTPException(401, "Сессия завершена. Выполните вход.")
        return BrowserSession(
            user=self._user(row), csrf_token=row["csrf"], expires_at=row["expires_at"]
        )

    def logout(self, token: str) -> None:
        """Удаляет серверную сессию, поэтому копия старой cookie больше не работает."""
        with self.database.connection(self.path) as conn:
            conn.execute(
                "DELETE FROM web_sessions WHERE token_hash = ?", (digest(token),)
            )

    @staticmethod
    def principal(session: BrowserSession) -> Principal:
        """Передаёт проверенную identity существующему RBAC без клиентских claims."""
        return Principal(
            subject=session.user.username,
            roles=session.user.roles,
            auth_method="cookie",
        )

    @staticmethod
    def _user(row) -> UserAccount:
        """Исключает password_hash даже при SELECT всех столбцов хранилища."""
        return UserAccount(
            username=row["username"],
            display_name=row["display_name"],
            roles=json.loads(row["roles"]),
            active=bool(row["active"]),
        )

    @staticmethod
    def _roles(roles: list[str]) -> None:
        """Сервисная роль не может назначаться браузерному пользователю."""
        if not roles or not set(roles) <= ROLES:
            raise HTTPException(422, "Недопустимый набор ролей.")
