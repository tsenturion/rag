"""Проверяет браузерный контракт и настоящий worker без генерации LLM и embeddings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
from uuid import uuid4

import httpx2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def checked(response: httpx2.Response, status: int = 200) -> httpx2.Response:
    """Ошибка сообщает только путь и статус, не публикуя тело с credentials."""
    if response.status_code != status:
        raise RuntimeError(
            f"{response.request.url.path}: ожидался {status}, получен {response.status_code}"
        )
    return response


def verify(base_url: str, browser_url: str | None) -> dict:
    """Временные аккаунты проходят login/CSRF/ownership/CLI и в конце блокируются."""
    key = os.getenv("SUPPORT_SERVICE_API_KEY")
    if not key:
        raise ValueError(
            "Задайте SUPPORT_SERVICE_API_KEY для создания проверочных аккаунтов."
        )
    report: dict = {"passed": False, "checks": []}
    names: list[str] = []
    with (
        httpx2.Client(
            base_url=base_url, trust_env=False, timeout=30, headers={"X-API-Key": key}
        ) as admin,
        httpx2.Client(base_url=base_url, trust_env=False, timeout=30) as browser,
    ):
        checked(admin.get("/ready"))
        password = secrets.token_urlsafe(32)
        source_id = memory_id = None
        try:
            username = "web-check-" + uuid4().hex[:16]
            checked(
                admin.post(
                    "/v1/admin/users",
                    json={
                        "username": username,
                        "password": password,
                        "display_name": "Проверка браузерного API",
                        "roles": ["operator"],
                    },
                ),
                201,
            )
            names.append(username)
            login = checked(
                browser.post(
                    "/v1/auth/login", json={"username": username, "password": password}
                )
            )
            assert "httponly" in login.headers["set-cookie"].lower()
            browser.headers["X-CSRF-Token"] = login.json()["csrf_token"]
            assert checked(browser.get("/v1/auth/me")).json()["subject"] == username
            assert (
                checked(browser.get("/v1/auth/session")).json()["user"]["username"]
                == username
            )
            checked(
                browser.post(
                    "/v1/memories",
                    headers={"X-CSRF-Token": "invalid"},
                    json={"key": "smoke", "value": "not stored"},
                ),
                403,
            )
            report["checks"].append("login_csrf_session")
            memory_id = checked(
                browser.post(
                    "/v1/memories",
                    json={
                        "key": "web-smoke",
                        "value": "Проверка сохранения через HTTP",
                    },
                ),
                201,
            ).json()["id"]
            source_id = checked(
                browser.post(
                    "/v1/knowledge/sources",
                    params={"filename": "web-smoke.txt"},
                    content="Проверка загрузки исходника без вызова LLM.".encode(),
                ),
                201,
            ).json()["id"]
            checked(browser.get(f"/v1/knowledge/sources/{source_id}"))
            checked(browser.get("/v1/memories"))
            report["checks"].append("resources")
            profiles = checked(browser.get("/v1/operations/profiles")).json()
            assert any(
                p["id"] == "tuning-inspect" and p["worker_ready"] for p in profiles
            )
            payload = {"profile": "tuning-inspect", "idempotency_key": str(uuid4())}
            job = checked(browser.post("/v1/operations", json=payload), 202).json()
            repeated = checked(browser.post("/v1/operations", json=payload), 202).json()
            assert repeated["id"] == job["id"]
            deadline = time.monotonic() + 180
            while (
                job["status"] in {"queued", "running"} and time.monotonic() < deadline
            ):
                time.sleep(1)
                job = checked(browser.get(f"/v1/operations/{job['id']}")).json()
            assert job["status"] == "completed", (
                f"Операция закончилась в состоянии {job['status']}"
            )
            assert job["exit_code"] == 0
            logs = checked(browser.get(f"/v1/operations/{job['id']}/logs")).json()
            assert logs["text"]
            report["operation_id"] = job["id"]
            report["checks"].append("persistent_worker_cli")
            if browser_url:
                environment = dict(
                    os.environ,
                    WEB_E2E_URL=browser_url,
                    WEB_E2E_USERNAME=username,
                    WEB_E2E_PASSWORD=password,
                )
                subprocess.run(
                    [
                        "node",
                        "node_modules/@playwright/test/cli.js",
                        "test",
                        "--grep",
                        "live smoke",
                    ],
                    cwd=ROOT / "frontend",
                    env=environment,
                    check=True,
                )
                report["checks"].append("live_browser_desktop_mobile")
            checked(browser.delete(f"/v1/memories/{memory_id}"), 204)
            memory_id = None
            checked(browser.delete(f"/v1/knowledge/sources/{source_id}"), 204)
            source_id = None
            checked(admin.patch(f"/v1/admin/users/{username}", json={"active": False}))
            checked(browser.get("/v1/auth/session"), 401)
            report["checks"].append("revocation")
            report["passed"] = True
            return report
        finally:
            # Проверка оставляет отчёт операции, но не активные аккаунты/секреты.
            if memory_id:
                browser.delete(f"/v1/memories/{memory_id}")
            if source_id:
                browser.delete(f"/v1/knowledge/sources/{source_id}")
            for name in names:
                checked(admin.patch(f"/v1/admin/users/{name}", json={"active": False}))


def main() -> int:
    """Принимает адреса уже запущенных API и Vite; модели не скачиваются и не вызываются."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--browser-url",
        help="Адрес Vite для дополнительного настоящего браузерного входа.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/agent/web-smoke.json"
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    report = verify(args.base_url, args.browser_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
