from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone

import httpx2
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_app.config import WeatherConfig

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
APPID_RE = re.compile(r"([?&]appid=)[^&\\s']+")
logging.getLogger("httpx2").setLevel(logging.WARNING)


class WeatherInput(BaseModel):
    city: str | None = Field(
        default=None,
        description="Название города. Используй город пользователя, если он есть в запросе.",
    )


def weather_tool(config: WeatherConfig) -> StructuredTool:
    def get_weather(city: str | None = None) -> str:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            return json.dumps(
                {
                    "error": "missing_api_key",
                    "message": f"Укажите {config.api_key_env} в .env или переменных окружения.",
                },
                ensure_ascii=False,
            )

        target_city = (city or config.default_city).strip()
        try:
            with httpx2.Client(timeout=config.timeout_seconds) as client:
                response = client.get(
                    OPENWEATHER_URL,
                    params={
                        "q": target_city,
                        "appid": api_key,
                        "units": config.default_units,
                        "lang": config.language,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return json.dumps(
                {
                    "error": "weather_request_failed",
                    "city": target_city,
                    "message": _redact_weather_error(str(exc)),
                },
                ensure_ascii=False,
            )

        result = {
            "city": payload.get("name", target_city),
            "country": (payload.get("sys") or {}).get("country"),
            "temperature": (payload.get("main") or {}).get("temp"),
            "feels_like": (payload.get("main") or {}).get("feels_like"),
            "humidity": (payload.get("main") or {}).get("humidity"),
            "wind_speed": (payload.get("wind") or {}).get("speed"),
            "description": (
                (payload.get("weather") or [{}])[0].get("description")
                if payload.get("weather")
                else None
            ),
            "units": config.default_units,
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        name="get_weather",
        description=(
            "Используй этот tool, когда пользователь спрашивает текущую погоду. "
            "На входе название города; если пользователь не указал город, используй город по умолчанию."
        ),
        func=get_weather,
        args_schema=WeatherInput,
    )


def _redact_weather_error(message: str) -> str:
    return APPID_RE.sub(r"\1<redacted>", message)
