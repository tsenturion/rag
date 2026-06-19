from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import httpx2
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_app.config import WeatherConfig

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


class WeatherInput(BaseModel):
    city: str | None = Field(
        default=None,
        description="City name. Use the user's city if it is present in the request.",
    )


def weather_tool(config: WeatherConfig) -> StructuredTool:
    def get_weather(city: str | None = None) -> str:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            return json.dumps(
                {
                    "error": "missing_api_key",
                    "message": f"Set {config.api_key_env} in .env or environment.",
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
                    "message": str(exc),
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
            "Use this tool when the user asks about current weather. "
            "Input is a city name; if the user does not provide one, use default city."
        ),
        func=get_weather,
        args_schema=WeatherInput,
    )
