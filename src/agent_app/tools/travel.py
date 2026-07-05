from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class TravelBudgetInput(BaseModel):
    city: str = Field(description="Город поездки.")
    days: int = Field(default=1, ge=1, le=60, description="Количество дней поездки.")
    hotel_per_night: float = Field(default=0.0, ge=0.0, description="Стоимость проживания за ночь.")
    meals_per_day: float = Field(default=0.0, ge=0.0, description="Расходы на питание в день.")
    transport_total: float = Field(default=0.0, ge=0.0, description="Общие расходы на транспорт.")
    extra_per_day: float = Field(default=0.0, ge=0.0, description="Дополнительные расходы в день.")
    currency: str = Field(default="RUB", description="Валюта расчёта.")


class PackingAdvisorInput(BaseModel):
    city: str = Field(description="Город поездки.")
    trip_goal: str = Field(default="деловая поездка", description="Цель поездки.")
    days: int = Field(default=1, ge=1, le=60, description="Количество дней поездки.")
    temperature: float | None = Field(default=None, description="Температура воздуха, если известна.")
    weather_description: str | None = Field(default=None, description="Описание погоды, если известно.")


def calculate_travel_budget(
    city: str,
    days: int = 1,
    hotel_per_night: float = 0.0,
    meals_per_day: float = 0.0,
    transport_total: float = 0.0,
    extra_per_day: float = 0.0,
    currency: str = "RUB",
) -> str:
    nights = max(days - 1, 0)
    hotel_total = hotel_per_night * nights
    meals_total = meals_per_day * days
    extra_total = extra_per_day * days
    total = hotel_total + meals_total + transport_total + extra_total
    result = {
        "city": city,
        "days": days,
        "nights": nights,
        "currency": currency,
        "items": {
            "hotel_total": round(hotel_total, 2),
            "meals_total": round(meals_total, 2),
            "transport_total": round(transport_total, 2),
            "extra_total": round(extra_total, 2),
        },
        "total": round(total, 2),
        "average_per_day": round(total / days, 2),
    }
    return json.dumps(result, ensure_ascii=False)


def advise_packing(
    city: str,
    trip_goal: str = "деловая поездка",
    days: int = 1,
    temperature: float | None = None,
    weather_description: str | None = None,
) -> str:
    items = [
        "паспорт или другой документ",
        "зарядные устройства",
        "аптечка первой необходимости",
        "комплект одежды по дням",
    ]
    goal = trip_goal.lower()
    weather = (weather_description or "").lower()
    if "дел" in goal or "встреч" in goal or "конференц" in goal:
        items.extend(["деловая одежда", "ноутбук", "блокнот для встреч"])
    if temperature is not None and temperature <= 5:
        items.extend(["тёплая куртка", "перчатки", "тёплая обувь"])
    elif temperature is not None and temperature >= 25:
        items.extend(["лёгкая одежда", "бутылка воды", "солнцезащитные очки"])
    if any(marker in weather for marker in ("дожд", "rain", "снег", "snow")):
        items.extend(["зонт", "непромокаемая обувь"])
    if days >= 3:
        items.append("запасной комплект одежды")

    result = {
        "city": city,
        "trip_goal": trip_goal,
        "days": days,
        "weather_description": weather_description,
        "temperature": temperature,
        "items": sorted(set(items)),
    }
    return json.dumps(result, ensure_ascii=False)


def travel_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="calculate_travel_budget",
            description=(
                "Считает бюджет поездки по дням: проживание, питание, транспорт "
                "и дополнительные расходы."
            ),
            func=calculate_travel_budget,
            args_schema=TravelBudgetInput,
        ),
        StructuredTool.from_function(
            name="advise_packing",
            description=(
                "Подбирает список вещей для поездки с учётом города, цели, срока "
                "и погодных условий."
            ),
            func=advise_packing,
            args_schema=PackingAdvisorInput,
        ),
    ]
