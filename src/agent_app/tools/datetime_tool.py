from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class DateTimeInput(BaseModel):
    timezone: str = Field(
        default="Asia/Yekaterinburg",
        description="IANA timezone, for example Asia/Yekaterinburg or UTC.",
    )


def current_datetime(timezone: str = "Asia/Yekaterinburg") -> str:
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("UTC")
        timezone = "UTC"
    now = datetime.now(tz)
    return now.isoformat(timespec="seconds") + f" ({timezone})"


def datetime_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="current_datetime",
        description="Use this tool when the user asks for current date or time.",
        func=current_datetime,
        args_schema=DateTimeInput,
    )
