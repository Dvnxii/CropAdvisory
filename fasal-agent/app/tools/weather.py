"""
Tool 3: Live weather API — current conditions + short-term forecast for a
given location, via Open-Meteo (no API key required, so the project runs
out of the box; swap the base URL for a keyed provider in production).
"""
from __future__ import annotations

from typing import Dict

import requests

from app import config


def get_weather_forecast(latitude: float, longitude: float, days: int = 3) -> Dict:
    """
    Fetches current weather and a short-term daily forecast for a location.

    Exposed to the LLM as the callable tool `get_weather_forecast`.
    """
    days = max(1, min(days, 7))
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "forecast_days": days,
        "timezone": "auto",
    }

    resp = requests.get(config.WEATHER_API_BASE, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    current = data.get("current", {})
    daily = data.get("daily", {})

    forecast = []
    dates = daily.get("time", [])
    for i, date in enumerate(dates):
        forecast.append(
            {
                "date": date,
                "temp_max_c": daily.get("temperature_2m_max", [None] * len(dates))[i],
                "temp_min_c": daily.get("temperature_2m_min", [None] * len(dates))[i],
                "precipitation_mm": daily.get("precipitation_sum", [None] * len(dates))[i],
                "wind_speed_max_kmh": daily.get("wind_speed_10m_max", [None] * len(dates))[i],
            }
        )

    return {
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
        },
        "daily_forecast": forecast,
        "source": "Open-Meteo",
    }


# --- Tool schema exposed to the Gemini function-calling loop ---
TOOL_SCHEMA = {
    "name": "get_weather_forecast",
    "description": (
        "Fetches current conditions and a daily forecast (temperature, "
        "precipitation, wind, humidity) for a farm location. Use this for "
        "questions about irrigation timing, spraying windows, frost risk, "
        "or any advice that depends on near-term weather."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number", "description": "Farm location latitude"},
            "longitude": {"type": "number", "description": "Farm location longitude"},
            "days": {"type": "integer", "description": "Forecast days ahead, 1-7. Defaults to 3."},
        },
        "required": ["latitude", "longitude"],
    },
}
