"""
Tool 2: ML regression tool — predicts soil organic carbon (%) from
soil-test and site parameters using a trained RandomForestRegressor
(see scripts/train_soil_model.py).

Loaded once at import time and reused across requests.
"""
from __future__ import annotations

from typing import Dict

import joblib

from app import config

_MODEL_BUNDLE = None


def _load_model():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        _MODEL_BUNDLE = joblib.load(config.SOIL_MODEL_PATH)
    return _MODEL_BUNDLE


def predict_soil_organic_carbon(
    ph: float,
    nitrogen_kg_ha: float,
    phosphorus_kg_ha: float,
    potassium_kg_ha: float,
    clay_pct: float,
    sand_pct: float,
    rainfall_mm: float,
    temperature_c: float,
    moisture_pct: float,
    depth_cm: float = 30,
) -> Dict:
    """
    Predicts soil organic carbon percentage from soil-test inputs.

    Exposed to the LLM as the callable tool `predict_soil_organic_carbon`.
    """
    bundle = _load_model()
    pipeline, features = bundle["pipeline"], bundle["features"]

    row = {
        "ph": ph,
        "nitrogen_kg_ha": nitrogen_kg_ha,
        "phosphorus_kg_ha": phosphorus_kg_ha,
        "potassium_kg_ha": potassium_kg_ha,
        "clay_pct": clay_pct,
        "sand_pct": sand_pct,
        "rainfall_mm": rainfall_mm,
        "temperature_c": temperature_c,
        "moisture_pct": moisture_pct,
        "depth_cm": depth_cm,
    }

    import pandas as pd

    X = pd.DataFrame([[row[f] for f in features]], columns=features)
    pred = float(pipeline.predict(X)[0])

    if pred < 0.5:
        band = "low"
    elif pred < 1.5:
        band = "moderate"
    else:
        band = "high"

    return {
        "predicted_soc_percent": round(pred, 3),
        "fertility_band": band,
        "note": "Estimated by a RandomForestRegressor trained on soil-test and site parameters.",
    }


# --- Tool schema exposed to the Gemini function-calling loop ---
TOOL_SCHEMA = {
    "name": "predict_soil_organic_carbon",
    "description": (
        "Predicts soil organic carbon (SOC %) — a key indicator of soil "
        "fertility and health — from soil-test values (pH, N, P, K, texture) "
        "and site conditions (rainfall, temperature, moisture, sample depth). "
        "Use this when the farmer gives or asks about soil-test numbers, or "
        "when soil health/fertility needs to be quantified rather than "
        "looked up from general guidance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ph": {"type": "number", "description": "Soil pH, typically 4.5-9.0"},
            "nitrogen_kg_ha": {"type": "number", "description": "Available nitrogen, kg/ha"},
            "phosphorus_kg_ha": {"type": "number", "description": "Available phosphorus, kg/ha"},
            "potassium_kg_ha": {"type": "number", "description": "Available potassium, kg/ha"},
            "clay_pct": {"type": "number", "description": "Clay content, percent"},
            "sand_pct": {"type": "number", "description": "Sand content, percent"},
            "rainfall_mm": {"type": "number", "description": "Recent/seasonal rainfall, mm"},
            "temperature_c": {"type": "number", "description": "Average temperature, Celsius"},
            "moisture_pct": {"type": "number", "description": "Soil moisture, percent"},
            "depth_cm": {"type": "number", "description": "Sample depth in cm (default 30)"},
        },
        "required": [
            "ph",
            "nitrogen_kg_ha",
            "phosphorus_kg_ha",
            "potassium_kg_ha",
            "clay_pct",
            "sand_pct",
            "rainfall_mm",
            "temperature_c",
            "moisture_pct",
        ],
    },
}
