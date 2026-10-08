"""
Trains the soil organic carbon (SOC) regression model used by the agent's
`predict_soil_organic_carbon` tool, and saves it with joblib.

Run:  python scripts/train_soil_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app import config

FEATURES = [
    "ph",
    "nitrogen_kg_ha",
    "phosphorus_kg_ha",
    "potassium_kg_ha",
    "clay_pct",
    "sand_pct",
    "rainfall_mm",
    "temperature_c",
    "moisture_pct",
    "depth_cm",
]
TARGET = "soil_organic_carbon_pct"


def main():
    df = pd.read_csv(config.SOIL_TRAINING_CSV)
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE: {mae:.4f}  |  Test R^2: {r2:.4f}")

    joblib.dump({"pipeline": pipeline, "features": FEATURES}, config.SOIL_MODEL_PATH)
    print(f"Saved model to {config.SOIL_MODEL_PATH}")


if __name__ == "__main__":
    main()
