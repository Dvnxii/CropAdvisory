"""
Generates a synthetic-but-realistic soil samples dataset for training the
soil organic carbon (SOC) regression model.

Features are loosely based on published soil-science ranges for Indian
agricultural soils (pH ~5.5-8.5, clay-dominant vs sandy textures, etc.).
Real deployments would swap this for ICAR / SoilGrids / state soil-health-card
data — this script exists purely to make the project runnable end-to-end
without requiring a licensed dataset.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 1200

ph = np.random.normal(6.8, 0.9, N).clip(4.5, 9.0)
nitrogen_kg_ha = np.random.normal(280, 90, N).clip(50, 600)
phosphorus_kg_ha = np.random.normal(22, 10, N).clip(2, 80)
potassium_kg_ha = np.random.normal(240, 80, N).clip(40, 600)
clay_pct = np.random.uniform(5, 60, N)
sand_pct = np.random.uniform(10, 80, N)
rainfall_mm = np.random.normal(950, 350, N).clip(150, 2500)
temperature_c = np.random.normal(26, 4, N).clip(10, 42)
moisture_pct = np.random.uniform(8, 45, N)
depth_cm = np.random.choice([15, 30, 45, 60], N)

# Synthetic ground truth generating function for SOC (%) with realistic
# agronomic relationships + noise: more N and moisture, finer texture (clay),
# and higher rainfall generally correlate with higher organic carbon;
# sandy, hot, low-moisture soils trend lower.
soc = (
    0.35
    + 0.0018 * nitrogen_kg_ha
    + 0.012 * clay_pct
    - 0.006 * sand_pct
    + 0.0006 * rainfall_mm
    + 0.015 * moisture_pct
    - 0.01 * temperature_c
    - 0.05 * np.abs(ph - 6.8)
    - 0.002 * depth_cm
    + np.random.normal(0, 0.18, N)
).clip(0.1, 3.5)

df = pd.DataFrame(
    {
        "ph": ph.round(2),
        "nitrogen_kg_ha": nitrogen_kg_ha.round(1),
        "phosphorus_kg_ha": phosphorus_kg_ha.round(1),
        "potassium_kg_ha": potassium_kg_ha.round(1),
        "clay_pct": clay_pct.round(1),
        "sand_pct": sand_pct.round(1),
        "rainfall_mm": rainfall_mm.round(1),
        "temperature_c": temperature_c.round(1),
        "moisture_pct": moisture_pct.round(1),
        "depth_cm": depth_cm,
        "soil_organic_carbon_pct": soc.round(3),
    }
)

df.to_csv("data/soil_samples.csv", index=False)
print(f"Wrote {len(df)} rows to data/soil_samples.csv")
print(df.describe().T[["mean", "std", "min", "max"]])
