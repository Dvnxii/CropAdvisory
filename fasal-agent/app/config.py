"""
Central configuration for the Fasal Agentic Crop Advisory System.

All secrets are read from environment variables (see .env.example).
Nothing here should ever contain a hardcoded key.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- LLM ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

# --- Vector store (ChromaDB) ---
CHROMA_PERSIST_DIR = str(BASE_DIR / "data" / "chroma_db")
CHROMA_COLLECTION_NAME = "agronomy_knowledge"
KNOWLEDGE_DOCS_DIR = BASE_DIR / "data" / "agronomy_knowledge"

# --- ML model (soil organic carbon regressor) ---
SOIL_MODEL_PATH = str(BASE_DIR / "data" / "soil_oc_model.joblib")
SOIL_TRAINING_CSV = str(BASE_DIR / "data" / "soil_samples.csv")

# --- Weather (Open-Meteo — free, no API key required) ---
WEATHER_API_BASE = "https://api.open-meteo.com/v1/forecast"

# --- Agent loop ---
MAX_TOOL_CALL_ROUNDS = int(os.getenv("MAX_TOOL_CALL_ROUNDS", "6"))
