"""
Offline sanity tests for the pieces of the agent that don't require a live
Gemini API key: the tool implementations, their schemas, and the dispatch
table the manual function-calling loop in app/agent.py relies on.

Run: python -m pytest tests/ -v   (or) python tests/test_tool_dispatch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import _TOOL_IMPLEMENTATIONS, _TOOL_SCHEMAS, _execute_tool


def test_all_schemas_have_matching_implementations():
    schema_names = {s["name"] for s in _TOOL_SCHEMAS}
    impl_names = set(_TOOL_IMPLEMENTATIONS.keys())
    assert schema_names == impl_names, f"{schema_names} != {impl_names}"
    print("OK: tool schemas <-> implementations are in sync:", schema_names)


def test_vector_search_dispatch():
    result = _execute_tool(
        "search_agronomy_knowledge", {"query": "nitrogen deficiency yellow leaves", "top_k": 2}
    )
    assert isinstance(result, list) and len(result) > 0
    assert "text" in result[0] and "topic" in result[0]
    print("OK: search_agronomy_knowledge ->", result[0]["topic"])


def test_soil_model_dispatch():
    args = dict(
        ph=6.2,
        nitrogen_kg_ha=250,
        phosphorus_kg_ha=18,
        potassium_kg_ha=220,
        clay_pct=40,
        sand_pct=25,
        rainfall_mm=1100,
        temperature_c=25,
        moisture_pct=30,
    )
    result = _execute_tool("predict_soil_organic_carbon", args)
    assert "predicted_soc_percent" in result and "fertility_band" in result
    print("OK: predict_soil_organic_carbon ->", result)


def test_unknown_tool_reports_error_not_raises():
    result = _execute_tool("not_a_real_tool", {})
    assert "error" in result
    print("OK: unknown tool handled gracefully ->", result)


def test_weather_tool_reports_error_not_raises_on_no_network():
    # In this sandbox, egress to open-meteo.com is blocked; the important
    # behavioral contract is that _execute_tool NEVER raises — it reports
    # the failure back to the model as a normal tool result.
    result = _execute_tool("get_weather_forecast", {"latitude": 29.22, "longitude": 79.53})
    assert isinstance(result, dict)
    print("OK: weather tool failure surfaced as data, not exception ->", result)


if __name__ == "__main__":
    test_all_schemas_have_matching_implementations()
    test_vector_search_dispatch()
    test_soil_model_dispatch()
    test_unknown_tool_reports_error_not_raises()
    test_weather_tool_reports_error_not_raises_on_no_network()
    print("\nAll offline dispatch tests passed.")
