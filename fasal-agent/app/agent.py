"""
The agentic core of the Fasal Crop Advisory System.

Unlike a fixed RAG pipeline (retrieve -> stuff into prompt -> generate),
this agent gives the LLM three tools and lets it decide, at each turn,
whether to call a tool, which one, with what arguments, and how many
rounds of tool use it needs before it has enough information to answer.

The Gemini SDK's automatic function-calling is explicitly disabled
(`automatic_function_calling={"disable": True}`), so every tool call is
intercepted here, executed, logged, and fed back to the model manually.
This is what makes the agent's reasoning auditable: `run_agent()` returns
the full ordered trace of (tool name, arguments, result) alongside the
final answer, instead of hiding tool execution inside the SDK.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from google import genai
from google.genai import types

from app import config
from app.tools import soil_model, vector_store, weather

SYSTEM_INSTRUCTION = """\
You are Fasal, an agentic crop advisory assistant for Indian farmers and \
agronomists. You have three tools:

1. search_agronomy_knowledge — domain knowledge (diseases, pests, fertilizer, \
irrigation practice, soil health guidance).
2. predict_soil_organic_carbon — a trained ML regression model that estimates \
soil organic carbon (%) from soil-test numbers the user provides.
3. get_weather_forecast — live current conditions and a short-term forecast \
for a location (lat/long).

Decide for yourself which tools are relevant to the question, in what order, \
and how many times to call them — do not call a tool that isn't needed, and \
call as many as are needed to give a grounded, specific answer. If the user \
gives soil-test numbers, use the regression tool rather than guessing. If the \
question depends on near-term conditions (spraying, irrigation timing, frost), \
check the weather tool. Combine tool results into one clear, actionable answer \
for a farmer — plain language, concrete next steps, and cite which tool a \
number came from when it matters (e.g. "the model estimates..." vs "current \
conditions show...").
"""

_TOOL_IMPLEMENTATIONS = {
    "search_agronomy_knowledge": lambda **kwargs: vector_store.vector_search(**kwargs),
    "predict_soil_organic_carbon": lambda **kwargs: soil_model.predict_soil_organic_carbon(**kwargs),
    "get_weather_forecast": lambda **kwargs: weather.get_weather_forecast(**kwargs),
}

_TOOL_SCHEMAS = [
    vector_store.TOOL_SCHEMA,
    soil_model.TOOL_SCHEMA,
    weather.TOOL_SCHEMA,
]


def _build_gemini_tools() -> List[types.Tool]:
    function_declarations = [
        types.FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters=schema["parameters"],
        )
        for schema in _TOOL_SCHEMAS
    ]
    return [types.Tool(function_declarations=function_declarations)]


def _execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    impl = _TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        result = impl(**args)
        # tool functions may return a list (vector_search) or dict
        return result if isinstance(result, (dict, list)) else {"result": result}
    except Exception as exc:  # tool failures are reported back to the model, not raised
        return {"error": str(exc)}


def run_agent(user_message: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
    """
    Runs the manual function-calling loop for one user turn.

    Returns:
        {
          "answer": str,
          "tool_trace": [
              {"round": int, "tool": str, "arguments": dict,
               "result": Any, "latency_ms": float},
              ...
          ],
          "rounds_used": int,
        }
    """
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    tools = _build_gemini_tools()

    contents: List[types.Content] = []
    for turn in history or []:
        contents.append(
            types.Content(role=turn["role"], parts=[types.Part(text=turn["content"])])
        )
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools,
        # Manual function calling: the SDK will NOT auto-execute tool calls.
        # We intercept every FunctionCall part ourselves below.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    tool_trace: List[Dict[str, Any]] = []
    rounds = 0

    while rounds < config.MAX_TOOL_CALL_ROUNDS:
        rounds += 1
        response = client.models.generate_content(
            model=config.GEMINI_MODEL, contents=contents, config=gen_config
        )

        candidate = response.candidates[0]
        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if getattr(part, "function_call", None) is not None
        ]

        # No tool calls this round -> the model is ready to answer.
        if not function_calls:
            final_text = "".join(
                part.text for part in candidate.content.parts if getattr(part, "text", None)
            )
            return {
                "answer": final_text,
                "tool_trace": tool_trace,
                "rounds_used": rounds,
            }

        # Echo the model's turn (including its function-call parts) back into
        # the conversation, then append one function_response per call.
        contents.append(candidate.content)

        response_parts = []
        for call in function_calls:
            args = dict(call.args or {})
            start = time.perf_counter()
            result = _execute_tool(call.name, args)
            latency_ms = round((time.perf_counter() - start) * 1000, 1)

            tool_trace.append(
                {
                    "round": rounds,
                    "tool": call.name,
                    "arguments": args,
                    "result": result,
                    "latency_ms": latency_ms,
                }
            )

            response_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"content": json.dumps(result, default=str)}
                )
            )

        contents.append(types.Content(role="user", parts=response_parts))

    # Round budget exhausted — force a final answer from whatever we have.
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text="Please give your best final answer now based on the tool results above."
                )
            ],
        )
    )
    final_response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    final_text = "".join(
        part.text
        for part in final_response.candidates[0].content.parts
        if getattr(part, "text", None)
    )
    return {"answer": final_text, "tool_trace": tool_trace, "rounds_used": rounds}
