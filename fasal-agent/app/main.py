"""
FastAPI entrypoint for the Fasal Agentic Crop Advisory System.

POST /advise is the main endpoint: it runs the manual Gemini function-calling
agent loop and returns both the final advice and the full tool-call trace,
so a caller (or a debugging UI) can audit exactly which tools the LLM chose,
in what order, with what arguments, and what each one returned.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import config
from app.agent import run_agent
from app.tools.vector_store import build_index

app = FastAPI(
    title="Fasal Agentic Crop Advisory System",
    description=(
        "A multi-tool AI agent where the LLM autonomously selects and "
        "sequences tools — vector search, an ML regression model, and a "
        "live weather API — instead of following a fixed RAG pipeline."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    role: str = Field(..., description="'user' or 'model'")
    content: str


class AdviseRequest(BaseModel):
    message: str = Field(..., description="The farmer's question.")
    history: Optional[List[ChatTurn]] = Field(
        default=None, description="Prior turns for multi-turn conversations."
    )


class ToolTraceEntry(BaseModel):
    round: int
    tool: str
    arguments: dict
    result: object
    latency_ms: float


class AdviseResponse(BaseModel):
    answer: str
    tool_trace: List[ToolTraceEntry]
    rounds_used: int


@app.on_event("startup")
def _ensure_index():
    # Idempotent: only rebuilds if the collection is empty.
    build_index()


@app.get("/health")
def health():
    return {"status": "ok", "model": config.GEMINI_MODEL}


@app.post("/advise", response_model=AdviseResponse)
def advise(req: AdviseRequest):
    if not config.GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured. Set it in your environment or .env file.",
        )
    history = [turn.model_dump() for turn in req.history] if req.history else None
    try:
        result = run_agent(req.message, history=history)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent run failed: {exc}") from exc
    return result


@app.post("/reindex")
def reindex():
    """Forces a rebuild of the agronomy knowledge vector index."""
    count = build_index(force=True)
    return {"chunks_indexed": count}
