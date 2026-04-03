"""FastAPI server for the IncidentIQ environment."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from starlette.responses import JSONResponse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.environment import IncidentIQEnv
from env.models import Action, Observation, ResetResult, StepResult


# ── Request / response bodies ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="Task identifier to start (defaults to first task)")
    seed: Optional[int] = Field(None, description="Random seed for determinism")


class StepRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier from /reset")
    action: Action = Field(..., description="Action to execute")


class ResetResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier for subsequent calls")
    observation: Any = Field(..., description="Initial observation")
    task_id: str = Field(..., description="Task identifier")
    task_description: str = Field(..., description="Human-readable task description")


# ── Application ─────────────────────────────────────────────────────────────

env = IncidentIQEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("IncidentIQ server ready", flush=True)
    yield


app = FastAPI(
    title="IncidentIQ",
    description="Production incident response RL environment",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict:
    return {
        "name": "IncidentIQ",
        "version": "1.1.0",
        "description": "Production incident response RL environment",
        "docs": "/docs",
        "endpoints": ["/health", "/metadata", "/schema", "/reset", "/step", "/state", "/tasks", "/mcp"],
    }

@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "version": "1.1.0"}


@app.get("/metadata")
async def metadata() -> dict:
    return {
        "name": "incidentiq",
        "description": (
            "Production incident response environment. An AI agent acts as an "
            "on-call SRE, diagnosing and remediating software failures across "
            "a simulated microservice architecture."
        ),
        "version": "1.1.0",
        "domain": "sre",
        "author": "Zewx77",
    }


@app.get("/schema")
async def schema() -> dict:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "done": {"type": "boolean"},
                "service_states": {"type": "object"},
                "cumulative_reward": {"type": "number"},
            },
        },
    }


@app.post("/reset")
async def reset(body: ResetRequest) -> dict:
    # Default to first available task if none specified
    task_id = body.task_id
    if not task_id:
        available = env.get_tasks()
        if not available:
            raise HTTPException(status_code=500, detail="No tasks available")
        task_id = available[0]["task_id"]

    try:
        session_id, result = env.reset(task_id, body.seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "session_id": session_id,
        "observation": result.observation.model_dump(),
        "task_id": result.task_id,
        "task_description": result.task_description,
    }


@app.post("/step")
async def step(body: StepRequest) -> dict:
    try:
        result = env.step(body.session_id, body.action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return result.model_dump()


@app.get("/state")
async def state_root() -> dict:
    """Return state for the most recent session (OpenEnv runtime contract)."""
    if env._sessions:
        session_id = list(env._sessions.keys())[-1]
        return env.get_state(session_id)
    return {"task_id": "", "step_count": 0, "max_steps": 0, "done": False, "service_states": {}, "cumulative_reward": 0.0}


@app.get("/state/{session_id}")
async def state(session_id: str) -> dict:
    try:
        return env.get_state(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/mcp")
async def mcp_endpoint(body: dict = {}) -> JSONResponse:
    """MCP JSON-RPC 2.0 endpoint (minimal implementation)."""
    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "name": "incidentiq",
            "version": "1.1.0",
        },
    })


@app.get("/tasks")
async def tasks() -> List[dict]:
    return env.get_tasks()


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
