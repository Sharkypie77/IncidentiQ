"""FastAPI server for the IncidentIQ environment."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional
from starlette.responses import JSONResponse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.environment import IncidentIQEnv
from env.models import (
    Action,
    Deployment,
    LogLine,
    MetricPoint,
    Observation,
    Reward,
    ResetResult,
    ServiceHealth,
    StepResult,
)


# ── Request / response bodies ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="Task identifier to start (defaults to first task)")
    seed: Optional[int] = Field(None, description="Random seed for determinism")


class StepRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier from /reset")
    action: Action = Field(..., description="Action to execute")


class ResetResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier for subsequent calls")
    observation: Observation = Field(..., description="Initial observation")
    task_id: str = Field(..., description="Task identifier")
    task_description: str = Field(..., description="Human-readable task description")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Server health status")
    version: str = Field(..., description="Server version string")


class MetadataResponse(BaseModel):
    """Environment metadata."""
    name: str = Field(..., description="Environment name")
    description: str = Field(..., description="Environment description")
    version: str = Field(..., description="Environment version")
    domain: str = Field(..., description="Domain of the environment")
    author: str = Field(..., description="Author of the environment")


class RootResponse(BaseModel):
    """Root endpoint response."""
    name: str = Field(..., description="Environment name")
    version: str = Field(..., description="Environment version")
    description: str = Field(..., description="Environment description")
    docs: str = Field(..., description="Documentation endpoint")
    endpoints: List[str] = Field(..., description="Available API endpoints")


class SchemaResponse(BaseModel):
    """Schema endpoint response."""
    action: Dict[str, Any] = Field(..., description="JSON Schema of the Action model")
    observation: Dict[str, Any] = Field(..., description="JSON Schema of the Observation model")
    state: Dict[str, Any] = Field(..., description="JSON Schema of the state object")


class TaskInfo(BaseModel):
    """Information about an available task."""
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    description: str = Field(..., description="Detailed task description")
    difficulty: str = Field("", description="Task difficulty level")
    max_steps: int = Field(20, description="Maximum number of steps allowed")


class StateResponse(BaseModel):
    """Current environment state."""
    task_id: str = Field(..., description="Active task identifier")
    step_count: int = Field(..., description="Number of steps taken")
    max_steps: int = Field(..., description="Maximum steps allowed")
    done: bool = Field(..., description="Whether the episode has ended")
    service_states: Dict[str, Any] = Field(..., description="Current service states")
    cumulative_reward: float = Field(..., description="Cumulative reward so far")
    action_log_length: int = Field(0, description="Number of actions taken so far")
    ground_truth: Optional[Dict[str, Any]] = Field(None, description="Ground truth revealed after episode ends")
    action_log: Optional[List[Dict[str, Any]]] = Field(None, description="Full action log revealed after episode ends")


class McpResponse(BaseModel):
    """MCP JSON-RPC 2.0 response."""
    jsonrpc: str = Field("2.0", description="JSON-RPC version")
    id: Any = Field(1, description="Request identifier")
    result: Any = Field(..., description="Response payload")


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

@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    return RootResponse(
        name="IncidentIQ",
        version="1.1.0",
        description="Production incident response RL environment",
        docs="/docs",
        endpoints=["/health", "/metadata", "/schema", "/reset", "/step", "/state", "/tasks", "/mcp"],
    )

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", version="1.1.0")


@app.get("/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    return MetadataResponse(
        name="incidentiq",
        description=(
            "Production incident response environment. An AI agent acts as an "
            "on-call SRE, diagnosing and remediating software failures across "
            "a simulated microservice architecture."
        ),
        version="1.1.0",
        domain="sre",
        author="Zewx77",
    )


@app.get("/schema", response_model=SchemaResponse)
async def schema() -> SchemaResponse:
    return SchemaResponse(
        action=Action.model_json_schema(),
        observation=Observation.model_json_schema(),
        state={
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
    )


@app.post("/reset", response_model=ResetResponse)
async def reset(body: ResetRequest) -> ResetResponse:
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

    return ResetResponse(
        session_id=session_id,
        observation=result.observation,
        task_id=result.task_id,
        task_description=result.task_description,
    )


@app.post("/step", response_model=StepResult)
async def step(body: StepRequest) -> StepResult:
    try:
        result = env.step(body.session_id, body.action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return result


@app.get("/state", response_model=StateResponse)
async def state_root() -> StateResponse:
    """Return state for the most recent session (OpenEnv runtime contract)."""
    if env.state:
        session_id = list(env.state.keys())[-1]
        data = env.get_state(session_id)
        return StateResponse(**data)
    return StateResponse(task_id="", step_count=0, max_steps=0, done=False, service_states={}, cumulative_reward=0.0)


@app.get("/state/{session_id}", response_model=StateResponse)
async def state(session_id: str) -> StateResponse:
    try:
        data = env.get_state(session_id)
        return StateResponse(**data)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/mcp", response_model=McpResponse, description="MCP protocol stub. Core environment interaction uses /reset and /step.")
async def mcp_endpoint(body: dict = {}) -> McpResponse:
    """MCP protocol stub. Core environment interaction uses /reset and /step."""
    return McpResponse(
        jsonrpc="2.0",
        id=body.get("id", 1),
        result={
            "name": "incidentiq",
            "version": "1.1.0",
        },
    )


@app.get("/tasks", response_model=List[TaskInfo])
async def tasks() -> List[TaskInfo]:
    raw_tasks = env.get_tasks()
    return [TaskInfo(**t) for t in raw_tasks]


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
