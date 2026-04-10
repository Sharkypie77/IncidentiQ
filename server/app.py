"""FastAPI server for the IncidentIQ environment."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from env.environment import IncidentIQEnv
from env.state_machine import DEPENDENCY_GRAPH, SERVICES
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


class TimelineEvent(BaseModel):
    """Single event in an incident timeline."""

    time_marker: str = Field(..., description="Relative or absolute time marker for the event")
    category: str = Field(..., description="Event category such as deployment, anomaly, or action")
    severity: Literal["info", "warning", "critical"] = Field(..., description="Event severity")
    detail: str = Field(..., description="Human-readable event description")


class TimelineResponse(BaseModel):
    """Reconstructed timeline for an incident session."""

    session_id: str = Field(..., description="Session identifier")
    task_id: str = Field(..., description="Task identifier")
    events: List[TimelineEvent] = Field(..., description="Ordered timeline events")
    summary: str = Field(..., description="Condensed narrative of incident progression")


class RootCauseCandidate(BaseModel):
    """Likely root-cause candidate with evidence."""

    service: str = Field(..., description="Candidate service")
    probability: float = Field(..., description="Relative probability in [0, 1]")
    evidence: List[str] = Field(..., description="Top evidence points supporting this candidate")


class RootCauseTreeResponse(BaseModel):
    """Top root-cause candidates for a session."""

    session_id: str = Field(..., description="Session identifier")
    task_id: str = Field(..., description="Task identifier")
    candidates: List[RootCauseCandidate] = Field(..., description="Top ranked root-cause candidates")


class AskIncidentRequest(BaseModel):
    """Question payload for lightweight incident Q&A."""

    question: str = Field(..., description="Natural language question about the incident")
    session_id: Optional[str] = Field(None, description="Session identifier (defaults to latest)")


class AskIncidentResponse(BaseModel):
    """Answer payload for incident Q&A."""

    session_id: str = Field(..., description="Session identifier")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Concise answer")
    confidence: float = Field(..., description="Confidence estimate in [0, 1]")
    supporting_signals: List[str] = Field(..., description="Signals used to derive the answer")


class McpResponse(BaseModel):
    """MCP JSON-RPC 2.0 response."""
    jsonrpc: str = Field("2.0", description="JSON-RPC version")
    id: Any = Field(1, description="Request identifier")
    result: Any = Field(..., description="Response payload")


# ── Application ─────────────────────────────────────────────────────────────

env = IncidentIQEnv()
UI_FILE = Path(__file__).with_name("index.html")
REVERSE_DEPS: Dict[str, List[str]] = {svc: [] for svc in DEPENDENCY_GRAPH}
for service, deps in DEPENDENCY_GRAPH.items():
    for dep in deps:
        REVERSE_DEPS.setdefault(dep, []).append(service)


def _get_session_id_or_404(session_id: Optional[str]) -> str:
    resolved = session_id or env.last_session_id
    if not resolved:
        raise HTTPException(status_code=404, detail="No active session")
    with env._lock:
        if resolved not in env.state:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {resolved}")
    return resolved


def _score_candidate(service_name: str, svc: Any) -> tuple[float, List[str]]:
    baseline = SERVICES.get(service_name, {})
    base_err = float(baseline.get("error_rate", 0.001))
    base_p99 = float(baseline.get("p99", 1.0))

    score = 0.0
    evidence: List[str] = []

    status = getattr(svc, "status", "healthy")
    cpu_pct = float(getattr(svc, "cpu_pct", 0.0))
    mem_pct = float(getattr(svc, "mem_pct", 0.0))
    p99_ms = float(getattr(svc, "p99_ms", 0.0))
    error_rate = float(getattr(svc, "error_rate", 0.0))
    active_connections = int(getattr(svc, "active_connections", 0))

    if status == "down":
        score += 2.0
        evidence.append("Service status is down")
    elif status == "degraded":
        score += 1.2
        evidence.append("Service status is degraded")

    if error_rate >= max(base_err * 5, 0.01):
        score += min(error_rate * 25.0, 2.0)
        evidence.append(f"Elevated error rate ({error_rate:.3f})")

    if base_p99 > 0 and p99_ms >= base_p99 * 2.0:
        score += min((p99_ms / base_p99) * 0.35, 1.5)
        evidence.append(f"p99 latency spike ({p99_ms:.0f}ms vs baseline {base_p99:.0f}ms)")

    if cpu_pct >= 85:
        score += 0.8
        evidence.append(f"High CPU usage ({cpu_pct:.1f}%)")
    if mem_pct >= 85:
        score += 0.8
        evidence.append(f"High memory usage ({mem_pct:.1f}%)")
    if active_connections >= 80:
        score += 0.7
        evidence.append(f"Connection pressure ({active_connections} active)")

    fanout = len(REVERSE_DEPS.get(service_name, []))
    if fanout > 0 and (status != "healthy" or error_rate >= max(base_err * 4, 0.01)):
        score += min(0.2 * fanout, 0.8)
        evidence.append(f"High blast potential ({fanout} downstream services)")

    if not evidence:
        evidence.append("No strong anomalous signal observed")

    return max(score, 0.01), evidence[:3]


def _build_root_cause_candidates(session_id: str, top_k: int = 3) -> List[RootCauseCandidate]:
    with env._lock:
        ep = env.state[session_id]

    raw: List[tuple[str, float, List[str]]] = []
    for service_name, svc in ep.service_states.items():
        score, evidence = _score_candidate(service_name, svc)
        raw.append((service_name, score, evidence))

    raw.sort(key=lambda x: x[1], reverse=True)
    top = raw[: max(1, top_k)]
    top_total = sum(score for _, score, _ in top) or 1.0

    return [
        RootCauseCandidate(
            service=service,
            probability=round(score / top_total, 4),
            evidence=evidence,
        )
        for service, score, evidence in top
    ]


def _build_timeline(session_id: str, max_action_events: int = 6) -> List[TimelineEvent]:
    with env._lock:
        ep = env.state[session_id]

    events: List[TimelineEvent] = []
    deployments = sorted(
        ep.recent_deployments,
        key=lambda d: (d.get("deployed_at", "") if isinstance(d, dict) else str(d)),
    )
    for dep in deployments:
        service = dep.get("service", "unknown")
        version = dep.get("version", "?")
        deployed_at = dep.get("deployed_at", "unknown-time")
        change_summary = dep.get("change_summary", "deployment event")
        events.append(
            TimelineEvent(
                time_marker=deployed_at,
                category="deployment",
                severity="info",
                detail=f"{service} deployed {version}: {change_summary}",
            )
        )

    for service_name, svc in ep.service_states.items():
        base_err = float(SERVICES.get(service_name, {}).get("error_rate", 0.001))
        has_anomaly = (
            svc.status != "healthy"
            or svc.error_rate >= max(base_err * 4, 0.01)
            or svc.p99_ms >= float(SERVICES.get(service_name, {}).get("p99", 1.0)) * 2.0
        )
        if not has_anomaly:
            continue
        severity: Literal["warning", "critical"] = "critical" if svc.status == "down" else "warning"
        events.append(
            TimelineEvent(
                time_marker="incident-window",
                category="anomaly",
                severity=severity,
                detail=(
                    f"{service_name} status={svc.status}, p99={svc.p99_ms:.0f}ms, "
                    f"error_rate={svc.error_rate:.3f}, cpu={svc.cpu_pct:.1f}%"
                ),
            )
        )

    for entry in ep.action_log[-max_action_events:]:
        action_name = entry.get("action", "unknown")
        reward = float(entry.get("reward", 0.0))
        events.append(
            TimelineEvent(
                time_marker=f"step-{entry.get('step', '?')}",
                category="action",
                severity="info",
                detail=f"Agent executed '{action_name}' (reward {reward:+.2f})",
            )
        )

    return events


def _build_timeline_summary(events: List[TimelineEvent]) -> str:
    deploy_count = sum(1 for e in events if e.category == "deployment")
    anomaly_count = sum(1 for e in events if e.category == "anomaly")
    action_count = sum(1 for e in events if e.category == "action")
    return (
        f"Timeline reconstructed with {deploy_count} deployment events, "
        f"{anomaly_count} anomaly signals, and {action_count} agent actions."
    )


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

@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(UI_FILE)


@app.get("/api", response_model=RootResponse)
async def root() -> RootResponse:
    return RootResponse(
        name="IncidentIQ",
        version="1.1.0",
        description="Production incident response RL environment",
        docs="/docs",
        endpoints=[
            "/",
            "/api",
            "/health",
            "/metadata",
            "/schema",
            "/reset",
            "/step",
            "/state",
            "/tasks",
            "/timeline/{session_id}",
            "/root-cause-tree/{session_id}",
            "/ask_incident",
            "/mcp",
        ],
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
async def reset(body: Optional[ResetRequest] = None) -> ResetResponse:
    if body is None:
        body = ResetRequest()
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
    if env.last_session_id and env.last_session_id in env.state:
        data = env.get_state(env.last_session_id)
        return StateResponse(**data)
    return StateResponse(task_id="", step_count=0, max_steps=0, done=False, service_states={}, cumulative_reward=0.0)


@app.get("/state/{session_id}", response_model=StateResponse)
async def state(session_id: str) -> StateResponse:
    try:
        data = env.get_state(session_id)
        return StateResponse(**data)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/timeline/{session_id}", response_model=TimelineResponse)
async def timeline(session_id: str) -> TimelineResponse:
    resolved = _get_session_id_or_404(session_id)
    with env._lock:
        ep = env.state[resolved]
    events = _build_timeline(resolved)
    return TimelineResponse(
        session_id=resolved,
        task_id=ep.task_id,
        events=events,
        summary=_build_timeline_summary(events),
    )


@app.get("/root-cause-tree/{session_id}", response_model=RootCauseTreeResponse)
async def root_cause_tree(session_id: str) -> RootCauseTreeResponse:
    resolved = _get_session_id_or_404(session_id)
    with env._lock:
        ep = env.state[resolved]
    return RootCauseTreeResponse(
        session_id=resolved,
        task_id=ep.task_id,
        candidates=_build_root_cause_candidates(resolved, top_k=3),
    )


@app.post("/ask_incident", response_model=AskIncidentResponse)
async def ask_incident(body: AskIncidentRequest) -> AskIncidentResponse:
    resolved = _get_session_id_or_404(body.session_id)
    with env._lock:
        ep = env.state[resolved]

    candidates = _build_root_cause_candidates(resolved, top_k=3)
    top = candidates[0]
    timeline_events = _build_timeline(resolved)
    question = body.question.strip()
    q = question.lower()
    supporting_signals = list(top.evidence)

    if "timeline" in q or "happen" in q or "sequence" in q:
        key_events = [e.detail for e in timeline_events[:3]]
        answer = "Incident timeline highlights: " + " | ".join(key_events) if key_events else "No significant timeline events found yet."
    elif "affected" in q or "blast" in q or "impact" in q:
        affected = [
            name
            for name, svc in ep.service_states.items()
            if svc.status != "healthy" or svc.error_rate >= 0.01
        ]
        answer = (
            f"Affected services: {', '.join(affected)}."
            if affected
            else "No clearly affected services are visible yet."
        )
        supporting_signals = [f"{name} shows degraded health" for name in affected[:3]] or supporting_signals
    elif "fix" in q or "next" in q or "recommend" in q or "remediate" in q:
        suggestions = {
            "postgres": "Check max_connections and apply a config patch if reduced unexpectedly.",
            "auth-service": "Restart auth-service and verify redis pool settings.",
            "order-service": "Inspect recent order-service config/deploy changes and rollback or patch.",
            "analytics-service": "Inspect batch cache TTL and patch unbounded cache settings.",
            "api-gateway": "Validate upstream dependency health before restarting gateway pods.",
        }
        answer = f"Top suspected root cause is {top.service} ({top.probability:.0%}). Recommended next action: {suggestions.get(top.service, 'Run targeted config and deployment checks on the suspected service.')}"
    else:
        answer = (
            f"Most likely root cause is {top.service} ({top.probability:.0%}) based on "
            f"{'; '.join(top.evidence[:2]).lower()}."
        )

    return AskIncidentResponse(
        session_id=resolved,
        question=question,
        answer=answer,
        confidence=top.probability,
        supporting_signals=supporting_signals,
    )


@app.post("/mcp", response_model=McpResponse, description="MCP protocol stub. Core environment interaction uses /reset and /step.")
async def mcp_endpoint(body: Optional[dict] = None) -> McpResponse:
    """MCP protocol stub. Core environment interaction uses /reset and /step."""
    if body is None:
        body = {}
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
