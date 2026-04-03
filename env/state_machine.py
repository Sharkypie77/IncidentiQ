"""Service state machine and episode state for IncidentIQ."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Baseline healthy states for each service ────────────────────────────────

SERVICES: Dict[str, Dict[str, Any]] = {
    "api-gateway": {
        "p50": 12, "p99": 45, "cpu": 20, "mem": 30,
        "error_rate": 0.001, "active_connections": 20,
    },
    "order-service": {
        "p50": 45, "p99": 120, "cpu": 35, "mem": 45,
        "error_rate": 0.002, "active_connections": 15,
    },
    "auth-service": {
        "p50": 8, "p99": 25, "cpu": 15, "mem": 25,
        "error_rate": 0.001, "active_connections": 10,
    },
    "postgres": {
        "p50": 5, "p99": 18, "cpu": 25, "mem": 60,
        "error_rate": 0.000, "active_connections": 12,
    },
    "analytics-service": {
        "p50": 200, "p99": 800, "cpu": 40, "mem": 55,
        "error_rate": 0.005, "active_connections": 8,
    },
}

DEPENDENCY_GRAPH: Dict[str, List[str]] = {
    "api-gateway": ["order-service", "auth-service"],
    "order-service": ["postgres"],
    "auth-service": [],
    "postgres": [],
    "analytics-service": [],
}

DECAY = 0.4


@dataclass
class ServiceState:
    """Mutable state of a single service during an episode."""

    name: str
    status: str = "healthy"
    cpu_pct: float = 0.0
    mem_pct: float = 0.0
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0

    @classmethod
    def from_baseline(cls, name: str) -> "ServiceState":
        """Create a ServiceState initialised to the service's healthy baseline."""
        b = SERVICES[name]
        return cls(
            name=name,
            status="healthy",
            cpu_pct=b["cpu"],
            mem_pct=b["mem"],
            p50_ms=b["p50"],
            p99_ms=b["p99"],
            error_rate=b["error_rate"],
            active_connections=b["active_connections"],
        )


@dataclass
class GroundTruth:
    """Holds the correct answers for an episode so the grader can score."""

    root_cause_service: str
    root_cause_mechanism: str
    failure_type: str
    correct_remediation: str
    affected_services: List[str]
    red_herring_services: List[str]
    failure_started_offset_minutes: int = 10
    config_entries: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)
    extra_deployments: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class EpisodeState:
    """Full mutable state of one episode."""

    task_id: str
    step_count: int
    max_steps: int
    done: bool
    service_states: Dict[str, ServiceState]
    ground_truth: GroundTruth
    action_log: List[dict]
    cumulative_reward: float
    seed: int
    rng: random.Random
    alert_summary: str = ""
    recent_deployments: List[Dict[str, str]] = field(default_factory=list)
    rewarded_services: set = field(default_factory=set)


# ── Failure propagation ─────────────────────────────────────────────────────

def _reverse_deps() -> Dict[str, List[str]]:
    """Build reverse dependency map: service -> list of services that depend on it."""
    rev: Dict[str, List[str]] = {s: [] for s in SERVICES}
    for svc, deps in DEPENDENCY_GRAPH.items():
        for d in deps:
            rev[d].append(svc)
    return rev


def _apply_failure(
    state: ServiceState,
    severity: float,
    failure_type: str,
    rng: random.Random,
) -> ServiceState:
    """Mutate *state* in-place based on failure type and severity, then add noise."""
    baseline = SERVICES[state.name]

    if failure_type == "cpu_saturation":
        state.p99_ms = baseline["p99"] * (1 + severity * 34)
        state.cpu_pct = baseline["cpu"] + severity * 55
        state.error_rate = baseline["error_rate"] * (1 + severity * 4)
        state.p50_ms = baseline["p50"] * (1 + severity * 5)

    elif failure_type == "connection_exhaustion":
        state.p99_ms = baseline["p99"] * (1 + severity * 20)
        state.error_rate = baseline["error_rate"] * (1 + severity * 15)
        state.active_connections = int(100 * severity)
        state.p50_ms = baseline["p50"] * (1 + severity * 3)

    elif failure_type == "memory_leak":
        state.mem_pct = baseline["mem"] + severity * 40
        state.p99_ms = baseline["p99"] * (1 + severity * 3)
        state.p50_ms = baseline["p50"] * (1 + severity * 1)

    elif failure_type == "race_condition":
        state.error_rate = baseline["error_rate"] + severity * 0.03
        state.p50_ms = baseline["p50"] * (1 + severity * 0.1)
        state.p99_ms = baseline["p99"] * (1 + severity * 0.2)

    elif failure_type == "cascaded_timeout":
        state.p99_ms = baseline["p99"] * (1 + severity * 8)
        state.error_rate = baseline["error_rate"] * (1 + severity * 6)
        state.p50_ms = baseline["p50"] * (1 + severity * 2)
        if severity > 0.3:
            state.status = "degraded"

    # Determine overall status based on severity
    if failure_type != "cascaded_timeout" and failure_type != "race_condition":
        if severity > 0.7:
            state.status = "degraded"
        if severity > 0.95:
            state.status = "down"

    # Add ±10 % noise to all metric values
    def noisy(val: float) -> float:
        return val * rng.uniform(0.9, 1.1)

    state.cpu_pct = max(0.0, min(100.0, noisy(state.cpu_pct)))
    state.mem_pct = max(0.0, min(100.0, noisy(state.mem_pct)))
    state.p50_ms = max(0.0, noisy(state.p50_ms))
    state.p99_ms = max(0.0, noisy(state.p99_ms))
    state.error_rate = max(0.0, min(1.0, noisy(state.error_rate)))
    state.active_connections = max(0, int(noisy(float(state.active_connections))))

    return state


def propagate_failure(
    root_service: str,
    severity: float,
    failure_type: str,
    rng: Optional[random.Random] = None,
) -> Dict[str, ServiceState]:
    """
    Propagate a failure from *root_service* through the dependency graph.

    Returns a dict of ``{service_name: ServiceState}`` for **all** services,
    with affected services having degraded metrics.
    """
    if rng is None:
        rng = random.Random(42)

    # Start every service at baseline
    states: Dict[str, ServiceState] = {
        name: ServiceState.from_baseline(name) for name in SERVICES
    }

    # Apply primary failure to the root service
    _apply_failure(states[root_service], severity, failure_type, rng)

    # Walk reverse deps (BFS) to propagate cascading impact
    reverse = _reverse_deps()
    visited = {root_service}
    frontier = [(root_service, severity)]

    while frontier:
        next_frontier: list = []
        for src, sev in frontier:
            for dependent in reverse.get(src, []):
                if dependent not in visited:
                    visited.add(dependent)
                    cascaded_sev = sev * DECAY
                    _apply_failure(
                        states[dependent],
                        cascaded_sev,
                        "cascaded_timeout",
                        rng,
                    )
                    next_frontier.append((dependent, cascaded_sev))
        frontier = next_frontier

    return states
