"""Pydantic v2 models for the IncidentIQ environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ServiceHealth(BaseModel):
    """Health snapshot of a single microservice."""

    name: str = Field(..., description="Name of the service")
    status: Literal["healthy", "degraded", "down"] = Field(
        ..., description="Current operational status"
    )
    cpu_pct: float = Field(..., description="CPU utilization percentage (0-100)")
    mem_pct: float = Field(..., description="Memory utilization percentage (0-100)")
    p50_ms: float = Field(..., description="Median (p50) response latency in milliseconds")
    p99_ms: float = Field(..., description="99th-percentile response latency in milliseconds")
    error_rate: float = Field(..., description="Fraction of requests resulting in errors (0.0-1.0)")
    active_connections: int = Field(
        ..., description="Number of currently active connections"
    )


class LogLine(BaseModel):
    """A single structured log entry."""

    timestamp: str = Field(..., description="ISO-8601 timestamp of the log entry")
    service: str = Field(..., description="Name of the emitting service")
    level: Literal["INFO", "WARN", "ERROR"] = Field(
        ..., description="Log severity level"
    )
    message: str = Field(..., description="Human-readable log message")


class MetricPoint(BaseModel):
    """A single data point in a time series."""

    timestamp: str = Field(..., description="ISO-8601 timestamp of the data point")
    value: float = Field(..., description="Metric value at this timestamp")


class Deployment(BaseModel):
    """Record of a service deployment."""

    service: str = Field(..., description="Name of the deployed service")
    version: str = Field(..., description="Semantic version string of the deployment")
    deployed_at: str = Field(..., description="ISO-8601 timestamp of deployment")
    deployed_by: str = Field(..., description="Username or system that triggered the deploy")
    change_summary: str = Field(..., description="Brief description of what changed")


class Observation(BaseModel):
    """What the agent sees at each step."""

    alert_summary: str = Field(..., description="Human-readable summary of the triggering alert")
    service_health: Dict[str, ServiceHealth] = Field(
        ..., description="Current health of every service keyed by service name"
    )
    recent_logs: List[LogLine] = Field(
        ..., description="Most recent log entries across all services"
    )
    metrics: Dict[str, List[MetricPoint]] = Field(
        ..., description="Time-series metrics keyed by metric name"
    )
    dependency_graph: Dict[str, List[str]] = Field(
        ..., description="Service dependency graph: service -> list of dependencies"
    )
    recent_deployments: List[Deployment] = Field(
        ..., description="Recent deployment records across all services"
    )
    step_number: int = Field(..., description="Current step index (0-based)")
    steps_remaining: int = Field(..., description="Number of steps left before timeout")
    last_action_result: Optional[str] = Field(
        None, description="Result string from the previous action, None on first step"
    )


VALID_ACTIONS = [
    "query_logs",
    "query_metrics",
    "query_traces",
    "check_deployment",
    "check_config",
    "hypothesize",
    "remediate",
    "close_incident",
]


class Action(BaseModel):
    """An action the agent wants to take."""

    action: str = Field(
        ...,
        description="One of the 8 valid action types: query_logs, query_metrics, "
        "query_traces, check_deployment, check_config, hypothesize, remediate, close_incident",
    )
    params: Dict[str, Any] = Field(
        ..., description="Action-specific parameters (see docs for each action type)"
    )


class Reward(BaseModel):
    """Reward signal returned after each step."""

    value: float = Field(..., description="Reward value for the current step")
    reason: str = Field(..., description="Human-readable explanation of the reward")
    cumulative: float = Field(..., description="Cumulative reward over the episode so far")


class StepResult(BaseModel):
    """Result of executing a single step in the environment."""

    observation: Observation = Field(..., description="Updated observation after the action")
    reward: Reward = Field(..., description="Reward signal for this step")
    done: bool = Field(..., description="True if the episode has ended")
    info: Dict[str, Any] = Field(
        ..., description="Additional metadata (task_id, step_count, etc.)"
    )


class ResetResult(BaseModel):
    """Result of resetting the environment for a new episode."""

    observation: Observation = Field(..., description="Initial observation for the episode")
    task_id: str = Field(..., description="Identifier of the task being run")
    task_description: str = Field(..., description="Human-readable task description")
