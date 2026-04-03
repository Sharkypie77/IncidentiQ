"""Synthetic metric time-series generator for IncidentIQ."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from env.models import MetricPoint
from env.state_machine import SERVICES, GroundTruth, ServiceState


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def generate_metrics(
    service: str,
    state: ServiceState,
    rng: random.Random,
    ground_truth: GroundTruth,
) -> Dict[str, List[MetricPoint]]:
    """
    Return four metric time-series for *service*, each with 30 data points
    (one per minute over the last 30 minutes).

    Shape:
    - Points 0-19 (first 20 min): near-baseline with ±5 % noise
    - Points 20-29 (last 10 min):
        * Affected services ramp toward degraded values
        * Red-herring services show slow gradual increase from point 0
    """
    base = SERVICES.get(service)
    if base is None:
        return {}

    now = datetime(2026, 4, 3, 7, 30, 0, tzinfo=timezone.utc)
    start = now - timedelta(minutes=30)

    is_affected = (
        service == ground_truth.root_cause_service
        or service in ground_truth.affected_services
    )
    is_red_herring = service in ground_truth.red_herring_services

    metrics: Dict[str, List[MetricPoint]] = {
        "cpu_pct": [],
        "p99_latency_ms": [],
        "error_rate": [],
        "active_connections": [],
    }

    for i in range(30):
        ts = _iso(start + timedelta(minutes=i))
        progress = i / 29.0  # 0.0 → 1.0 over the window

        if is_red_herring:
            # Slow gradual increase from point 1 — distinguishes from incident
            ramp = 1.0 + 0.15 * progress
            cpu_val = base["cpu"] * ramp * rng.uniform(0.95, 1.05)
            p99_val = base["p99"] * ramp * rng.uniform(0.95, 1.05)
            err_val = base["error_rate"] * ramp * rng.uniform(0.95, 1.05)
            conn_val = base["active_connections"] * ramp * rng.uniform(0.95, 1.05)

        elif is_affected and i >= 20:
            # Ramp toward degraded state values over last 10 minutes
            ramp_progress = (i - 20) / 9.0  # 0.0 → 1.0 within ramp window
            cpu_val = base["cpu"] + (state.cpu_pct - base["cpu"]) * ramp_progress
            p99_val = base["p99"] + (state.p99_ms - base["p99"]) * ramp_progress
            err_val = base["error_rate"] + (state.error_rate - base["error_rate"]) * ramp_progress
            conn_val = base["active_connections"] + (
                state.active_connections - base["active_connections"]
            ) * ramp_progress
            # Add ±5 % noise
            cpu_val *= rng.uniform(0.95, 1.05)
            p99_val *= rng.uniform(0.95, 1.05)
            err_val *= rng.uniform(0.95, 1.05)
            conn_val *= rng.uniform(0.95, 1.05)
        else:
            # Near-baseline with ±5 % noise
            cpu_val = base["cpu"] * rng.uniform(0.95, 1.05)
            p99_val = base["p99"] * rng.uniform(0.95, 1.05)
            err_val = base["error_rate"] * rng.uniform(0.95, 1.05)
            conn_val = base["active_connections"] * rng.uniform(0.95, 1.05)

        metrics["cpu_pct"].append(MetricPoint(
            timestamp=ts, value=round(max(0.0, min(100.0, cpu_val)), 2)
        ))
        metrics["p99_latency_ms"].append(MetricPoint(
            timestamp=ts, value=round(max(0.0, p99_val), 2)
        ))
        metrics["error_rate"].append(MetricPoint(
            timestamp=ts, value=round(max(0.0, min(1.0, err_val)), 6)
        ))
        metrics["active_connections"].append(MetricPoint(
            timestamp=ts, value=round(max(0.0, conn_val), 0)
        ))

    return metrics
