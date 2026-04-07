"""Synthetic log generator for IncidentIQ."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

from env.utils import iso_timestamp as _iso

from env.models import LogLine
from env.state_machine import GroundTruth, ServiceState




def _random_req_id(rng: random.Random) -> str:
    return uuid.UUID(int=rng.getrandbits(128), version=4).hex[:12]


def _random_order_id(rng: random.Random) -> str:
    return f"ORD-{rng.randint(100000, 999999)}"


def _random_cust_id(rng: random.Random) -> str:
    return f"cust_{rng.randint(1000, 9999)}"


def _random_uid(rng: random.Random) -> str:
    return f"user_{rng.randint(10000, 99999)}"


# ── Per-service log template generators ─────────────────────────────────────

def _api_gateway_logs(
    state: ServiceState,
    rng: random.Random,
    gt: GroundTruth,
    base_time: datetime,
    n: int,
) -> List[LogLine]:
    logs: List[LogLine] = []
    endpoints = ["orders", "users", "products", "health", "auth/token"]
    for i in range(n):
        ts = _iso(base_time - timedelta(seconds=rng.randint(0, 1800)))
        req_id = _random_req_id(rng)
        roll = rng.random()

        if state.status == "down":
            ms = int(state.p99_ms * rng.uniform(0.8, 1.5))
            logs.append(LogLine(
                timestamp=ts, service="api-gateway", level="ERROR",
                message=f"upstream timeout order-service after {ms}ms req_id={req_id}",
            ))
        elif state.status == "degraded" and roll < 0.4:
            ms = int(state.p99_ms * rng.uniform(0.6, 1.2))
            if roll < 0.2:
                logs.append(LogLine(
                    timestamp=ts, service="api-gateway", level="ERROR",
                    message=f"upstream timeout order-service after {ms}ms req_id={req_id}",
                ))
            else:
                logs.append(LogLine(
                    timestamp=ts, service="api-gateway", level="WARN",
                    message=f"high latency detected p99={ms}ms threshold=500ms",
                ))
        else:
            ep = rng.choice(endpoints)
            ms = int(state.p50_ms * rng.uniform(0.5, 2.0))
            status_code = 200 if rng.random() > state.error_rate else 503
            logs.append(LogLine(
                timestamp=ts, service="api-gateway", level="INFO",
                message=f"GET /api/v1/{ep} {status_code} {ms}ms req_id={req_id}",
            ))
    return logs


def _order_service_logs(
    state: ServiceState,
    rng: random.Random,
    gt: GroundTruth,
    base_time: datetime,
    n: int,
) -> List[LogLine]:
    logs: List[LogLine] = []
    is_root = gt.root_cause_service == "order-service"

    for i in range(n):
        ts = _iso(base_time - timedelta(seconds=rng.randint(0, 1800)))
        roll = rng.random()

        if state.status == "down":
            ms = int(state.p99_ms * rng.uniform(0.8, 1.5))
            logs.append(LogLine(
                timestamp=ts, service="order-service", level="ERROR",
                message=f"db query timeout after {ms}ms query=SELECT * FROM orders WHERE customer_id=?",
            ))
        elif state.status == "degraded" and roll < 0.4:
            if roll < 0.15:
                ms = int(state.p99_ms * rng.uniform(0.7, 1.3))
                logs.append(LogLine(
                    timestamp=ts, service="order-service", level="ERROR",
                    message=f"db query timeout after {ms}ms query=SELECT * FROM orders WHERE customer_id=?",
                ))
            elif roll < 0.25:
                rows = rng.randint(50000, 500000)
                logs.append(LogLine(
                    timestamp=ts, service="order-service", level="WARN",
                    message=f"slow query detected: full table scan orders rows_examined={rows}",
                ))
            else:
                wait = rng.randint(500, 5000)
                logs.append(LogLine(
                    timestamp=ts, service="order-service", level="ERROR",
                    message=f"connection pool exhausted waiting={wait}ms",
                ))
        else:
            order_id = _random_order_id(rng)
            cust_id = _random_cust_id(rng)
            ms = int(state.p50_ms * rng.uniform(0.5, 2.0))
            logs.append(LogLine(
                timestamp=ts, service="order-service", level="INFO",
                message=f"POST /orders/{order_id} 200 {ms}ms customer={cust_id}",
            ))

        # Inject root-cause symptom logs for cpu_saturation
        if is_root and gt.failure_type == "cpu_saturation" and i < 3:
            symptom_ts = _iso(base_time - timedelta(seconds=rng.randint(0, 300)))
            rows = rng.randint(200000, 800000)
            logs.append(LogLine(
                timestamp=symptom_ts, service="order-service", level="WARN",
                message=f"slow query detected: full table scan orders rows_examined={rows}",
            ))

        # Inject duplicate transaction logs for race_condition
        if is_root and gt.failure_type == "race_condition" and i < 4:
            tx_id = f"tx_{rng.randint(100000, 999999)}"
            t1 = base_time - timedelta(seconds=rng.randint(60, 600))
            t2 = t1 + timedelta(milliseconds=rng.randint(150, 250))
            logs.append(LogLine(
                timestamp=_iso(t1), service="order-service", level="INFO",
                message=f"transaction {tx_id} processed",
            ))
            logs.append(LogLine(
                timestamp=_iso(t2), service="order-service", level="WARN",
                message=f"duplicate transaction detected: {tx_id} already processed 200ms ago — idempotency key missing",
            ))

        # Inject connection pool exhaustion logs for Task 4 (postgres root cause)
        if is_root and gt.failure_type == "connection_exhaustion" and i < 3:
            symptom_ts = _iso(base_time - timedelta(seconds=rng.randint(0, 300)))
            wait = rng.randint(2000, 8000)
            logs.append(LogLine(
                timestamp=symptom_ts, service="order-service", level="ERROR",
                message=f"database connection pool exhausted: waited {wait}ms — all connections to postgres in use",
            ))

    return logs


def _auth_service_logs(
    state: ServiceState,
    rng: random.Random,
    gt: GroundTruth,
    base_time: datetime,
    n: int,
) -> List[LogLine]:
    logs: List[LogLine] = []
    is_root = gt.root_cause_service == "auth-service"

    for i in range(n):
        ts = _iso(base_time - timedelta(seconds=rng.randint(0, 1800)))
        roll = rng.random()

        if state.status == "down":
            ms = int(state.p99_ms * rng.uniform(0.8, 1.5))
            uid = _random_uid(rng)
            logs.append(LogLine(
                timestamp=ts, service="auth-service", level="ERROR",
                message=f"redis TIMEOUT after {ms}ms op=GET session:{uid}",
            ))
        elif state.status == "degraded" and roll < 0.4:
            uid = _random_uid(rng)
            if roll < 0.15:
                ms = int(state.p99_ms * rng.uniform(0.7, 1.3))
                logs.append(LogLine(
                    timestamp=ts, service="auth-service", level="ERROR",
                    message=f"redis TIMEOUT after {ms}ms op=GET session:{uid}",
                ))
            elif roll < 0.25:
                n_conns = rng.randint(85, 100)
                logs.append(LogLine(
                    timestamp=ts, service="auth-service", level="ERROR",
                    message=f"connection pool exhausted redis connections={n_conns}/100",
                ))
            else:
                pct = rng.randint(80, 99)
                logs.append(LogLine(
                    timestamp=ts, service="auth-service", level="WARN",
                    message=f"redis pool utilization {pct}% threshold=80%",
                ))
        else:
            uid = _random_uid(rng)
            ms = int(state.p50_ms * rng.uniform(0.5, 2.0))
            logs.append(LogLine(
                timestamp=ts, service="auth-service", level="INFO",
                message=f"AUTH OK user={uid} token_ttl=3600s latency={ms}ms",
            ))

        # Inject root-cause symptom logs for connection_exhaustion
        if is_root and gt.failure_type == "connection_exhaustion" and i < 3:
            symptom_ts = _iso(base_time - timedelta(seconds=rng.randint(0, 300)))
            n_conns = rng.randint(90, 100)
            logs.append(LogLine(
                timestamp=symptom_ts, service="auth-service", level="ERROR",
                message=f"connection pool exhausted redis connections={n_conns}/100",
            ))

    return logs


def _postgres_logs(
    state: ServiceState,
    rng: random.Random,
    gt: GroundTruth,
    base_time: datetime,
    n: int,
) -> List[LogLine]:
    logs: List[LogLine] = []
    tables = ["orders", "customers", "inventory", "payments", "sessions"]
    query_hints = [
        "SELECT * FROM orders WHERE customer_id=?",
        "UPDATE inventory SET stock=stock-1 WHERE sku=?",
        "INSERT INTO payments (order_id, amount) VALUES (?, ?)",
        "SELECT COUNT(*) FROM sessions WHERE expired_at < NOW()",
    ]

    is_root = gt.root_cause_service == "postgres"

    for i in range(n):
        ts = _iso(base_time - timedelta(seconds=rng.randint(0, 1800)))
        roll = rng.random()

        if state.status == "down":
            curr = rng.randint(95, 100)
            logs.append(LogLine(
                timestamp=ts, service="postgres", level="ERROR",
                message=f"too many connections current={curr} max=100",
            ))
        elif state.status == "degraded" and roll < 0.4:
            if roll < 0.2:
                ms = int(state.p99_ms * rng.uniform(0.7, 1.3))
                hint = rng.choice(query_hints)
                logs.append(LogLine(
                    timestamp=ts, service="postgres", level="WARN",
                    message=f"slow query {ms}ms: {hint}",
                ))
            else:
                curr = rng.randint(80, 100)
                logs.append(LogLine(
                    timestamp=ts, service="postgres", level="ERROR",
                    message=f"too many connections current={curr} max=100",
                ))
        else:
            table = rng.choice(tables)
            rows = rng.randint(1, 500)
            ms = int(state.p50_ms * rng.uniform(0.5, 2.0))
            logs.append(LogLine(
                timestamp=ts, service="postgres", level="INFO",
                message=f"query OK table={table} rows={rows} duration={ms}ms",
            ))

        # Inject root-cause symptom logs for connection_exhaustion (Task 4)
        if is_root and gt.failure_type == "connection_exhaustion" and i < 3:
            symptom_ts = _iso(base_time - timedelta(seconds=rng.randint(0, 300)))
            curr = rng.randint(18, 20)
            logs.append(LogLine(
                timestamp=symptom_ts, service="postgres", level="ERROR",
                message=f"FATAL: too many clients already current={curr} max_connections=20 — connection rejected",
            ))
            logs.append(LogLine(
                timestamp=_iso(base_time - timedelta(seconds=rng.randint(300, 900))),
                service="postgres", level="WARN",
                message=f"connection pool near capacity: {curr}/20 active connections (max_connections was reduced from 100 to 20)",
            ))

    return logs


def _analytics_service_logs(
    state: ServiceState,
    rng: random.Random,
    gt: GroundTruth,
    base_time: datetime,
    n: int,
) -> List[LogLine]:
    """Slightly degraded logs — red herring in most tasks, but real root cause in Task 5."""
    logs: List[LogLine] = []
    is_red_herring = "analytics-service" in gt.red_herring_services
    is_root = gt.root_cause_service == "analytics-service"

    for i in range(n):
        # Red herring logs are timestamped 3-6 hours before the incident
        if is_red_herring:
            offset_seconds = rng.randint(3 * 3600, 6 * 3600)
        else:
            offset_seconds = rng.randint(0, 1800)
        ts = _iso(base_time - timedelta(seconds=offset_seconds))
        roll = rng.random()

        if roll < 0.35:
            pct = rng.randint(72, 92)
            mb = rng.randint(1200, 2800)
            logs.append(LogLine(
                timestamp=ts, service="analytics-service", level="WARN",
                message=f"memory usage {pct}% heap_used={mb}MB",
            ))
        elif roll < 0.65:
            gc_ms = rng.randint(80, 350)
            logs.append(LogLine(
                timestamp=ts, service="analytics-service", level="WARN",
                message=f"GC pause {gc_ms}ms generation=old",
            ))
        else:
            rows = rng.randint(10000, 500000)
            sec = rng.randint(2, 45)
            logs.append(LogLine(
                timestamp=ts, service="analytics-service", level="INFO",
                message=f"batch job completed rows={rows} duration={sec}s",
            ))

        # Inject root-cause symptom logs for memory_leak (Task 5)
        if is_root and gt.failure_type == "memory_leak" and i < 4:
            symptom_ts = _iso(base_time - timedelta(seconds=rng.randint(0, 300)))
            mb = rng.randint(3200, 3900)
            cache_entries = rng.randint(500000, 2000000)
            logs.append(LogLine(
                timestamp=symptom_ts, service="analytics-service", level="ERROR",
                message=f"OOM warning: heap_used={mb}MB limit=4096MB batch_cache_entries={cache_entries} — cache growing unbounded (batch_cache_ttl=0)",
            ))
            if i < 2:
                logs.append(LogLine(
                    timestamp=_iso(base_time - timedelta(seconds=rng.randint(300, 1200))),
                    service="analytics-service", level="ERROR",
                    message=f"container killed by OOM killer: memory={mb}MB exceeds limit — batch cache not evicting (ttl=0)",
                ))

    return logs


# ── Public API ──────────────────────────────────────────────────────────────

_GENERATORS = {
    "api-gateway": _api_gateway_logs,
    "order-service": _order_service_logs,
    "auth-service": _auth_service_logs,
    "postgres": _postgres_logs,
    "analytics-service": _analytics_service_logs,
}


def generate_logs(
    service: str,
    state: ServiceState,
    rng: random.Random,
    ground_truth: GroundTruth,
    n: int = 25,
) -> List[LogLine]:
    """Generate *n* realistic-looking synthetic log lines for *service*."""
    base_time = datetime(2026, 4, 3, 7, 30, 0, tzinfo=timezone.utc)

    gen = _GENERATORS.get(service)
    if gen is None:
        return []

    logs = gen(state, rng, ground_truth, base_time, n)

    # Sort by timestamp descending (most recent first)
    logs.sort(key=lambda ll: ll.timestamp, reverse=True)

    # Cap at 30 entries
    return logs[:30]
