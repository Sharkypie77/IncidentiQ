"""Task 4 — Database connection limit (medium)."""

from __future__ import annotations

import random

from env.state_machine import (
    EpisodeState,
    GroundTruth,
    propagate_failure,
)
from tasks.base import BaseTask


class Task4DbConnectionLimit(BaseTask):
    task_id = "task4_db_connection_limit"
    name = "Database connection pool saturation"
    difficulty = "medium"
    max_steps = 20
    description = (
        "Postgres max_connections was reduced from 100 to 20 by a config change "
        "2 days ago. Connection pool is now saturated under normal load, cascading "
        "to order-service and api-gateway. Analytics-service memory is a red herring."
    )

    def build_episode(self, seed: int) -> EpisodeState:
        rng = random.Random(seed)

        config_entries = {
            "postgres": {
                "max_connections": {
                    "value": "20",
                    "last_changed": "2026-04-01T09:30:00.000Z",
                    "changed_by": "dba-team",
                },
            },
        }

        ground_truth = GroundTruth(
            root_cause_service="postgres",
            root_cause_mechanism="max_connections_reduced",
            failure_type="connection_exhaustion",
            correct_remediation="config_patch",
            affected_services=["postgres", "order-service", "api-gateway"],
            red_herring_services=["analytics-service"],
            failure_started_offset_minutes=2880,  # 2 days ago
            config_entries=config_entries,
        )

        service_states = propagate_failure(
            root_service="postgres",
            severity=0.80,
            failure_type="connection_exhaustion",
            rng=rng,
        )

        # Red herring: analytics-service memory elevated
        analytics = service_states["analytics-service"]
        analytics.mem_pct = min(100.0, analytics.mem_pct + 20)
        analytics.p99_ms = analytics.p99_ms * 1.25

        alert_summary = (
            "ALERT: order-service reporting database connection timeouts. "
            "postgres active_connections at capacity. "
            "API gateway 503 error rate rising. "
            "analytics-service memory utilisation elevated."
        )

        recent_deployments = [
            {
                "service": "order-service",
                "version": "v2.5.0",
                "deployed_at": "2026-04-02T14:00:00.000Z",
                "deployed_by": "deploy-bot",
                "change_summary": "Added new order status tracking endpoint",
            },
        ]

        return EpisodeState(
            task_id=self.task_id,
            step_count=0,
            max_steps=self.max_steps,
            done=False,
            service_states=service_states,
            ground_truth=ground_truth,
            action_log=[],
            cumulative_reward=0.0,
            seed=seed,
            rng=rng,
            alert_summary=alert_summary,
            recent_deployments=recent_deployments,
        )

    def grade(self, episode_state: EpisodeState) -> float:
        """Deterministic grader for Task 4."""
        action_log = episode_state.action_log
        gt = episode_state.ground_truth

        close_action = None
        for entry in action_log:
            if entry.get("action") == "close_incident":
                close_action = entry
                break

        if close_action is None:
            score = 0.0
            for entry in action_log:
                act = entry.get("action", "")
                params = entry.get("params", {})
                svc = params.get("service", "")
                if act in ("query_logs", "query_metrics") and svc == gt.root_cause_service:
                    score = max(score, 0.05)
                if act == "check_config" and svc == gt.root_cause_service:
                    score = max(score, 0.10)
                if act == "hypothesize" and params.get("root_cause_service") == gt.root_cause_service:
                    score = max(score, 0.10)
            return max(0.0, min(score, 0.25))

        params = close_action.get("params", {})
        score = 0.0

        # +0.25 correct root cause service
        if params.get("root_cause_service") == "postgres":
            score += 0.25

        # +0.20 mechanism mentions connection/max_connections/pool/limit
        mechanism = str(params.get("mechanism", "")).lower()
        if any(kw in mechanism for kw in ("connection", "max_connections", "pool", "limit", "config")):
            score += 0.20

        # +0.20 correct remediation
        if params.get("remediation_taken") == "config_patch":
            score += 0.20

        # +0.10 check_config called on postgres
        config_checked = any(
            e.get("action") == "check_config" and e.get("params", {}).get("service") == "postgres"
            for e in action_log
        )
        if config_checked:
            score += 0.10

        # +0.10 order-service in blast_radius
        blast = params.get("blast_radius", [])
        if "order-service" in blast:
            score += 0.10

        # +0.05 no red herrings in blast_radius
        if not any(rh in blast for rh in gt.red_herring_services):
            score += 0.05

        # +0.10 * efficiency
        efficiency = 1.0 - episode_state.step_count / episode_state.max_steps
        score += 0.10 * max(0.0, efficiency)

        return max(0.0, min(score, 1.0))
