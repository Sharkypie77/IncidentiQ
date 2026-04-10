"""Task 1 — Single-service CPU saturation (easy)."""

from __future__ import annotations

import random
from typing import List

from env.log_generator import generate_logs
from env.metric_generator import generate_metrics
from env.state_machine import (
    EpisodeState,
    GroundTruth,
    ServiceState,
    propagate_failure,
)
from tasks.base import BaseTask


class Task1CpuSaturation(BaseTask):
    task_id = "task1_cpu_saturation"
    name = "Single service CPU saturation"
    difficulty = "easy"
    max_steps = 15
    description = (
        "The order-service is experiencing CPU saturation due to a missing "
        "database index. Identify the root cause and correct remediation."
    )

    def build_episode(self, seed: int) -> EpisodeState:
        rng = random.Random(seed)

        ground_truth = GroundTruth(
            root_cause_service="order-service",
            root_cause_mechanism="missing_index",
            failure_type="cpu_saturation",
            correct_remediation="rollback",
            affected_services=["order-service", "api-gateway"],
            red_herring_services=["analytics-service"],
            failure_started_offset_minutes=10,
        )

        service_states = propagate_failure(
            root_service="order-service",
            severity=0.85,
            failure_type="cpu_saturation",
            rng=rng,
        )

        # Make analytics-service look slightly concerning (red herring memory leak, started 4h ago)
        analytics = service_states["analytics-service"]
        analytics.mem_pct = min(100.0, analytics.mem_pct + 20)
        analytics.p99_ms = analytics.p99_ms * 1.3

        alert_summary = (
            "ALERT: order-service p99 latency exceeded 4000ms threshold. "
            "API gateway reporting elevated 503 error rate. "
            "analytics-service memory utilisation elevated."
        )

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
            recent_deployments=[],
        )

    def grade(self, episode_state: EpisodeState) -> float:
        """Deterministic grader for Task 1."""
        action_log = episode_state.action_log
        gt = episode_state.ground_truth

        # Find the close_incident action (if any)
        close_action = None
        for entry in action_log:
            if entry.get("action") == "close_incident":
                close_action = entry
                break

        if close_action is None:
            # Partial credit from investigation actions
            score = 0.0
            for entry in action_log:
                act = entry.get("action", "")
                svc = entry.get("params", {}).get("service", "")
                if act in ("query_logs", "query_metrics") and svc == gt.root_cause_service:
                    score = max(score, 0.05)
                if act in ("query_logs", "query_metrics") and svc in gt.affected_services:
                    score = max(score, 0.03)
                if act == "check_deployment" and svc == gt.root_cause_service:
                    score = max(score, 0.05)
                if act == "hypothesize" and entry.get("params", {}).get("root_cause_service") == gt.root_cause_service:
                    score = max(score, 0.10)
            return max(0.0, min(score, 0.25))

        # Full grading from close_incident params
        params = close_action.get("params", {})
        score = 0.0

        if params.get("root_cause_service") == "order-service":
            score += 0.35

        mechanism = str(params.get("mechanism", "")).lower()
        if any(kw in mechanism for kw in ("index", "table scan", "full scan", "missing_index")):
            score += 0.10

        if params.get("remediation_taken") == "rollback":
            score += 0.25

        blast = params.get("blast_radius", [])
        if "order-service" in blast and "api-gateway" in blast:
            score += 0.15

        if "analytics-service" not in blast:
            score += 0.15

        # Efficiency bonus
        efficiency = 1.0 - episode_state.step_count / episode_state.max_steps
        score += 0.10 * max(0.0, efficiency)

        return max(0.0, min(score, 1.0))
