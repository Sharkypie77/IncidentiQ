"""Task 2 — Cascading failure across services (medium)."""

from __future__ import annotations

import random

from env.state_machine import (
    EpisodeState,
    GroundTruth,
    propagate_failure,
)
from tasks.base import BaseTask


class Task2CascadingFailure(BaseTask):
    task_id = "task2_cascading_failure"
    name = "Cascading failure across services"
    difficulty = "medium"
    max_steps = 20
    description = (
        "Auth-service Redis pool exhaustion cascades to API gateway 503s. "
        "A red herring (analytics memory leak) is also present."
    )

    def build_episode(self, seed: int) -> EpisodeState:
        rng = random.Random(seed)

        ground_truth = GroundTruth(
            root_cause_service="auth-service",
            root_cause_mechanism="redis_pool_exhaustion",
            failure_type="connection_exhaustion",
            correct_remediation="restart",
            affected_services=["auth-service", "api-gateway"],
            red_herring_services=["analytics-service"],
            failure_started_offset_minutes=15,
        )

        service_states = propagate_failure(
            root_service="auth-service",
            severity=0.90,
            failure_type="connection_exhaustion",
            rng=rng,
        )

        # Red herring: analytics-service memory elevated
        analytics = service_states["analytics-service"]
        analytics.mem_pct = min(100.0, analytics.mem_pct + 18)
        analytics.p99_ms = analytics.p99_ms * 1.2

        # order-service shows slightly elevated errors from cascaded timeout
        # (already handled by propagation since api-gateway depends on order-service,
        #  but order-service doesn't depend on auth-service directly, so we add a mild effect)
        order = service_states["order-service"]
        order.error_rate = max(order.error_rate, 0.005)
        order.p99_ms = order.p99_ms * 1.1

        alert_summary = (
            "ALERT: API gateway 503 error rate exceeded 5% threshold. "
            "auth-service response times spiking. "
            "Multiple services showing elevated latency."
        )

        # Red herring deployment: order-service v2.4.1 deployed 45 minutes ago
        recent_deployments = [
            {
                "service": "order-service",
                "version": "v2.4.1",
                "deployed_at": "2026-04-03T06:45:00.000Z",
                "deployed_by": "deploy-bot",
                "change_summary": "Optimised order lookup queries and added request caching",
            }
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
        """Deterministic grader for Task 2."""
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
                svc = entry.get("params", {}).get("service", "")
                if act in ("query_logs", "query_metrics") and svc == gt.root_cause_service:
                    score = max(score, 0.05)
                if act in ("query_logs", "query_metrics") and svc in gt.affected_services:
                    score = max(score, 0.03)
                if act == "check_deployment" and svc == gt.root_cause_service:
                    score = max(score, 0.05)
                if act == "hypothesize" and entry.get("params", {}).get("root_cause_service") == gt.root_cause_service:
                    score = max(score, 0.10)
            return min(score, 0.25)

        params = close_action.get("params", {})
        score = 0.0

        correct_root = params.get("root_cause_service") == "auth-service"

        # +0.25 correct root cause service
        if correct_root:
            score += 0.25

        # +0.20 mechanism contains redis/pool/connection
        mechanism = str(params.get("mechanism", "")).lower()
        if any(kw in mechanism for kw in ("redis", "pool", "connection")):
            score += 0.20

        # +0.20 correct remediation (must target auth-service)
        remediation = str(params.get("remediation_taken", "")).lower()
        if remediation == "restart" and correct_root:
            score += 0.20

        # +0.15 order-service deployment NOT blamed (only if root cause is correct)
        if correct_root:
            if "order-service deployment" not in mechanism and "order-service deploy" not in mechanism:
                score += 0.15

        # +0.10 api-gateway in blast_radius
        blast = params.get("blast_radius", [])
        if "api-gateway" in blast:
            score += 0.10

        # +0.10 * efficiency
        efficiency = 1.0 - episode_state.step_count / episode_state.max_steps
        score += 0.10 * max(0.0, efficiency)

        return min(score, 1.0)
