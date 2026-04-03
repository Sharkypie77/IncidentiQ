"""Task 3 — Silent data corruption (hard)."""

from __future__ import annotations

import random

from env.state_machine import (
    EpisodeState,
    GroundTruth,
    propagate_failure,
)
from tasks.base import BaseTask


class Task3SilentCorruption(BaseTask):
    task_id = "task3_silent_corruption"
    name = "Silent data corruption"
    difficulty = "hard"
    max_steps = 30
    description = (
        "A payment race condition causes 3% double-charges with no outage. "
        "Root cause is a config change 6 days ago, not a recent deployment."
    )

    def build_episode(self, seed: int) -> EpisodeState:
        rng = random.Random(seed)

        # Config entry for order-service payment-handler (the actual root cause)
        config_entries = {
            "order-service": {
                "payment-handler": {
                    "value": "retry_on_timeout=true",
                    "last_changed": "2026-03-28T14:00:00.000Z",
                    "changed_by": "platform-team",
                },
            },
        }

        ground_truth = GroundTruth(
            root_cause_service="order-service",
            root_cause_mechanism="duplicate_webhook_idempotency",
            failure_type="race_condition",
            correct_remediation="config_patch",
            affected_services=["order-service"],
            red_herring_services=["analytics-service"],
            failure_started_offset_minutes=8640,  # 6 days ago
            config_entries=config_entries,
        )

        service_states = propagate_failure(
            root_service="order-service",
            severity=0.30,
            failure_type="race_condition",
            rng=rng,
        )

        # All services should show "healthy" status (subtle corruption)
        for svc in service_states.values():
            svc.status = "healthy"

        # Red herring: analytics-service memory elevated
        analytics = service_states["analytics-service"]
        analytics.mem_pct = min(100.0, analytics.mem_pct + 15)

        # Red herring: postgres slightly elevated query times (routine maintenance)
        pg = service_states["postgres"]
        pg.p99_ms = pg.p99_ms * 1.15
        pg.p50_ms = pg.p50_ms * 1.10

        alert_summary = (
            "Billing reconciliation discrepancy detected: 3.1% transaction mismatch. "
            "No service outage reported. All services report healthy status."
        )

        # Red herring deployments
        recent_deployments = [
            {
                "service": "message-broker",
                "version": "v3.2.0",
                "deployed_at": "2026-04-01T09:00:00.000Z",
                "deployed_by": "infra-bot",
                "change_summary": "Upgraded message broker to support new partition scheme",
            },
            {
                "service": "analytics-service",
                "version": "v1.8.3",
                "deployed_at": "2026-04-02T16:00:00.000Z",
                "deployed_by": "analytics-team",
                "change_summary": "Added new batch processing metrics dashboard",
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
        """Deterministic grader for Task 3."""
        action_log = episode_state.action_log
        gt = episode_state.ground_truth

        close_action = None
        for entry in action_log:
            if entry.get("action") == "close_incident":
                close_action = entry
                break

        if close_action is None:
            # Partial credit from investigation trail
            score = 0.0
            queried_root = False
            checked_config = False
            correct_hypothesis = False
            for entry in action_log:
                act = entry.get("action", "")
                params = entry.get("params", {})
                svc = params.get("service", "")
                if act in ("query_logs", "query_metrics") and svc == gt.root_cause_service:
                    queried_root = True
                if act == "check_config" and svc == gt.root_cause_service:
                    checked_config = True
                if act == "check_config":
                    score = max(score, 0.03)  # at least tried checking config
                if act == "hypothesize" and params.get("root_cause_service") == gt.root_cause_service:
                    correct_hypothesis = True
            if queried_root:
                score += 0.05
            if checked_config:
                score += 0.10
            if correct_hypothesis:
                score += 0.10
            return min(score, 0.25)

        params = close_action.get("params", {})
        score = 0.0

        # +0.20 correct root cause service
        if params.get("root_cause_service") == "order-service":
            score += 0.20

        # +0.20 mechanism contains idempotency/duplicate/race/webhook/retry/timeout/config/payment
        mechanism = str(params.get("mechanism", "")).lower()
        if any(kw in mechanism for kw in (
            "idempotency", "duplicate", "race", "webhook",
            "retry", "timeout", "config", "payment",
        )):
            score += 0.20

        # +0.20 correct remediation
        if params.get("remediation_taken") == "config_patch":
            score += 0.20

        # +0.15 check_config was called on order-service
        config_checked = False
        for entry in action_log:
            if entry.get("action") == "check_config":
                if entry.get("params", {}).get("service") == "order-service":
                    config_checked = True
                    break
        if config_checked:
            score += 0.15

        # +0.15 no red-herring services in blast_radius
        blast = params.get("blast_radius", [])
        if not any(rh in blast for rh in gt.red_herring_services):
            score += 0.15

        # +0.10 * efficiency
        efficiency = 1.0 - episode_state.step_count / episode_state.max_steps
        score += 0.10 * max(0.0, efficiency)

        return min(score, 1.0)
