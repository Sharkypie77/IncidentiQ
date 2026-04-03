"""Task 5 — Memory leak in analytics pipeline (medium-hard).

The twist: analytics-service is usually the red herring in every other task.
Here, it's the ACTUAL root cause. This tests whether an agent has learned to
blindly ignore analytics-service or whether it can adapt its reasoning.
"""

from __future__ import annotations

import random

from env.state_machine import (
    EpisodeState,
    GroundTruth,
    propagate_failure,
)
from tasks.base import BaseTask


class Task5MemoryLeakAnalytics(BaseTask):
    task_id = "task5_memory_leak_analytics"
    name = "Analytics pipeline memory leak"
    difficulty = "medium-hard"
    max_steps = 25
    description = (
        "Analytics-service has a genuine memory leak caused by unbounded batch "
        "caching. Unlike other tasks where analytics-service is a red herring, "
        "here it IS the root cause. The agent must overcome its prior bias "
        "against investigating analytics-service. No other services are "
        "genuinely failing — postgres slow queries are routine maintenance."
    )

    def build_episode(self, seed: int) -> EpisodeState:
        rng = random.Random(seed)

        config_entries = {
            "analytics-service": {
                "batch_cache_ttl": {
                    "value": "0",  # disabled = unbounded cache growth
                    "last_changed": "2026-03-30T11:00:00.000Z",
                    "changed_by": "analytics-team",
                },
                "max_batch_size": {
                    "value": "10000",
                    "last_changed": "2026-03-30T11:00:00.000Z",
                    "changed_by": "analytics-team",
                },
            },
        }

        ground_truth = GroundTruth(
            root_cause_service="analytics-service",
            root_cause_mechanism="unbounded_batch_cache",
            failure_type="memory_leak",
            correct_remediation="config_patch",
            affected_services=["analytics-service"],
            red_herring_services=["postgres"],
            failure_started_offset_minutes=7200,  # 5 days ago
            config_entries=config_entries,
        )

        service_states = propagate_failure(
            root_service="analytics-service",
            severity=0.85,
            failure_type="memory_leak",
            rng=rng,
        )

        # Red herring: postgres having slow queries (routine vacuum/maintenance)
        pg = service_states["postgres"]
        pg.p99_ms = pg.p99_ms * 1.5
        pg.cpu_pct = min(100.0, pg.cpu_pct + 10)

        alert_summary = (
            "ALERT: analytics-service OOM killed twice in 24h. "
            "Container memory at 98%. Pipeline processing stalled. "
            "postgres p99 latency elevated (likely routine maintenance). "
            "All other services operating normally."
        )

        recent_deployments = [
            {
                "service": "postgres",
                "version": "v15.3-patch1",
                "deployed_at": "2026-04-02T03:00:00.000Z",
                "deployed_by": "dba-bot",
                "change_summary": "Applied security patch and ran VACUUM ANALYZE",
            },
            {
                "service": "analytics-service",
                "version": "v3.2.0",
                "deployed_at": "2026-03-28T10:00:00.000Z",
                "deployed_by": "analytics-team",
                "change_summary": "Upgraded batch processing engine",
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
        """Deterministic grader for Task 5."""
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
                    score += 0.10
                if act == "hypothesize" and params.get("root_cause_service") == gt.root_cause_service:
                    score += 0.10
            return min(score, 0.25)

        params = close_action.get("params", {})
        score = 0.0

        # +0.25 correct root cause service
        if params.get("root_cause_service") == "analytics-service":
            score += 0.25

        # +0.20 mechanism mentions memory/cache/batch/oom/unbounded/leak
        mechanism = str(params.get("mechanism", "")).lower()
        if any(kw in mechanism for kw in ("memory", "cache", "batch", "oom", "unbounded", "leak")):
            score += 0.20

        # +0.20 correct remediation
        if params.get("remediation_taken") == "config_patch":
            score += 0.20

        # +0.10 check_config called on analytics-service
        config_checked = any(
            e.get("action") == "check_config" and e.get("params", {}).get("service") == "analytics-service"
            for e in action_log
        )
        if config_checked:
            score += 0.10

        # +0.10 analytics-service metrics were queried
        metrics_queried = any(
            e.get("action") == "query_metrics" and e.get("params", {}).get("service") == "analytics-service"
            for e in action_log
        )
        if metrics_queried:
            score += 0.10

        # +0.05 no red herrings in blast_radius
        blast = params.get("blast_radius", [])
        if not any(rh in blast for rh in gt.red_herring_services):
            score += 0.05

        # +0.10 * efficiency
        efficiency = 1.0 - episode_state.step_count / episode_state.max_steps
        score += 0.10 * max(0.0, efficiency)

        return min(score, 1.0)
