"""Core IncidentIQ environment: reset, step, state."""

from __future__ import annotations

import json
import random
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from env.log_generator import generate_logs
from env.metric_generator import generate_metrics
from env.models import (
    Action,
    Deployment,
    LogLine,
    MetricPoint,
    Observation,
    ResetResult,
    Reward,
    ServiceHealth,
    StepResult,
    VALID_ACTIONS,
)
from env.reward import RewardCalculator
from env.state_machine import (
    DEPENDENCY_GRAPH,
    EpisodeState,
    GroundTruth,
    ServiceState,
)
from tasks.base import BaseTask, STRICT_SCORE_EPS
from tasks.task1_cpu_saturation import Task1CpuSaturation
from tasks.task2_cascading_failure import Task2CascadingFailure
from tasks.task3_silent_corruption import Task3SilentCorruption
from tasks.task4_db_connection_limit import Task4DbConnectionLimit
from tasks.task5_memory_leak_analytics import Task5MemoryLeakAnalytics

MAX_SESSIONS = 100  # Evict oldest sessions beyond this limit


class IncidentIQEnv:
    """In-memory reinforcement-learning environment for incident response."""

    def __init__(self) -> None:
        self.state: OrderedDict[str, EpisodeState] = OrderedDict()
        self.last_session_id: Optional[str] = None
        self._lock = threading.Lock()
        self.tasks: Dict[str, BaseTask] = {
            "task1_cpu_saturation": Task1CpuSaturation(),
            "task2_cascading_failure": Task2CascadingFailure(),
            "task3_silent_corruption": Task3SilentCorruption(),
            "task4_db_connection_limit": Task4DbConnectionLimit(),
            "task5_memory_leak_analytics": Task5MemoryLeakAnalytics(),
        }
        self.reward_calculator = RewardCalculator()

    def _evict_old_sessions(self) -> None:
        """Remove oldest sessions if we exceed MAX_SESSIONS."""
        while len(self.state) > MAX_SESSIONS:
            self.state.popitem(last=False)  # Remove oldest (FIFO)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        ep: EpisodeState,
        last_action_result: Optional[str] = None,
    ) -> Observation:
        """Build an Observation from the current episode state."""
        service_health: Dict[str, ServiceHealth] = {}
        recent_logs: List[LogLine] = []
        all_metrics: Dict[str, List[MetricPoint]] = {}

        for name, svc in ep.service_states.items():
            service_health[name] = ServiceHealth(
                name=name,
                status=svc.status,
                cpu_pct=round(svc.cpu_pct, 2),
                mem_pct=round(svc.mem_pct, 2),
                p50_ms=round(svc.p50_ms, 2),
                p99_ms=round(svc.p99_ms, 2),
                error_rate=round(svc.error_rate, 6),
                active_connections=svc.active_connections,
            )

            # Generate a handful of recent logs per service for the overview
            logs = generate_logs(name, svc, ep.rng, ep.ground_truth, n=5)
            recent_logs.extend(logs)

            # Generate key metrics for each service
            svc_metrics = generate_metrics(name, svc, ep.rng, ep.ground_truth)
            for metric_name, points in svc_metrics.items():
                key = f"{name}/{metric_name}"
                all_metrics[key] = points

        # Sort logs by timestamp descending
        recent_logs.sort(key=lambda l: l.timestamp, reverse=True)
        recent_logs = recent_logs[:25]

        # Convert deployment dicts to Deployment models
        deployments = [
            Deployment(**d) if isinstance(d, dict) else d
            for d in ep.recent_deployments
        ]

        return Observation(
            alert_summary=ep.alert_summary,
            service_health=service_health,
            recent_logs=recent_logs,
            metrics=all_metrics,
            dependency_graph=DEPENDENCY_GRAPH,
            recent_deployments=deployments,
            step_number=ep.step_count,
            steps_remaining=ep.max_steps - ep.step_count,
            last_action_result=last_action_result,
        )

    def _execute_action(
        self, action: Action, ep: EpisodeState
    ) -> str:
        """Execute an action and return the result string."""
        params = action.params
        act = action.action

        if act == "query_logs":
            service = params.get("service", "")
            pattern = params.get("pattern", "")
            svc_state = ep.service_states.get(service)
            if svc_state is None:
                return json.dumps({"error": f"Unknown service: {service}"})
            logs = generate_logs(service, svc_state, ep.rng, ep.ground_truth, n=25)
            if pattern:
                filtered = [
                    l for l in logs if pattern.lower() in l.message.lower()
                ]
                if filtered:
                    logs = filtered
            logs = logs[:10]
            return json.dumps([l.model_dump() for l in logs], indent=2)

        elif act == "query_metrics":
            service = params.get("service", "")
            metric = params.get("metric", "")
            window_minutes = int(params.get("window_minutes", 30))
            svc_state = ep.service_states.get(service)
            if svc_state is None:
                return json.dumps({"error": f"Unknown service: {service}"})
            all_metrics = generate_metrics(service, svc_state, ep.rng, ep.ground_truth)
            points = all_metrics.get(metric, [])
            if window_minutes < 30 and points:
                # Keep only last N minutes
                keep = min(window_minutes, 30)
                points = points[-keep:]
            return json.dumps([p.model_dump() for p in points], indent=2)

        elif act == "query_traces":
            service = params.get("service", "")
            svc_state = ep.service_states.get(service)
            if svc_state is None:
                return json.dumps({"error": f"Unknown service: {service}"})
            traces = self._generate_traces(service, svc_state, ep)
            return json.dumps(traces, indent=2)

        elif act == "check_deployment":
            service = params.get("service", "")
            deployments = [
                d for d in ep.recent_deployments
                if (d.get("service") if isinstance(d, dict) else d.service) == service
            ]
            if not deployments:
                # Generate a baseline deployment record
                deployments = [
                    {
                        "service": service,
                        "version": "v2.3.0",
                        "deployed_at": "2026-03-30T10:00:00.000Z",
                        "deployed_by": "deploy-bot",
                        "change_summary": "Routine dependency updates",
                    }
                ]
            return json.dumps(deployments, indent=2)

        elif act == "check_config":
            service = params.get("service", "")
            key = params.get("key", "")
            gt = ep.ground_truth
            # Check if ground truth has a config entry for this service+key
            if (
                service in gt.config_entries
                and key in gt.config_entries[service]
            ):
                entry = gt.config_entries[service][key]
                result = {
                    "service": service,
                    "key": key,
                    "value": entry["value"],
                    "last_changed": entry["last_changed"],
                    "changed_by": entry["changed_by"],
                }
            else:
                # Plausible-looking default config
                result = {
                    "service": service,
                    "key": key,
                    "value": "default",
                    "last_changed": "2026-03-01T12:00:00.000Z",
                    "changed_by": "platform-team",
                }
            return json.dumps(result, indent=2)

        elif act == "hypothesize":
            return "Hypothesis recorded."

        elif act == "remediate":
            rtype = params.get("type", "unknown")
            target = params.get("target", "unknown")
            return f"Remediation action '{rtype}' applied to {target}."

        elif act == "close_incident":
            # Terminal reward is calculated by the caller
            return ""  # Will be overwritten by caller

        return json.dumps({"error": f"Unknown action: {act}"})

    def _generate_traces(
        self, service: str, svc_state: ServiceState, ep: EpisodeState
    ) -> dict:
        """Generate 3 synthetic traces for the given service."""
        rng = ep.rng
        traces = []
        deps = DEPENDENCY_GRAPH.get(service, [])

        for i in range(3):
            trace_id = uuid.uuid4().hex[:16]
            spans = [
                {
                    "service": service,
                    "duration_ms": int(svc_state.p50_ms * rng.uniform(0.8, 2.5)),
                    "status_code": 200 if rng.random() > svc_state.error_rate else 500,
                }
            ]
            for dep in deps:
                dep_state = ep.service_states.get(dep)
                if dep_state:
                    dur = int(dep_state.p50_ms * rng.uniform(0.8, 2.5))
                    sc = 200 if rng.random() > dep_state.error_rate else 500
                    # Make degraded services show high duration
                    if dep_state.status in ("degraded", "down"):
                        dur = int(dep_state.p99_ms * rng.uniform(0.9, 1.5))
                        sc = 500 if rng.random() < 0.5 else 503
                    spans.append({
                        "service": dep,
                        "duration_ms": dur,
                        "status_code": sc,
                    })
            traces.append({"trace_id": trace_id, "spans": spans})

        return {"service": service, "traces": traces}

    # ── public API ──────────────────────────────────────────────────────────

    def reset(
        self, task_id: str, seed: Optional[int] = None
    ) -> tuple[str, ResetResult]:
        """Reset the environment for a new episode. Returns (session_id, result)."""
        if task_id not in self.tasks:
            raise KeyError(f"Unknown task_id: {task_id}")

        if seed is None:
            seed = random.randint(1, 999999)

        task = self.tasks[task_id]
        episode = task.build_episode(seed)

        session_id = uuid.uuid4().hex

        with self._lock:
            self.state[session_id] = episode
            self.last_session_id = session_id
            self._evict_old_sessions()

        observation = self._build_observation(episode)
        result = ResetResult(
            observation=observation,
            task_id=task_id,
            task_description=task.description,
        )
        return session_id, result

    def step(self, session_id: str, action: Action) -> StepResult:
        """Execute one step in the environment."""
        with self._lock:
            if session_id not in self.state:
                raise KeyError(f"Unknown session_id: {session_id}")
            ep = self.state[session_id]

        if ep.done:
            observation = self._build_observation(ep)
            if ep.task_id in self.tasks:
                done_score = self.tasks[ep.task_id].grade(ep)
            else:
                done_score = self._clamp_score(ep.cumulative_reward)
            return StepResult(
                observation=observation,
                reward=Reward(value=0.0, reason="episode already done", cumulative=round(done_score, 6)),
                done=True,
                info={"task_id": ep.task_id, "step_count": ep.step_count},
            )

        # Validate action type
        if action.action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action.action}'. Must be one of: {VALID_ACTIONS}"
            )

        # Execute the action
        action_result = self._execute_action(action, ep)

        # Calculate step reward
        step_reward, step_reason = self.reward_calculator.calculate_step_reward(
            action, action.params, ep
        )

        terminal_reward = 0.0
        terminal_reason = ""

        if action.action == "close_incident":
            terminal_reward, terminal_reason = self.reward_calculator.calculate_terminal_reward(
                action.params, ep
            )
            ep.done = True
            action_result = f"Incident closed. Final score: {terminal_reward:.2f}"

        total_reward = step_reward + terminal_reward
        ep.cumulative_reward += total_reward
        ep.step_count += 1

        # Append to action log
        ep.action_log.append({
            "step": ep.step_count,
            "action": action.action,
            "params": action.params,
            "reward": total_reward,
        })

        # Check if episode timed out
        if ep.step_count >= ep.max_steps:
            ep.done = True

        observation = self._build_observation(ep, last_action_result=action_result)
        reason = step_reason
        if terminal_reason:
            reason = f"{step_reason}; TERMINAL: {terminal_reason}"

        # When the episode ends, report the proper grade score (clamped to
        # strict (0, 1)) instead of the raw cumulative reward, so the
        # platform always sees a valid task score.
        if ep.done and ep.task_id in self.tasks:
            reported_cumulative = self.tasks[ep.task_id].grade(ep)
        else:
            reported_cumulative = self._clamp_score(ep.cumulative_reward)

        return StepResult(
            observation=observation,
            reward=Reward(
                value=round(total_reward, 4),
                reason=reason,
                cumulative=round(reported_cumulative, 6),
            ),
            done=ep.done,
            info={"task_id": ep.task_id, "step_count": ep.step_count},
        )

    @staticmethod
    def _clamp_score(value: float) -> float:
        """Clamp a value to the strict open interval (0, 1)."""
        return max(STRICT_SCORE_EPS, min(float(value), 1.0 - STRICT_SCORE_EPS))

    def get_state(self, session_id: str) -> dict:
        """Return the full episode state as a JSON-serializable dict."""
        with self._lock:
            if session_id not in self.state:
                raise KeyError(f"Unknown session_id: {session_id}")
            ep = self.state[session_id]

        # When done, compute the proper grade score via the task grader;
        # otherwise report the raw cumulative reward clamped to (0, 1).
        if ep.done and ep.task_id in self.tasks:
            score = self.tasks[ep.task_id].grade(ep)
        else:
            score = self._clamp_score(ep.cumulative_reward)

        result: Dict[str, Any] = {
            "task_id": ep.task_id,
            "step_count": ep.step_count,
            "max_steps": ep.max_steps,
            "done": ep.done,
            "service_states": {
                name: {
                    "name": s.name,
                    "status": s.status,
                    "cpu_pct": round(s.cpu_pct, 2),
                    "mem_pct": round(s.mem_pct, 2),
                    "p50_ms": round(s.p50_ms, 2),
                    "p99_ms": round(s.p99_ms, 2),
                    "error_rate": round(s.error_rate, 6),
                    "active_connections": s.active_connections,
                }
                for name, s in ep.service_states.items()
            },
            "cumulative_reward": round(score, 6),
            "action_log_length": len(ep.action_log),
        }

        # Only reveal ground truth after episode is done
        if ep.done:
            result["ground_truth"] = {
                "root_cause_service": ep.ground_truth.root_cause_service,
                "root_cause_mechanism": ep.ground_truth.root_cause_mechanism,
                "failure_type": ep.ground_truth.failure_type,
                "correct_remediation": ep.ground_truth.correct_remediation,
                "affected_services": ep.ground_truth.affected_services,
                "red_herring_services": ep.ground_truth.red_herring_services,
            }
            result["action_log"] = ep.action_log

        return result

    def get_tasks(self) -> List[dict]:
        """Return metadata about all available tasks."""
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "description": t.description,
            }
            for t in self.tasks.values()
        ]
