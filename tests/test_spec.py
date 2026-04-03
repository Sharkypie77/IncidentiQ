"""Tests for the IncidentIQ OpenEnv spec compliance."""

from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import IncidentIQEnv
from env.models import Action

SEED = 42


def _make_env() -> IncidentIQEnv:
    return IncidentIQEnv()


# ── 1. reset() returns valid ResetResult ────────────────────────────────────

def test_reset_returns_valid_result():
    env = _make_env()
    session_id, result = env.reset("task1_cpu_saturation", seed=SEED)

    assert isinstance(session_id, str) and len(session_id) > 0
    assert result.task_id == "task1_cpu_saturation"
    assert len(result.task_description) > 0

    obs = result.observation
    assert obs.alert_summary and len(obs.alert_summary) > 0
    assert len(obs.service_health) == 5
    assert obs.step_number == 0
    assert obs.steps_remaining > 0


# ── 2. step() returns reward value in [-0.15, 0.10] ────────────────────────

def test_step_reward_range():
    env = _make_env()
    session_id, _ = env.reset("task1_cpu_saturation", seed=SEED)

    action = Action(
        action="query_logs",
        params={"service": "order-service", "pattern": "error"},
    )
    result = env.step(session_id, action)

    assert -0.15 <= result.reward.value <= 0.10, (
        f"reward {result.reward.value} outside [-0.15, 0.10]"
    )
    assert isinstance(result.done, bool)
    assert result.observation is not None


# ── 3. state() returns dict with all required keys ─────────────────────────

def test_state_returns_required_keys():
    env = _make_env()
    session_id, _ = env.reset("task2_cascading_failure", seed=SEED)

    state = env.get_state(session_id)

    required_keys = [
        "task_id", "step_count", "max_steps", "done",
        "service_states", "cumulative_reward", "action_log_length",
    ]
    for key in required_keys:
        assert key in state, f"Missing key: {key}"

    assert state["task_id"] == "task2_cascading_failure"
    assert state["done"] is False


# ── 4. Invalid action type returns structured error ─────────────────────────

def test_invalid_action_raises_value_error():
    env = _make_env()
    session_id, _ = env.reset("task1_cpu_saturation", seed=SEED)

    action = Action(action="invalid_action_type", params={})
    try:
        env.step(session_id, action)
        assert False, "Expected ValueError for invalid action"
    except ValueError as e:
        assert "invalid_action_type" in str(e).lower() or "Invalid action" in str(e)


# ── 5. step() after done returns done=True without changing score ───────────

def test_step_after_done_is_noop():
    env = _make_env()
    session_id, _ = env.reset("task1_cpu_saturation", seed=SEED)

    # Close the incident immediately
    close_action = Action(
        action="close_incident",
        params={
            "root_cause_service": "order-service",
            "mechanism": "test",
            "remediation_taken": "rollback",
            "blast_radius": ["order-service"],
            "summary": "test close",
        },
    )
    result1 = env.step(session_id, close_action)
    assert result1.done is True
    score_after_close = result1.reward.cumulative

    # Try stepping again
    action2 = Action(
        action="query_logs",
        params={"service": "order-service", "pattern": "error"},
    )
    result2 = env.step(session_id, action2)
    assert result2.done is True
    assert result2.reward.value == 0.0
    assert result2.reward.cumulative == score_after_close


# ── 6. All 5 tasks are registered and produce valid observations ────────────

ALL_TASK_IDS = [
    "task1_cpu_saturation",
    "task2_cascading_failure",
    "task3_silent_corruption",
    "task4_db_connection_limit",
    "task5_memory_leak_analytics",
]


def test_all_tasks_reset_valid():
    env = _make_env()

    # get_tasks should return all 5
    task_list = env.get_tasks()
    returned_ids = {t["task_id"] for t in task_list}
    for tid in ALL_TASK_IDS:
        assert tid in returned_ids, f"Task {tid} not in get_tasks() response"

    # Each task should reset without errors and return a valid observation
    for tid in ALL_TASK_IDS:
        session_id, result = env.reset(tid, seed=SEED)
        assert isinstance(session_id, str) and len(session_id) > 0, f"{tid}: bad session_id"
        assert result.task_id == tid
        assert len(result.task_description) > 0, f"{tid}: empty description"

        obs = result.observation
        assert obs.alert_summary and len(obs.alert_summary) > 0, f"{tid}: empty alert"
        assert len(obs.service_health) == 5, f"{tid}: expected 5 services, got {len(obs.service_health)}"
        assert obs.step_number == 0, f"{tid}: step_number != 0"
        assert obs.steps_remaining > 0, f"{tid}: steps_remaining <= 0"
