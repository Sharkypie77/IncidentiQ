"""Tests for the IncidentIQ task graders — all 5 tasks."""

from __future__ import annotations

from tasks.task1_cpu_saturation import Task1CpuSaturation
from tasks.task2_cascading_failure import Task2CascadingFailure
from tasks.task3_silent_corruption import Task3SilentCorruption
from tasks.task4_db_connection_limit import Task4DbConnectionLimit
from tasks.task5_memory_leak_analytics import Task5MemoryLeakAnalytics

SEED = 42

ALL_TASKS = (
    Task1CpuSaturation,
    Task2CascadingFailure,
    Task3SilentCorruption,
    Task4DbConnectionLimit,
    Task5MemoryLeakAnalytics,
)


# ── 1. All graders return float in [0.0, 1.0] on a fresh episode ───────────

def test_graders_return_valid_float():
    for TaskCls in ALL_TASKS:
        task = TaskCls()
        episode = task.build_episode(SEED)
        score = task.grade(episode)
        assert isinstance(score, float), f"{task.task_id}: score is not float"
        assert 0.0 <= score <= 1.0, f"{task.task_id}: score {score} out of [0,1]"


# ── 2. Graders are deterministic (same seed -> same score) ──────────────────

def test_graders_deterministic():
    for TaskCls in ALL_TASKS:
        task = TaskCls()
        ep1 = task.build_episode(SEED)
        score1 = task.grade(ep1)
        ep2 = task.build_episode(SEED)
        score2 = task.grade(ep2)
        assert score1 == score2, (
            f"{task.task_id}: non-deterministic ({score1} != {score2})"
        )


# ── 3. Perfect agent scores >= 0.85 ────────────────────────────────────────

def test_perfect_agent_scores_high():
    # Task 1: CPU Saturation (Easy)
    t1 = Task1CpuSaturation()
    ep1 = t1.build_episode(SEED)
    ep1.action_log = [
        {"step": 1, "action": "query_logs", "params": {"service": "order-service", "pattern": "error"}, "reward": 0.08},
        {"step": 2, "action": "check_deployment", "params": {"service": "order-service"}, "reward": 0.03},
        {"step": 3, "action": "close_incident", "params": {
            "root_cause_service": "order-service",
            "mechanism": "missing_index causing full table scans",
            "remediation_taken": "rollback",
            "blast_radius": ["order-service", "api-gateway"],
            "summary": "Rolled back order-service deployment to fix missing index.",
        }, "reward": 0.5},
    ]
    ep1.step_count = 3
    score1 = t1.grade(ep1)
    assert score1 >= 0.85, f"Task1 perfect agent scored {score1}, expected >= 0.85"

    # Task 2: Cascading Failure (Medium)
    t2 = Task2CascadingFailure()
    ep2 = t2.build_episode(SEED)
    ep2.action_log = [
        {"step": 1, "action": "query_logs", "params": {"service": "auth-service", "pattern": "redis"}, "reward": 0.08},
        {"step": 2, "action": "close_incident", "params": {
            "root_cause_service": "auth-service",
            "mechanism": "redis pool exhaustion",
            "remediation_taken": "restart",
            "blast_radius": ["auth-service", "api-gateway"],
            "summary": "Restarted auth-service to clear exhausted redis connection pool.",
        }, "reward": 0.5},
    ]
    ep2.step_count = 2
    score2 = t2.grade(ep2)
    assert score2 >= 0.85, f"Task2 perfect agent scored {score2}, expected >= 0.85"

    # Task 3: Silent Corruption (Hard)
    t3 = Task3SilentCorruption()
    ep3 = t3.build_episode(SEED)
    ep3.action_log = [
        {"step": 1, "action": "check_config", "params": {"service": "order-service", "key": "payment-handler"}, "reward": 0.03},
        {"step": 2, "action": "close_incident", "params": {
            "root_cause_service": "order-service",
            "mechanism": "duplicate webhook idempotency failure causing race condition",
            "remediation_taken": "config_patch",
            "blast_radius": ["order-service"],
            "summary": "Patched payment-handler config to enforce idempotency keys.",
        }, "reward": 0.5},
    ]
    ep3.step_count = 2
    score3 = t3.grade(ep3)
    assert score3 >= 0.85, f"Task3 perfect agent scored {score3}, expected >= 0.85"

    # Task 4: DB Connection Limit (Medium)
    t4 = Task4DbConnectionLimit()
    ep4 = t4.build_episode(SEED)
    ep4.action_log = [
        {"step": 1, "action": "query_logs", "params": {"service": "postgres", "pattern": "connection"}, "reward": 0.08},
        {"step": 2, "action": "check_config", "params": {"service": "postgres", "key": "max_connections"}, "reward": 0.03},
        {"step": 3, "action": "close_incident", "params": {
            "root_cause_service": "postgres",
            "mechanism": "max_connections reduced from 100 to 20 causing connection pool saturation",
            "remediation_taken": "config_patch",
            "blast_radius": ["postgres", "order-service", "api-gateway"],
            "summary": "Patched postgres max_connections back to 100.",
        }, "reward": 0.5},
    ]
    ep4.step_count = 3
    score4 = t4.grade(ep4)
    assert score4 >= 0.85, f"Task4 perfect agent scored {score4}, expected >= 0.85"

    # Task 5: Memory Leak Analytics (Medium-Hard)
    t5 = Task5MemoryLeakAnalytics()
    ep5 = t5.build_episode(SEED)
    ep5.action_log = [
        {"step": 1, "action": "query_metrics", "params": {"service": "analytics-service", "metric": "cpu_pct", "window_minutes": 30}, "reward": 0.08},
        {"step": 2, "action": "check_config", "params": {"service": "analytics-service", "key": "batch_cache_ttl"}, "reward": 0.03},
        {"step": 3, "action": "close_incident", "params": {
            "root_cause_service": "analytics-service",
            "mechanism": "unbounded batch cache causing memory leak and OOM kills",
            "remediation_taken": "config_patch",
            "blast_radius": ["analytics-service"],
            "summary": "Set batch_cache_ttl to 3600 to bound cache growth.",
        }, "reward": 0.5},
    ]
    ep5.step_count = 3
    score5 = t5.grade(ep5)
    assert score5 >= 0.85, f"Task5 perfect agent scored {score5}, expected >= 0.85"


# ── 4. Empty action log scores <= 0.20 ─────────────────────────────────────

def test_empty_action_log_low_score():
    for TaskCls in ALL_TASKS:
        task = TaskCls()
        episode = task.build_episode(SEED)
        episode.action_log = []
        score = task.grade(episode)
        assert score <= 0.20, (
            f"{task.task_id}: empty log scored {score}, expected <= 0.20"
        )


# ── 5. Agent that only targets red herring scores < 0.10 ───────────────────

def test_red_herring_only_agent_low_score():
    # Tasks 1-3: red herring is analytics-service
    for TaskCls in (Task1CpuSaturation, Task2CascadingFailure, Task3SilentCorruption):
        task = TaskCls()
        episode = task.build_episode(SEED)
        episode.action_log = [
            {"step": 1, "action": "query_logs", "params": {"service": "analytics-service", "pattern": "memory"}, "reward": -0.08},
            {"step": 2, "action": "query_metrics", "params": {"service": "analytics-service", "metric": "cpu_pct", "window_minutes": 30}, "reward": -0.08},
            {"step": 3, "action": "close_incident", "params": {
                "root_cause_service": "analytics-service",
                "mechanism": "memory leak",
                "remediation_taken": "restart",
                "blast_radius": ["analytics-service"],
                "summary": "Restarted analytics-service due to memory leak.",
            }, "reward": 0.0},
        ]
        episode.step_count = 3
        score = task.grade(episode)
        assert score < 0.10, (
            f"{task.task_id}: red-herring-only agent scored {score}, expected < 0.10"
        )

    # Task 4: red herring is analytics-service
    t4 = Task4DbConnectionLimit()
    ep4 = t4.build_episode(SEED)
    ep4.action_log = [
        {"step": 1, "action": "query_logs", "params": {"service": "analytics-service", "pattern": "memory"}, "reward": -0.08},
        {"step": 2, "action": "close_incident", "params": {
            "root_cause_service": "analytics-service",
            "mechanism": "memory leak",
            "remediation_taken": "restart",
            "blast_radius": ["analytics-service"],
            "summary": "Restarted analytics-service.",
        }, "reward": 0.0},
    ]
    ep4.step_count = 2
    score4 = t4.grade(ep4)
    assert score4 < 0.10, (
        f"task4_db_connection_limit: red-herring-only agent scored {score4}, expected < 0.10"
    )

    # Task 5: red herring is postgres (analytics-service is the REAL root cause here)
    t5 = Task5MemoryLeakAnalytics()
    ep5 = t5.build_episode(SEED)
    ep5.action_log = [
        {"step": 1, "action": "query_logs", "params": {"service": "postgres", "pattern": "slow"}, "reward": -0.08},
        {"step": 2, "action": "query_metrics", "params": {"service": "postgres", "metric": "p99_latency_ms", "window_minutes": 30}, "reward": -0.08},
        {"step": 3, "action": "close_incident", "params": {
            "root_cause_service": "postgres",
            "mechanism": "slow queries from routine maintenance",
            "remediation_taken": "restart",
            "blast_radius": ["postgres"],
            "summary": "Restarted postgres to clear slow queries.",
        }, "reward": 0.0},
    ]
    ep5.step_count = 3
    score5 = t5.grade(ep5)
    assert score5 < 0.10, (
        f"task5_memory_leak_analytics: red-herring-only agent scored {score5}, expected < 0.10"
    )


# ── 6. Task 5 twist: agent that ignores analytics-service scores low ───────

def test_task5_ignoring_analytics_scores_low():
    """Task 5 flips the script — analytics-service IS the root cause.
    An agent blindly ignoring it should score poorly."""
    t5 = Task5MemoryLeakAnalytics()
    ep5 = t5.build_episode(SEED)
    ep5.action_log = [
        {"step": 1, "action": "query_logs", "params": {"service": "order-service", "pattern": "error"}, "reward": 0.0},
        {"step": 2, "action": "query_logs", "params": {"service": "auth-service", "pattern": "error"}, "reward": 0.0},
        {"step": 3, "action": "close_incident", "params": {
            "root_cause_service": "order-service",
            "mechanism": "unknown service degradation",
            "remediation_taken": "restart",
            "blast_radius": ["order-service"],
            "summary": "Restarted order-service.",
        }, "reward": 0.0},
    ]
    ep5.step_count = 3
    score5 = t5.grade(ep5)
    assert score5 < 0.15, (
        f"task5: agent ignoring analytics-service scored {score5}, expected < 0.15"
    )


# ── 7. Task 4 requires config investigation for high score ─────────────────

def test_task4_config_check_bonus():
    """Task 4 gives +0.10 bonus for checking postgres config."""
    t4 = Task4DbConnectionLimit()
    ep_no_config = t4.build_episode(SEED)
    ep_no_config.action_log = [
        {"step": 1, "action": "close_incident", "params": {
            "root_cause_service": "postgres",
            "mechanism": "connection pool saturation",
            "remediation_taken": "config_patch",
            "blast_radius": ["postgres", "order-service", "api-gateway"],
            "summary": "Patched config.",
        }, "reward": 0.5},
    ]
    ep_no_config.step_count = 1
    score_no_config = t4.grade(ep_no_config)

    ep_with_config = t4.build_episode(SEED)
    ep_with_config.action_log = [
        {"step": 1, "action": "check_config", "params": {"service": "postgres", "key": "max_connections"}, "reward": 0.03},
        {"step": 2, "action": "close_incident", "params": {
            "root_cause_service": "postgres",
            "mechanism": "connection pool saturation",
            "remediation_taken": "config_patch",
            "blast_radius": ["postgres", "order-service", "api-gateway"],
            "summary": "Patched config.",
        }, "reward": 0.5},
    ]
    ep_with_config.step_count = 2
    score_with_config = t4.grade(ep_with_config)

    assert score_with_config > score_no_config, (
        f"task4: config check should boost score ({score_with_config} vs {score_no_config})"
    )
