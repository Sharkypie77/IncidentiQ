"""Module-level grader functions for OpenEnv platform discovery.

Each function accepts an EpisodeState and returns a float in (0, 1).
The platform resolves graders via the ``grader`` field in openenv.yaml.
"""

from __future__ import annotations

from env.state_machine import EpisodeState
from tasks.task1_cpu_saturation import Task1CpuSaturation
from tasks.task2_cascading_failure import Task2CascadingFailure
from tasks.task3_silent_corruption import Task3SilentCorruption
from tasks.task4_db_connection_limit import Task4DbConnectionLimit
from tasks.task5_memory_leak_analytics import Task5MemoryLeakAnalytics

_t1 = Task1CpuSaturation()
_t2 = Task2CascadingFailure()
_t3 = Task3SilentCorruption()
_t4 = Task4DbConnectionLimit()
_t5 = Task5MemoryLeakAnalytics()


def grade_task1(episode_state: EpisodeState) -> float:
    """Grader for task1_cpu_saturation. Returns float in (0, 1)."""
    return _t1.grade(episode_state)


def grade_task2(episode_state: EpisodeState) -> float:
    """Grader for task2_cascading_failure. Returns float in (0, 1)."""
    return _t2.grade(episode_state)


def grade_task3(episode_state: EpisodeState) -> float:
    """Grader for task3_silent_corruption. Returns float in (0, 1)."""
    return _t3.grade(episode_state)


def grade_task4(episode_state: EpisodeState) -> float:
    """Grader for task4_db_connection_limit. Returns float in (0, 1)."""
    return _t4.grade(episode_state)


def grade_task5(episode_state: EpisodeState) -> float:
    """Grader for task5_memory_leak_analytics. Returns float in (0, 1)."""
    return _t5.grade(episode_state)
