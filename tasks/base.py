"""Abstract base class for IncidentIQ tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from env.state_machine import EpisodeState

STRICT_SCORE_EPS = 0.01


class BaseTask(ABC):
    """
    Every task defines:

    * ``task_id`` – unique identifier used in the API
    * ``name`` – human-readable name
    * ``difficulty`` – easy / medium / hard
    * ``max_steps`` – episode budget
    * ``description`` – one-paragraph description of the scenario

    Subclasses must implement ``build_episode`` and ``grade``.
    """

    task_id: str = ""
    name: str = ""
    difficulty: str = ""
    max_steps: int = 0
    description: str = ""

    @staticmethod
    def clamp_score(score: float) -> float:
        """Clamp score to the strict open interval (0, 1)."""
        return max(STRICT_SCORE_EPS, min(float(score), 1.0 - STRICT_SCORE_EPS))

    @abstractmethod
    def build_episode(self, seed: int) -> EpisodeState:
        """Build and return a fresh :class:`EpisodeState` for this task."""
        ...

    @abstractmethod
    def grade(self, episode_state: EpisodeState) -> float:
        """
        Score a completed (or timed-out) episode.

        Returns a float in ``(0.0, 1.0)``.  Must be deterministic:
        the same ``episode_state`` always produces the same score.
        Must **not** call any LLM.
        """
        ...
