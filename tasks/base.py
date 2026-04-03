"""Abstract base class for IncidentIQ tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from env.state_machine import EpisodeState


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

    @abstractmethod
    def build_episode(self, seed: int) -> EpisodeState:
        """Build and return a fresh :class:`EpisodeState` for this task."""
        ...

    @abstractmethod
    def grade(self, episode_state: EpisodeState) -> float:
        """
        Score a completed (or timed-out) episode.

        Returns a float in ``[0.0, 1.0]``.  Must be deterministic:
        the same ``episode_state`` always produces the same score.
        Must **not** call any LLM.
        """
        ...
