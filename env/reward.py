"""Reward calculation for the IncidentIQ environment."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from env.models import Action
from env.state_machine import EpisodeState


class RewardCalculator:
    """Deterministic reward calculator — never calls an LLM."""

    @staticmethod
    def calculate_step_reward(
        action: Action,
        params: Dict[str, Any],
        state: EpisodeState,
    ) -> Tuple[float, str]:
        """
        Return ``(reward_value, reason_string)`` for a single step.

        Applies the first matching rule; clamps to [-0.15, +0.10].
        """
        gt = state.ground_truth
        act = action.action
        reward = 0.0
        reason = "neutral action"

        # Check for repeated action (same action + params already in action_log)
        for entry in state.action_log:
            if entry.get("action") == act and entry.get("params") == params:
                reward = -0.01
                reason = "repeated action – penalise loop"
                return (max(-0.15, min(0.10, reward)), reason)

        # ── Query logs ──
        if act == "query_logs":
            svc = params.get("service", "")
            reward_key = f"logs:{svc}"
            if svc in gt.red_herring_services:
                reward = -0.08
                reason = f"queried logs of red-herring service ({svc})"
            elif svc == gt.root_cause_service:
                if reward_key not in state.rewarded_services:
                    state.rewarded_services.add(reward_key)
                    reward = 0.08
                    reason = f"queried logs of root-cause service ({svc})"
                else:
                    reward = 0.0
                    reason = f"already rewarded for logs of {svc}"
            elif svc in gt.affected_services:
                if reward_key not in state.rewarded_services:
                    state.rewarded_services.add(reward_key)
                    reward = 0.05
                    reason = f"queried logs of affected service ({svc})"
                else:
                    reward = 0.0
                    reason = f"already rewarded for logs of {svc}"
            else:
                reward = 0.0
                reason = f"queried logs of {svc}"

        # ── Query metrics ──
        elif act == "query_metrics":
            svc = params.get("service", "")
            reward_key = f"metrics:{svc}"
            if svc in gt.red_herring_services:
                reward = -0.08
                reason = f"queried metrics of red-herring service ({svc})"
            elif svc == gt.root_cause_service:
                if reward_key not in state.rewarded_services:
                    state.rewarded_services.add(reward_key)
                    reward = 0.08
                    reason = f"queried metrics of root-cause service ({svc})"
                else:
                    reward = 0.0
                    reason = f"already rewarded for metrics of {svc}"
            elif svc in gt.affected_services:
                if reward_key not in state.rewarded_services:
                    state.rewarded_services.add(reward_key)
                    reward = 0.05
                    reason = f"queried metrics of affected service ({svc})"
                else:
                    reward = 0.0
                    reason = f"already rewarded for metrics of {svc}"
            else:
                reward = 0.0
                reason = f"queried metrics of {svc}"

        # ── Query traces ──
        elif act == "query_traces":
            svc = params.get("service", "")
            if svc in gt.red_herring_services:
                reward = -0.08
                reason = f"traced red-herring service ({svc})"
            else:
                reward = 0.0
                reason = f"queried traces for {svc}"

        # ── Check deployment ──
        elif act == "check_deployment":
            svc = params.get("service", "")
            if svc == gt.root_cause_service:
                reward = 0.03
                reason = f"checked deployments of root-cause service ({svc})"
            elif svc in gt.red_herring_services:
                reward = -0.08
                reason = f"checked deployments of red-herring service ({svc})"
            else:
                reward = 0.0
                reason = f"checked deployments for {svc}"

        # ── Check config ──
        elif act == "check_config":
            svc = params.get("service", "")
            if svc == gt.root_cause_service:
                reward = 0.03
                reason = f"checked config of root-cause service ({svc})"
            elif svc in gt.red_herring_services:
                reward = -0.08
                reason = f"checked config of red-herring service ({svc})"
            else:
                reward = 0.0
                reason = f"checked config for {svc}"

        # ── Hypothesize ──
        elif act == "hypothesize":
            svc = params.get("root_cause_service", "")
            if svc == gt.root_cause_service:
                reward = 0.10
                reason = "hypothesis targets the correct root-cause service"
            elif svc in gt.affected_services:
                reward = 0.05
                reason = "hypothesis targets an affected (but not root-cause) service"
            elif svc in gt.red_herring_services:
                reward = -0.08
                reason = "hypothesis targets a red-herring service"
            else:
                reward = 0.0
                reason = f"hypothesis targets {svc}"

        # ── Remediate ──
        elif act == "remediate":
            target = params.get("target", "")
            rtype = params.get("type", "")
            all_affected = [gt.root_cause_service] + list(gt.affected_services)
            healthy_services = [
                s for s in [
                    "api-gateway", "order-service", "auth-service",
                    "postgres", "analytics-service",
                ]
                if s not in all_affected and s not in gt.red_herring_services
            ]
            # Red-herring check MUST come before the generic unaffected check,
            # because red herrings are not in all_affected either.
            if target in gt.red_herring_services:
                reward = -0.08
                reason = f"remediation targets a red-herring service ({target})"
            elif target not in all_affected:
                reward = -0.10
                reason = f"remediation applied to unaffected service ({target})"
            elif rtype == "restart" and target in healthy_services:
                reward = -0.05
                reason = f"unnecessary restart of healthy service ({target})"
            else:
                reward = 0.0
                reason = f"remediation '{rtype}' on {target}"

        # ── Close incident (step reward is 0; terminal reward handled separately) ──
        elif act == "close_incident":
            reward = 0.0
            reason = "incident closed — terminal reward applies"

        return (max(-0.15, min(0.10, reward)), reason)

    @staticmethod
    def calculate_terminal_reward(
        close_params: Dict[str, Any],
        state: EpisodeState,
    ) -> Tuple[float, str]:
        """
        Return ``(reward_value, reason_string)`` for the terminal close_incident action.

        Sums all matching bonuses; clamps to [0.0, 0.60].
        """
        gt = state.ground_truth
        total = 0.0
        reasons: list[str] = []

        # +0.25 correct root cause service
        if close_params.get("root_cause_service") == gt.root_cause_service:
            total += 0.25
            reasons.append("+0.25 correct root_cause_service")

        # +0.15 mechanism match (substring, case-insensitive)
        mechanism = str(close_params.get("mechanism", "")).lower()
        if gt.root_cause_mechanism.lower() in mechanism:
            total += 0.15
            reasons.append("+0.15 mechanism matches ground truth")

        # +0.15 correct remediation
        if close_params.get("remediation_taken") == gt.correct_remediation:
            total += 0.15
            reasons.append("+0.15 correct remediation_taken")

        # +0.10 exact blast radius match
        claimed_blast = set(close_params.get("blast_radius", []))
        actual_blast = set(gt.affected_services)
        if claimed_blast == actual_blast:
            total += 0.10
            reasons.append("+0.10 exact blast_radius match")

        # +0.05 no red herrings in blast radius
        if not any(rh in claimed_blast for rh in gt.red_herring_services):
            total += 0.05
            reasons.append("+0.05 no red-herring services in blast_radius")

        # +0.10 * efficiency bonus
        efficiency = 1.0 - state.step_count / state.max_steps
        eff_bonus = round(0.10 * efficiency, 4)
        total += eff_bonus
        reasons.append(f"+{eff_bonus:.4f} efficiency bonus ({state.step_count}/{state.max_steps} steps)")

        # Clamp close evaluation reward to [0.0, 0.60].
        total = max(0.0, min(0.60, total))
        return (total, "; ".join(reasons))
