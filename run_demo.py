#!/usr/bin/env python3
"""
IncidentIQ Demo — One-command demonstration of the environment.

Usage:
    # Start the server first:
    uvicorn server.app:app --port 7860

    # Then run:
    python run_demo.py

This script runs a deterministic rule-based agent through all 5 tasks,
showing step-by-step investigation, decisions, and final scoring.
No LLM or API key required.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List

import httpx

ENV_URL = "http://localhost:7860"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Colors ──────────────────────────────────────────────────────────────────

class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"

def header(text: str) -> str:
    return f"\n{C.BOLD}{C.CYAN}{'═' * 70}{C.RESET}\n{C.BOLD}{C.WHITE}  {text}{C.RESET}\n{C.BOLD}{C.CYAN}{'═' * 70}{C.RESET}"

def subheader(text: str) -> str:
    return f"\n{C.BOLD}{C.BLUE}── {text} ──{C.RESET}"

def step_line(step: int, action: str, target: str, reasoning: str) -> str:
    return (
        f"  {C.BOLD}{C.WHITE}Step {step:2d}{C.RESET} │ "
        f"{C.YELLOW}{action:<18}{C.RESET} │ "
        f"{C.CYAN}{target:<20}{C.RESET} │ "
        f"{C.DIM}{reasoning}{C.RESET}"
    )

def reward_badge(val: float) -> str:
    if val > 0:
        return f"{C.GREEN}+{val:.3f}{C.RESET}"
    elif val < 0:
        return f"{C.RED}{val:.3f}{C.RESET}"
    return f"{C.DIM} 0.000{C.RESET}"

def result_badge(success: bool) -> str:
    if success:
        return f"{C.BG_GREEN}{C.WHITE} ✓ SOLVED {C.RESET}"
    return f"{C.BG_RED}{C.WHITE} ✗ FAILED {C.RESET}"


# ── Rule-based policy ────────────────────────────────────────────────────────

def pick_service_from_queried(observation: Dict[str, Any], queried_services: List[str]) -> str:
    service_health = observation.get("service_health", {})
    if not service_health:
        return "api-gateway"

    picked_service = "api-gateway"
    best_score = -1.0
    for service_name in queried_services:
        health = service_health.get(service_name)
        if not health:
            continue
        score = float(health.get("error_rate", 0.0)) * float(health.get("p99_ms", 0.0))
        if score > best_score:
            best_score = score
            picked_service = service_name
    return picked_service

TASK_NAMES = {
    "task1_cpu_saturation": ("Task 1: CPU Saturation", "easy"),
    "task2_cascading_failure": ("Task 2: Cascading Failure", "medium"),
    "task3_silent_corruption": ("Task 3: Silent Corruption", "hard"),
    "task4_db_connection_limit": ("Task 4: DB Connection Limit", "medium"),
    "task5_memory_leak_analytics": ("Task 5: Memory Leak (Twist)", "medium-hard"),
}


# ── Demo Runner ─────────────────────────────────────────────────────────────

def run_task_demo(client: httpx.Client, task_id: str) -> Dict[str, Any]:
    """Run the expert policy on one task and return results."""
    name, difficulty = TASK_NAMES[task_id]
    print(header(f"{name}  [{difficulty}]"))

    # Reset
    r = client.post("/reset", json={"task_id": task_id, "seed": 42})
    r.raise_for_status()
    reset_data = r.json()
    session_id = reset_data["session_id"]
    obs = reset_data["observation"]

    print(f"\n  {C.BOLD}Alert:{C.RESET} {C.YELLOW}{obs['alert_summary']}{C.RESET}")
    print(f"  {C.BOLD}Services:{C.RESET} {', '.join(obs['service_health'].keys())}")
    print(f"  {C.BOLD}Max steps:{C.RESET} {obs['steps_remaining']}")

    # Service health summary
    print(subheader("Initial Service Health"))
    for svc_name, health in obs["service_health"].items():
        status = health["status"]
        color = C.GREEN if status == "healthy" else C.YELLOW if status == "degraded" else C.RED
        print(f"    {color}●{C.RESET} {svc_name:<20} {color}{status:<10}{C.RESET} "
              f"cpu={health['cpu_pct']:5.1f}%  mem={health['mem_pct']:5.1f}%  "
              f"err={health['error_rate']:.4f}  p99={health['p99_ms']:.0f}ms")

    # Step through rule-based policy
    print(subheader("Agent Investigation"))
    print(f"  {C.DIM}{'Step':>6} │ {'Action':<18} │ {'Target':<20} │ Reasoning{C.RESET}")
    print(f"  {C.DIM}{'─' * 6}─┼─{'─' * 18}─┼─{'─' * 20}─┼─{'─' * 30}{C.RESET}")

    rewards = []
    done = False
    step_num = 1
    queried_services: List[str] = list(obs.get("service_health", {}).keys())

    def execute_action(action_dict: Dict[str, Any], reasoning: str) -> None:
        nonlocal obs, done, step_num
        if done:
            return
        target = action_dict["params"].get("service", action_dict["params"].get("target", "—"))
        print(step_line(step_num, action_dict["action"], target, reasoning))
        try:
            resp = client.post("/step", json={"session_id": session_id, "action": action_dict})
            resp.raise_for_status()
            step_data = resp.json()
        except Exception as e:
            print(f"    {C.RED}ERROR: {e}{C.RESET}")
            rewards.append(0.0)
            step_num += 1
            return

        reward_val = step_data["reward"]["value"]
        reward_reason = step_data["reward"]["reason"]
        done = step_data["done"]
        rewards.append(reward_val)
        print(f"         │ {reward_badge(reward_val)} │ {C.DIM}{reward_reason[:50]}{C.RESET}")
        obs = step_data["observation"]
        step_num += 1
        time.sleep(0.1)

    # Step 1
    execute_action(
        {"action": "query_metrics", "params": {"service": "api-gateway", "metric": "error_rate", "window_minutes": 30}},
        "Start at entry point: inspect api-gateway metrics",
    )

    # Step 2
    execute_action(
        {"action": "query_logs", "params": {"service": "api-gateway", "pattern": "error"}},
        "Inspect api-gateway logs for upstream failure hints",
    )

    # Step 3: pick service with highest error_rate * p99_ms across observed services
    picked_service = pick_service_from_queried(obs, queried_services)

    # Steps 4-10
    close_action = {
        "action": "close_incident",
        "params": {
            "root_cause_service": picked_service,
            "mechanism": "high error rate and latency detected via metric analysis",
            "remediation_taken": "rollback",
            "blast_radius": [picked_service, "api-gateway"],
            "summary": f"Incident closed after investigating {picked_service}",
        },
    }
    picked_status = (
        obs.get("service_health", {})
        .get(picked_service, {})
        .get("status", "healthy")
    )
    if picked_status == "healthy":
        dependencies = list(obs.get("dependency_graph", {}).get("api-gateway", []))
        dep_target = pick_service_from_queried(obs, dependencies) if dependencies else "api-gateway"
        follow_up_actions: List[tuple[Dict[str, Any], str]] = [
            (
                {"action": "query_metrics", "params": {"service": dep_target, "metric": "error_rate", "window_minutes": 30}},
                f"Check one gateway dependency before closing: {dep_target}",
            ),
            (close_action, "Close incident with final report"),
        ]
    elif picked_service == "analytics-service":
        follow_up_actions = [
            (
                {"action": "query_logs", "params": {"service": picked_service, "pattern": "error"}},
                "Deepen evidence on selected service with error logs",
            ),
            (
                {"action": "check_deployment", "params": {"service": picked_service}},
                "Check latest deployment for regression signal",
            ),
            (
                {"action": "check_config", "params": {"service": picked_service, "key": "payment-handler"}},
                "Check critical config key for drift",
            ),
            (
                {
                    "action": "hypothesize",
                    "params": {
                        "root_cause_service": picked_service,
                        "mechanism": "high error rate and latency detected via metric analysis",
                        "confidence": 0.8,
                    },
                },
                "Form hypothesis from metric+log evidence",
            ),
            (
                {
                    "action": "remediate",
                    "params": {
                        "type": "rollback",
                        "target": picked_service,
                        "details": "rolling back last deployment",
                    },
                },
                "Apply rollback remediation",
            ),
            (close_action, "Close incident with final report"),
        ]
    else:
        follow_up_actions = [
            (close_action, "Close incident with final report"),
        ]
    for action_dict, reasoning in follow_up_actions:
        if done:
            break
        execute_action(action_dict, reasoning)

    # Results
    total_reward = sum(rewards)
    ceiling = 1.05
    score = min(max(total_reward / ceiling, 0.0), 1.0)
    success = score >= 0.5

    print(subheader("Result"))
    print(f"  {result_badge(success)} {C.BOLD}Score: {score:.2%}{C.RESET} "
          f"│ Steps: {len(rewards)} │ Total reward: {total_reward:.4f}")

    return {
        "task_id": task_id,
        "name": name,
        "difficulty": difficulty,
        "success": success,
        "score": score,
        "steps": len(rewards),
        "total_reward": total_reward,
        "rewards": rewards,
    }


def print_summary(results: List[Dict]) -> None:
    """Print the final benchmark summary table."""
    print(header("BENCHMARK SUMMARY"))

    # Table header
    print(f"\n  {C.BOLD}{'Task':<35} {'Diff':<12} {'Steps':>5} {'Score':>8} {'Reward':>9} {'Result':>10}{C.RESET}")
    print(f"  {'─' * 35} {'─' * 12} {'─' * 5} {'─' * 8} {'─' * 9} {'─' * 10}")

    total_score = 0
    total_steps = 0
    successes = 0

    for r in results:
        badge = f"{C.GREEN}✓ PASS{C.RESET}" if r["success"] else f"{C.RED}✗ FAIL{C.RESET}"
        score_color = C.GREEN if r["score"] >= 0.6 else C.YELLOW if r["score"] >= 0.3 else C.RED
        print(f"  {r['name']:<35} {r['difficulty']:<12} {r['steps']:>5} "
              f"{score_color}{r['score']:>7.1%}{C.RESET} {r['total_reward']:>+8.4f}  {badge}")
        total_score += r["score"]
        total_steps += r["steps"]
        if r["success"]:
            successes += 1

    avg_score = total_score / len(results) if results else 0
    avg_steps = total_steps / len(results) if results else 0

    print(f"  {'─' * 35} {'─' * 12} {'─' * 5} {'─' * 8} {'─' * 9} {'─' * 10}")
    score_color = C.GREEN if avg_score >= 0.6 else C.YELLOW
    print(f"  {C.BOLD}{'AVERAGE':<35} {'—':<12} {avg_steps:>5.1f} "
          f"{score_color}{avg_score:>7.1%}{C.RESET}  "
          f"{C.BOLD}{successes}/{len(results)} solved{C.RESET}")

    # Key metrics
    print(f"\n  {C.BOLD}Key Metrics:{C.RESET}")
    print(f"    Success Rate:   {C.BOLD}{successes}/{len(results)}{C.RESET} ({successes/len(results)*100:.0f}%)")
    print(f"    Average Score:  {C.BOLD}{avg_score:.1%}{C.RESET}")
    print(f"    Average Steps:  {C.BOLD}{avg_steps:.1f}{C.RESET}")
    print(f"    Total Steps:    {C.BOLD}{total_steps}{C.RESET}")
    print()


def main() -> None:
    print(header("IncidentIQ — Production Incident Response RL Environment"))
    print(f"""
  {C.BOLD}{C.WHITE}What is IncidentIQ?{C.RESET}
  {C.DIM}A benchmark where AI agents act as on-call SREs, diagnosing and fixing
  production incidents across a simulated microservice architecture — with
  deterministic grading, red-herring hardening, and no LLM in the loop.{C.RESET}

  {C.BOLD}Running:{C.RESET} Rule-based policy on all 5 tasks (no LLM required)
  {C.BOLD}Server:{C.RESET} {ENV_URL}
""")

    # Health check
    client = httpx.Client(base_url=ENV_URL, timeout=30.0)
    try:
        r = client.get("/health")
        r.raise_for_status()
        health = r.json()
        print(f"  {C.GREEN}●{C.RESET} Server healthy (v{health.get('version', '?')})")
    except Exception as e:
        print(f"  {C.RED}●{C.RESET} Server not reachable: {e}")
        print(f"\n  {C.YELLOW}Start the server first:{C.RESET}")
        print(f"    uvicorn server.app:app --port 7860\n")
        sys.exit(1)

    # Run all tasks
    results = []
    task_ids = list(TASK_NAMES.keys())

    for task_id in task_ids:
        result = run_task_demo(client, task_id)
        results.append(result)

    # Summary
    print_summary(results)

    # Save results to JSON
    output = {
        "environment": "incidentiq",
        "agent": "rule-based policy (deterministic)",
        "tasks": results,
        "summary": {
            "success_rate": sum(r["success"] for r in results) / len(results),
            "avg_score": sum(r["score"] for r in results) / len(results),
            "avg_steps": sum(r["steps"] for r in results) / len(results),
            "total_tasks": len(results),
            "tasks_solved": sum(r["success"] for r in results),
        },
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  {C.DIM}Results saved to benchmark_results.json{C.RESET}\n")


if __name__ == "__main__":
    main()
