#!/usr/bin/env python3
"""
IncidentIQ Demo — One-command demonstration of the environment.

Usage:
    # Start the server first:
    uvicorn server.app:app --port 7860

    # Then run:
    python run_demo.py

This script runs a deterministic expert agent through all 5 tasks,
showing step-by-step investigation, decisions, and final scoring.
No LLM or API key required — uses a built-in expert policy.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Tuple

import httpx

ENV_URL = "http://localhost:7860"

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


# ── Expert Policies ─────────────────────────────────────────────────────────
# Each task has a scripted optimal sequence mimicking expert SRE reasoning.

EXPERT_POLICIES: Dict[str, List[Tuple[Dict, str]]] = {
    "task1_cpu_saturation": [
        ({"action": "query_metrics", "params": {"service": "order-service", "metric": "cpu_pct", "window_minutes": 30}},
         "Alert mentions order-service → check CPU metrics first"),
        ({"action": "query_logs", "params": {"service": "order-service", "pattern": "cpu"}},
         "CPU is high → look for CPU-related errors in logs"),
        ({"action": "query_traces", "params": {"service": "order-service"}},
         "Check request traces for slow spans"),
        ({"action": "check_deployment", "params": {"service": "order-service"}},
         "Check if a recent deployment caused this"),
        ({"action": "hypothesize", "params": {"root_cause_service": "order-service", "mechanism": "CPU saturation from missing database index", "confidence": 0.85}},
         "Evidence points to order-service CPU + deployment"),
        ({"action": "remediate", "params": {"type": "rollback", "target": "order-service", "details": "Roll back to previous version with DB index"}},
         "Rollback the bad deployment"),
        ({"action": "close_incident", "params": {"root_cause_service": "order-service", "mechanism": "CPU saturation from missing database index after deployment", "remediation_taken": "rollback", "blast_radius": ["order-service", "api-gateway"], "summary": "order-service deployed without DB index causing CPU saturation. Rolled back."}},
         "Root cause confirmed → close incident"),
    ],
    "task2_cascading_failure": [
        ({"action": "query_logs", "params": {"service": "api-gateway", "pattern": "503"}},
         "Alert mentions 503s → check API gateway logs"),
        ({"action": "query_metrics", "params": {"service": "api-gateway", "metric": "error_rate", "window_minutes": 30}},
         "Confirm error rate spike on api-gateway"),
        ({"action": "query_traces", "params": {"service": "api-gateway"}},
         "Trace requests to find upstream failure"),
        ({"action": "query_logs", "params": {"service": "auth-service", "pattern": "redis"}},
         "Traces show auth-service failing → check Redis connection issues"),
        ({"action": "query_metrics", "params": {"service": "auth-service", "metric": "active_connections", "window_minutes": 30}},
         "Check auth-service connection pool saturation"),
        ({"action": "check_config", "params": {"service": "auth-service", "key": "redis_pool_size"}},
         "Verify Redis pool configuration"),
        ({"action": "hypothesize", "params": {"root_cause_service": "auth-service", "mechanism": "Redis connection pool exhaustion cascading to API gateway", "confidence": 0.9}},
         "Redis pool exhausted → cascading 503s"),
        ({"action": "remediate", "params": {"type": "restart", "target": "auth-service", "details": "Restart to clear exhausted Redis connection pool"}},
         "Restart auth-service to reset pool"),
        ({"action": "close_incident", "params": {"root_cause_service": "auth-service", "mechanism": "Redis pool exhaustion cascading to api-gateway 503s", "remediation_taken": "restart", "blast_radius": ["auth-service", "api-gateway"], "summary": "auth-service Redis pool exhausted, cascading 503s to api-gateway. Restarted auth-service."}},
         "Cascade root cause = auth-service → close"),
    ],
    "task3_silent_corruption": [
        ({"action": "query_logs", "params": {"service": "order-service", "pattern": "payment"}},
         "Alert mentions payment issues → check order-service logs"),
        ({"action": "query_metrics", "params": {"service": "order-service", "metric": "error_rate", "window_minutes": 30}},
         "Check if error rate reflects corruption"),
        ({"action": "query_logs", "params": {"service": "order-service", "pattern": "duplicate"}},
         "Look for duplicate transaction evidence"),
        ({"action": "check_deployment", "params": {"service": "order-service"}},
         "Check deployments — but issue may predate them"),
        ({"action": "check_config", "params": {"service": "order-service", "key": "payment-handler"}},
         "Silent issue? Check payment config — may be a config change"),
        ({"action": "hypothesize", "params": {"root_cause_service": "order-service", "mechanism": "Payment race condition from config change causing double-charges", "confidence": 0.85}},
         "Config changed 6 days ago → race condition"),
        ({"action": "remediate", "params": {"type": "config_patch", "target": "order-service", "details": "Restore payment handler to use idempotency keys"}},
         "Patch config to re-enable safety checks"),
        ({"action": "close_incident", "params": {"root_cause_service": "order-service", "mechanism": "Payment race condition from config change disabling idempotency", "remediation_taken": "config_patch", "blast_radius": ["order-service"], "summary": "Config change 6 days ago disabled payment idempotency, causing 3% double-charges. Patched config."}},
         "Silent corruption resolved → close"),
    ],
    "task4_db_connection_limit": [
        ({"action": "query_logs", "params": {"service": "postgres", "pattern": "connection"}},
         "Alert mentions DB issues → check postgres connection logs"),
        ({"action": "query_metrics", "params": {"service": "postgres", "metric": "active_connections", "window_minutes": 30}},
         "Check connection pool usage on postgres"),
        ({"action": "query_logs", "params": {"service": "order-service", "pattern": "connection"}},
         "Check if order-service is failing due to DB connections"),
        ({"action": "check_config", "params": {"service": "postgres", "key": "max_connections"}},
         "Config may have reduced max_connections"),
        ({"action": "hypothesize", "params": {"root_cause_service": "postgres", "mechanism": "max_connections reduced from 100 to 20 causing pool saturation", "confidence": 0.9}},
         "Config confirms: max_connections cut to 20"),
        ({"action": "remediate", "params": {"type": "config_patch", "target": "postgres", "details": "Restore max_connections from 20 back to 100"}},
         "Patch config to restore connection limit"),
        ({"action": "close_incident", "params": {"root_cause_service": "postgres", "mechanism": "max_connections reduced from 100 to 20 by config change", "remediation_taken": "config_patch", "blast_radius": ["postgres", "order-service", "api-gateway"], "summary": "postgres max_connections reduced to 20 by config change 2 days ago. Restored to 100."}},
         "DB connection limit restored → close"),
    ],
    "task5_memory_leak_analytics": [
        ({"action": "query_metrics", "params": {"service": "analytics-service", "metric": "cpu_pct", "window_minutes": 30}},
         "Alert mentions analytics → DON'T dismiss it this time"),
        ({"action": "query_logs", "params": {"service": "analytics-service", "pattern": "memory"}},
         "Check for memory-related errors in analytics"),
        ({"action": "query_metrics", "params": {"service": "analytics-service", "metric": "active_connections", "window_minutes": 30}},
         "Check resource utilization trends"),
        ({"action": "check_config", "params": {"service": "analytics-service", "key": "batch_cache_ttl"}},
         "Check if cache TTL was misconfigured"),
        ({"action": "hypothesize", "params": {"root_cause_service": "analytics-service", "mechanism": "Unbounded batch cache causing memory leak", "confidence": 0.85}},
         "Cache TTL disabled → unbounded memory growth"),
        ({"action": "remediate", "params": {"type": "config_patch", "target": "analytics-service", "details": "Restore batch_cache_ttl to enable cache expiration"}},
         "Re-enable cache expiration"),
        ({"action": "close_incident", "params": {"root_cause_service": "analytics-service", "mechanism": "Unbounded batch cache memory leak from disabled TTL", "remediation_taken": "config_patch", "blast_radius": ["analytics-service"], "summary": "analytics-service batch_cache_ttl was disabled, causing unbounded memory growth. Restored TTL config."}},
         "Memory leak fixed → close"),
    ],
}

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

    # Step through expert policy
    print(subheader("Agent Investigation"))
    print(f"  {C.DIM}{'Step':>6} │ {'Action':<18} │ {'Target':<20} │ Reasoning{C.RESET}")
    print(f"  {C.DIM}{'─' * 6}─┼─{'─' * 18}─┼─{'─' * 20}─┼─{'─' * 30}{C.RESET}")

    policy = EXPERT_POLICIES[task_id]
    rewards = []
    done = False

    for step_num, (action_dict, reasoning) in enumerate(policy, 1):
        if done:
            break

        target = action_dict["params"].get("service", action_dict["params"].get("target", "—"))
        print(step_line(step_num, action_dict["action"], target, reasoning))

        # Execute step
        try:
            r = client.post("/step", json={"session_id": session_id, "action": action_dict})
            r.raise_for_status()
            step_data = r.json()
        except Exception as e:
            print(f"    {C.RED}ERROR: {e}{C.RESET}")
            rewards.append(0.0)
            continue

        reward_val = step_data["reward"]["value"]
        reward_reason = step_data["reward"]["reason"]
        done = step_data["done"]
        cumulative = step_data["reward"]["cumulative"]
        rewards.append(reward_val)

        # Show reward inline
        print(f"         │ {reward_badge(reward_val)} │ {C.DIM}{reward_reason[:50]}{C.RESET}")

        obs = step_data["observation"]
        time.sleep(0.1)  # Slight delay for visual effect

    # Results
    total_reward = sum(rewards)
    ceiling = {"task1_cpu_saturation": 0.85, "task2_cascading_failure": 0.85,
               "task3_silent_corruption": 0.90, "task4_db_connection_limit": 0.85,
               "task5_memory_leak_analytics": 0.85}.get(task_id, 0.85)
    score = min(max(total_reward / ceiling, 0.01), 0.99)
    success = score >= 0.5

    # Get final state with ground truth
    state_r = client.get(f"/state/{session_id}")
    state_data = state_r.json()

    print(subheader("Result"))
    print(f"  {result_badge(success)} {C.BOLD}Score: {score:.2%}{C.RESET} "
          f"│ Steps: {len(rewards)} │ Total reward: {total_reward:.4f}")

    if "ground_truth" in state_data and state_data["ground_truth"]:
        gt = state_data["ground_truth"]
        print(f"\n  {C.DIM}Ground Truth:{C.RESET}")
        print(f"    Root cause: {C.BOLD}{gt.get('root_cause_service','?')}{C.RESET}")
        print(f"    Mechanism:  {gt.get('root_cause_mechanism','?')}")
        print(f"    Fix:        {gt.get('correct_remediation','?')}")

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

  {C.BOLD}Running:{C.RESET} Expert policy on all 5 tasks (no LLM required)
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
    task_ids = list(EXPERT_POLICIES.keys())

    for task_id in task_ids:
        result = run_task_demo(client, task_id)
        results.append(result)

    # Summary
    print_summary(results)

    # Save results to JSON
    output = {
        "environment": "incidentiq",
        "agent": "expert-policy (deterministic)",
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
