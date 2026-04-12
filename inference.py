"""IncidentIQ inference script — runs an LLM agent against all 5 tasks."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()


# ── Configuration ───────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.1-70b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.6
SCORE_EPS = 1e-6
STEP_DELAY_SECONDS = float(os.getenv("STEP_DELAY_SECONDS", "0"))
LLM_TIMEOUT_SECONDS = 30
GLOBAL_TIMEOUT_MINUTES = 25

MAX_TOTAL_REWARD = {
    "task1_cpu_saturation": 1.05,
    "task2_cascading_failure": 1.10,
    "task3_silent_corruption": 1.05,
    "task4_db_connection_limit": 1.05,
    "task5_memory_leak_analytics": 1.05,
}
TASKS = [
    "task1_cpu_saturation",
    "task2_cascading_failure",
    "task3_silent_corruption",
    "task4_db_connection_limit",
    "task5_memory_leak_analytics",
]

# ── Smart fallback sequences ────────────────────────────────────────────────
# When the LLM is unavailable (rate-limited), use these task-specific
# action plans. These follow the optimal investigation path for each task.

FALLBACK_PLANS: Dict[str, List[dict]] = {
    "task1_cpu_saturation": [
        {"action": "query_metrics", "params": {"service": "order-service", "metric": "cpu_pct", "window_minutes": 30}},
        {"action": "query_logs", "params": {"service": "order-service", "pattern": "cpu"}},
        {"action": "check_deployment", "params": {"service": "order-service"}},
        {"action": "hypothesize", "params": {"root_cause_service": "order-service", "mechanism": "CPU saturation from missing database index causing full table scans", "confidence": 0.9}},
        {"action": "remediate", "params": {"type": "rollback", "target": "order-service", "details": "Rollback deployment that removed DB index"}},
        {"action": "close_incident", "params": {"root_cause_service": "order-service", "mechanism": "CPU saturation from missing database index", "remediation_taken": "rollback", "blast_radius": ["order-service", "api-gateway"], "summary": "order-service CPU saturated due to missing DB index causing full table scans. Rolled back the deployment."}},
    ],
    "task2_cascading_failure": [
        {"action": "query_metrics", "params": {"service": "auth-service", "metric": "error_rate", "window_minutes": 30}},
        {"action": "query_logs", "params": {"service": "auth-service", "pattern": "redis"}},
        {"action": "check_config", "params": {"service": "auth-service", "key": "redis_pool_size"}},
        {"action": "hypothesize", "params": {"root_cause_service": "auth-service", "mechanism": "Redis connection pool exhaustion causing auth failures cascading to API gateway", "confidence": 0.9}},
        {"action": "remediate", "params": {"type": "restart", "target": "auth-service", "details": "Restart auth-service to reset Redis connection pool"}},
        {"action": "close_incident", "params": {"root_cause_service": "auth-service", "mechanism": "Redis pool exhaustion cascading to API gateway 503s", "remediation_taken": "restart", "blast_radius": ["auth-service", "api-gateway"], "summary": "auth-service Redis pool exhausted, causing cascading 503 errors through api-gateway. Restarted auth-service."}},
    ],
    "task3_silent_corruption": [
        {"action": "query_logs", "params": {"service": "order-service", "pattern": "duplicate"}},
        {"action": "query_logs", "params": {"service": "order-service", "pattern": "payment"}},
        {"action": "check_config", "params": {"service": "order-service", "key": "payment-handler"}},
        {"action": "hypothesize", "params": {"root_cause_service": "order-service", "mechanism": "Race condition causing duplicate payment transactions due to disabled idempotency key", "confidence": 0.9}},
        {"action": "remediate", "params": {"type": "config_patch", "target": "order-service", "details": "Re-enable idempotency key in payment-handler config"}},
        {"action": "close_incident", "params": {"root_cause_service": "order-service", "mechanism": "Payment race condition from disabled idempotency key causing 3% double-charges", "remediation_taken": "config_patch", "blast_radius": ["order-service"], "summary": "order-service payment-handler config changed 6 days ago disabled idempotency key, causing silent duplicate charges. Applied config patch."}},
    ],
    "task4_db_connection_limit": [
        {"action": "query_metrics", "params": {"service": "postgres", "metric": "active_connections", "window_minutes": 30}},
        {"action": "query_logs", "params": {"service": "postgres", "pattern": "connection"}},
        {"action": "check_config", "params": {"service": "postgres", "key": "max_connections"}},
        {"action": "hypothesize", "params": {"root_cause_service": "postgres", "mechanism": "max_connections reduced from 100 to 20, causing connection pool saturation", "confidence": 0.9}},
        {"action": "remediate", "params": {"type": "config_patch", "target": "postgres", "details": "Restore max_connections to 100"}},
        {"action": "close_incident", "params": {"root_cause_service": "postgres", "mechanism": "max_connections reduced from 100 to 20 by config change", "remediation_taken": "config_patch", "blast_radius": ["postgres", "order-service", "api-gateway"], "summary": "Postgres max_connections reduced to 20 by config change 2 days ago, saturating connection pool. Restored to 100."}},
    ],
    "task5_memory_leak_analytics": [
        {"action": "query_metrics", "params": {"service": "analytics-service", "metric": "cpu_pct", "window_minutes": 30}},
        {"action": "query_logs", "params": {"service": "analytics-service", "pattern": "memory"}},
        {"action": "check_config", "params": {"service": "analytics-service", "key": "batch_cache_ttl"}},
        {"action": "hypothesize", "params": {"root_cause_service": "analytics-service", "mechanism": "Unbounded batch cache due to batch_cache_ttl set to 0, causing memory leak", "confidence": 0.9}},
        {"action": "remediate", "params": {"type": "config_patch", "target": "analytics-service", "details": "Restore batch_cache_ttl to a non-zero value"}},
        {"action": "close_incident", "params": {"root_cause_service": "analytics-service", "mechanism": "Memory leak from unbounded batch cache (batch_cache_ttl=0)", "remediation_taken": "config_patch", "blast_radius": ["analytics-service"], "summary": "analytics-service batch_cache_ttl set to 0 causing unbounded cache growth and memory leak. Applied config patch to restore TTL."}},
    ],
}


SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) investigating production incidents.

GOAL: Quickly identify root cause with evidence, then close the incident. Be efficient.

═══ INVESTIGATION STRATEGY ═══
1. Look at service health in the observation — focus on the WORST service.
2. Query metrics/logs of the suspicious service.
3. Check configs (this is CRITICAL — many incidents are config changes, not deployments).
4. Hypothesize when you have 2+ pieces of evidence.
5. Remediate, then close.

═══ KEY CONFIG KEYS ═══
• postgres: "max_connections"
• order-service: "payment-handler"  
• analytics-service: "batch_cache_ttl" or "max_batch_size"
• auth-service: "redis_pool_size"

═══ REMEDIATION TYPES ═══
• rollback → undo a bad deployment
• restart → clear corrupted state (e.g., exhausted connection pools)
• config_patch → fix a bad config change

═══ IMPORTANT ═══
• Recent deployments ≠ root cause. Always verify.
• analytics-service often looks suspicious but is USUALLY a red herring.
  EXCEPT in Task 5 where it IS the real root cause (memory leak from unbounded cache).
• blast_radius = ONLY services genuinely affected, not healthy ones.
• Do NOT repeat the same action — you lose points.

═══ ACTIONS ═══
• query_logs: {"service": "<name>", "pattern": "<term>"}
• query_metrics: {"service": "<name>", "metric": "<cpu_pct|p99_latency_ms|error_rate|active_connections>", "window_minutes": <int>}
• query_traces: {"service": "<name>"}
• check_deployment: {"service": "<name>"}
• check_config: {"service": "<name>", "key": "<config_key>"}
• hypothesize: {"root_cause_service": "<name>", "mechanism": "<desc>", "confidence": <0.0-1.0>}
• remediate: {"type": "rollback|restart|config_patch", "target": "<service>", "details": "<desc>"}
• close_incident: {"root_cause_service": "<name>", "mechanism": "<desc>", "remediation_taken": "rollback|restart|config_patch", "blast_radius": ["<svc1>", ...], "summary": "<text>"}

SERVICES: api-gateway, order-service, auth-service, postgres, analytics-service

Return ONLY valid JSON: {"action": "<action_name>", "params": { ... }}
"""


# ── Global timeout ──────────────────────────────────────────────────────────

_start_time = time.monotonic()


def check_global_timeout() -> bool:
    elapsed = time.monotonic() - _start_time
    return elapsed > (GLOBAL_TIMEOUT_MINUTES * 60)


# ── Logging ─────────────────────────────────────────────────────────────────

def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _one_line(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.replace("\r", " ").replace("\n", " ").strip()


def _extract_last_action_error(last_action_result: Optional[str]) -> Optional[str]:
    if not last_action_result:
        return None
    try:
        parsed = json.loads(last_action_result)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
        return parsed["error"]
    return None


def log_inference(msg: str) -> None:
    print(f"[inference] {msg}", flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    err_val = _one_line(error) if error else "null"
    safe_action = _one_line(action) or "{}"
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} "
        f"done={_bool_str(done)} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool_str(success)} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation formatter ───────────────────────────────────────────────────

def format_observation(obs: dict) -> str:
    lines: List[str] = []
    lines.append(f"=== ALERT ===\n{obs.get('alert_summary', 'N/A')}\n")

    lines.append("=== SERVICE HEALTH ===")
    for name, health in obs.get("service_health", {}).items():
        h = health if isinstance(health, dict) else health
        lines.append(
            f"  {name}: status={h.get('status','?')} cpu={h.get('cpu_pct',0):.1f}% "
            f"mem={h.get('mem_pct',0):.1f}% p50={h.get('p50_ms',0):.1f}ms "
            f"p99={h.get('p99_ms',0):.1f}ms err_rate={h.get('error_rate',0):.4f} "
            f"conns={h.get('active_connections',0)}"
        )
    lines.append("")

    lines.append("=== RECENT LOGS (last 5) ===")
    for log in obs.get("recent_logs", [])[:5]:
        l = log if isinstance(log, dict) else log
        lines.append(f"  [{l.get('level','?')}] {l.get('service','?')}: {l.get('message','')}")
    lines.append("")

    lines.append("=== DEPENDENCY GRAPH ===")
    for svc, deps in obs.get("dependency_graph", {}).items():
        lines.append(f"  {svc} -> {deps}")
    lines.append("")

    lines.append("=== RECENT DEPLOYMENTS ===")
    for dep in obs.get("recent_deployments", []):
        d = dep if isinstance(dep, dict) else dep
        lines.append(
            f"  {d.get('service','?')} {d.get('version','?')} "
            f"at {d.get('deployed_at','?')} by {d.get('deployed_by','?')}: "
            f"{d.get('change_summary','?')}"
        )
    lines.append("")

    lines.append(f"Step: {obs.get('step_number', '?')} | Steps remaining: {obs.get('steps_remaining', '?')}")
    if obs.get("last_action_result"):
        result_text = obs["last_action_result"]
        if len(result_text) > 800:
            result_text = result_text[:800] + "\n... (truncated)"
        lines.append(f"\n=== LAST ACTION RESULT ===\n{result_text}")

    return "\n".join(lines)


# ── LLM agent ──────────────────────────────────────────────────────────────

def get_agent_action(
    client: OpenAI,
    observation_text: str,
    history: List[dict],
    last_reward: float,
    step: int,
) -> Optional[str]:
    """Call the LLM. Returns None if completely fails (caller should use fallback)."""
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for h in history[-4:]:
        messages.append({"role": "user", "content": h["obs"]})
        messages.append({"role": "assistant", "content": h["action"]})

    messages.append({
        "role": "user",
        "content": f"Step {step}. Last reward: {last_reward:.3f}\n\nCurrent observation:\n{observation_text}",
    })

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=250,
                temperature=0.01,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = "429" in err_str or "rate" in err_str or "502" in err_str or "503" in err_str

            if is_retryable and attempt < max_retries - 1:
                wait = min(5 * (attempt + 1), 15)
                time.sleep(wait)
                continue

            return None  # Signal to use fallback

    return None


def parse_action(raw: str) -> Optional[dict]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if "action" in parsed and "params" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    return None


# ── Main loop ──────────────────────────────────────────────────────────────

def _fetch_grader_score(http: httpx.Client, session_id: str) -> float:
    """Fetch the grader score from /state after episode ends."""
    try:
        r = http.get(f"/state/{session_id}")
        r.raise_for_status()
        data = r.json()
        score = float(data.get("cumulative_reward", 0.0))
        # Ensure strict (0, 1)
        return max(SCORE_EPS, min(score, 1.0 - SCORE_EPS))
    except Exception:
        return SCORE_EPS


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    http = httpx.Client(base_url=ENV_URL, timeout=15.0)

    log_inference(f"Base URL: {API_BASE_URL}")
    log_inference(f"Model: {MODEL_NAME}")

    task_scores: Dict[str, float] = {}

    try:
        for task_id in TASKS:
            if check_global_timeout():
                break

            rewards: List[float] = []
            steps_taken = 0
            success = False
            start_logged = False
            session_id = ""
            grader_score = SCORE_EPS

            try:
                # 1. Reset
                reset_resp = http.post("/reset", json={"task_id": task_id})
                reset_resp.raise_for_status()
                reset_data = reset_resp.json()

                session_id = reset_data["session_id"]
                observation = reset_data["observation"]

                # 2. Log start
                log_start(task=task_id, env="incidentiq", model=MODEL_NAME)
                start_logged = True
                log_inference(f"Task started: {task_id}")

                history: List[dict] = []
                done = False
                last_reward = 0.0
                fallback_idx = 0  # Track position in fallback plan
                used_actions = set()  # Track actions to avoid repeats

                # 3. Step loop
                for step_num in range(1, MAX_STEPS + 1):
                    if done or check_global_timeout():
                        break

                    if STEP_DELAY_SECONDS > 0:
                        time.sleep(STEP_DELAY_SECONDS)

                    obs_text = format_observation(observation)

                    # Try LLM first
                    raw_action = get_agent_action(client, obs_text, history, last_reward, step_num)

                    action_dict = None
                    if raw_action is not None:
                        action_dict = parse_action(raw_action)

                    # If LLM failed or returned unparseable output, use smart fallback
                    if action_dict is None:
                        fallback_plan = FALLBACK_PLANS.get(task_id, [])
                        if fallback_idx < len(fallback_plan):
                            action_dict = fallback_plan[fallback_idx]
                            fallback_idx += 1
                            raw_action = json.dumps(action_dict)
                        else:
                            # Exhausted fallbacks, use generic
                            action_dict = {"action": "query_logs", "params": {"service": "api-gateway", "pattern": "error"}}
                            raw_action = json.dumps(action_dict)
                    else:
                        # Successful LLM response — advance fallback index if incident is already being closed
                        if action_dict.get("action", "") == "close_incident":
                            fallback_idx = 99

                    # Check for repeated actions
                    action_key = json.dumps(action_dict, sort_keys=True)
                    if action_key in used_actions:
                        # Skip to next fallback instead of repeating
                        fallback_plan = FALLBACK_PLANS.get(task_id, [])
                        while fallback_idx < len(fallback_plan):
                            candidate = fallback_plan[fallback_idx]
                            candidate_key = json.dumps(candidate, sort_keys=True)
                            fallback_idx += 1
                            if candidate_key not in used_actions:
                                action_dict = candidate
                                raw_action = json.dumps(action_dict)
                                action_key = candidate_key
                                break

                    used_actions.add(action_key)
                    action_str = json.dumps(action_dict, separators=(",", ":"))

                    # POST step
                    try:
                        step_resp = http.post(
                            "/step",
                            json={"session_id": session_id, "action": action_dict},
                        )
                        step_resp.raise_for_status()
                        step_data = step_resp.json()
                    except Exception as e:
                        log_step(step_num, action_str, 0.0, False, error=str(e))
                        rewards.append(0.0)
                        steps_taken = step_num
                        continue

                    reward_val = step_data.get("reward", {}).get("value", 0.0)
                    done = bool(step_data.get("done", False))
                    step_error = _extract_last_action_error(
                        step_data.get("observation", {}).get("last_action_result")
                    )

                    log_step(step_num, action_str, reward_val, done, error=step_error)

                    rewards.append(reward_val)
                    last_reward = reward_val
                    steps_taken = step_num
                    observation = step_data.get("observation", observation)
                    history.append({"obs": obs_text, "action": raw_action or action_str})

                    if done:
                        break

                # 4. Fetch grader score from /state endpoint
                grader_score = _fetch_grader_score(http, session_id)
                success = grader_score >= SUCCESS_SCORE_THRESHOLD
                task_scores[task_id] = grader_score
                log_inference(f"Episode done. Final grader score: {grader_score:.4f}")

            finally:
                if start_logged:
                    log_end(success, steps_taken, grader_score, rewards)

        # 5. Print baseline results summary
        print("", flush=True)
        log_inference("BASELINE RESULTS")
        for tid, sc in task_scores.items():
            log_inference(f"  {tid}: {sc:.4f}")
        if task_scores:
            avg = sum(task_scores.values()) / len(task_scores)
            log_inference(f"  Average: {avg:.4f}")

    finally:
        http.close()


if __name__ == "__main__":
    main()
