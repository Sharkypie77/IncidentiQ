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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.6
STEP_DELAY_SECONDS = float(os.getenv("STEP_DELAY_SECONDS", "12"))

# Per-task reward ceilings: realistic max step rewards + terminal max (0.60)
MAX_TOTAL_REWARD = {
    "task1_cpu_saturation": 0.85,
    "task2_cascading_failure": 0.85,
    "task3_silent_corruption": 0.90,
    "task4_db_connection_limit": 0.85,
    "task5_memory_leak_analytics": 0.85,
}
TASKS = [
    "task1_cpu_saturation",
    "task2_cascading_failure",
    "task3_silent_corruption",
    "task4_db_connection_limit",
    "task5_memory_leak_analytics",
]

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) investigating production incidents in a distributed microservice system.

Your goal is NOT to guess — your goal is to PROVE the root cause using evidence.

═══ CORE RULES (MANDATORY) ═══

1. ALWAYS investigate before hypothesizing.
   - Never jump to conclusions from the alert alone.

2. PRIORITIZE SYSTEMATIC DEBUGGING:
   Follow this order:
   (a) Identify most impacted service from metrics/logs
   (b) Check its dependencies
   (c) Check logs → metrics → traces → deployments → configs

3. CONFIG CHECK IS CRITICAL:
   - Before closing any incident, you MUST call:
     check_config on your suspected root cause service
   - Many incidents are caused by configuration changes, NOT deployments.
   - Config keys to check:
     - postgres: "max_connections"
     - order-service: "payment-handler"
     - analytics-service: "batch_cache_ttl" or "max_batch_size"
     - auth-service: "redis_pool_size"

4. DO NOT TRUST CORRELATION:
   - Recent deployments ≠ root cause
   - High resource usage ≠ root cause
   - Always verify causation with logs/configs

5. HANDLE RED HERRINGS:
   - Some services may look suspicious but are unrelated
   - Ignore services with old timestamps or weak/indirect signals
   - Focus on strongest, most recent, causal evidence
   - Do NOT blindly assume analytics-service is always a red herring.
     In some incidents it IS the real root cause (e.g., genuine memory leak from unbounded caching).

6. SPECIAL CASE — SILENT FAILURES:
   - If system appears "healthy" but issue exists:
     → suspect race conditions or config bugs
     → check configs even if no errors are visible
     → look for duplicate transactions (idempotency failures)
     → check config changes from days ago (not recent deploys)

7. MINIMIZE STEPS:
   - Each action costs time. Be precise, not exhaustive.
   - Do NOT repeat the same action with the same params — you get penalized.

═══ DECISION PROCESS (FOLLOW STRICTLY) ═══

At each step ask yourself:
1. What is the MOST suspicious service right now?
2. What evidence do I have?
3. What evidence am I missing?
4. What is the NEXT BEST action to confirm or reject my hypothesis?

═══ WHEN TO HYPOTHESIZE ═══

Only hypothesize when:
- You have at least 2 strong pieces of evidence
- You have checked logs or metrics of that service

═══ WHEN TO CLOSE INCIDENT ═══

Only close when ALL are true:
- Root cause service is identified
- Mechanism is clear (cpu, memory, config, connection, race condition)
- Config has been checked if relevant
- Remediation matches the cause:
  - rollback → undo a bad deployment
  - restart → clear corrupted state (e.g., exhausted connection pools)
  - config_patch → fix a bad config change (reduced limits, disabled TTLs, etc.)
- blast_radius = ONLY services genuinely affected by THIS incident, NOT healthy ones

═══ AVAILABLE ACTIONS ═══
• query_logs: {"service": "<name>", "pattern": "<search_term>"}
• query_metrics: {"service": "<name>", "metric": "<cpu_pct|p99_latency_ms|error_rate|active_connections>", "window_minutes": <int>}
• query_traces: {"service": "<name>"}
• check_deployment: {"service": "<name>"}
• check_config: {"service": "<name>", "key": "<config_key>"}
• hypothesize: {"root_cause_service": "<name>", "mechanism": "<description>", "confidence": <0.0-1.0>}
• remediate: {"type": "rollback|restart|config_patch", "target": "<service>", "details": "<description>"}
• close_incident: {"root_cause_service": "<name>", "mechanism": "<description>", "remediation_taken": "rollback|restart|config_patch", "blast_radius": ["<svc1>", ...], "summary": "<text>"}

═══ SERVICES ═══
api-gateway, order-service, auth-service, postgres, analytics-service

═══ OUTPUT FORMAT (STRICT) ═══
Return ONLY valid JSON. No prose, no markdown, no explanation.
{"action": "<action_name>", "params": { ... }}
"""


# ── Logging functions (exact spec format) ───────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(
        json.dumps({"type": "[START]", "task": task, "env": env, "model": model}),
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    print(
        json.dumps({
            "type": "[STEP]",
            "step": step,
            "action": action,
            "reward": round(reward, 4),
            "done": done,
            "error": error,
        }),
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        json.dumps({
            "type": "[END]",
            "success": success,
            "steps": steps,
            "score": round(score, 4),
            "total_reward": round(sum(rewards), 4),
            "rewards": [round(r, 4) for r in rewards],
        }),
        flush=True,
    )


# ── Observation formatter ───────────────────────────────────────────────────

def format_observation(obs: dict) -> str:
    """Format a raw observation dict into a readable text string for the LLM."""
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

    lines.append("=== RECENT LOGS (last 10) ===")
    for log in obs.get("recent_logs", [])[:10]:
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
        if len(result_text) > 1500:
            result_text = result_text[:1500] + "\n... (truncated)"
        lines.append(f"\n=== LAST ACTION RESULT ===\n{result_text}")

    return "\n".join(lines)


# ── LLM agent ──────────────────────────────────────────────────────────────

def get_agent_action(
    client: OpenAI,
    observation_text: str,
    history: List[dict],
    last_reward: float,
    step: int,
) -> str:
    """Call the LLM and return its raw text response with retry on rate limits."""
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include last 6 turns for context
    for h in history[-6:]:
        messages.append({"role": "user", "content": h["obs"]})
        messages.append({"role": "assistant", "content": h["action"]})

    messages.append({
        "role": "user",
        "content": f"Step {step}. Last reward: {last_reward:.3f}\n\nCurrent observation:\n{observation_text}",
    })

    max_retries = 15
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=300,
                temperature=0.01,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e).lower()
            # Handle rate limits AND router upstream failures which are extremely common on OpenRouter free tiers
            is_retryable = "429" in err_str or "rate" in err_str or "provider" in err_str or "502" in err_str or "503" in err_str
            
            if is_retryable and attempt < max_retries - 1:
                # Exponential backoff capped at 2 minutes
                wait = min(8 * (2 ** attempt), 120) 
                print(f"[DEBUG] API congested (attempt {attempt+1}/{max_retries}). Waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
                
            print(f"[DEBUG] LLM call completely failed after {attempt+1} attempts: {e}", flush=True)
            return json.dumps({
                "action": "query_logs",
                "params": {"service": "api-gateway", "pattern": "error"},
            })

    return json.dumps({
        "action": "query_logs",
        "params": {"service": "api-gateway", "pattern": "error"},
    })


def parse_action(raw: str) -> dict:
    """Parse the LLM response into an action dict, with fallback."""
    # Strip markdown code fences if present
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

    # Fallback
    return {
        "action": "query_logs",
        "params": {"service": "api-gateway", "pattern": "error"},
    }


# ── Main loop ──────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    http = httpx.Client(base_url=ENV_URL, timeout=30.0)

    for task_idx, task_id in enumerate(TASKS):
        if task_idx > 0:
            print("\n" + "=" * 60 + "\n", flush=True)

        # 1. Reset
        reset_resp = http.post("/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()

        session_id = reset_data["session_id"]
        observation = reset_data["observation"]

        # 2. Log start
        log_start(task=task_id, env="incidentiq", model=MODEL_NAME)

        history: List[dict] = []
        rewards: List[float] = []
        done = False
        last_reward = 0.0
        steps_taken = 0

        # 3. Step loop
        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            # Extreme pacing to respect very strict free tier API limits (e.g. 5 requests per min)
            time.sleep(STEP_DELAY_SECONDS)

            obs_text = format_observation(observation)
            raw_action = get_agent_action(client, obs_text, history, last_reward, step_num)
            action_dict = parse_action(raw_action)
            action_str = json.dumps(action_dict)

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
            done = step_data.get("done", False)

            log_step(step_num, action_str, reward_val, done)

            rewards.append(reward_val)
            last_reward = reward_val
            steps_taken = step_num

            # Update observation for next turn
            observation = step_data.get("observation", observation)

            # Append to history
            history.append({"obs": obs_text, "action": raw_action})

            if done:
                break

        # 4–6. Score and log end
        total_reward = sum(rewards)
        ceiling = MAX_TOTAL_REWARD.get(task_id, 0.85)
        score = min(max(total_reward / ceiling, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()