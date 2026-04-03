---
title: IncidentIQ
emoji: 🚨
colorFrom: pink
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
  - sre
  - incident-response
  - reinforcement-learning
---

# IncidentIQ — Production Incident Response RL Environment

IncidentIQ is an OpenEnv-compliant reinforcement learning environment that simulates production software incidents across a microservice architecture. An AI agent plays the role of an on-call Site Reliability Engineer (SRE), receiving alerts, investigating service health through logs, metrics, traces, and deployment histories, and ultimately diagnosing and remediating the root cause of an incident.

The environment is designed to evaluate an agent's ability to reason under uncertainty, resist red-herring distractions, efficiently gather evidence, and take decisive corrective action. Five tasks of increasing difficulty test progressively deeper reasoning and operational expertise — from straightforward CPU saturation to subtle silent data corruption, and even a task that inverts prior assumptions about which service is suspicious.

## Environment Description

IncidentIQ simulates a production microservice architecture consisting of five services: **api-gateway**, **order-service**, **auth-service**, **postgres**, and **analytics-service**. These services have realistic baseline performance characteristics and a defined dependency graph. When a failure occurs in one service, it propagates through the dependency graph, creating cascading degradation that mirrors real-world incident patterns.

Why does this matter? Incident response is one of the highest-stakes activities in software engineering, yet it is rarely practised in a controlled, repeatable setting. IncidentIQ provides a sandbox where AI agents (and humans) can develop and benchmark their diagnostic reasoning, triage skills, and remediation judgement — all without risking production systems.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Deterministic rewards** | No LLM in grading — same actions always produce same scores |
| **Red herring hardening** | analytics-service looks suspicious in most tasks but is rarely the cause |
| **Task 5 twist** | analytics-service IS the root cause — tests whether agents blindly ignore it |
| **Seeded RNG per episode** | Reproducible scenarios for benchmarking |
| **Ground truth hidden until done** | Agents can't cheat by reading `/state` |
| **Efficiency bonus** | Rewards agents that solve faster |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `alert_summary` | `str` | Human-readable summary of the triggering alert |
| `service_health` | `Dict[str, ServiceHealth]` | Current health snapshot of every service |
| `recent_logs` | `List[LogLine]` | Most recent log entries across all services |
| `metrics` | `Dict[str, List[MetricPoint]]` | Time-series metrics keyed by `{service}/{metric}` |
| `dependency_graph` | `Dict[str, List[str]]` | Service dependency graph |
| `recent_deployments` | `List[Deployment]` | Recent deployment records |
| `step_number` | `int` | Current step index (0-based) |
| `steps_remaining` | `int` | Steps left before timeout |
| `last_action_result` | `Optional[str]` | Result from the previous action |

## Action Space

| Action | Params | Description |
|--------|--------|-------------|
| `query_logs` | `{service, pattern}` | Search logs for a service by pattern |
| `query_metrics` | `{service, metric, window_minutes}` | Get time-series metrics for a specific metric |
| `query_traces` | `{service}` | Get distributed traces for a service |
| `check_deployment` | `{service}` | Get recent deployment records |
| `check_config` | `{service, key}` | Check a configuration value |
| `hypothesize` | `{root_cause_service, mechanism, confidence}` | Record a hypothesis |
| `remediate` | `{type, target, details}` | Take a remediation action (rollback/restart/config_patch) |
| `close_incident` | `{root_cause_service, mechanism, remediation_taken, blast_radius, summary}` | Close the incident and trigger final grading |

## Tasks

| ID | Difficulty | Max Steps | Root Cause | Description |
|----|-----------|-----------|------------|-------------|
| `task1_cpu_saturation` | Easy | 15 | order-service | CPU saturation from missing DB index. Remediation: rollback |
| `task2_cascading_failure` | Medium | 20 | auth-service | Redis pool exhaustion cascading to API gateway 503s. Remediation: restart |
| `task3_silent_corruption` | Hard | 30 | order-service | Payment race condition causing 3% double-charges with no outage. Config change 6 days ago. Remediation: config_patch |
| `task4_db_connection_limit` | Medium | 20 | postgres | max_connections reduced from 100→20 by config change 2 days ago. Remediation: config_patch |
| `task5_memory_leak_analytics` | Medium-Hard | 25 | analytics-service | **Twist**: analytics-service IS the root cause (unbounded batch cache). Remediation: config_patch |

### Task Design Philosophy

- **Tasks 1-4**: `analytics-service` is always a red herring — it looks suspicious but is never the root cause
- **Task 5**: Flips the script — `analytics-service` IS the genuine root cause. Tests whether agents adapted vs. learned a shortcut

## Reward Function

### Step Rewards

| Condition | Reward | Description |
|-----------|--------|-------------|
| Query logs/metrics of root-cause service | +0.08 | Targeted investigation |
| Query logs/metrics of affected service | +0.05 | Relevant investigation |
| Hypothesis targets root-cause service | +0.10 | Correct hypothesis |
| Hypothesis targets affected service | +0.05 | Partial hypothesis |
| Check deployment of root-cause service | +0.03 | Good investigation |
| Remediation on wrong service | -0.10 | Misguided action |
| Restart of healthy service | -0.05 | Unnecessary action |
| Any action targeting red-herring service | -0.08 | Distraction penalty |
| Repeated action (same action+params) | -0.01 | Loop penalty |
| All other valid actions | +0.00 | Neutral exploration |

Step rewards are clamped to **[-0.15, +0.10]**.

### Terminal Rewards (on `close_incident`)

| Condition | Reward |
|-----------|--------|
| Correct `root_cause_service` | +0.25 |
| Mechanism matches ground truth | +0.15 |
| Correct `remediation_taken` | +0.15 |
| Exact `blast_radius` match | +0.10 |
| No red-herring services in blast radius | +0.05 |
| Efficiency bonus | +0.10 × (1 - steps/max_steps) |

Terminal rewards are clamped to **[0.0, 0.60]**.

## Setup and Usage

### Docker

```bash
# Build the image
docker build -t incidentiq .

# Run the server
docker run -p 7860:7860 incidentiq

# Run inference (in another terminal)
docker run --network host \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e HF_TOKEN=your-token \
  -e MODEL_NAME=gpt-4o-mini \
  -e ENV_URL=http://localhost:7860 \
  incidentiq python inference.py
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run tests (13 tests)
pytest tests/ -v

# Run inference (all 5 tasks)
API_BASE_URL=https://api.openai.com/v1 HF_TOKEN=your-token python inference.py
```

### Quick Smoke Test

```bash
# Reset a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1_cpu_saturation"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<id>","action":{"action":"query_logs","params":{"service":"order-service","pattern":"error"}}}'
```

## Baseline Scores

| Task | Model | Score | Notes |
|------|-------|-------|-------|
| `task1_cpu_saturation` | Qwen2.5-72B-Instruct | 0.68 | ✅ Solved correctly |
| `task2_cascading_failure` | Qwen2.5-72B-Instruct | 0.77 | ✅ Solved correctly |
| `task3_silent_corruption` | Qwen2.5-72B-Instruct | 0.00 | ❌ Fell for red herrings |
| `task4_db_connection_limit` | — | — | Not yet benchmarked |
| `task5_memory_leak_analytics` | — | — | Not yet benchmarked |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api-inference.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use for inference |
| `HF_TOKEN` | `""` | API key / HuggingFace token |
| `ENV_URL` | `http://localhost:7860` | IncidentIQ server URL |

## File Structure

```
incidentiq/
├── Dockerfile              # Multi-stage Python 3.11-slim build
├── openenv.yaml            # OpenEnv manifest (5 tasks)
├── requirements.txt        # Dependencies
├── inference.py            # LLM agent loop (all 5 tasks)
├── server/
│   └── app.py              # FastAPI server (5 routes)
├── env/
│   ├── environment.py      # Core RL environment
│   ├── models.py           # Pydantic v2 models
│   ├── state_machine.py    # Service states, failure propagation
│   ├── reward.py           # Deterministic reward calculator
│   ├── log_generator.py    # Synthetic logs with fault injection
│   └── metric_generator.py # 4-metric time series
├── tasks/
│   ├── base.py             # Abstract base class
│   ├── task1_cpu_saturation.py
│   ├── task2_cascading_failure.py
│   ├── task3_silent_corruption.py
│   ├── task4_db_connection_limit.py
│   └── task5_memory_leak_analytics.py
└── tests/
    ├── test_spec.py         # 6 OpenEnv spec compliance tests
    └── test_graders.py      # 7 grader accuracy tests
```
