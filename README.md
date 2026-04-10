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
# IncidentIQ
IncidentIQ is a reinforcement learning environment for production incident response in a microservice system.
Instead of treating incident handling as a one-shot diagnosis, IncidentIQ models it as a sequential decision process:
an agent inspects logs/metrics/traces/config/deployments, forms hypotheses, applies remediation, and closes the incident.
Rewards are deterministic and grading is fully programmatic (no LLM in the grader).
## At a Glance
| Item | Value |
|---|---|
| Project type | Reinforcement learning environment |
| Domain | SRE / production incident response |
| Interaction style | Multi-step `reset -> step -> close_incident` |
| Grading | Deterministic, programmatic, no LLM in grader |
| Task count | 5 |
| Difficulty structure | Easy, Medium, Hard, Twist |
| Built-in UI | `GET /` |
| API index | `GET /api` |
| Score bounds | Strictly clamped to `(0.01, 0.99)` |
## What This Project Is
| Question | Answer |
|---|---|
| Is this a model? | No. It is an environment an agent interacts with. |
| Is this only a benchmark sheet? | No. It includes environment state, reward logic, and episode flow. |
| Is grading LLM-based? | No. Grading is deterministic and reproducible. |
| Why RL for this? | Because step-to-step choices affect final incident quality and closure score. |
## Why This Problem Matters
Real incidents are rarely solved by one log line.
The hardest failures include noisy symptoms, red herrings, cascading dependencies, and config-vs-deploy ambiguity.
IncidentIQ is designed to measure whether an agent can reason through those constraints rather than pattern-match.
## Why Reinforcement Learning Fits
In incident response:
- a good step at time `t` depends on what was discovered earlier
- locally plausible actions can hurt terminal score
- correctness is often clear only at close time
RL fits this well: sequential actions, intermediate rewards, and delayed terminal grading.
## Environment Overview
IncidentIQ simulates five services:
`api-gateway`, `order-service`, `auth-service`, `postgres`, `analytics-service`.
Failures propagate through a dependency graph and create realistic cascade behavior.
### Episode Flow
| Step | What happens | Why it matters |
|---|---|---|
| 1 | Client calls `POST /reset` with `task_id` | Starts a fresh incident session |
| 2 | Environment returns an observation | Agent sees health, logs, metrics, traces, deployments |
| 3 | Agent calls `POST /step` with an action | Agent investigates or remediates |
| 4 | Environment returns reward + new observation | Agent gets deterministic learning signal |
| 5 | Agent calls `close_incident` action | Terminal grading is applied |
### Action Space
| Action | Params | Purpose |
|---|---|---|
| `query_logs` | `{service, pattern}` | Inspect service logs |
| `query_metrics` | `{service, metric, window_minutes}` | Inspect time-series metrics |
| `query_traces` | `{service}` | Inspect distributed traces |
| `check_deployment` | `{service}` | Inspect deployment history |
| `check_config` | `{service, key}` | Inspect configuration state |
| `hypothesize` | `{root_cause_service, mechanism, confidence}` | Record root-cause hypothesis |
| `remediate` | `{type, target, details}` | Apply rollback/restart/config patch |
| `close_incident` | `{root_cause_service, mechanism, remediation_taken, blast_radius, summary}` | End episode and trigger terminal grading |
### Observation Space
| Field | Type | Meaning |
|---|---|---|
| `alert_summary` | `str` | Triggering incident summary |
| `service_health` | `Dict[str, ServiceHealth]` | Current service status snapshot |
| `recent_logs` | `List[LogLine]` | Recent logs across services |
| `metrics` | `Dict[str, List[MetricPoint]]` | Time-series metrics |
| `dependency_graph` | `Dict[str, List[str]]` | Service dependency graph |
| `recent_deployments` | `List[Deployment]` | Recent deployments |
| `step_number` | `int` | Current step index |
| `steps_remaining` | `int` | Remaining budget |
| `last_action_result` | `Optional[str]` | Last action result summary |
### Example Interaction
```text
POST /reset {"task_id":"task1_cpu_saturation"}
-> returns session_id + initial observation

POST /step {"session_id":"...","action":{"action":"query_logs","params":{"service":"order-service","pattern":"error"}}}
-> returns reward + next observation

POST /step {"session_id":"...","action":{"action":"close_incident","params":{...}}}
-> returns terminal reward and done=true
```
## Benchmark Tasks
| Task ID | Difficulty | Max Steps | Root Cause | Main challenge |
|---|---|---:|---|---|
| `task1_cpu_saturation` | Easy | 15 | `order-service` | Missing DB index causing CPU saturation |
| `task2_cascading_failure` | Medium | 20 | `auth-service` | Redis pool exhaustion causing cascade |
| `task3_silent_corruption` | Hard | 30 | `order-service` | Silent corruption from config race condition |
| `task4_db_connection_limit` | Medium | 20 | `postgres` | Reduced DB connection limit |
| `task5_memory_leak_analytics` | Medium-Hard | 25 | `analytics-service` | Twist task where analytics is true root cause |
### Task Design Philosophy
| Design choice | Why it exists |
|---|---|
| Tasks 1-4 keep analytics as red herring | Tests distraction resistance |
| Task 5 flips analytics to real cause | Tests adaptation vs shortcut behavior |
| Config-heavy root causes | Forces deeper reasoning beyond log keywords |
## Reward Design
### Per-step Rewards
| Condition | Reward |
|---|---:|
| Query logs/metrics on root-cause service | +0.08 |
| Query logs/metrics on affected service | +0.05 |
| Correct hypothesis target | +0.10 |
| Partial hypothesis target | +0.05 |
| Check deployment/config on root-cause service | +0.03 |
| Wrong remediation target | -0.10 |
| Restart healthy service | -0.05 |
| Target red-herring service | -0.08 |
| Repeated same action+params | -0.01 |
Step rewards are clamped to `[-0.15, +0.10]`.
### Terminal Rewards (`close_incident`)
| Component | Reward |
|---|---:|
| Correct `root_cause_service` | +0.25 |
| Correct mechanism | +0.15 |
| Correct remediation | +0.15 |
| Exact blast radius | +0.10 |
| No red-herring blast entries | +0.05 |
| Efficiency bonus | `+0.10 * (1 - steps/max_steps)` |
Terminal rewards are clamped to `[0.0, 0.60]`.
## Deterministic Grading and Score Bounds
| Rule | Effect |
|---|---|
| No randomness in grader | Same episode history always yields same score |
| No LLM in grading loop | Reproducible and audit-friendly evaluation |
| Final score clamp | Strictly `(0.01, 0.99)` |
This strict score range is required by benchmark validators that reject `0.0` and `1.0`.
## Reward System Validation Snapshot
| Factor | Current value | Source |
|---|---|---|
| Step reward clamp | `[-0.15, +0.10]` | `env/reward.py` |
| Terminal reward clamp | `[0.001, 0.599]` | `env/reward.py` |
| Final task score clamp | `(0.01, 0.99)` | task graders + `run_demo.py` + `inference.py` |
| Benchmark success rate | `100% (5/5)` | `benchmark_results.json` |
| Average benchmark score | `0.99` | `benchmark_results.json` |
| Average benchmark steps | `7.6` | `benchmark_results.json` |
| Reward/score tests | `3 passed` | `test_step_reward_range`, `test_graders_return_valid_float`, `test_perfect_agent_scores_high` |
### Per-task Reward Snapshot (Expert Policy)
| Task | Steps | Total Reward | Score |
|---|---:|---:|---:|
| `task1_cpu_saturation` | 7 | 0.889 | 0.99 |
| `task2_cascading_failure` | 9 | 0.989 | 0.99 |
| `task3_silent_corruption` | 8 | 0.919 | 0.99 |
| `task4_db_connection_limit` | 7 | 0.939 | 0.99 |
| `task5_memory_leak_analytics` | 7 | 0.889 | 0.99 |
| **Average** | **7.6** | **0.925** | **0.99** |
## API Reference
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Browser UI console |
| `GET` | `/api` | API metadata and endpoint list |
| `GET` | `/health` | Health check |
| `GET` | `/metadata` | Environment metadata |
| `GET` | `/schema` | Action/observation/state schemas |
| `GET` | `/tasks` | List benchmark tasks |
| `POST` | `/reset` | Start episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | State for latest session |
| `GET` | `/state/{session_id}` | State for a specific session |
| `GET` | `/timeline/{session_id}` | Reconstructed incident timeline |
| `GET` | `/root-cause-tree/{session_id}` | Ranked root-cause candidates |
| `POST` | `/ask_incident` | Natural-language incident Q&A |
| `POST` | `/mcp` | MCP protocol stub |
## UI Console
Open `http://localhost:7860` after server start.
The UI supports:
- reset task sessions
- run arbitrary actions
- inspect timeline/root-cause tree
- ask natural-language incident questions
## Recorded Inference Run (Recent)
| Task | Success | Steps |
|---|---|---:|
| `task1_cpu_saturation` | ✅ | 6 |
| `task2_cascading_failure` | ✅ | 6 |
| `task3_silent_corruption` | ✅ | 7 |
| `task4_db_connection_limit` | ✅ | 7 |
| `task5_memory_leak_analytics` | ✅ | 5 |
| **Summary** | **5/5 solved** | **6.2 avg** |
### Example Output
```text
[START] task=task1_cpu_saturation env=incidentiq model=meta/llama-3.1-70b-instruct
[STEP] step=1 action={"action":"query_metrics","params":{"service":"order-service","metric":"cpu_pct","window_minutes":10}} reward=0.08 done=false error=null
[STEP] step=2 action={"action":"query_logs","params":{"service":"order-service","pattern":"slow query detected"}} reward=0.08 done=false error=null
[STEP] step=6 action={"action":"close_incident","params":{"root_cause_service":"order-service","mechanism":"CPU saturation from missing database index","remediation_taken":"rollback","blast_radius":["order-service","api-gateway"],"summary":"order-service CPU saturated due to missing DB index causing full table scans. Rolled back the deployment."}} reward=0.60 done=true error=null
[END] success=true steps=6 rewards=0.08,0.08,0.03,0.10,0.00,0.60
```
## Running the Project
### Environment Variables
| Variable | Default | Required | Description |
|---|---|---|---|
| `API_BASE_URL` | `https://integrate.api.nvidia.com/v1` | No | LLM API base URL |
| `MODEL_NAME` | `meta/llama-3.1-70b-instruct` | No | Model name for inference |
| `HF_TOKEN` | — | Yes | Provider API key |
| `ENV_URL` | `http://localhost:7860` | No | Environment server URL |
| `STEP_DELAY_SECONDS` | `0` | No | Delay between inference steps |
| `RESCUE_START_STEP` | `6` | No | Inference rescue mode start |
| `FORCE_CLOSE_STEP` | `9` | No | Forced close step |
### Quick Local Run
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
# open http://localhost:7860
python run_demo.py
```
### Docker Run
```bash
docker build -t incidentiq .
docker run -p 7860:7860 incidentiq
```
### LLM Inference
```bash
API_BASE_URL=https://integrate.api.nvidia.com/v1 HF_TOKEN=your-key python inference.py
```
### Tests
```bash
pytest tests/ -v
```
## Hugging Face Space Usage
Set your environment URL to the live Space and run inference:
```bash
ENV_URL=https://Zewx77-incidentiq.hf.space HF_TOKEN=your-key python inference.py
```
Basic checks:
```bash
curl https://Zewx77-incidentiq.hf.space/health
curl https://Zewx77-incidentiq.hf.space/tasks
```
## Repository Structure
| Path | Purpose |
|---|---|
| `openenv.yaml` | OpenEnv manifest |
| `server/app.py` | FastAPI app + API/UI routes |
| `server/index.html` | Browser UI |
| `env/environment.py` | Core episode loop |
| `env/reward.py` | Deterministic reward calculator |
| `tasks/*.py` | Task definitions and grading logic |
| `inference.py` | LLM inference loop |
| `run_demo.py` | Deterministic expert run |
| `tests/` | Spec, grader, and server tests |
## Submission Strengths
| Strength | Why it matters |
|---|---|
| Real incident-response domain | Strong practical relevance |
| Deterministic evaluation | Reproducible scoring for judges |
| Curriculum tasks + twist | Tests deeper reasoning and adaptation |
| Multi-signal observability | Logs + metrics + traces + deploy + config |
| Built-in UI + API | Easy to demo and inspect behavior |
## Recent Fixes
| Fix | Reason |
|---|---|
| Strict score clamp to `(0.01, 0.99)` | Prevent validator rejection |
| Inference output normalization | Ensures parser-safe `[START]/[STEP]/[END]` lines |
| Rescue mode in inference | Improves completion reliability under drift |
| Root UI + `/api` split | Better user experience with preserved API index |
| Timeline/root-cause/Q&A endpoints | Better explainability and demo value |
