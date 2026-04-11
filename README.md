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

# 🚨 IncidentIQ

**A training ground for AI agents that solve production outages.**

IncidentIQ is an interactive simulation where an AI agent plays the role of an on-call engineer responding to a software system failure. The agent must investigate the problem step by step — reading logs, checking metrics, inspecting configurations — then diagnose the root cause and fix it, just like a real engineer would during a live incident.

Everything is scored automatically. No human judgement is needed to evaluate the AI's performance, and the scoring never uses another AI. This makes results **fully reproducible and auditable**.

---

## 🎯 What Does It Do?

| Question | Answer |
|----------|--------|
| What is this? | A simulated environment where an AI agent investigates and fixes software outages |
| Is this an AI model? | No — it's a testing environment that *any* AI model can be plugged into |
| How is performance scored? | Automatically, using fixed rules — no AI is involved in grading |
| Why does this matter? | Real outages are messy and multi-step. This tests whether AI can handle that complexity |

---

## 💡 Why This Problem Matters

When real software systems break, the cause is rarely obvious. Engineers face:

- **Noisy signals** — many things look wrong at once, but only one is the real cause
- **Red herrings** — some services appear broken but are actually fine
- **Chain reactions** — one failure causes others, hiding the true origin
- **Subtle config changes** — sometimes the problem isn't a code bug, but a setting that was quietly changed days ago

IncidentIQ is designed to test whether an AI agent can navigate these real-world challenges instead of just pattern-matching on keywords.

---

## 🏗️ How It Works

The environment simulates **five microservices** that depend on each other — just like a real production system. When something breaks, failures can cascade across services, creating realistic symptoms.

### How an Episode Plays Out

| Step | What happens |
|------|-------------|
| **1. Start** | The AI receives an alert about a system problem |
| **2. Investigate** | The AI inspects logs, metrics, configurations, and deployment history |
| **3. Diagnose** | The AI forms a hypothesis about what went wrong |
| **4. Fix** | The AI applies a fix (rollback, restart, or config change) |
| **5. Close** | The AI submits its final diagnosis and resolution |

At each step, the AI receives a **score** telling it whether its choices are helpful or not. The final score reflects the accuracy of the diagnosis and the quality of the fix.

### What the AI Can Do

| Action | What it does |
|--------|-------------|
| Read logs | Look at recent log messages from any service |
| Check metrics | View CPU, memory, latency, and error rate over time |
| Check traces | Inspect request flow across services |
| Check deployments | See what was recently deployed and by whom |
| Check config | Look at configuration settings and when they last changed |
| Hypothesize | Record a theory about the root cause |
| Fix the problem | Apply a rollback, restart, or configuration patch |
| Close the incident | Submit final diagnosis and end the episode |

---

## 📋 The Five Challenge Tasks

Each task presents a different type of outage, ranging from simple to tricky:

| # | Task | Difficulty | What's Happening |
|---|------|-----------|-----------------|
| 1 | CPU Saturation | Easy | A missing database index is overloading one service |
| 2 | Cascading Failure | Medium | A connection pool runs dry and takes down other services |
| 3 | Silent Corruption | Hard | A config change from 6 days ago is silently causing double-charges — no outage visible |
| 4 | Database Connection Limit | Medium | A database setting was quietly reduced, starving all services of connections |
| 5 | Memory Leak (Twist) | Medium-Hard | The service that was always a red herring in other tasks is *actually* the real problem this time |

### Why the tasks are designed this way

- **Tasks 1–4** include a distraction: the analytics service always *looks* suspicious but is never the real problem. This tests whether the AI blindly follows noise.
- **Task 5** flips the script — analytics *is* the real cause. This tests whether the AI can adapt instead of relying on shortcuts.
- Several tasks involve **config changes, not code deployments** — forcing the AI to dig deeper than just checking "what was deployed recently."

---

## 📊 Scoring System

### During the Investigation (per-step scoring)

| Good actions | Score |
|-------------|-------|
| Querying logs or metrics for the right service | +0.08 |
| Correctly identifying the root cause | +0.10 |
| Checking the right deployment or config | +0.03 |

| Bad actions | Score |
|------------|-------|
| Investigating a red herring service | −0.08 |
| Applying a fix to the wrong service | −0.10 |
| Repeating the same action | −0.01 |

### At the End (final diagnosis scoring)

| What's evaluated | Score |
|-----------------|-------|
| Correctly identifying the broken service | +0.25 |
| Correctly explaining *why* it broke | +0.15 |
| Applying the right fix | +0.15 |
| Correctly listing all affected services | +0.10 |
| Not blaming innocent services | +0.05 |
| Solving it efficiently (fewer steps = more bonus) | up to +0.10 |

**Final task scores are always between 0.01 and 0.99** (strictly between 0 and 1).

---

## 📈 Baseline Performance

These scores were achieved by running an AI agent (Llama 3.1 70B) against the environment:

| Task | Difficulty | AI Agent Score |
|------|-----------|---------------|
| CPU Saturation | Easy | 0.72 |
| Cascading Failure | Medium | 0.63 |
| Silent Corruption | Hard | 0.85 |
| DB Connection Limit | Medium | 0.70 |
| Memory Leak (Twist) | Medium-Hard | 0.77 |

---

## 🖥️ Interactive UI

Opening the server in a browser (`http://localhost:7860`) gives you a visual console where you can:

- Start any task and watch the AI investigate
- Manually run actions to explore the environment
- View a timeline of what happened during the incident
- See ranked root-cause candidates
- Ask natural-language questions about the incident

---

## 🔌 API Endpoints

| Endpoint | What it does |
|----------|-------------|
| `GET /` | Opens the browser UI |
| `GET /health` | Health check |
| `GET /tasks` | Lists all available tasks |
| `POST /reset` | Starts a new incident session |
| `POST /step` | Sends an action and gets the result |
| `GET /state` | Returns the current state of the session |
| `GET /timeline/{id}` | Shows the incident timeline |
| `GET /root-cause-tree/{id}` | Shows ranked root-cause candidates |
| `POST /ask_incident` | Ask a question about the incident |

---

## 🚀 How to Run

### Quick Start (Local)

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860 in your browser
```

### Using Docker

```bash
docker build -t incidentiq .
docker run -p 7860:7860 incidentiq
```

### Running the AI Agent

```bash
API_BASE_URL=https://integrate.api.nvidia.com/v1 HF_TOKEN=your-key python inference.py
```

### Running Tests

```bash
pytest tests/ -v
```

---

## 📂 Project Structure

| File / Folder | What it does |
|--------------|-------------|
| `openenv.yaml` | Project manifest (metadata, task list, grader references) |
| `server/app.py` | The web server with all API endpoints |
| `server/index.html` | The browser-based UI |
| `env/environment.py` | Core logic: episode management, actions, state |
| `env/reward.py` | Scoring rules for each step |
| `tasks/*.py` | The five challenge tasks and their grading logic |
| `tasks/grader.py` | Grading functions referenced by the platform |
| `inference.py` | AI agent loop (connects to an LLM to solve tasks) |
| `run_demo.py` | A simple rule-based agent for testing |
| `tests/` | Automated tests for grading, rewards, and API correctness |

---

## ✅ Key Strengths

| Strength | Why it matters |
|----------|---------------|
| **Real-world problem domain** | Incident response is a critical skill — both for humans and AI |
| **Fully deterministic scoring** | Every run with the same inputs produces the same score — no randomness, no AI in the grader |
| **Progressive difficulty with a twist** | Tasks get harder, and the final task deliberately breaks assumptions the agent may have formed |
| **Rich observability data** | Logs, metrics, traces, deployments, and configs — just like a real production system |
| **Built-in UI and API** | Easy to demo, easy to inspect, easy to understand what's happening |
