"""OpenEnv spec compliance validation — checks yaml + live endpoints."""
import sys, json, yaml, httpx

URL = "http://localhost:7860"
PASS, FAIL, WARN = 0, 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  ⚠️  {name} — {detail}")

# ── 1. YAML structure ───────────────────────────────────────────────────────
print("\n📄 openenv.yaml validation")
with open("openenv.yaml") as f:
    cfg = yaml.safe_load(f)

check("has 'name'", "name" in cfg)
check("has 'version'", "version" in cfg)
check("has 'description'", "description" in cfg and len(cfg.get("description","")) > 10)
check("has 'domain'", "domain" in cfg)
check("has 'tags'", "tags" in cfg and isinstance(cfg["tags"], list))
check("has 'author'", "author" in cfg and cfg["author"] != "your-hf-username", 
      f"author='{cfg.get('author','')}'")
check("has 'tasks' list", "tasks" in cfg and isinstance(cfg["tasks"], list) and len(cfg["tasks"]) >= 1)

if "tasks" in cfg:
    for t in cfg["tasks"]:
        check(f"task '{t.get('id','')}' has required fields",
              all(k in t for k in ("id","name","difficulty","max_steps","description")),
              f"missing: {[k for k in ('id','name','difficulty','max_steps','description') if k not in t]}")

check("has 'api' section", "api" in cfg)
if "api" in cfg:
    api = cfg["api"]
    check("api.base_url defined", "base_url" in api)
    check("api.reset_endpoint", "reset_endpoint" in api and api.get("reset_endpoint") == "/reset")
    check("api.step_endpoint", "step_endpoint" in api and api.get("step_endpoint") == "/step")
    check("api.state_endpoint", "state_endpoint" in api and api.get("state_endpoint") == "/state")

# ── 2. Dockerfile ───────────────────────────────────────────────────────────
print("\n🐳 Dockerfile validation")
with open("Dockerfile") as f:
    df = f.read()
check("Dockerfile exists", True)
check("EXPOSE 7860", "EXPOSE 7860" in df)
check("CMD uses uvicorn", "uvicorn" in df)
check("port 7860 in CMD", "7860" in df)

# ── 3. Live endpoint validation ─────────────────────────────────────────────
print("\n🌐 Live endpoint validation")
c = httpx.Client(base_url=URL, timeout=10)

try:
    r = c.get("/health")
    check("GET /health returns 200", r.status_code == 200)
    body = r.json()
    check("/health has 'status'", "status" in body)
except Exception as e:
    check("GET /health reachable", False, str(e))

try:
    r = c.get("/tasks")
    check("GET /tasks returns 200", r.status_code == 200)
    tasks = r.json()
    check("/tasks returns list", isinstance(tasks, list))
    check("/tasks has >= 1 task", len(tasks) >= 1)
    for t in tasks:
        check(f"/tasks: '{t['task_id']}' has all fields",
              all(k in t for k in ("task_id","name","difficulty","max_steps","description")))
except Exception as e:
    check("GET /tasks reachable", False, str(e))

# ── 4. reset + step + state flow ────────────────────────────────────────────
print("\n🔄 Reset → Step → State flow")
try:
    task_id = tasks[0]["task_id"]
    r = c.post("/reset", json={"task_id": task_id, "seed": 42})
    check("POST /reset returns 200", r.status_code == 200)
    data = r.json()
    check("/reset has 'session_id'", "session_id" in data and len(data["session_id"]) > 0)
    check("/reset has 'observation'", "observation" in data)
    sid = data["session_id"]
    obs = data["observation"]

    # Observation structure
    check("observation.alert_summary", "alert_summary" in obs and len(obs["alert_summary"]) > 0)
    check("observation.service_health", "service_health" in obs and len(obs["service_health"]) >= 1)
    check("observation.recent_logs", "recent_logs" in obs and isinstance(obs["recent_logs"], list))
    check("observation.metrics", "metrics" in obs)
    check("observation.dependency_graph", "dependency_graph" in obs)
    check("observation.recent_deployments", "recent_deployments" in obs)
    check("observation.step_number", "step_number" in obs and obs["step_number"] == 0)
    check("observation.steps_remaining", "steps_remaining" in obs and obs["steps_remaining"] > 0)

    # Step
    r = c.post("/step", json={"session_id": sid, "action": {"action": "query_logs", "params": {"service": "order-service", "pattern": "error"}}})
    check("POST /step returns 200", r.status_code == 200)
    step_data = r.json()
    check("/step has 'observation'", "observation" in step_data)
    check("/step has 'reward'", "reward" in step_data)
    check("/step has 'done'", "done" in step_data and isinstance(step_data["done"], bool))
    check("/step has 'info'", "info" in step_data)

    reward = step_data["reward"]
    check("reward.value is float", isinstance(reward["value"], (int, float)))
    check("reward in [-0.15, 0.10]", -0.15 <= reward["value"] <= 0.10,
          f"got {reward['value']}")
    check("reward.reason is string", isinstance(reward.get("reason",""), str))
    check("reward.cumulative is float", isinstance(reward.get("cumulative", 0), (int, float)))

    # State
    r = c.get(f"/state/{sid}")
    check("GET /state returns 200", r.status_code == 200)
    state = r.json()
    for key in ("task_id","step_count","max_steps","done","service_states","cumulative_reward"):
        check(f"/state has '{key}'", key in state)

    # Invalid action
    r = c.post("/step", json={"session_id": sid, "action": {"action": "bad_action", "params": {}}})
    check("invalid action returns 422", r.status_code == 422)

    # Invalid session
    r = c.post("/step", json={"session_id": "nonexistent", "action": {"action": "query_logs", "params": {"service": "x", "pattern": "x"}}})
    check("invalid session returns 404", r.status_code == 404)

except Exception as e:
    check("reset/step/state flow", False, str(e))

# ── 5. Inference script check ───────────────────────────────────────────────
print("\n🤖 inference.py validation")
with open("inference.py", encoding="utf-8") as f:
    src = f.read()
check("inference.py exists", True)
check("no hardcoded HF tokens", "hf_" not in src,
      "found hardcoded HF token string")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed, {WARN} warnings")
if FAIL > 0:
    print("❌ VALIDATION FAILED")
    sys.exit(1)
else:
    print("✅ ALL CHECKS PASSED")
    sys.exit(0)
