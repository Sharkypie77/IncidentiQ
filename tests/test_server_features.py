"""Tests for additional IncidentIQ server intelligence endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app


def _new_session(client: TestClient) -> str:
    resp = client.post("/reset", json={"task_id": "task1_cpu_saturation", "seed": 42})
    assert resp.status_code == 200
    return resp.json()["session_id"]


def test_root_serves_ui_html():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "IncidentIQ Console" in resp.text


def test_api_root_returns_metadata_json():
    client = TestClient(app)
    resp = client.get("/api")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "IncidentIQ"
    assert "/ask_incident" in data["endpoints"]


def test_timeline_endpoint_returns_events():
    client = TestClient(app)
    session_id = _new_session(client)

    # Generate at least one action event in timeline.
    step_resp = client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": {"action": "query_logs", "params": {"service": "order-service", "pattern": "error"}},
        },
    )
    assert step_resp.status_code == 200

    resp = client.get(f"/timeline/{session_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_id"] == session_id
    assert data["task_id"] == "task1_cpu_saturation"
    assert isinstance(data["events"], list)
    assert len(data["events"]) >= 1
    assert "summary" in data and len(data["summary"]) > 0


def test_root_cause_tree_endpoint_returns_ranked_candidates():
    client = TestClient(app)
    session_id = _new_session(client)

    resp = client.get(f"/root-cause-tree/{session_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_id"] == session_id
    assert isinstance(data["candidates"], list)
    assert len(data["candidates"]) == 3
    probs = [c["probability"] for c in data["candidates"]]
    assert probs[0] >= probs[1] >= probs[2]
    assert 0.95 <= sum(probs) <= 1.05


def test_ask_incident_endpoint_returns_answer():
    client = TestClient(app)
    session_id = _new_session(client)

    resp = client.post(
        "/ask_incident",
        json={"session_id": session_id, "question": "What is the most likely root cause?"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_id"] == session_id
    assert "root cause" in data["answer"].lower()
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["supporting_signals"], list)

