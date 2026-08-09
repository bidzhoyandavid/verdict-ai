"""API-level tests. The graph is stubbed — these cover the HTTP layer,
tenant scoping and the run lifecycle, not the analysis itself.
"""

from __future__ import annotations

import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from api.main import app


class _Snapshot:
    def __init__(self, values: dict):
        self.values = values


class FakeGraph:
    """Emits the same shapes LangGraph does: node updates, then a final state."""

    def __init__(self, chunks: list[dict], final: dict | None = None):
        self.chunks = chunks
        self.final = final or {}
        self.calls: list = []

    def stream(self, payload, config, stream_mode="updates"):
        self.calls.append(payload)
        yield from self.chunks

    def get_state(self, config):
        return _Snapshot(self.final)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth(client):
    """A fresh company + admin; returns the auth header."""
    email = f"user-{time.time_ns()}@acme.io"
    response = client.post(
        "/auth/signup",
        json={"email": email, "password": "password123", "name": "Мария Иванова", "company": f"Acme {time.time_ns()}"},
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _upload_dataset(client, auth) -> str:
    df = pd.DataFrame({"group": ["a", "b"] * 10, "metric": list(range(20)), "user_id": list(range(20))})
    buffer = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    response = client.post(
        "/files/datasets", headers=auth, files={"file": ("data.csv", buffer, "text/csv")}
    )
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


def _wait_for_status(client, auth, test_id: str, expected: set[str], timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/tests/{test_id}", headers=auth).json()
        if body["status"] in expected:
            return body
        time.sleep(0.05)
    raise AssertionError(f"test stayed in {body['status']!r}, expected one of {expected}")


def test_requires_authentication(client):
    assert client.get("/tests").status_code == 401
    assert client.get("/tests", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_signup_rejects_duplicate_email(client, auth):
    me = client.get("/auth/me", headers=auth).json()
    response = client.post(
        "/auth/signup", json={"email": me["email"], "password": "password123", "company": "Other"}
    )
    assert response.status_code == 409


def test_first_user_is_admin(client, auth):
    assert client.get("/auth/me", headers=auth).json()["role"] == "Admin"


def test_tests_are_scoped_to_the_company(client, auth):
    other = client.post(
        "/auth/signup",
        json={"email": f"other-{time.time_ns()}@x.io", "password": "password123", "company": "Other Co"},
    ).json()
    other_auth = {"Authorization": f"Bearer {other['access_token']}"}

    test_id = client.post("/tests", headers=auth, json={"name": "Mine"}).json()["id"]

    assert client.get(f"/tests/{test_id}", headers=auth).status_code == 200
    # 404, not 403 — a foreign id must not be distinguishable from a missing one.
    assert client.get(f"/tests/{test_id}", headers=other_auth).status_code == 404
    assert client.get("/tests", headers=other_auth).json() == []


def test_upload_rejects_unsupported_extension(client, auth):
    response = client.post(
        "/files/datasets", headers=auth, files={"file": ("notes.txt", io.BytesIO(b"x"), "text/plain")}
    )
    assert response.status_code == 415


def test_create_test_with_dataset_runs_the_graph(client, auth, monkeypatch):
    graph = FakeGraph(
        chunks=[
            {"load": {"group_col": "group", "metric_col": "metric"}},
            {"insight": {"messages": [AIMessage(content="Вариант B значимо лучше.")]}},
        ],
        final={"test_result": {"p_value": 0.02, "decision": "significant", "effect": {"relative_lift": 0.21}}},
    )
    monkeypatch.setattr("api.runner.get_graph", lambda: graph)

    dataset_id = _upload_dataset(client, auth)
    created = client.post(
        "/tests",
        headers=auth,
        json={"name": "Карточка товара", "hypothesis": "Новый макет поднимет конверсию", "dataset_id": dataset_id},
    ).json()

    body = _wait_for_status(client, auth, created["id"], {"done", "failed"})
    assert body["status"] == "done"
    assert body["results"]["short"].startswith("treatment +21.0%")

    messages = client.get(f"/tests/{created['id']}/messages", headers=auth).json()
    assert [m["role"] for m in messages] == ["user", "agent"]
    assert messages[1]["text"] == "Вариант B значимо лучше."


def test_interrupt_pauses_the_test_and_resume_continues_it(client, auth, monkeypatch):
    pausing = FakeGraph(chunks=[{"__interrupt__": [type("I", (), {"value": {"kind": "outlier_review"}})()]}])
    monkeypatch.setattr("api.runner.get_graph", lambda: pausing)

    dataset_id = _upload_dataset(client, auth)
    created = client.post("/tests", headers=auth, json={"name": "T", "dataset_id": dataset_id}).json()

    body = _wait_for_status(client, auth, created["id"], {"awaiting_input", "failed"})
    assert body["status"] == "awaiting_input"
    assert body["pending_interrupt"] == {"kind": "outlier_review"}
    # A new message must not be accepted while a question is pending.
    assert client.post(f"/tests/{created['id']}/messages", headers=auth, json={"text": "hi"}).status_code == 409

    resuming = FakeGraph(chunks=[{"insight": {"messages": [AIMessage(content="Готово.")]}}], final={})
    monkeypatch.setattr("api.runner.get_graph", lambda: resuming)
    assert (
        client.post(
            f"/tests/{created['id']}/resume", headers=auth, json={"decision": {"method": "winsorize"}}
        ).status_code
        == 202
    )

    body = _wait_for_status(client, auth, created["id"], {"done", "failed"})
    assert body["status"] == "done"
    assert resuming.calls and getattr(resuming.calls[0], "resume", None) == {"method": "winsorize"}


def test_failing_run_marks_the_test_failed(client, auth, monkeypatch):
    class Exploding(FakeGraph):
        def stream(self, payload, config, stream_mode="updates"):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    monkeypatch.setattr("api.runner.get_graph", lambda: Exploding([]))

    dataset_id = _upload_dataset(client, auth)
    created = client.post("/tests", headers=auth, json={"name": "T", "dataset_id": dataset_id}).json()

    body = _wait_for_status(client, auth, created["id"], {"failed", "done"})
    assert body["status"] == "failed"
    assert "boom" in body["error"]


def test_only_admin_can_invite(client, auth):
    invited = client.post("/team", headers=auth, json={"email": f"a-{time.time_ns()}@acme.io", "role": "Analyst"})
    assert invited.status_code == 201

    # The invitee (Analyst) cannot invite further members.
    from api.db import SessionLocal
    from api.models import User
    from api.security import issue_token

    with SessionLocal() as session:
        analyst = session.get(User, invited.json()["id"])
        analyst_auth = {"Authorization": f"Bearer {issue_token(analyst.id, analyst.company_id)}"}

    assert client.post("/team", headers=analyst_auth, json={"email": "b@acme.io"}).status_code == 403
