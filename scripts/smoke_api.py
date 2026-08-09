"""End-to-end smoke check against a running API.

Signs up, uploads a dataset, creates a test, follows the SSE stream and
auto-answers any HITL interrupt with the agent's own recommendation — i.e.
exactly what the frontend will do, minus the UI.

    uvicorn api.main:app --reload        # in another terminal
    python scripts/smoke_api.py --csv path/to/data.csv

Exits non-zero if the run does not reach `done`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import httpx

TIMEOUT = httpx.Timeout(10.0, read=300.0)


def sse_events(response: httpx.Response) -> Iterator[tuple[str, dict]]:
    """Minimal SSE parser: enough for `event:`/`data:` pairs."""
    event = "message"
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            yield event, json.loads(line[5:].strip())
            event = "message"


def log(label: str, payload: Any = "") -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)[:400]
    print(f"[{label}] {text}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--csv", required=True, help="CSV/Parquet with group + metric columns")
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--metric-col", default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--hypothesis", default="Проверяем, влияет ли вариант на метрику.")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=TIMEOUT)

    health = client.get("/health").json()
    log("health", health)
    if health["checkpointer"] == "memory":
        log("warn", "MemorySaver: paused runs will not survive a restart")

    email = f"smoke-{int(time.time())}@example.io"
    signup = client.post(
        "/auth/signup",
        json={"email": email, "password": "password123", "name": "Smoke Test", "company": f"Smoke {int(time.time())}"},
    )
    signup.raise_for_status()
    token = signup.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    log("signup", email)

    path = Path(args.csv)
    with path.open("rb") as handle:
        upload = client.post("/files/datasets", headers=headers, files={"file": (path.name, handle)})
    upload.raise_for_status()
    dataset = upload.json()
    log("dataset", f"{dataset['dataset_id']} rows={dataset['n_rows']} cols={dataset['columns']}")

    created = client.post(
        "/tests",
        headers=headers,
        json={
            "name": f"Smoke {path.stem}",
            "hypothesis": args.hypothesis,
            "dataset_id": dataset["dataset_id"],
            "group_col": args.group_col,
            "metric_col": args.metric_col,
            "id_col": args.id_col,
        },
    )
    created.raise_for_status()
    test_id = created.json()["id"]
    log("test", test_id)

    status = _follow(client, headers, test_id, args.base_url, token)

    messages = client.get(f"/tests/{test_id}/messages", headers=headers).json()
    print("\n--- chat ---")
    for message in messages:
        print(f"{message['author']}: {message['text']}\n")

    final = client.get(f"/tests/{test_id}", headers=headers).json()
    print("--- result ---")
    print(json.dumps(final.get("results"), ensure_ascii=False, indent=2))

    if status != "done":
        log("FAIL", f"final status={status} error={final.get('error')}")
        return 1
    log("OK", "run finished")
    return 0


def _follow(client: httpx.Client, headers: dict, test_id: str, base_url: str, token: str) -> str:
    """Consume the SSE stream, answering interrupts, until the run ends."""
    while True:
        with client.stream("GET", f"/tests/{test_id}/stream", params={"token": token}) as response:
            response.raise_for_status()
            for event, data in sse_events(response):
                if event == "step.done":
                    log("step", data["step"])
                elif event == "message":
                    log("message", data["text"][:200])
                elif event == "interrupt":
                    payload = data["payload"]
                    choice = _auto_answer(payload)
                    log("interrupt", f"{payload.get('kind')} -> {choice}")
                    client.post(f"/tests/{test_id}/resume", headers=headers, json={"decision": choice}).raise_for_status()
                    # The resumed run publishes on the same channel; keep reading.
                elif event == "run.finished":
                    log("finished", data["status"])
                    return data["status"]
                elif event == "state":
                    log("state", data["status"])


def _auto_answer(payload: dict) -> dict:
    """Pick whatever the agent recommended — mimics a user clicking the
    highlighted option."""
    options = {o["method"]: o for o in payload.get("options", [])}
    recommended = payload.get("recommendation")
    chosen = options.get(recommended) or next(iter(options.values()), {"method": "none", "params": {}})
    return {"method": chosen["method"], "params": chosen.get("params", {})}


if __name__ == "__main__":
    sys.exit(main())
