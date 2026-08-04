"""Terminal harness to exercise the agent graph without any UI.

Usage:
    ANTHROPIC_API_KEY=... .venv/Scripts/python.exe scripts/run_cli.py \
        --csv tests/fixtures_sample_ab_test.csv --group-col group --metric-col revenue

If --group-col/--metric-col are omitted, the agent's own column detection
(heuristic + LLM fallback) is used instead.
"""

from __future__ import annotations

import argparse
import sys
import uuid

# Windows terminals often default to a non-UTF-8 codepage; force UTF-8 output
# so Cyrillic (and other non-ASCII) agent text doesn't come out as mojibake.
sys.stdout.reconfigure(encoding="utf-8")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.graph import build_graph
from app.nodes.data_loader import load_file
from app.state import empty_state


def _print_messages(result: dict) -> None:
    for msg in result.get("messages", []):
        content = getattr(msg, "content", None)
        if content:
            print(f"\n[agent] {content}")


def _handle_interrupt(graph, config, interrupt_payload: dict) -> dict:
    print(f"\n--- HITL: {interrupt_payload['kind']} ---")
    print(f"metric: {interrupt_payload['metric_col']}  outlier_share: {interrupt_payload['outlier_share']:.2%}")
    for i, opt in enumerate(interrupt_payload["options"]):
        flag = " <- recommended" if opt["method"] == interrupt_payload["recommendation"] else ""
        print(f"  [{i}] {opt['method']}: n_affected={opt['n_affected']} share={opt['share_affected']:.2%}{flag}")

    choice = input("Pick option index: ").strip()
    chosen = interrupt_payload["options"][int(choice)]
    return graph.invoke(
        Command(resume={"method": chosen["method"], "params": chosen["params"]}), config
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--metric-col", default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--company-id", default="_template")
    args = parser.parse_args()

    llm = ChatAnthropic(model="claude-sonnet-5")
    graph = build_graph(llm, guardrail_specs=[])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    df = load_file(args.csv)
    state = empty_state(args.company_id)
    state["raw_data"] = df
    state["group_col"] = args.group_col
    state["metric_col"] = args.metric_col
    state["id_col"] = args.id_col
    state["data_source"] = "file"
    state["messages"] = [HumanMessage(content="Загружен новый датасет, начни анализ.")]

    result = graph.invoke(state, config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        result = _handle_interrupt(graph, config, payload)

    _print_messages(result)

    print("\n--- follow-up questions (blank line to exit) ---")
    while True:
        text = input("> ").strip()
        if not text:
            break
        result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = _handle_interrupt(graph, config, payload)
        _print_messages(result)


if __name__ == "__main__":
    main()
