"""Non-interactive batch runner: runs the full A/B pipeline once per metric
column against a fixed group/id column, printing the final agent message
and key intermediate results for each metric."""

from __future__ import annotations

import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.graph import build_graph
from app.nodes.data_loader import load_file
from app.state import empty_state

CSV = "tests/t_2845_ab_test_Moscow_2weeks_short_rents_winsorized.csv"
GROUP_COL = "model_version"
ID_COL = "car_id"
METRICS = [
    "total_car_revenue",
    "total_distance_travelled",
    "car_exploitation_time_hour",
    "total_est_margin",
    "car_ava_time_hour",
    "total_car_revenue_per_hour_in_exploitation",
]


def run_one(graph, df, metric: str) -> None:
    print(f"\n{'=' * 80}\nMETRIC: {metric}\n{'=' * 80}")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = empty_state("_template")
    state["raw_data"] = df
    state["group_col"] = GROUP_COL
    state["metric_col"] = metric
    state["id_col"] = ID_COL
    state["data_source"] = "file"
    state["messages"] = [HumanMessage(content="Загружен новый датасет, начни анализ.")]

    try:
        result = graph.invoke(state, config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            options = {o["method"]: o for o in payload["options"]}
            chosen = options[payload["recommendation"]]
            print(
                f"[HITL interrupt: {payload['kind']}] outlier_share={payload['outlier_share']:.2%} "
                f"-> auto-picking recommended method={chosen['method']} params={chosen['params']}"
            )
            result = graph.invoke(
                Command(resume={"method": chosen["method"], "params": chosen["params"]}), config
            )
        for msg in result.get("messages", []):
            content = getattr(msg, "content", None)
            if content:
                print(f"[agent] {content}")
        print(f"[srm_result] {result.get('srm_result')}")
        print(f"[recommendation] {result.get('recommendation')}")
        print(f"[test_result] {result.get('test_result')}")
        print(f"[guardrail_results] {result.get('guardrail_results')}")
        print(f"[needs_clarification] {result.get('needs_clarification')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}")


def main() -> None:
    llm = ChatAnthropic(model="claude-sonnet-5")
    graph = build_graph(llm, guardrail_specs=[])
    df = load_file(CSV)
    df = df.dropna(subset=[GROUP_COL, ID_COL])
    for metric in METRICS:
        run_one(graph, df, metric)


if __name__ == "__main__":
    main()
