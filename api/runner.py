"""Runs the graph off the request thread and streams its progress.

A run is: feed input into the graph thread, forward every node update as an
SSE event, persist agent messages, and land the test in a terminal status
(`done`/`failed`) or in `awaiting_input` when the graph interrupts for HITL.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from api import events
from api.db import SessionLocal
from api.graph_runtime import get_graph, thread_config
from api.models import Message, Test
from api.serializers import jsonable
from app.state import STEP_ORDER

logger = logging.getLogger(__name__)

AGENT_AUTHOR = "Verdict AI"


def message_text(message: Any) -> str:
    """ChatAnthropic content is a string or a list of content blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def start_run(test_id: str, payload: Any) -> None:
    """Kick off a graph run in a worker thread.

    `payload` is either an initial state dict or a `Command(resume=...)`.
    """
    thread = threading.Thread(target=_run, args=(test_id, payload), daemon=True)
    thread.start()


def resume_run(test_id: str, decision: Any) -> None:
    start_run(test_id, Command(resume=decision))


def opening_prompt(test: Test) -> str:
    """Первая реплика, уходящая в граф.

    Одну гипотезу роутер принимает за вопрос и уводит сразу в insight, минуя
    анализ, — поэтому намерение задаётся явно, а гипотеза идёт контекстом.
    """
    hypothesis = (test.hypothesis or "").strip()
    prompt = "Загружен новый датасет, начни анализ."
    return f"{prompt} Гипотеза: {hypothesis}" if hypothesis else prompt


def build_initial_state(test: Test, user_text: str) -> dict:
    from app.state import empty_state

    state = empty_state(test.company_id)
    state["dataset_id"] = test.dataset_id
    state["data_source"] = "file"
    config = test.config or {}
    state["group_col"] = config.get("group_col")
    state["metric_col"] = config.get("metric_col")
    state["id_col"] = config.get("id_col")
    state["messages"] = [HumanMessage(content=user_text)]
    return state


def build_turn_input(test: Test, user_text: str) -> dict:
    """Вход графа для очередной реплики в уже существующем чате.

    Полное начальное состояние тут слать нельзя: его None-поля перезапишут
    результаты, лежащие в чекпоинте, и агент ответит "данных нет". Отправляем
    только сообщение — остальное граф возьмёт из своего треда.
    """
    snapshot = get_graph().get_state(thread_config(test.thread_id))
    if snapshot.values.get("dataset_id"):
        return {"messages": [HumanMessage(content=user_text)]}
    # Треда ещё нет (например, первый прогон не стартовал) — начинаем с нуля.
    return build_initial_state(test, user_text)


def _run(test_id: str, payload: Any) -> None:
    graph = get_graph()
    channel = test_id

    with SessionLocal() as session:
        test = session.get(Test, test_id)
        if test is None:
            return
        thread_id = test.thread_id
        _set_status(session, test, "analyzing", pending_interrupt=None, error=None)

    events.publish(channel, "run.started", {"test_id": test_id, "steps": list(STEP_ORDER)})
    config = thread_config(thread_id)
    interrupted = False
    last_node: str | None = None

    try:
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupted = True
                    _handle_interrupt(test_id, channel, update)
                    continue
                if not isinstance(update, dict):
                    continue

                last_node = node
                events.publish(channel, "step.done", {"step": node, "payload": jsonable(_publishable(update))})
                _persist_messages(test_id, channel, update.get("messages") or [])

        if not interrupted:
            _finish(test_id, channel, config, last_node)
    except Exception as exc:  # graph failures must surface, not vanish in a thread
        logger.exception("run failed for test %s", test_id)
        with SessionLocal() as session:
            test = session.get(Test, test_id)
            if test is not None:
                _set_status(session, test, "failed", error=str(exc))
        events.publish(channel, "run.finished", {"status": "failed", "error": str(exc)})


def _handle_interrupt(test_id: str, channel: str, update: Any) -> None:
    payload = update[0].value if isinstance(update, (list, tuple)) and update else update
    payload = jsonable(payload)
    with SessionLocal() as session:
        test = session.get(Test, test_id)
        if test is not None:
            _set_status(session, test, "awaiting_input", pending_interrupt=payload)
    events.publish(channel, "interrupt", {"test_id": test_id, "payload": payload})


def _finish(test_id: str, channel: str, config: dict, last_node: str | None) -> None:
    snapshot = get_graph().get_state(config).values

    # `clarify` — тоже терминальная нода графа, но она не завершает анализ,
    # а задаёт вопрос в чате. Показывать "Готово" в этом случае — врать.
    status = "clarifying" if last_node == "clarify" else "done"

    results = jsonable(
        {
            "test_result": snapshot.get("test_result"),
            "srm_result": snapshot.get("srm_result"),
            "recommendation": snapshot.get("recommendation"),
            "guardrail_results": snapshot.get("guardrail_results") or [],
        }
    )
    has_results = any(results[key] for key in ("test_result", "srm_result"))

    with SessionLocal() as session:
        test = session.get(Test, test_id)
        if test is not None:
            if has_results:
                test.results = results
            _set_status(session, test, status, pending_interrupt=None)
    events.publish(channel, "run.finished", {"status": status, "results": results if has_results else None})


def _persist_messages(test_id: str, channel: str, messages: list) -> int:
    """Store agent messages produced by a node and echo them to subscribers.

    Human messages are already persisted by the route that accepted them.
    """
    stored = 0
    with SessionLocal() as session:
        for message in messages:
            if isinstance(message, HumanMessage):
                continue
            text = message_text(message).strip()
            if not text:
                continue
            row = Message(test_id=test_id, role="agent", author=AGENT_AUTHOR, text=text)
            session.add(row)
            session.commit()
            events.publish(
                channel,
                "message",
                {"id": row.id, "role": "agent", "author": AGENT_AUTHOR, "text": text},
            )
            stored += 1
    return stored


def _set_status(session, test: Test, status: str, **fields: Any) -> None:
    test.status = status
    for key, value in fields.items():
        setattr(test, key, value)
    session.add(test)
    session.commit()


def _publishable(update: dict) -> dict:
    """Node updates carry LangChain message objects; drop them here — messages
    go out as their own `message` events."""
    return {k: v for k, v in update.items() if k != "messages"}
