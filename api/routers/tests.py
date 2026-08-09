"""Tests = chats. Everything the main screen and the "All tests" table need.

Reads are plain SQL against the product tables — listing tests never wakes
the graph up. Only `/messages` and `/resume` start a run.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from api import events, runner
from api.deps import SessionDep, TestDep, UserDep
from api.models import Message, Test
from api.schemas import MessageOut, NewTestRequest, ResumeRequest, SendMessageRequest, TestOut
from api.serializers import initials, message_out, test_out

router = APIRouter(prefix="/tests", tags=["tests"])

SSE_KEEPALIVE_SECONDS = 15.0


@router.get("", response_model=list[TestOut])
def list_tests(user: UserDep, session: SessionDep) -> list[TestOut]:
    rows = session.scalars(
        select(Test).where(Test.company_id == user.company_id).order_by(Test.created_at.desc())
    ).all()
    return [test_out(row) for row in rows]


@router.post("", response_model=TestOut, status_code=status.HTTP_201_CREATED)
def create_test(payload: NewTestRequest, user: UserDep, session: SessionDep) -> TestOut:
    test = Test(
        company_id=user.company_id,
        created_by=user.id,
        name=payload.name.strip() or "Без названия",
        hypothesis=payload.hypothesis.strip(),
        dataset_id=payload.dataset_id,
        status="queued",
        config={
            "test_type": payload.test_type,
            "groups": payload.groups,
            "tracker": payload.tracker,
            "segment": payload.segment,
            "start_date": payload.start_date,
            "end_date": payload.end_date,
            "group_col": payload.group_col,
            "metric_col": payload.metric_col,
            "id_col": payload.id_col,
        },
    )
    session.add(test)
    session.commit()

    # Анализ стартует сам. В чате показываем гипотезу пользователя, а в граф
    # уходит явная инструкция — иначе роутер может счесть её вопросом.
    if test.dataset_id:
        if test.hypothesis:
            session.add(
                Message(test_id=test.id, role="user", author=user.name or user.email, text=test.hypothesis)
            )
            session.commit()
        runner.start_run(test.id, runner.build_initial_state(test, runner.opening_prompt(test)))

    return test_out(test)


@router.get("/{test_id}", response_model=TestOut)
def get_test(test: TestDep) -> TestOut:
    return test_out(test)


@router.get("/{test_id}/messages", response_model=list[MessageOut])
def get_messages(test: TestDep, user: UserDep, session: SessionDep) -> list[MessageOut]:
    rows = session.scalars(
        select(Message).where(Message.test_id == test.id).order_by(Message.created_at)
    ).all()
    user_initials = initials(user.name, user.email)
    return [message_out(row, user_initials) for row in rows]


@router.post("/{test_id}/messages", response_model=MessageOut, status_code=status.HTTP_202_ACCEPTED)
def send_message(
    payload: SendMessageRequest, test: TestDep, user: UserDep, session: SessionDep
) -> MessageOut:
    if test.status == "analyzing":
        raise HTTPException(status.HTTP_409_CONFLICT, "a run is already in progress")
    if test.status == "awaiting_input":
        raise HTTPException(status.HTTP_409_CONFLICT, "answer the pending question first")
    if not test.dataset_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "test has no dataset attached")

    row = Message(test_id=test.id, role="user", author=user.name or user.email, text=payload.text)
    session.add(row)
    session.commit()

    runner.start_run(test.id, runner.build_turn_input(test, payload.text))
    return message_out(row, initials(user.name, user.email))


@router.post("/{test_id}/resume", status_code=status.HTTP_202_ACCEPTED)
def resume(payload: ResumeRequest, test: TestDep, session: SessionDep) -> dict:
    """Answer a pending HITL interrupt (e.g. the chosen outlier treatment)."""
    if test.status != "awaiting_input":
        raise HTTPException(status.HTTP_409_CONFLICT, "no pending question")

    test.pending_interrupt = None
    session.add(test)
    session.commit()

    runner.resume_run(test.id, payload.decision)
    return {"status": "accepted"}


@router.get("/{test_id}/stream")
async def stream(test: TestDep) -> StreamingResponse:
    """SSE feed of the run: step.done, message, interrupt, run.finished."""
    queue = events.subscribe(test.id)

    async def event_source() -> AsyncIterator[bytes]:
        try:
            # Initial frame so a late subscriber renders current state immediately.
            yield _sse("state", test_out(test).model_dump(mode="json"))
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                yield _sse(payload["event"], payload["data"])
        finally:
            events.unsubscribe(test.id, queue)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n".encode("utf-8")
