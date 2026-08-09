"""Wire contracts. Field names mirror `verdict front/app/src/types.ts` so the
frontend can drop its mock layer without renaming anything."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field

Role = Literal["Admin", "Analyst", "Product", "Marketer", "Other"]
TestStatus = Literal["queued", "analyzing", "awaiting_input", "clarifying", "done", "failed"]


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str = ""
    company: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    role: Role
    initials: str
    company_id: str
    onboarded: bool


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class GroupResult(BaseModel):
    group: str
    conversion: str
    delta: str
    good: bool | None = None


class TestResults(BaseModel):
    groups: list[GroupResult] = []
    short: str = ""
    # Full graph output for the detailed view; the two fields above are the
    # compact form the tests table renders.
    raw: dict[str, Any] | None = None


class TestOut(BaseModel):
    id: str
    name: str
    hypothesis: str
    status: TestStatus
    decision: str
    date: str
    dataset_id: str | None = None
    results: TestResults | None = None
    pending_interrupt: dict[str, Any] | None = None
    error: str | None = None


class NewTestRequest(BaseModel):
    name: str
    hypothesis: str = ""
    test_type: str = ""
    groups: str = ""
    tracker: str = ""
    segment: str = ""
    start_date: str = ""
    end_date: str = ""
    dataset_id: str | None = None
    group_col: str | None = None
    metric_col: str | None = None
    id_col: str | None = None


class MessageOut(BaseModel):
    id: str
    role: Literal["agent", "user"]
    author: str
    text: str
    initials: str | None = None
    results: dict[str, Any] | None = None


class SendMessageRequest(BaseModel):
    text: str


class ResumeRequest(BaseModel):
    """Answer to a pending HITL interrupt — e.g. the chosen outlier option."""

    decision: dict[str, Any]


class DatasetOut(BaseModel):
    dataset_id: str
    n_rows: int
    columns: list[str]


class TeamMemberOut(BaseModel):
    id: str
    name: str
    email: str
    role: Role


class InviteRequest(BaseModel):
    email: EmailStr
    role: Role = "Analyst"
