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


class ResultRow(BaseModel):
    """Одна строка итоговой таблицы — одна метрика."""

    metric: str
    is_primary: bool = False
    control_group: str | None = None
    treatment_group: str | None = None
    control_value: float | None = None
    treatment_value: float | None = None
    n_control: int | None = None
    n_treatment: int | None = None
    absolute_diff: float | None = None
    relative_diff: float | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    relative_ci_low: float | None = None
    relative_ci_high: float | None = None
    significant: bool | None = None
    method: str | None = None
    warnings: list[str] = []


class CheckResult(BaseModel):
    """Одна проверка пайплайна: ok / warning / failed / skipped."""

    name: str
    status: Literal["ok", "warning", "failed", "skipped"]
    detail: str = ""


class Verdict(BaseModel):
    code: str
    label: str
    action: str
    metric: str | None = None
    relative_diff: float | None = None
    p_value: float | None = None
    blocking_checks: list[str] = []
    caveats: list[str] = []


class TestResults(BaseModel):
    rows: list[ResultRow] = []
    checks: list[CheckResult] = []
    verdict: Verdict | None = None
    # Компактная сводка для колонки "Результаты" в списке тестов.
    short: str = ""
    srm_detected: bool = False
    correction_applied: str | None = None
    power_verdict: str | None = None
    timeline_warnings: list[str] = []
    guardrail_violations: list[str] = []
    # Полный вывод графа для детального разбора.
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
    # Plotly-спеки; отдаются только на детальном эндпоинте.
    charts: list[dict[str, Any]] | None = None
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
    metric_cols: list[str] = []
    id_col: str | None = None
    timestamp_col: str | None = None
    # Явные роли групп: без них направление эффекта определяется по названиям.
    control_group: str | None = None
    treatment_group: str | None = None
    guardrail_specs: list[dict[str, Any]] = []
    ratio_metric_specs: list[dict[str, Any]] = []


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
