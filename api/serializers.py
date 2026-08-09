"""ORM rows -> wire shapes, including the compact result summary the tests
table renders."""

from __future__ import annotations

from typing import Any

from api.models import Message, Test, User
from api.schemas import GroupResult, MessageOut, TestOut, TestResults, UserOut


def initials(name: str, email: str) -> str:
    parts = [p for p in name.split() if p]
    if parts:
        return "".join(p[0].upper() for p in parts[:2])
    return email[:2].upper()


def user_out(user: User, onboarded: bool) -> UserOut:
    return UserOut(
        id=user.id,
        name=user.name or user.email.split("@")[0],
        email=user.email,
        role=user.role,  # type: ignore[arg-type]
        initials=initials(user.name, user.email),
        company_id=user.company_id,
        onboarded=onboarded,
    )


def _fmt_p(p_value: Any) -> str:
    if p_value is None:
        return ""
    return f"p={p_value:.3g}"


def summarize(results: dict[str, Any] | None) -> TestResults | None:
    """Squeeze the graph output into the two-line form the UI lists.

    SRM short-circuits the pipeline, so an SRM verdict is reported even when
    there is no stat test at all.
    """
    if not results:
        return None

    srm = results.get("srm_result") or {}
    if srm.get("has_srm"):
        return TestResults(groups=[], short="SRM: результат невалиден", raw=results)

    report = results.get("test_result") or {}
    if not report:
        return TestResults(groups=[], short="", raw=results)

    effect = report.get("effect") or {}
    lift = effect.get("relative_lift")
    lift_str = f"{lift * 100:+.1f}%" if isinstance(lift, (int, float)) else "—"
    p_str = _fmt_p(report.get("p_value"))
    significant = report.get("decision") == "significant"

    control = str(report.get("control_group", "control"))
    treatment = str(report.get("treatment_group", "treatment"))
    groups = [
        GroupResult(group=control, conversion="—", delta="—"),
        GroupResult(
            group=treatment,
            conversion="—",
            delta=", ".join(x for x in (lift_str, p_str) if x),
            good=significant and isinstance(lift, (int, float)) and lift > 0,
        ),
    ]
    short = ", ".join(x for x in (f"{treatment} {lift_str}", p_str) if x)
    return TestResults(groups=groups, short=short, raw=results)


def test_out(test: Test) -> TestOut:
    return TestOut(
        id=test.id,
        name=test.name,
        hypothesis=test.hypothesis,
        status=test.status,  # type: ignore[arg-type]
        decision=test.decision or "—",
        date=test.created_at.strftime("%d.%m.%Y"),
        dataset_id=test.dataset_id,
        results=summarize(test.results),
        pending_interrupt=test.pending_interrupt,
        error=test.error,
    )


def message_out(message: Message, author_initials: str | None = None) -> MessageOut:
    return MessageOut(
        id=message.id,
        role=message.role,  # type: ignore[arg-type]
        author=message.author,
        text=message.text,
        initials=author_initials if message.role == "user" else None,
        results=message.results,
    )
