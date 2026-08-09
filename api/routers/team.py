"""Team listing and invites. Invites create a user without a usable password
— the real invite-email flow is a later iteration."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from api.deps import SessionDep, UserDep
from api.models import User
from api.schemas import InviteRequest, TeamMemberOut
from api.security import hash_password

router = APIRouter(prefix="/team", tags=["team"])


@router.get("", response_model=list[TeamMemberOut])
def list_team(user: UserDep, session: SessionDep) -> list[TeamMemberOut]:
    rows = session.scalars(select(User).where(User.company_id == user.company_id).order_by(User.created_at)).all()
    return [
        TeamMemberOut(id=r.id, name=r.name or r.email.split("@")[0], email=r.email, role=r.role)  # type: ignore[arg-type]
        for r in rows
    ]


@router.post("", response_model=TeamMemberOut, status_code=status.HTTP_201_CREATED)
def invite(payload: InviteRequest, user: UserDep, session: SessionDep) -> TeamMemberOut:
    if user.role != "Admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "only an Admin can invite")
    if session.scalar(select(User).where(User.email == str(payload.email))) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")

    invited = User(
        company_id=user.company_id,
        email=str(payload.email),
        # Unusable until the invitee sets their own password.
        password_hash=hash_password(uuid.uuid4().hex),
        role=payload.role,
    )
    session.add(invited)
    session.commit()
    return TeamMemberOut(id=invited.id, name=invited.email.split("@")[0], email=invited.email, role=invited.role)  # type: ignore[arg-type]
