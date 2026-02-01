from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.database import get_db_session
from db.models import Session
from db.utterance_repo import get_utterances_for_session, get_gloss_tokens

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/history")
async def history(db: AsyncSession = Depends(get_db_session), limit: int = 20):
    stmt = select(Session).order_by(Session.started_at.desc()).limit(limit)
    res = await db.execute(stmt)
    sessions = res.scalars().all()

    out = []
    for s in sessions:
        utterances = await get_utterances_for_session(db, s.id)
        u_out = []
        for u in utterances:
            tokens = await get_gloss_tokens(db, u.id)
            u_out.append({
                "id": str(u.id),
                "session_id": str(u.session_id),
                "created_at": u.created_at.isoformat(),
                "transcript": u.transcript,
                "gloss": [t.token for t in tokens],
                "audio_path": u.audio_path,
                "json_path": u.json_path,
            })

        out.append({
            "id": str(s.id),
            "language": s.language,
            "status": s.status,
            "started_at": s.started_at.isoformat(),
            "ended_at": s.ended_at.isoformat() if s.ended_at else None,
            "utterances": u_out,
        })

    return out


@router.get("/session/{session_id}")
async def session_detail(session_id: str, db: AsyncSession = Depends(get_db_session)):
    s = await db.get(Session, session_id)
    if not s:
        return {"error": "Session not found"}

    utterances = await get_utterances_for_session(db, s.id)
    u_out = []
    for u in utterances:
        tokens = await get_gloss_tokens(db, u.id)
        u_out.append({
            "id": str(u.id),
            "created_at": u.created_at.isoformat(),
            "transcript": u.transcript,
            "gloss": [t.token for t in tokens],
            "audio_path": u.audio_path,
            "json_path": u.json_path,
        })

    return {
        "id": str(s.id),
        "language": s.language,
        "status": s.status,
        "started_at": s.started_at.isoformat(),
        "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        "utterances": u_out,
    }
