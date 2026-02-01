from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Session


async def create_session(db: AsyncSession, language: str = "en") -> Session:
    s = Session(language=language, status="recording")
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return s


async def mark_session_done(db: AsyncSession, session_id) -> Session:
    s = await db.get(Session, session_id)
    if s is None:
        raise ValueError("Session not found")

    s.status = "done"
    s.ended_at = datetime.utcnow()
    await db.commit()
    await db.refresh(s)
    return s
