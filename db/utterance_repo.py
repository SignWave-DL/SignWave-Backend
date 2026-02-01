from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Utterance, GlossToken


async def create_utterance(
    db: AsyncSession,
    session_id,
    audio_path: str,
    json_path: str,
    transcript: str,
    gloss: list[str],
) -> Utterance:
    u = Utterance(
        session_id=session_id,
        audio_path=audio_path,
        json_path=json_path,
        transcript=transcript,
    )
    db.add(u)
    await db.flush()  # to get u.id without committing yet

    tokens = [
        GlossToken(utterance_id=u.id, position=i, token=t)
        for i, t in enumerate(gloss)
    ]
    db.add_all(tokens)

    await db.commit()
    await db.refresh(u)
    return u


async def get_utterances_for_session(db: AsyncSession, session_id):
    stmt = select(Utterance).where(Utterance.session_id == session_id).order_by(Utterance.created_at.desc())
    res = await db.execute(stmt)
    return res.scalars().all()


async def get_gloss_tokens(db: AsyncSession, utterance_id):
    stmt = select(GlossToken).where(GlossToken.utterance_id == utterance_id).order_by(GlossToken.position.asc())
    res = await db.execute(stmt)
    return res.scalars().all()
