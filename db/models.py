import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import schema

class Session(schema):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    language: Mapped[str] = mapped_column(String(8), default="en")
    status: Mapped[str] = mapped_column(String(20), default="recording")

    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    utterances: Mapped[list["Utterance"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Utterance(schema):
    __tablename__ = "utterances"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"))

    audio_path: Mapped[str] = mapped_column(String(500))
    transcript: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped["Session"] = relationship(back_populates="utterances")
    gloss_tokens: Mapped[list["GlossToken"]] = relationship(back_populates="utterance", cascade="all, delete-orphan")


class GlossToken(schema):
    __tablename__ = "gloss_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    utterance_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("utterances.id", ondelete="CASCADE"))

    position: Mapped[int] = mapped_column(Integer)
    token: Mapped[str] = mapped_column(String(64))

    utterance: Mapped["Utterance"] = relationship(back_populates="gloss_tokens")
