from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

class WSResult(BaseModel):
    type: str = "result"
    session_id: uuid.UUID
    utterance_id: uuid.UUID
    transcript: str
    gloss: List[str]
    saved_audio_path: str

class UtteranceOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    transcript: str
    gloss: List[str]
    created_at: datetime
    audio_path: str

class SessionOut(BaseModel):
    id: uuid.UUID
    language: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    utterances: List[UtteranceOut]
