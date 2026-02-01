import os
import json
import uuid
from datetime import datetime

BASE_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
AUDIO_DIR = os.path.join(BASE_OUTPUT, "audio")
JSON_DIR = os.path.join(BASE_OUTPUT, "json")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)


def save_audio_bytes(audio_bytes: bytes, ext: str = "webm") -> str:
    """
    Saves raw bytes to outputs/audio/<uuid>.<ext>
    Returns path.
    """
    name = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(AUDIO_DIR, name)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def save_result_json(session_id: str, transcript: str, gloss: list[str]) -> str:
    """
    Saves outputs/json/<session_id>_<timestamp>.json
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = f"{session_id}_{ts}.json"
    path = os.path.join(JSON_DIR, name)

    payload = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "language": "en",
        "transcript": transcript,
        "gloss": gloss,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path
