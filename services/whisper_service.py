import os
import whisper

_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
_model = whisper.load_model(_MODEL_NAME)

def transcribe(path: str, language: str = "en") -> str:
    """
    Whisper transcription. language='en' forces English.
    fp16=False for CPU compatibility.
    """
    result = _model.transcribe(path, language=language, fp16=False)
    return (result.get("text") or "").strip()
