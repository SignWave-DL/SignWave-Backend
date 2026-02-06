import os
import torch
import whisper

_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

# Use GPU if available (RTX 3060)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = whisper.load_model(_MODEL_NAME, device=_device)

def transcribe(path: str, language: str = "en") -> str:
    """
    Whisper transcription. language='en' forces English.
    Uses GPU (CUDA) if available, otherwise falls back to CPU.
    """
    result = _model.transcribe(path, language=language, fp16=torch.cuda.is_available())
    return (result.get("text") or "").strip()
