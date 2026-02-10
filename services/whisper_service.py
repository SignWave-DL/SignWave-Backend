import os
import subprocess
import shutil
import numpy as np
import whisper

# Load model once
_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
_model = whisper.load_model(_MODEL_NAME)

import os
import subprocess
import shutil
import numpy as np
import whisper

FFMPEG_BIN = r"C:\Users\jfvdk\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
_model = whisper.load_model(_MODEL_NAME)

def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. PATH is not set correctly.")



def _webm_bytes_to_audio_array(webm_bytes: bytes) -> np.ndarray:
    """
    Decode WebM/Opus bytes to 16kHz mono float32 numpy array using ffmpeg (in memory).
    """
    _check_ffmpeg()

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1",
    ]

    process = subprocess.run(
        cmd,
        input=webm_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    audio_int16 = np.frombuffer(process.stdout, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32


def transcribe_bytes(audio_bytes: bytes, language: str = "en") -> str:
    """
    Transcribe audio bytes (WebM) directly.
    """
    audio = _webm_bytes_to_audio_array(audio_bytes)

    result = _model.transcribe(
        audio,
        language=language,
        fp16=False
    )

    return (result.get("text") or "").strip()
