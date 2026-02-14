import os
import subprocess
import shutil
import numpy as np
import whisper

POSSIBLE_FFMPEG_PATHS = [
    r"C:\Users\takie\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-essentials_build\bin",
    r"C:\Users\jfvdk\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
]

ffmpeg_found = False
for ffmpeg_path in POSSIBLE_FFMPEG_PATHS:
    ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg.exe")
    if os.path.exists(ffmpeg_exe):
        os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
        print(f"✅ FFmpeg found: {ffmpeg_exe}")
        ffmpeg_found = True
        break

if not ffmpeg_found:
    print("⚠️ Warning: FFmpeg not found in known paths!")

_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
print(f"🔄 Loading Whisper model: {_MODEL_NAME}")
_model = whisper.load_model(_MODEL_NAME)
print(f"✅ Whisper model loaded successfully")

def _check_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found. PATH is not set correctly.")
    print(f"✅ FFmpeg available at: {ffmpeg_path}")
    return ffmpeg_path



def _webm_bytes_to_audio_array(webm_bytes: bytes) -> np.ndarray:
    """
    Decode WebM/Opus bytes to 16kHz mono float32 numpy array using ffmpeg (in memory).
    """
    print(f"🔄 Decoding audio with FFmpeg (input size: {len(webm_bytes)} bytes)...")
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

    try:
        process = subprocess.run(
            cmd,
            input=webm_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        
        if process.stderr:
            stderr_text = process.stderr.decode('utf-8', errors='ignore')
            if stderr_text.strip():
                print(f"⚠️ FFmpeg stderr: {stderr_text}")
        
        audio_int16 = np.frombuffer(process.stdout, dtype=np.int16)
        print(f"✅ Audio decoded: {len(audio_int16)} samples ({len(audio_int16)/16000:.2f} seconds)")
        
        if len(audio_int16) == 0:
            raise RuntimeError("FFmpeg produced empty audio output. Input might be invalid WebM.")
        
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32
    
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode('utf-8', errors='ignore')
        print(f"❌ FFmpeg error: {stderr_text}")
        raise RuntimeError(f"FFmpeg failed to decode audio: {stderr_text}")


def transcribe_bytes(audio_bytes: bytes, language: str = "en") -> tuple[str, float]:
    """
    Transcribe audio bytes (WebM) directly.
    """
    print(f"🎤 Transcribing {len(audio_bytes)} bytes (language: {language})...")
    
    if len(audio_bytes) < 100:
        raise RuntimeError(f"Audio data too small: {len(audio_bytes)} bytes. Recording might have failed.")
    
    audio = _webm_bytes_to_audio_array(audio_bytes)

    print(f"🔄 Running Whisper transcription...")
    result = _model.transcribe(
        audio,
        language=language,
        fp16=False
    )

    transcript = (result.get("text") or "").strip()
    
    # Calculate confidence from segments (avg_logprob)
    # Whisper returns avg_logprob, we convert to probability: exp(avg_logprob)
    confidence = 0.0
    segments = result.get("segments", [])
    if segments:
        avg_logprobs = [seg.get("avg_logprob", -1.0) for seg in segments]
        # Average probability across segments
        probs = [np.exp(lp) for lp in avg_logprobs]
        confidence = sum(probs) / len(probs)
    
    print(f"✅ Transcription result: '{transcript}' (Confidence: {confidence:.2%})")
    return transcript, confidence
