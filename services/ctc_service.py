import io
import os
import subprocess
import tempfile
import torch
import soundfile as sf

from services.ctc_model import CTCTranscriber  # you’ll create this (see next section)

# Load once (IMPORTANT) so we don't reload model on every request
_ctc = None

def get_ctc():
    global _ctc
    if _ctc is None:
        # path to your saved checkpoint
        _ctc = CTCTranscriber(r"C:\Users\jfvdk\Desktop\Git\Deep-Learning\SignWave-Backend\services\checkpoints\ctc_asr.pt")

    return _ctc

def _webm_bytes_to_waveform(audio_bytes: bytes):
    """
    Convert webm/opus bytes -> (wav_tensor, sr) using ffmpeg.
    Requires ffmpeg installed and on PATH.
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as fin:
        fin.write(audio_bytes)
        in_path = fin.name

    out_path = in_path.replace(".webm", ".wav")

    try:
        # Convert to 16k mono wav
        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-ac", "1",
            "-ar", "16000",
            out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        data, sr = sf.read(out_path, dtype="float32")  # (T,) or (T,C)
        if data.ndim == 2:
            data = data.mean(axis=1)

        wav = torch.from_numpy(data)  # (T,)
        return wav, sr

    finally:
        # cleanup temp files
        try:
            os.remove(in_path)
        except:
            pass
        try:
            os.remove(out_path)
        except:
            pass

def transcribe_bytes_ctc(audio_bytes: bytes, language="en") -> str:
    """
    Transcribe audio bytes using your CTC model.
    (language param kept for compatibility; your model may ignore it)
    """
    wav, sr = _webm_bytes_to_waveform(audio_bytes)
    model = get_ctc()
    return model.transcribe(wav, sr)
