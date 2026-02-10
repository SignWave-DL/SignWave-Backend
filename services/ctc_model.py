import torch
import torch.nn as nn
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.sample_rate = sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, wav: torch.Tensor, sr: int):
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        feat = self.to_db(self.melspec(wav))  # (n_mels, frames)
        feat = feat.transpose(0, 1)           # (frames, n_mels)
        return feat

class CTCASR(nn.Module):
    def __init__(self, n_mels=80, hidden=256, layers=3, vocab_size=30, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)  # (B,T,V)

def greedy_decode(logits, itos, blank_id=0):
    pred = logits.argmax(dim=-1)
    texts = []
    for seq in pred:
        prev = None
        chars = []
        for p in seq.tolist():
            if p != blank_id and p != prev:
                chars.append(itos[p])
            prev = p
        texts.append("".join(chars).replace("  ", " ").strip())
    return texts

class CTCTranscriber:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        vocab = ckpt["vocab"]
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        self.blank_id = 0

        # If you saved extra config fields, use them; else default:
        self.sample_rate = ckpt.get("sample_rate", 16000)
        self.n_mels = ckpt.get("n_mels", 80)
        hidden = ckpt.get("hidden", 256)
        layers = ckpt.get("layers", 3)

        self.fe = FeatureExtractor(sample_rate=self.sample_rate, n_mels=self.n_mels)
        self.model = CTCASR(n_mels=self.n_mels, hidden=hidden, layers=layers, vocab_size=len(vocab))

        # support both save formats
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt["model"]
        self.model.load_state_dict(state)

        self.model.to(DEVICE).eval()

    @torch.no_grad()
    def transcribe(self, wav: torch.Tensor, sr: int) -> str:
        feat = self.fe(wav, sr)
        x = feat.unsqueeze(0).to(DEVICE)     # (1,T,F)
        logits = self.model(x).cpu()         # (1,T,V)
        return greedy_decode(logits, self.itos, blank_id=self.blank_id)[0]
