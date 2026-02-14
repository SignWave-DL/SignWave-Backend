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
    # logits: (B, T, V)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_probs, pred = torch.max(probs, dim=-1)  # (B, T)
    
    texts = []
    confidences = []
    
    for i in range(len(pred)):
        seq = pred[i].tolist()
        seq_probs = max_probs[i].tolist()
        
        prev = None
        chars = []
        char_probs = []
        
        for idx, p in enumerate(seq):
            if p != blank_id and p != prev:
                chars.append(itos[p])
                char_probs.append(seq_probs[idx])
            prev = p
            
        text = "".join(chars).replace("  ", " ").strip()
        texts.append(text)
        
        # Calculate mean confidence for the sentence
        if len(char_probs) > 0:
            avg_conf = sum(char_probs) / len(char_probs)
        else:
            avg_conf = 0.0
        confidences.append(avg_conf)
        
    return texts, confidences

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

        self.fe.to(DEVICE)
        self.model.to(DEVICE).eval()
        
        # Extract and display model info
        self.accuracy = ckpt.get("accuracy", None)
        self.test_accuracy = ckpt.get("test_accuracy", None)
        self.val_accuracy = ckpt.get("val_accuracy", None)
        self.train_accuracy = ckpt.get("train_accuracy", None)
        self.epoch = ckpt.get("epoch", None)
        self.loss = ckpt.get("loss", None)
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n" + "="*60)
        print("🤖 CTC MODEL INFO")
        print("="*60)
        print(f"📍 Device: {DEVICE}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"🔢 Vocabulary Size: {len(vocab)}")
        print(f"🎵 Sample Rate: {self.sample_rate} Hz")
        print(f"📊 Mel Features: {self.n_mels}")
        print(f"🧠 Hidden Units: {hidden}")
        print(f"📚 LSTM Layers: {layers}")
        print(f"⚙️  Total Parameters: {total_params:,}")
        print(f"🎯 Trainable Parameters: {trainable_params:,}")
        
        if self.epoch is not None:
            print(f"📈 Epoch: {self.epoch}")
        if self.loss is not None:
            print(f"📉 Loss: {self.loss:.4f}")
        if self.accuracy is not None:
            print(f"✅ Accuracy: {self.accuracy:.2%}")
        if self.train_accuracy is not None:
            print(f"🏋️  Train Accuracy: {self.train_accuracy:.2%}")
        if self.val_accuracy is not None:
            print(f"✔️  Val Accuracy: {self.val_accuracy:.2%}")
        if self.test_accuracy is not None:
            print(f"🎯 Test Accuracy: {self.test_accuracy:.2%}")
        
        print("="*60 + "\n")

    @torch.no_grad()
    def transcribe(self, wav: torch.Tensor, sr: int) -> tuple[str, float]:
        wav = wav.to(DEVICE)
        feat = self.fe(wav, sr)
        x = feat.unsqueeze(0)
        logits = self.model(x).cpu()
        
        texts, confidences = greedy_decode(logits, self.itos, blank_id=self.blank_id)
        return texts[0], confidences[0]
