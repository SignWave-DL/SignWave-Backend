import whisper
import torch
import time

def start_listening():
    # Aggressive hardware verification
    print("--- AUDIO SYSTEM STARTUP ---")
    if torch.cuda.is_available():
        print(f"✅ POWER DETECTED: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️ ALERT: GPU NOT DETECTED. Switching to CPU mode (SLOW).")
        device = "cpu"

    # Load model "small" for quick testing, or "medium" for accuracy
    # Available models: tiny, base, small, medium, large
    print("Loading the brain (Whisper)...")
    model = whisper.load_model("medium", device=device)
    print("✅ Brain loaded and ready.")

    # Simulation: Replace this with the path to your real audio file
    audio_file = "test.m4a" 
    
    try:
        start_time = time.time()
        # The fp16=False argument is sometimes necessary if it bugs on Windows
        result = model.transcribe(audio_file, fp16=False, task="translate")
        end_time = time.time()
        
        print(f"\n🗣️ AI HAS HEARD ({end_time - start_time:.2f}s):")
        print(f"Operation completed. Text: \n>> {result['text'].strip()}")
    except Exception as e:
        print(f"ERROR: You forgot to put the audio file in the folder!")
        print(e)

if __name__ == "__main__":
    start_listening()
