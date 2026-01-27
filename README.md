# SignWave Backend

## Current State

The SignWave Backend is the core processing engine that handles audio transcription using OpenAI's Whisper model. The backend integrates local audio processing capabilities with GPU/CPU optimization to convert audio input into text output.

### Current Features

#### Audio Transcription Engine
- Local Whisper model integration for audio-to-text conversion
- GPU acceleration support with automatic CPU fallback
- Support for multiple audio formats (m4a, mp3, wav, flac, ogg, aac, etc.)
- Language translation capability (converts non-English speech to English text)
- Configurable model sizes (tiny, base, small, medium, large)
- Processing time measurement and reporting

#### Hardware Optimization
- Automatic GPU (CUDA) detection and utilization
- Intelligent fallback to CPU for systems without GPU
- Device-specific model loading
- Memory-efficient processing options

## Architecture

The backend follows a modular architecture designed for incremental feature additions:

```
SignWave Backend
├── Audio Input Processing (Whisper)
├── Hardware Management (GPU/CPU)
└── Text Output
```

Each feature is designed to be independent and composable, allowing new components to be integrated seamlessly.

## Installation & Setup

### Prerequisites
- Python 3.8+
- FFmpeg (required for audio processing)
- CUDA 11.8 (optional, for GPU support)

### Installation Steps

1. **Install FFmpeg**
   ```bash
   # Windows
   winget install "FFmpeg (Essentials Build)"
   
   # Linux
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU Support (Optional)**
   ```bash
   python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
   ```

## Configuration

### Environment Variables
Currently, the backend uses hardcoded configurations. Future versions will support environment-based configuration.

### Current Settings
- **Whisper Model**: `medium` (can be changed in code)
- **Task Mode**: `translate` (convert any language to English)
- **GPU Support**: Automatic detection

## Usage

### Basic Operation
```bash
python main.py
```

### Configuration Changes
Edit `main.py` to modify:
- Model size (line 19): Change `"medium"` to `"tiny"`, `"base"`, `"small"`, or `"large"`
- Input audio file (line 21): Update `audio_file = "test.m4a"` path
- Processing task (line 30): Change `task="translate"` to `task="transcribe"` to keep original language

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| openai-whisper | 20250625 | Audio transcription |
| torch | 2.7.1+cu118 | Deep learning framework |
| numpy | 2.3.5 | Numerical operations |
| torchaudio | 2.7.1+cu118 | Audio processing |
| FFmpeg | Latest | Audio format conversion |

## Performance Characteristics

### Processing Speed
- **GPU (CUDA)**: 2-10x faster than CPU
- **CPU**: Slower but functional

### Memory Requirements
- **tiny model**: ~1GB
- **base model**: ~1.5GB
- **small model**: ~2GB
- **medium model**: ~5GB (current default)
- **large model**: ~10GB

## Future Features

New features will be added incrementally to this backend. Each addition will:
1. Extend current capabilities
2. Maintain backward compatibility where possible
3. Be documented in this README with implementation details
4. Include configuration options for customization

### Planned Enhancements
- Real-time audio streaming support
- Batch processing for multiple files
- REST API endpoints
- Database integration for result storage
- Advanced filtering and post-processing
- Parallel processing capabilities

## Output Format

Current output includes:
- Processing status messages
- Hardware device information
- Transcribed/translated text
- Processing duration in seconds

Example:
```
--- AUDIO SYSTEM STARTUP ---
✅ POWER DETECTED: NVIDIA GeForce RTX 3080
Loading the brain (Whisper)...
✅ Brain loaded and ready.

🗣️ AI HAS HEARD (12.45s):
Operation completed. Text: 
>> [transcribed/translated text here]
```

## Troubleshooting

### Common Issues

**FFmpeg not found**
- Ensure FFmpeg is properly installed and added to system PATH
- Restart terminal/IDE after installation

**Audio file not found**
- Verify the audio file exists in the correct directory
- Check the filename in the `audio_file` variable matches actual file

**GPU not detected**
- System will automatically fall back to CPU
- For GPU support, ensure NVIDIA drivers are installed
- Verify CUDA 11.8 compatibility with your GPU

**Memory errors**
- Use a smaller Whisper model (tiny, base, or small)
- Reduce audio file length
- Close other memory-intensive applications

## Directory Structure

```
SignWave-Backend/
├── main.py              # Main backend script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Contributing

When adding new features to the backend:
1. Update `requirements.txt` with any new dependencies
2. Document the new feature in this README
3. Maintain the modular architecture
4. Test with both GPU and CPU
5. Update example configurations

## License

Part of the SignWave project. Follow the main project license.
