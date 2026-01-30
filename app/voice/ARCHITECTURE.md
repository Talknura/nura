# Voice Pipeline Architecture

## Local Streaming Voice: VAD → STT → LLM → TTS

Target: **<500ms to first sentence heard** (not full response)

All components run locally - no cloud APIs required.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STREAMING VOICE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   MICROPHONE                                                                │
│      │                                                                      │
│      ▼                                                                      │
│   ┌─────────┐        ┌─────────┐                                           │
│   │   VAD   │ ─────▶ │   STT   │ (Faster Whisper)                          │
│   │ Silero  │        │         │                                           │
│   └─────────┘        └────┬────┘                                           │
│     ~1ms                  │ transcript (~150ms)                             │
│                           ▼                                                 │
│                    ┌─────────────┐                                          │
│                    │  BACKBONE   │ (Safety + Intent, parallel)             │
│                    │   <50ms     │                                          │
│                    └──────┬──────┘                                          │
│                           │                                                 │
│                           ▼                                                 │
│   ┌─────────┐     ┌──────────────┐     ┌─────────┐     ┌─────────┐         │
│   │   LLM   │ ──▶ │  SENTENCE    │ ──▶ │   TTS   │ ──▶ │ SPEAKER │         │
│   │  Phi    │     │   BUFFER     │     │ Kokoro  │     │         │         │
│   └─────────┘     └──────────────┘     └─────────┘     └─────────┘         │
│     tokens          sentences           audio           output              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Latency Breakdown

| Stage | Component | Target | Notes |
|-------|-----------|--------|-------|
| 1 | VAD | ~1ms | Silero VAD (ONNX) |
| 2 | STT | ~150ms | Faster Whisper base.en |
| 3 | Backbone | ~50ms | Safety + Intent parallel |
| 4 | LLM TTFT | ~100ms | Phi with GPU, greedy |
| 5 | Sentence Buffer | ~50ms | Wait for first sentence |
| 6 | TTS TTFA | ~100ms | Kokoro ONNX |
| **Total** | | **<500ms** | First sentence heard |

## Components

### 1. Voice Activity Detection (VAD)

Silero VAD detects when user starts and stops speaking.

```python
from app.voice import get_vad, VADConfig

vad = get_vad(VADConfig(
    sample_rate=16000,
    speech_threshold=0.5,
    min_silence_ms=500
))

# Process audio chunks
for chunk in audio_stream:
    event = vad.process_chunk(chunk)
    if event.event_type == "speech_end":
        # Process complete utterance
        process(event.audio)
```

**Features:**
- ~1ms per chunk (ONNX)
- Pre-speech audio buffering (300ms)
- Automatic utterance segmentation
- Configurable thresholds

### 2. Speech-to-Text (STT)

Faster Whisper for fast local transcription.

```python
from app.voice import FasterWhisperSTT

stt = FasterWhisperSTT(
    model_size="base.en",  # tiny.en, base.en, small.en
    device="cuda",
    compute_type="float16"
)

text = stt.transcribe_bytes(audio_bytes)
```

**Models:**

| Model | Size | Speed (GPU) | Accuracy |
|-------|------|-------------|----------|
| tiny.en | 39M | ~50ms | Good |
| base.en | 74M | ~100ms | Better |
| small.en | 244M | ~200ms | Best |

### 3. Text-to-Speech (TTS)

Kokoro TTS for natural local voice synthesis.

```python
from app.voice import get_kokoro_tts, TTSConfig

tts = get_kokoro_tts(TTSConfig(
    voice="af_heart",  # American Female - Heart
    speed=1.0,
    use_onnx=True
))

# Synthesize single sentence (fastest TTFA)
for chunk in tts.synthesize_sentence("Hello there!"):
    play_audio(chunk.audio)

# Or full text with sentence streaming
for chunk in tts.synthesize_stream("Hello. How are you today?"):
    play_audio(chunk.audio)
```

**Available Voices:**

| Voice ID | Description |
|----------|-------------|
| af_heart | American Female - Heart (warm) |
| af_bella | American Female - Bella |
| af_nicole | American Female - Nicole |
| am_adam | American Male - Adam |
| am_michael | American Male - Michael |
| bf_emma | British Female - Emma |
| bm_george | British Male - George |

### 4. Streaming Pipeline

Full pipeline with sentence-level streaming.

```python
from app.voice import get_streaming_pipeline, PipelineConfig

pipeline = get_streaming_pipeline(PipelineConfig(
    stt_model="base.en",
    tts_voice="af_heart",
    min_sentence_chars=20
))

# Set audio output callback
def on_audio(audio_bytes):
    speaker.play(audio_bytes)

pipeline.set_audio_callback(on_audio)

# Process audio
response, metrics = pipeline.process_audio(
    audio_bytes=recorded_audio,
    user_id=123
)

print(f"First sentence at: {metrics.tts_first_audio_ms}ms")
print(f"Response: {response}")
```

## Sentence Buffer Strategy

The key to <500ms first sentence is **not waiting for full LLM output**.

```
LLM Output:    "Hello. │ I'm here to help. │ What would you like..."
                       │                    │
                       ▼                    ▼
TTS Start:    First sentence       Second sentence
              synthesizes          synthesizes
              immediately          in background
                       │
                       ▼
User Hears:   Audio plays          More audio
              at ~450ms            continues
```

**Sentence Detection:**
- Looks for `.` `!` `?` followed by space or end
- Minimum 20 characters per sentence (avoids "Dr." etc.)
- Maximum 200 characters (force flush)

## Configuration

### Pipeline Config

```python
PipelineConfig(
    # VAD
    vad_sample_rate=16000,
    vad_chunk_ms=30,
    vad_speech_threshold=0.5,
    vad_silence_ms=500,

    # STT
    stt_model="base.en",
    stt_device="cuda",
    stt_compute_type="float16",

    # TTS
    tts_voice="af_heart",
    tts_speed=1.0,
    tts_use_onnx=True,

    # LLM
    llm_max_tokens=150,
    llm_temperature=0.6,

    # Sentence buffer
    min_sentence_chars=20,
    max_buffer_chars=200
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NURA_KOKORO_MODEL | Path to Kokoro ONNX model | Auto-download |
| NURA_KOKORO_VOICES | Path to voices file | Auto-download |
| NURA_KOKORO_LANG | Language code | "a" (American) |
| NURA_PHI_MODEL | Path to Phi GGML model | Required |

## Requirements

```txt
# Voice Pipeline
faster-whisper>=0.10.0
kokoro>=0.9.4
# or: kokoro-onnx>=0.4.0
soundfile>=0.12.0
sounddevice>=0.4.6
torch>=2.0.0
onnxruntime>=1.16.0
```

## Usage Examples

### Simple Voice Turn

```python
from app.voice import process_voice_turn

def on_audio(audio_bytes):
    # Play through speaker
    speaker.write(audio_bytes)

# Process recorded audio
response, metrics = process_voice_turn(
    audio_bytes=recorded_audio,
    user_id=123,
    on_audio=on_audio
)

print(f"Response: {response}")
print(f"First sentence latency: {metrics['tts_ttfa']}ms")
```

### Continuous Voice Streaming

```python
from app.voice import get_streaming_pipeline, get_vad

pipeline = get_streaming_pipeline()
vad = get_vad()

pipeline.set_audio_callback(speaker.write)

# Process microphone stream
for chunk in microphone.stream():
    event = vad.process_chunk(chunk)

    if event and event.event_type == "speech_end":
        response, metrics = pipeline.process_audio(
            event.audio,
            user_id=123
        )
        print(f"You said: {stt.transcribe_bytes(event.audio)}")
        print(f"Nura: {response}")
```

### With Backbone Integration

```python
from app.voice import get_streaming_pipeline
from app.integration import get_backbone

pipeline = get_streaming_pipeline()
backbone = get_backbone()

# The pipeline automatically uses backbone for:
# - Safety checks
# - Intent classification
# - Memory retrieval (for PAST_SELF_REFERENCE)
# - Async memory/adaptation writes

response, metrics = pipeline.process_audio(audio, user_id=123)

# Check backbone stats
print(backbone.get_stats())
```

## Performance Optimization

### For Fastest TTFA

1. Use `tiny.en` Whisper model (50ms vs 150ms)
2. Use ONNX backends (Kokoro ONNX, Silero ONNX)
3. Set `min_sentence_chars=15` (flush sentences faster)
4. Use GPU for LLM (`n_gpu_layers=-1`)

### For Best Quality

1. Use `small.en` Whisper model
2. Set `min_sentence_chars=30` (more complete sentences)
3. Use higher quality TTS voice

## File Structure

```
app/voice/
├── vad.py              # Silero VAD wrapper
├── kokoro_tts.py       # Kokoro TTS streaming
├── streaming_pipeline.py  # Full pipeline
├── __init__.py         # Module exports
└── ARCHITECTURE.md     # This file
```

## Design Principles

1. **Local First**: No cloud APIs, everything runs on device
2. **Sentence Streaming**: Don't wait for full response
3. **Parallel Operations**: VAD, backbone run in parallel where possible
4. **Lazy Loading**: Models load on first use
5. **Graceful Degradation**: Missing components don't break pipeline

## Frozen: January 2025

This architecture provides <500ms first-sentence latency with local models.

Sources:
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M)
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
