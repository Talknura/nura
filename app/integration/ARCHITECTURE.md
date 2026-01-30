# Nura Backbone Architecture

## Unified System: Voice + Engines + LLM

The Backbone Layer IS Nura - one unified system that handles everything from audio input to audio output.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NURA BACKBONE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   AUDIO IN ──▶ VAD ──▶ STT ──┐                                              │
│   (Microphone)  (Silero)  (Whisper)                                         │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    CRITICAL PATH (<50ms)                           │    │
│   │                                                                    │    │
│   │   ┌─────────────────────────────────────────────────────────┐     │    │
│   │   │     Safety        Intent        Temporal                │     │    │
│   │   │      ~1ms    +     ~5ms    +      ~6ms    = ~12ms      │     │    │
│   │   └─────────────────────────────────────────────────────────┘     │    │
│   │                              │                                    │    │
│   │                              ▼                                    │    │
│   │   ┌─────────────────────────────────────────────────────────┐     │    │
│   │   │              Retrieval (if PAST_SELF_REFERENCE)         │     │    │
│   │   │                        ~10-50ms                         │     │    │
│   │   └─────────────────────────────────────────────────────────┘     │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │              LLM + SENTENCE BUFFER + TTS STREAMING                 │    │
│   │                                                                    │    │
│   │   LLM tokens ──▶ Sentence Buffer ──▶ TTS ──▶ Audio Out            │    │
│   │                                                                    │    │
│   │   Target: <500ms to first sentence heard                          │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│                         AUDIO OUT ──▶ Speaker                               │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    ASYNC (After Response)                          │    │
│   │                                                                    │    │
│   │   Memory Write │ Adaptation │ Proactive │ HNSW Batch              │    │
│   │                        (non-blocking)                              │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Latency Breakdown

| Stage | Target | Component |
|-------|--------|-----------|
| VAD | ~1ms | Silero VAD (ONNX) |
| STT | ~150ms | Faster Whisper base.en |
| Safety + Intent | ~6ms | Parallel execution |
| Temporal | ~6ms | Time parsing |
| Retrieval | ~10-50ms | Only if needed |
| LLM TTFT | ~100ms | Phi with GPU |
| First Sentence | ~50ms | Buffer detection |
| TTS TTFA | ~100ms | Kokoro (ONNX) |
| **Total** | **<500ms** | First sentence heard |

## Usage

### Voice Turn (Primary Interface)

```python
from app.integration import get_backbone

backbone = get_backbone()

def on_audio(audio_bytes):
    speaker.write(audio_bytes)

result = backbone.process_voice_turn(
    audio_bytes=recorded_audio,
    user_id=123,
    on_audio=on_audio
)

print(f"User said: {result.transcript}")
print(f"Nura said: {result.response}")
print(f"First sentence at: {result.timing['tts_first_audio']}ms")
```

### Text Mode (Fallback)

```python
result = backbone.process(
    user_input="What did I tell you about my job?",
    user_id=123,
    llm_callable=my_llm_function
)

print(result.llm_output)
```

### Continuous VAD Streaming

```python
backbone = get_backbone()

for chunk in microphone.stream():
    event = backbone.process_vad_chunk(chunk)

    if event and event.event_type == "speech_end":
        result = backbone.process_voice_turn(
            event.audio,
            user_id=123,
            on_audio=speaker.write
        )
```

## Key Design: Sentence Streaming

The secret to <500ms first response is **not waiting for full LLM output**.

```
LLM generates:  "Hello. │ I'm Nura. │ How can I help?"
                        │            │
                        ▼            ▼
TTS starts:     First sentence   Second sentence
                immediately      in background
                        │
                        ▼
User hears:     Audio at ~450ms  More audio...
```

**Why sentences, not words?**
- Words are choppy and unnatural
- Sentences are complete thoughts
- Natural speech rhythm preserved

## All Components (Local, No Cloud)

| Component | Technology | Latency |
|-----------|------------|---------|
| VAD | Silero VAD (ONNX) | ~1ms/chunk |
| STT | Faster Whisper | ~150ms |
| Safety | Rule-based | ~1ms |
| Intent | Semantic + Fast Path | ~5ms |
| Temporal | Parser | ~6ms |
| Memory | SQLite + HNSW | async |
| Retrieval | HNSW search | ~10-50ms |
| Adaptation | Profile update | async |
| Proactive | Deferred eval | async |
| LLM | Phi (llama.cpp) | ~100ms TTFT |
| TTS | Kokoro (ONNX) | ~100ms TTFA |

## Async Operations

These run AFTER the response starts playing:

```python
# Automatically queued after voice turn
backbone.queue_async_operations(ctx, llm_output)

# Includes:
# - Memory write (if PERSONAL_STATE or EXPLICIT_MEMORY)
# - Adaptation update (if PERSONAL_STATE)
# - Proactive evaluation (deferred)
# - HNSW batch updates (every 5s or 10 items)
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| NURA_PHI_MODEL | Path to Phi GGML model |
| NURA_KOKORO_MODEL | Path to Kokoro ONNX (optional, auto-downloads) |
| NURA_KOKORO_VOICES | Path to voices file (optional) |

### Code Configuration

```python
# VAD settings (in backbone.py)
VADConfig(
    sample_rate=16000,
    speech_threshold=0.5,
    min_silence_ms=500
)

# TTS settings
TTSConfig(
    voice="af_heart",
    use_onnx=True
)
```

## File Structure

```
app/integration/
├── backbone.py          # THE Nura backbone (everything)
├── __init__.py          # Module exports
└── ARCHITECTURE.md      # This file

app/voice/               # Voice components (imported by backbone)
├── vad.py              # Silero VAD
├── kokoro_tts.py       # Kokoro TTS
└── streaming_pipeline.py  # Legacy (use backbone instead)
```

## Design Principles

1. **One System**: Voice + Engines + LLM are not separate - they're Nura
2. **Sentence Streaming**: TTS starts on first sentence, not full response
3. **Parallel Critical Path**: Safety and Intent run in parallel
4. **Async Everything Else**: Memory/Adaptation/Proactive after response
5. **Local First**: No cloud APIs, everything runs on device
6. **Graceful Degradation**: Missing components don't break the system

## Example: Full Voice Conversation

```python
from app.integration import get_backbone
import sounddevice as sd

backbone = get_backbone()

# Record 3 seconds
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='int16')
sd.wait()
audio_bytes = audio.tobytes()

# Process through Nura
def play_audio(chunk):
    # Queue audio for playback
    audio_queue.put(chunk)

result = backbone.process_voice_turn(
    audio_bytes=audio_bytes,
    user_id=1,
    on_audio=play_audio
)

print(f"Transcript: {result.transcript}")
print(f"Response: {result.response}")
print(f"Timing: {result.timing}")
# Timing: {'stt': 145.2, 'backbone': 48.3, 'llm_first_token': 95.1,
#          'llm_first_sentence': 142.8, 'tts_first_audio': 238.5, 'total': 512.3}
```

## Frozen: January 2025

This is the unified Nura backbone - Voice + Engines + LLM in one system.
