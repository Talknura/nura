"""
Voice Module - Local Streaming Voice Pipeline.

Full local voice pipeline: VAD → STT → LLM → TTS
Target: <500ms to first sentence heard

Components:
- VAD: Silero VAD (voice activity detection)
- STT: Faster Whisper (speech-to-text)
- LLM: Phi/Qwen streaming (language model)
- TTS: Kokoro (text-to-speech)

All components run locally, no cloud APIs required.
"""

from app.voice.vad import (
    VADConfig,
    VADEvent,
    SileroVAD,
    VADStream,
    get_vad,
)

from app.voice.kokoro_tts import (
    TTSConfig,
    TTSChunk,
    KokoroTTS,
    StreamingTTSBuffer,
    get_kokoro_tts,
    KOKORO_VOICES,
)

from app.voice.streaming_pipeline import (
    PipelineConfig,
    PipelineMetrics,
    StreamingVoicePipeline,
    FasterWhisperSTT,
    SentenceBuffer,
    get_streaming_pipeline,
    process_voice_turn,
)

__all__ = [
    # VAD
    "VADConfig",
    "VADEvent",
    "SileroVAD",
    "VADStream",
    "get_vad",
    # TTS
    "TTSConfig",
    "TTSChunk",
    "KokoroTTS",
    "StreamingTTSBuffer",
    "get_kokoro_tts",
    "KOKORO_VOICES",
    # Pipeline
    "PipelineConfig",
    "PipelineMetrics",
    "StreamingVoicePipeline",
    "FasterWhisperSTT",
    "SentenceBuffer",
    "get_streaming_pipeline",
    "process_voice_turn",
]
