"""
Streaming Voice Pipeline - VAD → STT → LLM → TTS

Target: <500ms to first sentence heard (not whole response)

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                      STREAMING VOICE PIPELINE                                │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │   MICROPHONE                                                                │
    │      │                                                                      │
    │      ▼                                                                      │
    │   ┌─────────┐        ┌─────────┐                                           │
    │   │   VAD   │ ─────▶ │   STT   │ (Faster Whisper streaming)                │
    │   └─────────┘        └────┬────┘                                           │
    │     ~1ms                  │ transcript                                      │
    │                           ▼                                                 │
    │                    ┌─────────────┐                                          │
    │                    │  BACKBONE   │ (Safety + Intent, parallel)             │
    │                    │   <50ms     │                                          │
    │                    └──────┬──────┘                                          │
    │                           │                                                 │
    │                           ▼                                                 │
    │   ┌─────────┐     ┌──────────────┐     ┌─────────┐     ┌─────────┐         │
    │   │   LLM   │ ──▶ │  SENTENCE    │ ──▶ │   TTS   │ ──▶ │ SPEAKER │         │
    │   │ Stream  │     │   BUFFER     │     │ Kokoro  │     │         │         │
    │   └─────────┘     └──────────────┘     └─────────┘     └─────────┘         │
    │     tokens          sentences           audio           output              │
    │                                                                             │
    │   ═══════════════════════════════════════════════════════════════════════  │
    │   Timeline: |--VAD--|--STT--|--Backbone--|--LLM+TTS streaming--|            │
    │             0      30ms    200ms        250ms                  500ms+       │
    │                                                    ▲                        │
    │                                        First sentence heard                 │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

Key Optimizations:
1. VAD detects speech end immediately (~1ms)
2. STT streams final result fast (~150ms)
3. Backbone runs Safety+Intent in parallel (~50ms)
4. LLM streams tokens
5. TTS synthesizes first sentence immediately (not full response)
6. Audio plays while LLM continues generating
"""

from __future__ import annotations

import os
import re
import time
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional, Callable, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

# Voice components
from app.voice.vad import SileroVAD, VADConfig, VADEvent, get_vad
from app.voice.kokoro_tts import KokoroTTS, TTSConfig, TTSChunk, get_kokoro_tts

# STT (Faster Whisper)
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[Pipeline] faster-whisper not available")

# Backbone integration
try:
    from app.integration.backbone import get_backbone, BackboneContext
    BACKBONE_AVAILABLE = True
except ImportError:
    BACKBONE_AVAILABLE = False
    print("[Pipeline] backbone not available")

# LLM streaming
try:
    from app.services.streaming_llm import StreamingLLM, LLMChunk
    from app.services.optimized_llm import FastPhiStreamingLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[Pipeline] streaming LLM not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for streaming voice pipeline."""
    # VAD settings
    vad_sample_rate: int = 16000
    vad_chunk_ms: int = 30
    vad_speech_threshold: float = 0.5
    vad_silence_ms: int = 500

    # STT settings
    stt_model: str = "base.en"  # tiny.en, base.en, small.en
    stt_device: str = "cuda"  # cuda, cpu
    stt_compute_type: str = "float16"  # float16, int8, float32

    # TTS settings
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    tts_use_onnx: bool = True

    # LLM settings
    llm_max_tokens: int = 150
    llm_temperature: float = 0.6

    # Pipeline settings
    min_sentence_chars: int = 20  # Minimum chars before considering sentence complete
    max_buffer_chars: int = 200  # Force flush buffer at this size


@dataclass
class PipelineMetrics:
    """Track pipeline latencies."""
    vad_start_ms: float = 0.0
    vad_end_ms: float = 0.0
    stt_start_ms: float = 0.0
    stt_end_ms: float = 0.0
    backbone_ms: float = 0.0
    llm_first_token_ms: float = 0.0
    llm_first_sentence_ms: float = 0.0
    tts_first_audio_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "vad": self.vad_end_ms - self.vad_start_ms,
            "stt": self.stt_end_ms - self.stt_start_ms,
            "backbone": self.backbone_ms,
            "llm_ttft": self.llm_first_token_ms,
            "llm_first_sentence": self.llm_first_sentence_ms,
            "tts_ttfa": self.tts_first_audio_ms,
            "total": self.total_ms,
            "target": "< 500ms"
        }


# =============================================================================
# FASTER WHISPER STT
# =============================================================================

class FasterWhisperSTT:
    """
    Faster Whisper STT with streaming output.

    Uses CTranslate2 for fast inference on GPU/CPU.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure_loaded(self):
        """Lazy load model."""
        if self._model is not None:
            return

        if not WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper not installed")

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        print(f"[STT] Faster Whisper loaded: {self.model_size} on {self.device}")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples (float32, normalized)
            sample_rate: Sample rate (must be 16000 for Whisper)

        Returns:
            Transcribed text
        """
        self._ensure_loaded()

        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if sample_rate != 16000:
            # Resample if needed
            import scipy.signal
            audio = scipy.signal.resample(
                audio,
                int(len(audio) * 16000 / sample_rate)
            )

        # Transcribe
        segments, info = self._model.transcribe(
            audio,
            beam_size=1,  # Faster
            language="en",
            vad_filter=False,  # We already have VAD
            without_timestamps=True
        )

        # Collect text
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe from raw PCM bytes (16-bit mono)."""
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.transcribe(audio, sample_rate)


# =============================================================================
# SENTENCE BUFFER
# =============================================================================

class SentenceBuffer:
    """
    Buffer LLM tokens and detect complete sentences.

    Flushes sentences to TTS as soon as they're complete,
    enabling low-latency first-sentence output.
    """

    def __init__(
        self,
        min_chars: int = 20,
        max_chars: int = 200
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""
        self._flushed_sentences: List[str] = []

    def add(self, text: str) -> Optional[str]:
        """
        Add text to buffer.

        Returns:
            Complete sentence if available, None otherwise
        """
        self._buffer += text

        # Check for sentence boundary
        sentence = self._extract_sentence()
        if sentence:
            self._flushed_sentences.append(sentence)
            return sentence

        # Force flush if buffer too large
        if len(self._buffer) >= self.max_chars:
            forced = self._buffer
            self._buffer = ""
            self._flushed_sentences.append(forced)
            return forced

        return None

    def _extract_sentence(self) -> Optional[str]:
        """Extract first complete sentence from buffer."""
        # Match sentence ending with . ! ? followed by space or end
        match = re.search(r'^(.*?[.!?])(?:\s+|$)', self._buffer)
        if match and len(match.group(1)) >= self.min_chars:
            sentence = match.group(1).strip()
            self._buffer = self._buffer[match.end():].lstrip()
            return sentence
        return None

    def flush(self) -> Optional[str]:
        """Flush remaining buffer content."""
        if self._buffer.strip():
            remaining = self._buffer.strip()
            self._buffer = ""
            return remaining
        return None

    @property
    def sentences_count(self) -> int:
        """Number of sentences flushed."""
        return len(self._flushed_sentences)


# =============================================================================
# STREAMING VOICE PIPELINE
# =============================================================================

class StreamingVoicePipeline:
    """
    Full streaming voice pipeline: VAD → STT → LLM → TTS

    Target: <500ms to first sentence heard
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize components
        self.vad = SileroVAD(VADConfig(
            sample_rate=self.config.vad_sample_rate,
            chunk_size_ms=self.config.vad_chunk_ms,
            speech_threshold=self.config.vad_speech_threshold,
            min_silence_ms=self.config.vad_silence_ms
        ))

        self.stt = FasterWhisperSTT(
            model_size=self.config.stt_model,
            device=self.config.stt_device,
            compute_type=self.config.stt_compute_type
        )

        self.tts = get_kokoro_tts(TTSConfig(
            voice=self.config.tts_voice,
            speed=self.config.tts_speed,
            use_onnx=self.config.tts_use_onnx
        ))

        # LLM (lazy loaded)
        self._llm = None

        # Backbone (lazy loaded)
        self._backbone = None

        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="VoicePipeline")

        # Audio output callback
        self._on_audio: Optional[Callable[[bytes], None]] = None

    def set_audio_callback(self, callback: Callable[[bytes], None]):
        """Set callback for audio output."""
        self._on_audio = callback

    def _get_llm(self):
        """Get LLM instance (lazy load)."""
        if self._llm is None and LLM_AVAILABLE:
            self._llm = FastPhiStreamingLLM()
        return self._llm

    def _get_backbone(self):
        """Get backbone instance (lazy load)."""
        if self._backbone is None and BACKBONE_AVAILABLE:
            self._backbone = get_backbone()
        return self._backbone

    def process_audio(
        self,
        audio_bytes: bytes,
        user_id: int,
        session_id: Optional[str] = None
    ) -> tuple[str, PipelineMetrics]:
        """
        Process audio through full pipeline.

        Args:
            audio_bytes: Raw PCM audio (16-bit, 16kHz, mono)
            user_id: User identifier
            session_id: Optional session ID

        Returns:
            (llm_response, metrics)
        """
        metrics = PipelineMetrics()
        start_time = time.perf_counter()

        # === STAGE 1: STT ===
        metrics.stt_start_ms = (time.perf_counter() - start_time) * 1000
        transcript = self.stt.transcribe_bytes(audio_bytes)
        metrics.stt_end_ms = (time.perf_counter() - start_time) * 1000

        if not transcript:
            return "", metrics

        print(f"[Pipeline] STT: '{transcript}' ({metrics.stt_end_ms:.0f}ms)")

        # === STAGE 2: BACKBONE (Safety + Intent) ===
        backbone_start = time.perf_counter()
        backbone = self._get_backbone()

        if backbone:
            ctx = backbone.process_critical_path(
                user_input=transcript,
                user_id=user_id,
                session_id=session_id
            )
            if not ctx.safety_passed:
                return "I'm sorry, I can't help with that.", metrics

            intent_result = ctx.intent_result
            retrieval_result = ctx.retrieval_result
        else:
            intent_result = {"primary_intent": "GENERAL_KNOWLEDGE"}
            retrieval_result = None

        metrics.backbone_ms = (time.perf_counter() - backbone_start) * 1000
        print(f"[Pipeline] Backbone: {metrics.backbone_ms:.0f}ms")

        # === STAGE 3: BUILD PROMPT ===
        prompt = self._build_prompt(transcript, intent_result, retrieval_result)

        # === STAGE 4: LLM + TTS STREAMING ===
        llm_output, audio_metrics = self._stream_llm_tts(prompt, start_time)

        # Update metrics
        metrics.llm_first_token_ms = audio_metrics.get("llm_first_token", 0)
        metrics.llm_first_sentence_ms = audio_metrics.get("llm_first_sentence", 0)
        metrics.tts_first_audio_ms = audio_metrics.get("tts_first_audio", 0)
        metrics.total_ms = (time.perf_counter() - start_time) * 1000

        print(f"[Pipeline] Total: {metrics.total_ms:.0f}ms, First sentence: {metrics.tts_first_audio_ms:.0f}ms")

        # === STAGE 5: ASYNC OPERATIONS ===
        if backbone:
            backbone.queue_async_operations(ctx, llm_output)

        return llm_output, metrics

    def _build_prompt(
        self,
        user_input: str,
        intent_result: Dict[str, Any],
        retrieval_result: Any
    ) -> str:
        """Build minimal LLM prompt for fast TTFT."""
        context = ""

        # Add retrieval context if available
        if retrieval_result and hasattr(retrieval_result, 'hits') and retrieval_result.hits:
            hits = retrieval_result.hits[:2]  # Top 2 only for speed
            context_parts = [hit.get('content', '')[:80] for hit in hits]
            context = "Context: " + " | ".join(context_parts) + "\n"

        # Minimal prompt
        return f"{context}User: {user_input}\nAssistant:"

    def _stream_llm_tts(
        self,
        prompt: str,
        pipeline_start: float
    ) -> tuple[str, Dict[str, float]]:
        """
        Stream LLM → TTS with sentence buffering.

        Returns (full_response, timing_metrics)
        """
        metrics = {
            "llm_first_token": 0,
            "llm_first_sentence": 0,
            "tts_first_audio": 0
        }

        llm = self._get_llm()
        if llm is None:
            # Fallback response
            fallback = "I'm here to help. What would you like to know?"
            self._synthesize_and_output(fallback)
            return fallback, metrics

        # Sentence buffer
        buffer = SentenceBuffer(
            min_chars=self.config.min_sentence_chars,
            max_chars=self.config.max_buffer_chars
        )

        # Stream LLM and TTS concurrently
        full_response = ""
        first_token_recorded = False
        first_sentence_recorded = False
        first_audio_recorded = False

        # TTS queue for concurrent synthesis
        tts_queue: queue.Queue[Optional[str]] = queue.Queue()
        tts_done = threading.Event()

        def tts_worker():
            nonlocal first_audio_recorded, metrics
            while True:
                sentence = tts_queue.get()
                if sentence is None:
                    break

                # Synthesize and output
                for chunk in self.tts.synthesize_sentence(sentence):
                    if chunk.audio and self._on_audio:
                        if not first_audio_recorded:
                            metrics["tts_first_audio"] = (time.perf_counter() - pipeline_start) * 1000
                            first_audio_recorded = True
                        self._on_audio(chunk.audio)

            tts_done.set()

        # Start TTS worker
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

        try:
            # Stream LLM tokens
            for chunk in llm.stream_generate(prompt):
                if not first_token_recorded:
                    metrics["llm_first_token"] = (time.perf_counter() - pipeline_start) * 1000
                    first_token_recorded = True

                # Get new text since last chunk
                new_text = chunk.text[len(full_response):]
                full_response = chunk.text

                # Add to sentence buffer
                sentence = buffer.add(new_text)
                if sentence:
                    if not first_sentence_recorded:
                        metrics["llm_first_sentence"] = (time.perf_counter() - pipeline_start) * 1000
                        first_sentence_recorded = True

                    # Send to TTS immediately
                    tts_queue.put(sentence)

                if chunk.is_final:
                    break

            # Flush remaining text
            remaining = buffer.flush()
            if remaining:
                tts_queue.put(remaining)

        except Exception as e:
            print(f"[Pipeline] LLM error: {e}")
            full_response = "I'm having trouble responding right now."

        # Signal TTS worker to stop
        tts_queue.put(None)
        tts_done.wait(timeout=10.0)

        return full_response.strip(), metrics

    def _synthesize_and_output(self, text: str):
        """Synthesize text and send to audio output."""
        if self._on_audio is None:
            return

        for chunk in self.tts.synthesize_stream(text):
            if chunk.audio:
                self._on_audio(chunk.audio)

    def process_stream(
        self,
        audio_stream: Iterator[bytes],
        user_id: int,
        session_id: Optional[str] = None
    ) -> Iterator[tuple[str, PipelineMetrics]]:
        """
        Process streaming audio with VAD.

        Yields (response, metrics) for each detected utterance.
        """
        self.vad.reset()

        for chunk in audio_stream:
            event = self.vad.process_chunk(chunk)

            if event and event.event_type == "speech_end":
                # Process complete utterance
                result = self.process_audio(
                    event.audio,
                    user_id,
                    session_id
                )
                yield result

    def shutdown(self):
        """Shutdown pipeline resources."""
        self._executor.shutdown(wait=False)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_pipeline_instance: Optional[StreamingVoicePipeline] = None


def get_streaming_pipeline(config: Optional[PipelineConfig] = None) -> StreamingVoicePipeline:
    """Get or create singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = StreamingVoicePipeline(config)
    return _pipeline_instance


def process_voice_turn(
    audio_bytes: bytes,
    user_id: int,
    on_audio: Callable[[bytes], None],
    session_id: Optional[str] = None
) -> tuple[str, Dict[str, float]]:
    """
    Process a single voice turn.

    Convenience function for simple usage.

    Args:
        audio_bytes: User's recorded audio (PCM 16-bit, 16kHz, mono)
        user_id: User identifier
        on_audio: Callback for audio output
        session_id: Optional session ID

    Returns:
        (response_text, latency_metrics)
    """
    pipeline = get_streaming_pipeline()
    pipeline.set_audio_callback(on_audio)

    response, metrics = pipeline.process_audio(audio_bytes, user_id, session_id)
    return response, metrics.to_dict()


__all__ = [
    "PipelineConfig",
    "PipelineMetrics",
    "StreamingVoicePipeline",
    "FasterWhisperSTT",
    "SentenceBuffer",
    "get_streaming_pipeline",
    "process_voice_turn",
]
