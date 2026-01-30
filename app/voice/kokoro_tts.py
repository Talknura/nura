"""
Kokoro TTS - Local Text-to-Speech with Streaming Support.

Kokoro is a lightweight (82M params) open-source TTS model that runs locally.
Supports sentence-level streaming for low-latency voice output.

Installation:
    pip install kokoro>=0.9.4 soundfile
    # or for ONNX (faster):
    pip install kokoro-onnx
"""

from __future__ import annotations

import os
import io
import re
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional, List, Callable
import numpy as np

# Lazy imports
_kokoro_pipeline = None
_kokoro_onnx = None


@dataclass
class TTSConfig:
    """Configuration for Kokoro TTS."""
    voice: str = "af_heart"  # Default voice (American Female)
    speed: float = 1.0  # Speech speed multiplier
    sample_rate: int = 24000  # Output sample rate
    use_onnx: bool = True  # Use ONNX for faster inference
    chunk_size: int = 4096  # Audio chunk size for streaming


# Available voices
KOKORO_VOICES = {
    # American English
    "af_heart": "American Female - Heart (warm)",
    "af_bella": "American Female - Bella",
    "af_nicole": "American Female - Nicole",
    "af_sarah": "American Female - Sarah",
    "af_sky": "American Female - Sky",
    "am_adam": "American Male - Adam",
    "am_michael": "American Male - Michael",
    # British English
    "bf_emma": "British Female - Emma",
    "bf_isabella": "British Female - Isabella",
    "bm_george": "British Male - George",
    "bm_lewis": "British Male - Lewis",
}


@dataclass(frozen=True)
class TTSChunk:
    """A chunk of synthesized audio."""
    audio: bytes  # Raw PCM audio (16-bit, mono)
    is_final: bool
    latency_ms: float
    sample_rate: int = 24000


def _load_kokoro_onnx():
    """Load Kokoro ONNX model (faster)."""
    global _kokoro_onnx
    if _kokoro_onnx is not None:
        return _kokoro_onnx

    try:
        from kokoro_onnx import Kokoro

        # Model paths (auto-download from HuggingFace if not present)
        model_path = os.getenv("NURA_KOKORO_MODEL", None)
        voices_path = os.getenv("NURA_KOKORO_VOICES", None)

        _kokoro_onnx = Kokoro(model_path, voices_path)
        print("[TTS] Kokoro ONNX loaded")
        return _kokoro_onnx
    except ImportError:
        print("[TTS] kokoro-onnx not installed, falling back to torch version")
        return None
    except Exception as e:
        print(f"[TTS] Failed to load Kokoro ONNX: {e}")
        return None


def _load_kokoro_pipeline():
    """Load Kokoro PyTorch pipeline."""
    global _kokoro_pipeline
    if _kokoro_pipeline is not None:
        return _kokoro_pipeline

    try:
        from kokoro import KPipeline

        # Use environment variable for language or default to American English
        lang = os.getenv("NURA_KOKORO_LANG", "a")  # 'a' = American English
        _kokoro_pipeline = KPipeline(lang_code=lang)
        print(f"[TTS] Kokoro pipeline loaded (lang={lang})")
        return _kokoro_pipeline
    except ImportError as e:
        print(f"[TTS] Kokoro not installed: {e}")
        print("[TTS] Install with: pip install kokoro>=0.9.4 soundfile")
        return None
    except Exception as e:
        print(f"[TTS] Failed to load Kokoro: {e}")
        return None


class KokoroTTS:
    """
    Kokoro TTS with sentence-level streaming.

    Features:
    - Local inference (no cloud)
    - Multiple voices
    - Sentence-level streaming for low latency
    - ONNX support for faster inference
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._onnx_model = None
        self._torch_pipeline = None
        self._initialized = False

    def _ensure_initialized(self):
        """Initialize TTS on first use."""
        if self._initialized:
            return

        if self.config.use_onnx:
            self._onnx_model = _load_kokoro_onnx()

        if self._onnx_model is None:
            self._torch_pipeline = _load_kokoro_pipeline()

        if self._onnx_model is None and self._torch_pipeline is None:
            raise RuntimeError("No TTS backend available. Install kokoro or kokoro-onnx")

        self._initialized = True

    def synthesize(self, text: str, voice: Optional[str] = None) -> Optional[bytes]:
        """
        Synthesize text to audio (blocking, full output).

        Args:
            text: Text to synthesize
            voice: Voice ID (optional, uses config default)

        Returns:
            Raw PCM audio bytes (24kHz, 16-bit, mono)
        """
        self._ensure_initialized()
        voice = voice or self.config.voice

        try:
            if self._onnx_model:
                # ONNX path
                samples, sample_rate = self._onnx_model.create(
                    text,
                    voice=voice,
                    speed=self.config.speed
                )
                # Convert to 16-bit PCM
                audio_int16 = (samples * 32767).astype(np.int16)
                return audio_int16.tobytes()
            else:
                # Torch path
                generator = self._torch_pipeline(
                    text,
                    voice=voice,
                    speed=self.config.speed
                )
                # Collect all audio
                all_audio = []
                for _, _, audio in generator:
                    all_audio.append(audio)

                if all_audio:
                    combined = np.concatenate(all_audio)
                    audio_int16 = (combined * 32767).astype(np.int16)
                    return audio_int16.tobytes()
                return None

        except Exception as e:
            print(f"[TTS] Synthesis error: {e}")
            return None

    def synthesize_stream(self, text: str, voice: Optional[str] = None) -> Iterator[TTSChunk]:
        """
        Stream synthesized audio by sentences.

        Splits text into sentences and synthesizes each one,
        yielding audio chunks as they're ready. This provides
        low-latency first-audio output.

        Args:
            text: Text to synthesize
            voice: Voice ID (optional)

        Yields:
            TTSChunk with audio data
        """
        self._ensure_initialized()
        voice = voice or self.config.voice
        start_time = time.perf_counter()

        # Split into sentences for streaming
        sentences = self._split_sentences(text)
        if not sentences:
            yield TTSChunk(
                audio=b"",
                is_final=True,
                latency_ms=(time.perf_counter() - start_time) * 1000
            )
            return

        for idx, sentence in enumerate(sentences):
            is_last = (idx == len(sentences) - 1)

            try:
                audio_bytes = self.synthesize(sentence, voice)
                if audio_bytes:
                    # Chunk the audio for streaming
                    chunk_size = self.config.chunk_size
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i:i + chunk_size]
                        latency = (time.perf_counter() - start_time) * 1000

                        is_final_chunk = is_last and (i + chunk_size >= len(audio_bytes))
                        yield TTSChunk(
                            audio=chunk,
                            is_final=is_final_chunk,
                            latency_ms=latency,
                            sample_rate=self.config.sample_rate
                        )

            except Exception as e:
                print(f"[TTS] Sentence synthesis error: {e}")
                continue

        # Ensure final marker
        yield TTSChunk(
            audio=b"",
            is_final=True,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            sample_rate=self.config.sample_rate
        )

    def synthesize_sentence(self, sentence: str, voice: Optional[str] = None) -> Iterator[TTSChunk]:
        """
        Synthesize a single sentence with chunked output.

        Optimized for first-sentence latency in streaming pipeline.
        """
        self._ensure_initialized()
        voice = voice or self.config.voice
        start_time = time.perf_counter()

        try:
            audio_bytes = self.synthesize(sentence, voice)
            if audio_bytes:
                # Emit first chunk immediately
                chunk_size = self.config.chunk_size
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    latency = (time.perf_counter() - start_time) * 1000
                    is_final = (i + chunk_size >= len(audio_bytes))

                    yield TTSChunk(
                        audio=chunk,
                        is_final=is_final,
                        latency_ms=latency,
                        sample_rate=self.config.sample_rate
                    )
        except Exception as e:
            print(f"[TTS] Error: {e}")

        yield TTSChunk(
            audio=b"",
            is_final=True,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            sample_rate=self.config.sample_rate
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming."""
        # Split on sentence boundaries
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @property
    def available_voices(self) -> dict:
        """Get available voice IDs."""
        return KOKORO_VOICES.copy()


class StreamingTTSBuffer:
    """
    Buffer for LLM -> TTS streaming with sentence detection.

    Accumulates LLM tokens and flushes complete sentences to TTS.
    This enables <500ms first-sentence latency.
    """

    def __init__(
        self,
        tts: KokoroTTS,
        on_audio: Callable[[bytes], None],
        voice: Optional[str] = None
    ):
        self.tts = tts
        self.on_audio = on_audio
        self.voice = voice
        self._buffer = ""
        self._sentences_sent = 0
        self._lock = threading.Lock()

        # TTS worker thread
        self._tts_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._running = True
        self._worker = threading.Thread(target=self._tts_worker, daemon=True)
        self._worker.start()

    def feed(self, text: str):
        """Feed text chunk from LLM."""
        with self._lock:
            self._buffer += text

        # Check for complete sentences
        self._flush_sentences()

    def _flush_sentences(self):
        """Flush complete sentences to TTS queue."""
        with self._lock:
            # Look for sentence boundaries
            pattern = r'([.!?])\s+'
            parts = re.split(pattern, self._buffer)

            # Reconstruct sentences
            sentences = []
            i = 0
            while i < len(parts) - 1:
                if i + 1 < len(parts) and parts[i + 1] in '.!?':
                    sentences.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    i += 1

            # Keep incomplete part in buffer
            if len(parts) > 0:
                # Last part is incomplete (no punctuation after it)
                remainder = parts[-1] if not (len(parts) >= 2 and parts[-1] in '.!?') else ""
                self._buffer = remainder

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    self._tts_queue.put(sentence)
                    self._sentences_sent += 1

    def _tts_worker(self):
        """Background TTS synthesis thread."""
        while self._running:
            try:
                sentence = self._tts_queue.get(timeout=0.1)
                if sentence is None:
                    break

                # Synthesize and emit audio
                for chunk in self.tts.synthesize_sentence(sentence, self.voice):
                    if chunk.audio:
                        self.on_audio(chunk.audio)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS Buffer] Error: {e}")

    def flush_remaining(self):
        """Flush any remaining text in buffer."""
        with self._lock:
            remaining = self._buffer.strip()
            self._buffer = ""

        if remaining:
            self._tts_queue.put(remaining)

    def finish(self):
        """Finish processing and wait for completion."""
        self.flush_remaining()
        self._tts_queue.put(None)  # Signal shutdown
        self._running = False
        self._worker.join(timeout=5.0)

    @property
    def sentences_processed(self) -> int:
        """Number of sentences sent to TTS."""
        return self._sentences_sent


# Singleton
_tts_instance: Optional[KokoroTTS] = None


def get_kokoro_tts(config: Optional[TTSConfig] = None) -> KokoroTTS:
    """Get or create singleton Kokoro TTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = KokoroTTS(config)
    return _tts_instance


__all__ = [
    "TTSConfig",
    "TTSChunk",
    "KokoroTTS",
    "StreamingTTSBuffer",
    "get_kokoro_tts",
    "KOKORO_VOICES",
]
