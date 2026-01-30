"""
Voice Activity Detection (VAD) using Silero VAD.

Detects when user starts and stops speaking for real-time voice pipeline.
Silero VAD is fast (~1ms per chunk) and accurate.
"""

from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Iterator, Optional, Callable, List
import numpy as np

# Lazy import torch to avoid startup delay
_vad_model = None
_vad_utils = None


def _load_silero_vad():
    """Lazy load Silero VAD model."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        import torch
        torch.set_num_threads(1)  # VAD is single-threaded

        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True  # Use ONNX for speed
        )
        _vad_model = model
        _vad_utils = utils
    return _vad_model, _vad_utils


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""
    sample_rate: int = 16000
    chunk_size_ms: int = 30  # 30ms chunks for real-time
    speech_threshold: float = 0.5
    silence_threshold: float = 0.35
    min_speech_ms: int = 250  # Minimum speech duration to trigger
    min_silence_ms: int = 500  # Silence duration to end utterance
    max_speech_ms: int = 30000  # Maximum utterance length (30s)
    pre_speech_padding_ms: int = 300  # Audio to keep before speech start


@dataclass
class VADEvent:
    """VAD detection event."""
    event_type: str  # "speech_start", "speech_end", "audio_chunk"
    audio: Optional[bytes] = None
    timestamp_ms: float = 0.0
    speech_probability: float = 0.0
    duration_ms: float = 0.0


class SileroVAD:
    """
    Real-time Voice Activity Detection using Silero VAD.

    Features:
    - ~1ms latency per chunk
    - Accurate speech detection
    - Pre-speech audio buffering
    - Automatic utterance segmentation
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._model = None
        self._get_speech_timestamps = None
        self._initialized = False

        # Chunk size in samples
        self.chunk_samples = int(self.config.sample_rate * self.config.chunk_size_ms / 1000)

        # State
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._audio_buffer: List[bytes] = []
        self._pre_speech_buffer: List[bytes] = []
        self._max_pre_speech_chunks = int(self.config.pre_speech_padding_ms / self.config.chunk_size_ms)

    def _ensure_initialized(self):
        """Initialize VAD model on first use."""
        if self._initialized:
            return

        try:
            self._model, utils = _load_silero_vad()
            self._get_speech_timestamps = utils[0]
            self._initialized = True
            print("[VAD] Silero VAD initialized (ONNX)")
        except Exception as e:
            print(f"[VAD] Failed to load Silero VAD: {e}")
            raise

    def _bytes_to_tensor(self, audio_bytes: bytes) -> 'torch.Tensor':
        """Convert audio bytes to torch tensor."""
        import torch

        # Assume 16-bit PCM audio
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_np = audio_np / 32768.0  # Normalize to [-1, 1]
        return torch.from_numpy(audio_np)

    def get_speech_prob(self, audio_chunk: bytes) -> float:
        """
        Get speech probability for audio chunk.

        Args:
            audio_chunk: Raw PCM audio bytes (16-bit, mono)

        Returns:
            Speech probability (0.0 to 1.0)
        """
        self._ensure_initialized()

        try:
            tensor = self._bytes_to_tensor(audio_chunk)
            prob = self._model(tensor, self.config.sample_rate).item()
            return prob
        except Exception as e:
            print(f"[VAD] Error getting speech prob: {e}")
            return 0.0

    def process_chunk(self, audio_chunk: bytes) -> Optional[VADEvent]:
        """
        Process audio chunk and detect speech events.

        Args:
            audio_chunk: Raw PCM audio bytes

        Returns:
            VADEvent if state changed, None otherwise
        """
        self._ensure_initialized()

        current_time = time.perf_counter() * 1000  # ms
        prob = self.get_speech_prob(audio_chunk)

        # Update pre-speech buffer (circular)
        self._pre_speech_buffer.append(audio_chunk)
        if len(self._pre_speech_buffer) > self._max_pre_speech_chunks:
            self._pre_speech_buffer.pop(0)

        if not self._is_speaking:
            # Looking for speech start
            if prob >= self.config.speech_threshold:
                self._is_speaking = True
                self._speech_start_time = current_time
                self._silence_start_time = None

                # Include pre-speech buffer
                self._audio_buffer = list(self._pre_speech_buffer)
                self._audio_buffer.append(audio_chunk)

                return VADEvent(
                    event_type="speech_start",
                    timestamp_ms=current_time,
                    speech_probability=prob
                )
            else:
                return None
        else:
            # Currently speaking, looking for end
            self._audio_buffer.append(audio_chunk)

            # Check speech duration limit
            speech_duration = current_time - self._speech_start_time
            if speech_duration >= self.config.max_speech_ms:
                return self._end_speech(current_time, prob, "max_duration")

            if prob < self.config.silence_threshold:
                # Potential silence
                if self._silence_start_time is None:
                    self._silence_start_time = current_time

                silence_duration = current_time - self._silence_start_time
                if silence_duration >= self.config.min_silence_ms:
                    return self._end_speech(current_time, prob, "silence")
            else:
                # Still speaking
                self._silence_start_time = None

            # Return audio chunk event for streaming
            return VADEvent(
                event_type="audio_chunk",
                audio=audio_chunk,
                timestamp_ms=current_time,
                speech_probability=prob
            )

    def _end_speech(self, current_time: float, prob: float, reason: str) -> VADEvent:
        """End current speech segment."""
        duration = current_time - self._speech_start_time

        # Collect all buffered audio
        all_audio = b''.join(self._audio_buffer)

        # Reset state
        self._is_speaking = False
        self._speech_start_time = None
        self._silence_start_time = None
        self._audio_buffer = []

        return VADEvent(
            event_type="speech_end",
            audio=all_audio,
            timestamp_ms=current_time,
            speech_probability=prob,
            duration_ms=duration
        )

    def reset(self):
        """Reset VAD state."""
        self._is_speaking = False
        self._speech_start_time = None
        self._silence_start_time = None
        self._audio_buffer = []
        self._pre_speech_buffer = []

    @property
    def is_speaking(self) -> bool:
        """Check if currently in speech segment."""
        return self._is_speaking


class VADStream:
    """
    Streaming VAD processor with async audio input.

    Usage:
        vad_stream = VADStream()
        vad_stream.start()

        # Feed audio chunks
        for chunk in audio_source:
            vad_stream.feed(chunk)

        # Get speech segments
        for event in vad_stream.events():
            if event.event_type == "speech_end":
                process_speech(event.audio)
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.vad = SileroVAD(config)
        self._audio_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._event_queue: queue.Queue[Optional[VADEvent]] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start VAD processing thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop VAD processing."""
        self._running = False
        self._audio_queue.put(None)  # Signal shutdown
        if self._thread:
            self._thread.join(timeout=1.0)

    def feed(self, audio_chunk: bytes):
        """Feed audio chunk to VAD."""
        if self._running:
            self._audio_queue.put(audio_chunk)

    def events(self) -> Iterator[VADEvent]:
        """Iterate over VAD events."""
        while True:
            try:
                event = self._event_queue.get(timeout=0.1)
                if event is None:
                    break
                yield event
            except queue.Empty:
                if not self._running:
                    break

    def _process_loop(self):
        """Background processing loop."""
        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                if chunk is None:
                    break

                event = self.vad.process_chunk(chunk)
                if event:
                    self._event_queue.put(event)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VAD] Processing error: {e}")

        self._event_queue.put(None)  # Signal end


# Singleton
_vad_instance: Optional[SileroVAD] = None


def get_vad(config: Optional[VADConfig] = None) -> SileroVAD:
    """Get or create singleton VAD instance."""
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = SileroVAD(config)
    return _vad_instance


__all__ = [
    "VADConfig",
    "VADEvent",
    "SileroVAD",
    "VADStream",
    "get_vad",
]
