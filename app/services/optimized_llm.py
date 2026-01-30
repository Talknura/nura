"""
Qwen3-4B Streaming LLM for Nura Voice Pipeline.

Optimized for:
- Human conversation with semantic understanding
- Reasoning capabilities ("thinking mode")
- Streaming STT → LLM → TTS with sentence buffering
- Context window: 4096 tokens (fits system + engines + user input)

Target: <500ms TTFT (Time To First Token)
"""

from __future__ import annotations

import os
import re
import time
import queue
import threading
from dataclasses import dataclass
from typing import Iterator, Optional, Callable, List

from app.services.streaming_llm import LLMChunk, PhiStreamingLLM


# =============================================================================
# CONFIGURATION
# =============================================================================

# Context budget (must fit in n_ctx)
CONTEXT_BUDGET = {
    "system": 200,
    "memory": 400,
    "retrieval": 300,
    "temporal": 100,
    "user_input": 200,
    "response": 300,
}
# Total: ~1,500 tokens, use 4096 context for safety margin


# =============================================================================
# QWEN3 STREAMING LLM
# =============================================================================

class Qwen3StreamingLLM(PhiStreamingLLM):
    """
    Qwen3-4B optimized for Nura voice pipeline.

    Features:
    - 4096 token context (fits all engine outputs)
    - Streaming token generation
    - Sentence-aware output for TTS
    - Conversation + reasoning optimized

    Supports:
    - Stream input from STT (text chunks)
    - Stream output for TTS (sentence chunks)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,  # Fits system + engines + response
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        max_new_tokens: int = 256,  # Longer responses for conversation
    ) -> None:
        # Use NURA_LLM_MODEL env var (more generic than NURA_PHI_MODEL)
        if model_path is None:
            model_path = os.getenv("NURA_LLM_MODEL") or os.getenv("NURA_PHI_MODEL")

        super().__init__(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            max_new_tokens=max_new_tokens
        )

    def stream_generate(self, prompt: str) -> Iterator[LLMChunk]:
        """
        Stream generate tokens from prompt.

        Yields LLMChunk with accumulated text as tokens arrive.
        Use with SentenceBuffer for TTS streaming.
        """
        llama = self._load()
        start = time.perf_counter()
        acc = ""

        # Qwen3 optimized parameters
        for chunk in llama(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.7,  # Balanced for conversation
            top_p=0.9,
            stream=True,
            repeat_penalty=1.1,
            stop=["User:", "\n\nUser:", "\nUser:", "<|im_end|>", "<|endoftext|>"],
        ):
            token = chunk["choices"][0]["text"]
            acc += token
            latency = (time.perf_counter() - start) * 1000.0
            yield LLMChunk(text=acc, is_final=False, latency_ms=latency)

        yield LLMChunk(text=acc, is_final=True, latency_ms=(time.perf_counter() - start) * 1000.0)

    def stream_generate_sentences(
        self,
        prompt: str,
        on_sentence: Callable[[str], None],
        min_sentence_chars: int = 20
    ) -> str:
        """
        Stream generate with sentence-level callbacks for TTS.

        Args:
            prompt: Full prompt text
            on_sentence: Called with each complete sentence (for TTS)
            min_sentence_chars: Minimum chars before flushing sentence

        Returns:
            Full response text
        """
        buffer = ""
        full_response = ""

        for chunk in self.stream_generate(prompt):
            new_text = chunk.text[len(full_response):]
            full_response = chunk.text
            buffer += new_text

            # Check for sentence boundary
            while True:
                match = re.search(r'^(.*?[.!?])(?:\s+|$)', buffer)
                if match and len(match.group(1)) >= min_sentence_chars:
                    sentence = match.group(1).strip()
                    buffer = buffer[match.end():].lstrip()
                    on_sentence(sentence)
                else:
                    break

            if chunk.is_final:
                break

        # Flush remaining buffer
        if buffer.strip():
            on_sentence(buffer.strip())

        return full_response.strip()


# =============================================================================
# SENTENCE BUFFER (for STT → LLM → TTS streaming)
# =============================================================================

class SentenceStreamBuffer:
    """
    Buffer for streaming LLM output to TTS by sentences.

    This enables the <500ms first-sentence latency:
    - LLM generates tokens
    - Buffer detects complete sentences
    - Each sentence immediately sent to TTS
    - Audio plays while LLM continues generating

    Usage:
        buffer = SentenceStreamBuffer(on_sentence=tts.synthesize)
        for chunk in llm.stream_generate(prompt):
            buffer.feed(chunk.text)
        buffer.flush()
    """

    def __init__(
        self,
        on_sentence: Callable[[str], None],
        min_chars: int = 20,
        max_chars: int = 200
    ):
        self.on_sentence = on_sentence
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""
        self._last_text = ""
        self._sentences_emitted = 0

    def feed(self, accumulated_text: str) -> None:
        """
        Feed accumulated LLM text (not delta).

        Args:
            accumulated_text: Full text so far from LLM
        """
        # Get new text since last feed
        new_text = accumulated_text[len(self._last_text):]
        self._last_text = accumulated_text
        self._buffer += new_text

        # Extract complete sentences
        self._flush_sentences()

    def _flush_sentences(self) -> None:
        """Flush complete sentences to callback."""
        while True:
            # Match sentence ending
            match = re.search(r'^(.*?[.!?])(?:\s+|$)', self._buffer)
            if match and len(match.group(1)) >= self.min_chars:
                sentence = match.group(1).strip()
                self._buffer = self._buffer[match.end():].lstrip()
                self.on_sentence(sentence)
                self._sentences_emitted += 1
            elif len(self._buffer) >= self.max_chars:
                # Force flush if buffer too large
                self.on_sentence(self._buffer.strip())
                self._buffer = ""
                self._sentences_emitted += 1
            else:
                break

    def flush(self) -> None:
        """Flush any remaining text in buffer."""
        if self._buffer.strip():
            self.on_sentence(self._buffer.strip())
            self._sentences_emitted += 1
            self._buffer = ""

    @property
    def sentences_count(self) -> int:
        """Number of sentences emitted."""
        return self._sentences_emitted


# =============================================================================
# STREAMING PIPELINE HELPER
# =============================================================================

class StreamingLLMPipeline:
    """
    Complete STT → LLM → TTS streaming pipeline.

    Integrates:
    - STT text input (can be partial/streaming)
    - LLM streaming generation
    - Sentence buffering
    - TTS output callbacks

    Usage:
        pipeline = StreamingLLMPipeline(llm, tts_callback)
        response = pipeline.process(prompt)
    """

    def __init__(
        self,
        llm: Qwen3StreamingLLM,
        on_audio: Callable[[bytes], None],
        tts_synthesize: Optional[Callable[[str], Iterator]] = None
    ):
        self.llm = llm
        self.on_audio = on_audio
        self.tts_synthesize = tts_synthesize

        # TTS queue for concurrent synthesis
        self._tts_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._tts_thread: Optional[threading.Thread] = None
        self._running = False

    def _start_tts_worker(self) -> None:
        """Start background TTS worker."""
        if self._tts_thread is not None and self._tts_thread.is_alive():
            return

        self._running = True
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

    def _tts_worker(self) -> None:
        """Background TTS synthesis worker."""
        while self._running:
            try:
                sentence = self._tts_queue.get(timeout=0.1)
                if sentence is None:
                    break

                if self.tts_synthesize:
                    for chunk in self.tts_synthesize(sentence):
                        if hasattr(chunk, 'audio') and chunk.audio:
                            self.on_audio(chunk.audio)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[LLMPipeline] TTS error: {e}")

    def _stop_tts_worker(self) -> None:
        """Stop TTS worker and wait for completion."""
        self._tts_queue.put(None)
        self._running = False
        if self._tts_thread:
            self._tts_thread.join(timeout=10.0)

    def process(self, prompt: str) -> str:
        """
        Process prompt through LLM with TTS streaming.

        Args:
            prompt: Full prompt for LLM

        Returns:
            Complete response text
        """
        self._start_tts_worker()

        def on_sentence(sentence: str):
            self._tts_queue.put(sentence)

        try:
            response = self.llm.stream_generate_sentences(
                prompt,
                on_sentence=on_sentence
            )
        except Exception as e:
            print(f"[LLMPipeline] LLM error: {e}")
            response = "I'm having trouble responding right now."

        self._stop_tts_worker()
        return response


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

# Singleton instance
_llm_instance: Optional[Qwen3StreamingLLM] = None


def get_streaming_llm(model_path: Optional[str] = None) -> Qwen3StreamingLLM:
    """
    Get Qwen3-4B streaming LLM instance.

    Auto-detects model path from environment:
    - NURA_LLM_MODEL (preferred)
    - NURA_PHI_MODEL (fallback)
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = Qwen3StreamingLLM(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # All GPU
            max_new_tokens=256
        )
    return _llm_instance


# Legacy alias
def get_optimized_llm(model_path: Optional[str] = None) -> Qwen3StreamingLLM:
    """Legacy alias for get_streaming_llm."""
    return get_streaming_llm(model_path)


# Keep old class name as alias for backward compatibility
FastPhiStreamingLLM = Qwen3StreamingLLM


__all__ = [
    "Qwen3StreamingLLM",
    "SentenceStreamBuffer",
    "StreamingLLMPipeline",
    "get_streaming_llm",
    "get_optimized_llm",
    "FastPhiStreamingLLM",  # Legacy alias
    "CONTEXT_BUDGET",
]
