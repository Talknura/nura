"""
Integration Module - Unified Nura Backbone.

The Backbone Layer IS Nura - coordinating:
    - Voice I/O (VAD, STT, TTS with sentence streaming)
    - All engines (Memory, Retrieval, Temporal, Adaptation, Proactive)
    - LLM streaming
    - Async operations for low latency

Target: <500ms to first sentence heard
"""

from app.integration.backbone import (
    BackboneLayer,
    BackboneContext,
    BackboneResult,
    BackboneTask,
    VoiceResult,
    AsyncOperationQueue,
    DeferredProactiveQueue,
    BatchedHNSWUpdater,
    PredictiveCache,
    get_backbone,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    PRIORITY_LOW,
    PRIORITY_IDLE,
)

# Legacy imports for backward compatibility
try:
    from app.integration.async_memory_queue import AsyncMemoryQueue
except ImportError:
    AsyncMemoryQueue = None

try:
    from app.integration.parallel_engine_executor import ParallelEngineExecutor
except ImportError:
    ParallelEngineExecutor = None

try:
    from app.integration.optimized_voice_pipeline import OptimizedVoicePipeline
except ImportError:
    OptimizedVoicePipeline = None

__all__ = [
    # Backbone Layer (primary)
    "BackboneLayer",
    "BackboneContext",
    "BackboneResult",
    "BackboneTask",
    "VoiceResult",
    "AsyncOperationQueue",
    "DeferredProactiveQueue",
    "BatchedHNSWUpdater",
    "PredictiveCache",
    "get_backbone",
    # Priorities
    "PRIORITY_CRITICAL",
    "PRIORITY_HIGH",
    "PRIORITY_NORMAL",
    "PRIORITY_LOW",
    "PRIORITY_IDLE",
    # Legacy (optional)
    "AsyncMemoryQueue",
    "ParallelEngineExecutor",
    "OptimizedVoicePipeline",
]
