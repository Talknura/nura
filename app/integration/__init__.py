"""
Integration Module.

Central coordination layer for all Nura engines with async optimizations.
The Backbone Layer coordinates all engines with minimal latency through:
    - Critical path processing (blocking operations)
    - Async operations (non-blocking, after response)
    - Deferred proactive evaluation
    - Batched HNSW updates
    - Predictive caching
"""

from app.integration.backbone import (
    BackboneLayer,
    BackboneContext,
    BackboneResult,
    BackboneTask,
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
