"""
Backbone Layer - Unified Nura System.

The backbone layer IS Nura - the central nervous system that coordinates:
- Voice I/O (VAD, STT, TTS)
- All engines (Memory, Retrieval, Temporal, Adaptation, Proactive)
- LLM streaming with sentence-level TTS
- Async operations for low latency

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                              NURA BACKBONE                                  │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │   AUDIO IN ──▶ VAD ──▶ STT ──┐                                              │
    │                              │                                              │
    │                              ▼                                              │
    │   ┌───────────────────────────────────────────────────────────────────┐    │
    │   │                    CRITICAL PATH (Blocking)                        │    │
    │   │         Safety + Intent (parallel) → Temporal → Retrieval          │    │
    │   │                        Target: <50ms                               │    │
    │   └───────────────────────────────────────────────────────────────────┘    │
    │                              │                                              │
    │                              ▼                                              │
    │   ┌───────────────────────────────────────────────────────────────────┐    │
    │   │              LLM STREAMING + SENTENCE BUFFER + TTS                 │    │
    │   │                  Target: <500ms to first sentence                  │    │
    │   └───────────────────────────────────────────────────────────────────┘    │
    │                              │                                              │
    │                              ▼                                              │
    │                         AUDIO OUT                                           │
    │                                                                             │
    │   ┌───────────────────────────────────────────────────────────────────┐    │
    │   │                    ASYNC AFTER RESPONSE                            │    │
    │   │   Memory Write │ Adaptation │ Proactive │ HNSW Update             │    │
    │   └───────────────────────────────────────────────────────────────────┘    │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

Latency Target: <500ms to first sentence heard
    - STT: ~150ms (Faster Whisper)
    - Backbone: ~50ms (Safety + Intent + Engines)
    - LLM TTFT: ~100ms
    - First Sentence: ~50ms buffer
    - TTS TTFA: ~100ms
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

# =============================================================================
# ENGINE IMPORTS (with graceful fallback)
# =============================================================================

# Memory Engine
try:
    from app.memory import get_memory_store
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# Retrieval Engine
try:
    from app.retrieval import RetrievalEngine, RetrievalState
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False

# Temporal Engine
try:
    from app.temporal import get_temporal_engine
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False

# Adaptation Engine
try:
    from app.adaptation import get_adaptation_engine
    ADAPTATION_AVAILABLE = True
except ImportError:
    ADAPTATION_AVAILABLE = False

# Proactive Engine
try:
    from app.proactive import get_proactive_engine
    PROACTIVE_AVAILABLE = True
except ImportError:
    PROACTIVE_AVAILABLE = False

# Orchestrator
try:
    from app.orchestrator import get_intent_gate, get_orchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

# Safety Layer
try:
    from app.guards.safety_layer import get_safety_layer
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False

# Vector/Embedding
try:
    from app.vector.embedder import embed_text
    EMBEDDER_AVAILABLE = True
except ImportError:
    EMBEDDER_AVAILABLE = False

# Voice Components (VAD, STT, TTS)
try:
    from app.voice.vad import SileroVAD, VADConfig, VADEvent
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

try:
    from app.voice.kokoro_tts import KokoroTTS, TTSConfig, TTSChunk
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False

# LLM Streaming
try:
    from app.services.streaming_llm import StreamingLLM, LLMChunk
    from app.services.optimized_llm import FastPhiStreamingLLM
    LLM_STREAMING_AVAILABLE = True
except ImportError:
    LLM_STREAMING_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

# Task priorities (lower = higher priority)
PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 1
PRIORITY_NORMAL = 2
PRIORITY_LOW = 3
PRIORITY_IDLE = 4

# Batch settings
HNSW_BATCH_SIZE = 10
HNSW_FLUSH_INTERVAL = 5.0  # seconds

# Worker counts
ASYNC_WORKERS = 3
CRITICAL_WORKERS = 2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BackboneTask:
    """A task for the backbone queue."""
    task_id: str
    operation: Callable[[], Any]
    priority: int = PRIORITY_NORMAL
    created_at: float = field(default_factory=time.time)
    category: str = "general"  # memory, adaptation, proactive, hnsw, logging

    def __lt__(self, other):
        return self.priority < other.priority


@dataclass
class BackboneContext:
    """Context for a backbone request."""
    user_id: int
    user_input: str
    session_id: str
    now: datetime

    # Results from critical path
    safety_passed: bool = True
    intent_result: Optional[Dict[str, Any]] = None
    temporal_context: Optional[Dict[str, Any]] = None
    temporal_rewrite: Optional[Dict[str, Any]] = None
    retrieval_result: Optional[Any] = None

    # Deferred operations
    memory_task_id: Optional[str] = None
    adaptation_task_id: Optional[str] = None
    proactive_task_id: Optional[str] = None
    hnsw_task_id: Optional[str] = None

    # Response
    llm_output: Optional[str] = None

    # Timing
    timing: Dict[str, float] = field(default_factory=dict)


@dataclass
class BackboneResult:
    """Result from backbone processing."""
    llm_output: str
    context: BackboneContext
    async_tasks: List[str] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# DEFERRED PROACTIVE QUEUE
# =============================================================================

class DeferredProactiveQueue:
    """
    Evaluate proactive engine AFTER response is delivered.
    Stores results for next turn.
    """

    def __init__(self):
        self._results_cache: Dict[int, Dict[str, Any]] = {}
        self._pending: Dict[int, bool] = {}
        self._lock = threading.Lock()

    def queue_evaluation(
        self,
        user_id: int,
        context: Dict[str, Any],
        async_queue: 'AsyncOperationQueue'
    ) -> str:
        """
        Queue proactive evaluation (non-blocking).

        Returns task_id for tracking.
        """
        with self._lock:
            self._pending[user_id] = True

        def _evaluate():
            try:
                if PROACTIVE_AVAILABLE:
                    engine = get_proactive_engine()
                    result = engine.evaluate(context)
                    result_dict = result if isinstance(result, dict) else result.__dict__

                    with self._lock:
                        self._results_cache[user_id] = result_dict
                        self._pending[user_id] = False
            except Exception as e:
                print(f"[DeferredProactive] Evaluation failed: {e}")
                with self._lock:
                    self._pending[user_id] = False

        return async_queue.enqueue(
            _evaluate,
            priority=PRIORITY_LOW,
            category="proactive"
        )

    def get_cached_result(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get result from previous turn's evaluation."""
        with self._lock:
            return self._results_cache.get(user_id)

    def is_pending(self, user_id: int) -> bool:
        """Check if evaluation is pending."""
        with self._lock:
            return self._pending.get(user_id, False)

    def clear_cache(self, user_id: int) -> None:
        """Clear cached result for user."""
        with self._lock:
            self._results_cache.pop(user_id, None)
            self._pending.pop(user_id, None)


# =============================================================================
# BATCHED HNSW UPDATER
# =============================================================================

class BatchedHNSWUpdater:
    """
    Batch HNSW index updates for efficiency.
    Updates every N seconds or N items, whichever comes first.
    """

    def __init__(
        self,
        flush_interval: float = HNSW_FLUSH_INTERVAL,
        batch_size: int = HNSW_BATCH_SIZE
    ):
        self._pending: Dict[Tuple[int, str], List[Tuple[np.ndarray, str]]] = defaultdict(list)
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="HNSWBatchFlusher",
            daemon=True
        )
        self._flush_thread.start()

    def queue_update(
        self,
        user_id: int,
        tier: str,
        vector: np.ndarray,
        memory_id: str
    ) -> None:
        """Queue HNSW update (non-blocking)."""
        with self._lock:
            key = (user_id, tier)
            self._pending[key].append((vector, memory_id))

            # Flush if batch size reached
            if len(self._pending[key]) >= self._batch_size:
                self._flush_tier_unsafe(user_id, tier)

    def _flush_loop(self) -> None:
        """Background thread that flushes periodically."""
        while self._running:
            time.sleep(self._flush_interval)
            self._flush_all()

    def _flush_all(self) -> None:
        """Flush all pending updates."""
        with self._lock:
            keys = list(self._pending.keys())
            for user_id, tier in keys:
                self._flush_tier_unsafe(user_id, tier)

    def _flush_tier_unsafe(self, user_id: int, tier: str) -> None:
        """Flush tier updates (must hold lock)."""
        key = (user_id, tier)
        updates = self._pending.pop(key, [])

        if not updates:
            return

        try:
            if MEMORY_AVAILABLE:
                memory_store = get_memory_store()
                # Get the appropriate index
                index = memory_store.get_hnsw_index(user_id, tier)
                if index is not None:
                    vectors = np.array([v for v, _ in updates])
                    ids = [int(id.split('_')[-1]) if '_' in id else hash(id) % (2**31)
                           for _, id in updates]
                    index.add(vectors, ids)
                    print(f"[BatchedHNSW] Flushed {len(updates)} updates for user={user_id} tier={tier}")
        except Exception as e:
            print(f"[BatchedHNSW] Flush failed: {e}")

    def shutdown(self) -> None:
        """Shutdown the updater."""
        self._running = False
        self._flush_all()
        self._flush_thread.join(timeout=2.0)


# =============================================================================
# PREDICTIVE CACHE
# =============================================================================

class PredictiveCache:
    """
    Pre-compute likely next operations during idle time.
    """

    def __init__(self):
        self._user_context: Dict[int, Dict[str, Any]] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def on_response_delivered(
        self,
        user_id: int,
        context: BackboneContext,
        async_queue: 'AsyncOperationQueue'
    ) -> None:
        """Start pre-computing for likely next turn."""
        # Cache context
        with self._lock:
            self._user_context[user_id] = {
                "last_input": context.user_input,
                "last_intent": context.intent_result,
                "session_id": context.session_id,
                "timestamp": context.now.isoformat(),
            }

        # Queue predictive operations
        async_queue.enqueue(
            lambda: self._precompute_embeddings(user_id, context),
            priority=PRIORITY_IDLE,
            category="predictive"
        )

    def _precompute_embeddings(self, user_id: int, context: BackboneContext) -> None:
        """Pre-compute embeddings for user's frequent patterns."""
        if not EMBEDDER_AVAILABLE:
            return

        try:
            # Pre-warm common follow-up patterns
            follow_up_patterns = [
                "yes",
                "no",
                "tell me more",
                "what else",
                "thank you",
                "that's helpful",
            ]

            for pattern in follow_up_patterns:
                cache_key = f"common:{pattern}"
                if cache_key not in self._embedding_cache:
                    emb = embed_text(pattern)
                    with self._lock:
                        self._embedding_cache[cache_key] = emb

        except Exception as e:
            print(f"[PredictiveCache] Pre-compute failed: {e}")

    def get_cached_context(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached context for user."""
        with self._lock:
            return self._user_context.get(user_id)

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        cache_key = f"common:{text.lower().strip()}"
        with self._lock:
            return self._embedding_cache.get(cache_key)


# =============================================================================
# ASYNC OPERATION QUEUE
# =============================================================================

class AsyncOperationQueue:
    """
    Central async queue for all non-blocking operations.
    Coordinates memory writes, adaptation, proactive, HNSW, and logging.
    """

    def __init__(self, num_workers: int = ASYNC_WORKERS):
        self._queue = queue.PriorityQueue()
        self._workers: List[threading.Thread] = []
        self._running = True
        self._task_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._lock = threading.Lock()

        # Category tracking
        self._category_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"queued": 0, "completed": 0, "failed": 0}
        )

        # Start workers
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncOp-Worker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)

    def _worker_loop(self) -> None:
        """Worker thread processing tasks."""
        while self._running:
            try:
                task = self._queue.get(timeout=1.0)
                if task is None:
                    break

                try:
                    task.operation()
                    with self._lock:
                        self._completed_count += 1
                        self._category_stats[task.category]["completed"] += 1
                except Exception as e:
                    print(f"[AsyncQueue] Task {task.task_id} failed: {e}")
                    with self._lock:
                        self._failed_count += 1
                        self._category_stats[task.category]["failed"] += 1
                finally:
                    self._queue.task_done()

            except queue.Empty:
                continue

    def enqueue(
        self,
        operation: Callable[[], Any],
        priority: int = PRIORITY_NORMAL,
        category: str = "general",
        task_id: Optional[str] = None
    ) -> str:
        """
        Enqueue an async operation (non-blocking).

        Returns task_id for tracking.
        """
        if task_id is None:
            with self._lock:
                self._task_count += 1
                task_id = f"async_{category}_{self._task_count}"

        task = BackboneTask(
            task_id=task_id,
            operation=operation,
            priority=priority,
            category=category
        )

        with self._lock:
            self._category_stats[category]["queued"] += 1

        self._queue.put(task)
        return task_id

    def enqueue_memory_write(
        self,
        user_id: int,
        role: str,
        text: str,
        session_id: str,
        ts: datetime,
        temporal_tags: Dict[str, Any],
        source: str = "backbone",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convenience method for memory writes."""
        def _write():
            if MEMORY_AVAILABLE:
                store = get_memory_store()
                store.ingest_event(
                    user_id=user_id,
                    role=role,
                    text=text,
                    session_id=session_id,
                    ts=ts,
                    temporal_tags=temporal_tags,
                    source=source,
                    metadata=metadata or {}
                )

        return self.enqueue(_write, priority=PRIORITY_HIGH, category="memory")

    def enqueue_adaptation_update(
        self,
        user_id: int,
        user_text: str
    ) -> str:
        """Convenience method for adaptation updates."""
        def _update():
            if ADAPTATION_AVAILABLE:
                engine = get_adaptation_engine()
                # Create simple metrics object
                class SimpleMetrics:
                    def __init__(self, text):
                        self.user_text = text
                        self.prefers_direct = False
                engine.update(user_id, SimpleMetrics(user_text))

        return self.enqueue(_update, priority=PRIORITY_NORMAL, category="adaptation")

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "queued": self._queue.qsize(),
                "total_tasks": self._task_count,
                "completed": self._completed_count,
                "failed": self._failed_count,
                "by_category": dict(self._category_stats),
            }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the queue."""
        self._running = False

        # Send shutdown signals
        for _ in self._workers:
            self._queue.put(None)

        if wait:
            for worker in self._workers:
                worker.join(timeout=2.0)


# =============================================================================
# BACKBONE LAYER
# =============================================================================

class BackboneLayer:
    """
    Central coordination layer for all Nura engines.

    Provides:
        - Critical path processing (blocking, fast)
        - Async operations (non-blocking, after response)
        - Deferred proactive evaluation
        - Batched HNSW updates
        - Predictive caching
    """

    def __init__(self):
        # Async components
        self._async_queue = AsyncOperationQueue(num_workers=ASYNC_WORKERS)
        self._deferred_proactive = DeferredProactiveQueue()
        self._batched_hnsw = BatchedHNSWUpdater()
        self._predictive_cache = PredictiveCache()

        # Engine instances (lazy loaded)
        self._memory_store = None
        self._temporal_engine = None
        self._adaptation_engine = None
        self._proactive_engine = None
        self._intent_gate = None
        self._safety_layer = None

        # Thread pool for parallel critical operations
        self._critical_executor = ThreadPoolExecutor(
            max_workers=CRITICAL_WORKERS,
            thread_name_prefix="CriticalPath"
        )

        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of engines."""
        if self._initialized:
            return

        if MEMORY_AVAILABLE:
            try:
                self._memory_store = get_memory_store()
            except Exception:
                pass

        if TEMPORAL_AVAILABLE:
            try:
                self._temporal_engine = get_temporal_engine()
            except Exception:
                pass

        if ADAPTATION_AVAILABLE:
            try:
                self._adaptation_engine = get_adaptation_engine()
            except Exception:
                pass

        if PROACTIVE_AVAILABLE:
            try:
                self._proactive_engine = get_proactive_engine()
            except Exception:
                pass

        if ORCHESTRATOR_AVAILABLE:
            try:
                self._intent_gate = get_intent_gate()
            except Exception:
                pass

        if SAFETY_AVAILABLE:
            try:
                self._safety_layer = get_safety_layer()
            except Exception:
                pass

        self._initialized = True

    # -------------------------------------------------------------------------
    # CRITICAL PATH (Blocking Operations)
    # -------------------------------------------------------------------------

    def process_critical_path(
        self,
        user_input: str,
        user_id: int,
        session_id: Optional[str] = None
    ) -> BackboneContext:
        """
        Process the critical path (blocking).

        This includes:
            1. Safety check (parallel with intent)
            2. Intent classification (parallel with safety)
            3. Temporal parsing
            4. Retrieval (if needed)

        Returns context with results for LLM.
        """
        self._ensure_initialized()
        start_time = time.perf_counter()

        now = datetime.now(timezone.utc)
        if session_id is None:
            session_id = f"backbone_{user_id}_{int(now.timestamp())}"

        ctx = BackboneContext(
            user_id=user_id,
            user_input=user_input,
            session_id=session_id,
            now=now,
        )

        # === PARALLEL: Safety + Intent ===
        safety_future = None
        intent_future = None

        if self._safety_layer:
            safety_future = self._critical_executor.submit(
                self._safety_layer.assess, str(user_id), user_input
            )

        if self._intent_gate:
            intent_future = self._critical_executor.submit(
                self._intent_gate.classify, user_input
            )

        # Wait for parallel results
        if safety_future:
            try:
                safety_result = safety_future.result(timeout=2.0)
                ctx.safety_passed = not safety_result.should_block
                ctx.timing["safety"] = (time.perf_counter() - start_time) * 1000
            except Exception:
                ctx.safety_passed = True  # Default to pass on error

        if intent_future:
            try:
                intent_result = intent_future.result(timeout=2.0)
                ctx.intent_result = intent_result.to_dict() if hasattr(intent_result, 'to_dict') else intent_result
                ctx.timing["intent"] = (time.perf_counter() - start_time) * 1000
            except Exception:
                ctx.intent_result = {"primary_intent": "GENERAL_KNOWLEDGE", "confidence": 0.5}

        if not ctx.safety_passed:
            return ctx

        # === TEMPORAL PARSING ===
        temporal_start = time.perf_counter()
        if self._temporal_engine:
            try:
                result = self._temporal_engine.parse(user_input, now)
                ctx.temporal_context = self._temporal_engine.generate_temporal_tags(now)
                ctx.temporal_rewrite = {
                    "start_ts": result.start_ts,
                    "end_ts": result.end_ts,
                    "granularity": result.granularity.value if result.granularity else None,
                    "window_days": result.retrieval_window_days,
                    "requires_clarification": result.requires_clarification,
                }
            except Exception:
                ctx.temporal_context = {}
                ctx.temporal_rewrite = {}
        ctx.timing["temporal"] = (time.perf_counter() - temporal_start) * 1000

        # === RETRIEVAL (if needed) ===
        primary_intent = ctx.intent_result.get("primary_intent", "GENERAL_KNOWLEDGE")
        if primary_intent == "PAST_SELF_REFERENCE" and RETRIEVAL_AVAILABLE:
            retrieval_start = time.perf_counter()
            try:
                if self._memory_store and self._temporal_engine:
                    retrieval_engine = RetrievalEngine(self._memory_store, self._temporal_engine)
                    state = RetrievalState(
                        user_id=user_id,
                        query=user_input,
                        now=now,
                        current_temporal_tags=ctx.temporal_context or {},
                        temporal_rewrite=ctx.temporal_rewrite or {},
                    )
                    ctx.retrieval_result = retrieval_engine.retrieve(state)
            except Exception as e:
                print(f"[Backbone] Retrieval failed: {e}")
            ctx.timing["retrieval"] = (time.perf_counter() - retrieval_start) * 1000

        ctx.timing["critical_total"] = (time.perf_counter() - start_time) * 1000
        return ctx

    # -------------------------------------------------------------------------
    # ASYNC OPERATIONS (After Response)
    # -------------------------------------------------------------------------

    def queue_async_operations(
        self,
        ctx: BackboneContext,
        llm_output: str
    ) -> List[str]:
        """
        Queue async operations to run after response.

        Returns list of task IDs.
        """
        task_ids = []
        primary_intent = ctx.intent_result.get("primary_intent", "GENERAL_KNOWLEDGE")

        # === Memory Write (if appropriate intent) ===
        if primary_intent in ["PERSONAL_STATE", "EXPLICIT_MEMORY_COMMAND"]:
            task_id = self._async_queue.enqueue_memory_write(
                user_id=ctx.user_id,
                role="user",
                text=ctx.user_input,
                session_id=ctx.session_id,
                ts=ctx.now,
                temporal_tags=ctx.temporal_context or {},
                source="backbone",
                metadata={"intent": primary_intent}
            )
            ctx.memory_task_id = task_id
            task_ids.append(task_id)

        # === Adaptation Update (if PERSONAL_STATE) ===
        if primary_intent == "PERSONAL_STATE":
            task_id = self._async_queue.enqueue_adaptation_update(
                user_id=ctx.user_id,
                user_text=ctx.user_input
            )
            ctx.adaptation_task_id = task_id
            task_ids.append(task_id)

        # === Deferred Proactive Evaluation ===
        if primary_intent == "PERSONAL_STATE" and not ctx.intent_result.get("ambiguity", False):
            proactive_context = self._build_proactive_context(ctx)
            task_id = self._deferred_proactive.queue_evaluation(
                user_id=ctx.user_id,
                context=proactive_context,
                async_queue=self._async_queue
            )
            ctx.proactive_task_id = task_id
            task_ids.append(task_id)

        # === Predictive Cache Update ===
        ctx.llm_output = llm_output
        self._predictive_cache.on_response_delivered(
            user_id=ctx.user_id,
            context=ctx,
            async_queue=self._async_queue
        )

        return task_ids

    def _build_proactive_context(self, ctx: BackboneContext) -> Dict[str, Any]:
        """Build context for proactive evaluation."""
        recent_memories = []
        if self._memory_store:
            try:
                recent = self._memory_store.recent(ctx.user_id, k=20)
                for mem in recent:
                    recent_memories.append({
                        "memory_id": mem.get("id"),
                        "content": mem.get("content"),
                        "memory_type": mem.get("memory_type"),
                        "importance": mem.get("importance"),
                        "created_at": mem.get("created_at"),
                        "focus": mem.get("content", "")[:100],
                    })
            except Exception:
                pass

        return {
            "user_id": str(ctx.user_id),
            "now_timestamp": ctx.now.isoformat(),
            "recent_memories": recent_memories,
            "temporal_tags": list((ctx.temporal_context or {}).keys()),
            "cooldown_state": {"last_asked_at": None, "asks_today": 0},
        }

    # -------------------------------------------------------------------------
    # HNSW BATCH UPDATES
    # -------------------------------------------------------------------------

    def queue_hnsw_update(
        self,
        user_id: int,
        tier: str,
        vector: np.ndarray,
        memory_id: str
    ) -> None:
        """Queue HNSW index update (batched)."""
        self._batched_hnsw.queue_update(user_id, tier, vector, memory_id)

    # -------------------------------------------------------------------------
    # PROACTIVE ACCESS
    # -------------------------------------------------------------------------

    def get_proactive_result(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached proactive result from previous turn."""
        return self._deferred_proactive.get_cached_result(user_id)

    def is_proactive_pending(self, user_id: int) -> bool:
        """Check if proactive evaluation is still pending."""
        return self._deferred_proactive.is_pending(user_id)

    # -------------------------------------------------------------------------
    # FULL PIPELINE
    # -------------------------------------------------------------------------

    def process(
        self,
        user_input: str,
        user_id: int,
        llm_callable: Callable[[BackboneContext], str],
        session_id: Optional[str] = None
    ) -> BackboneResult:
        """
        Full backbone processing pipeline.

        Args:
            user_input: User's message
            user_id: User identifier
            llm_callable: Function that takes context and returns LLM response
            session_id: Optional session ID

        Returns:
            BackboneResult with response and metadata
        """
        # === CRITICAL PATH ===
        ctx = self.process_critical_path(user_input, user_id, session_id)

        # Check safety
        if not ctx.safety_passed:
            return BackboneResult(
                llm_output="I'm sorry, I can't help with that.",
                context=ctx,
                timing=ctx.timing,
            )

        # === LLM RESPONSE ===
        llm_start = time.perf_counter()
        try:
            llm_output = llm_callable(ctx)
        except Exception as e:
            print(f"[Backbone] LLM failed: {e}")
            llm_output = self._generate_fallback(ctx)
        ctx.timing["llm"] = (time.perf_counter() - llm_start) * 1000

        # === ASYNC OPERATIONS (after response) ===
        async_tasks = self.queue_async_operations(ctx, llm_output)

        return BackboneResult(
            llm_output=llm_output,
            context=ctx,
            async_tasks=async_tasks,
            timing=ctx.timing,
        )

    def _generate_fallback(self, ctx: BackboneContext) -> str:
        """Generate fallback response."""
        if ctx.temporal_rewrite and ctx.temporal_rewrite.get("requires_clarification"):
            return "Could you clarify which specific time you mean?"

        if ctx.retrieval_result:
            return "I found some relevant context. What would you like to know?"

        return "I'm here to help. Could you tell me more?"

    # -------------------------------------------------------------------------
    # STATS & SHUTDOWN
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get backbone statistics."""
        return {
            "async_queue": self._async_queue.get_stats(),
            "proactive_cache_size": len(self._deferred_proactive._results_cache),
            "predictive_cache_size": len(self._predictive_cache._user_context),
        }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown backbone layer."""
        self._async_queue.shutdown(wait=wait)
        self._batched_hnsw.shutdown()
        self._critical_executor.shutdown(wait=wait)

    # -------------------------------------------------------------------------
    # VOICE I/O (Unified with Engine Processing)
    # -------------------------------------------------------------------------

    def _ensure_voice_initialized(self) -> None:
        """Initialize voice components on first use."""
        if not hasattr(self, '_vad'):
            self._vad = None
            self._stt = None
            self._tts = None
            self._llm = None

        if self._vad is None and VAD_AVAILABLE:
            try:
                self._vad = SileroVAD(VADConfig(
                    sample_rate=16000,
                    chunk_size_ms=30,
                    speech_threshold=0.5,
                    min_silence_ms=500
                ))
            except Exception as e:
                print(f"[Backbone] VAD init failed: {e}")

        if self._stt is None and STT_AVAILABLE:
            try:
                import os
                device = "cuda" if os.path.exists("/dev/nvidia0") or os.name == "nt" else "cpu"
                self._stt = WhisperModel("base.en", device=device, compute_type="float16" if device == "cuda" else "int8")
            except Exception as e:
                print(f"[Backbone] STT init failed: {e}")

        if self._tts is None and TTS_AVAILABLE:
            try:
                self._tts = KokoroTTS(TTSConfig(voice="af_heart", use_onnx=True))
            except Exception as e:
                print(f"[Backbone] TTS init failed: {e}")

        if self._llm is None and LLM_STREAMING_AVAILABLE:
            try:
                self._llm = FastPhiStreamingLLM()
            except Exception as e:
                print(f"[Backbone] LLM init failed: {e}")

    def process_voice_turn(
        self,
        audio_bytes: bytes,
        user_id: int,
        on_audio: Callable[[bytes], None],
        session_id: Optional[str] = None
    ) -> 'VoiceResult':
        """
        Process a complete voice turn: Audio In → Engines → Audio Out

        This is the unified entry point for voice interaction.
        Target: <500ms to first sentence heard.

        Args:
            audio_bytes: Raw PCM audio (16-bit, 16kHz, mono)
            user_id: User identifier
            on_audio: Callback for audio output (called with chunks)
            session_id: Optional session ID

        Returns:
            VoiceResult with transcript, response, and timing
        """
        self._ensure_initialized()
        self._ensure_voice_initialized()

        start_time = time.perf_counter()
        timing = {}

        # === STAGE 1: STT ===
        stt_start = time.perf_counter()
        transcript = self._transcribe(audio_bytes)
        timing["stt"] = (time.perf_counter() - stt_start) * 1000

        if not transcript:
            return VoiceResult(
                transcript="",
                response="",
                timing=timing,
                success=False
            )

        # === STAGE 2: CRITICAL PATH (Safety + Intent + Engines) ===
        ctx = self.process_critical_path(transcript, user_id, session_id)
        timing["backbone"] = ctx.timing.get("critical_total", 0)

        if not ctx.safety_passed:
            refusal = "I'm sorry, I can't help with that."
            self._synthesize_and_output(refusal, on_audio)
            return VoiceResult(
                transcript=transcript,
                response=refusal,
                timing=timing,
                success=True
            )

        # === STAGE 3: LLM + TTS STREAMING ===
        response, stream_timing = self._stream_llm_tts(ctx, on_audio, start_time)
        timing.update(stream_timing)

        # === STAGE 4: ASYNC OPERATIONS ===
        self.queue_async_operations(ctx, response)

        timing["total"] = (time.perf_counter() - start_time) * 1000
        return VoiceResult(
            transcript=transcript,
            response=response,
            timing=timing,
            success=True
        )

    def _transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio to text using Faster Whisper."""
        if self._stt is None:
            return ""

        try:
            # Convert bytes to float32 numpy array
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            segments, _ = self._stt.transcribe(
                audio,
                beam_size=1,
                language="en",
                vad_filter=False,
                without_timestamps=True
            )

            text = " ".join(seg.text.strip() for seg in segments)
            return text.strip()
        except Exception as e:
            print(f"[Backbone] STT error: {e}")
            return ""

    def _stream_llm_tts(
        self,
        ctx: BackboneContext,
        on_audio: Callable[[bytes], None],
        pipeline_start: float
    ) -> Tuple[str, Dict[str, float]]:
        """
        Stream LLM → TTS with sentence buffering.

        Synthesizes first sentence immediately for low latency.
        """
        import re
        import queue
        import threading

        timing = {
            "llm_first_token": 0,
            "llm_first_sentence": 0,
            "tts_first_audio": 0
        }

        if self._llm is None or self._tts is None:
            fallback = self._generate_fallback(ctx)
            self._synthesize_and_output(fallback, on_audio)
            return fallback, timing

        # Build minimal prompt
        prompt = self._build_voice_prompt(ctx)

        # Sentence buffer
        buffer = ""
        full_response = ""
        first_token_recorded = False
        first_sentence_recorded = False
        first_audio_recorded = False
        min_sentence_chars = 20

        # TTS queue for concurrent synthesis
        tts_queue: queue.Queue[Optional[str]] = queue.Queue()

        def tts_worker():
            nonlocal first_audio_recorded, timing
            while True:
                sentence = tts_queue.get()
                if sentence is None:
                    break
                for chunk in self._tts.synthesize_sentence(sentence):
                    if chunk.audio:
                        if not first_audio_recorded:
                            timing["tts_first_audio"] = (time.perf_counter() - pipeline_start) * 1000
                            first_audio_recorded = True
                        on_audio(chunk.audio)

        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

        try:
            for chunk in self._llm.stream_generate(prompt):
                if not first_token_recorded:
                    timing["llm_first_token"] = (time.perf_counter() - pipeline_start) * 1000
                    first_token_recorded = True

                new_text = chunk.text[len(full_response):]
                full_response = chunk.text
                buffer += new_text

                # Check for complete sentence
                match = re.search(r'^(.*?[.!?])(?:\s+|$)', buffer)
                if match and len(match.group(1)) >= min_sentence_chars:
                    sentence = match.group(1).strip()
                    buffer = buffer[match.end():].lstrip()

                    if not first_sentence_recorded:
                        timing["llm_first_sentence"] = (time.perf_counter() - pipeline_start) * 1000
                        first_sentence_recorded = True

                    tts_queue.put(sentence)

                if chunk.is_final:
                    break

            # Flush remaining buffer
            if buffer.strip():
                tts_queue.put(buffer.strip())

        except Exception as e:
            print(f"[Backbone] LLM stream error: {e}")
            full_response = self._generate_fallback(ctx)

        tts_queue.put(None)
        tts_thread.join(timeout=10.0)

        return full_response.strip(), timing

    def _build_voice_prompt(self, ctx: BackboneContext) -> str:
        """Build minimal prompt for voice (fast TTFT)."""
        context = ""

        if ctx.retrieval_result and hasattr(ctx.retrieval_result, 'hits') and ctx.retrieval_result.hits:
            hits = ctx.retrieval_result.hits[:2]
            parts = [h.get('content', '')[:80] for h in hits]
            context = "Context: " + " | ".join(parts) + "\n"

        return f"{context}User: {ctx.user_input}\nAssistant:"

    def _synthesize_and_output(self, text: str, on_audio: Callable[[bytes], None]) -> None:
        """Synthesize text and send to audio output."""
        if self._tts is None:
            return

        for chunk in self._tts.synthesize_stream(text):
            if chunk.audio:
                on_audio(chunk.audio)

    def process_vad_chunk(self, audio_chunk: bytes) -> Optional[VADEvent]:
        """
        Process audio chunk through VAD.

        Returns VADEvent if speech boundary detected.
        Use this for continuous microphone streaming.
        """
        self._ensure_voice_initialized()
        if self._vad is None:
            return None
        return self._vad.process_chunk(audio_chunk)


# =============================================================================
# VOICE RESULT
# =============================================================================

@dataclass
class VoiceResult:
    """Result from voice turn processing."""
    transcript: str
    response: str
    timing: Dict[str, float]
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcript": self.transcript,
            "response": self.response,
            "timing": self.timing,
            "success": self.success
        }


# =============================================================================
# SINGLETON & EXPORTS
# =============================================================================

_backbone_instance: Optional[BackboneLayer] = None


def get_backbone() -> BackboneLayer:
    """Get or create the singleton backbone layer."""
    global _backbone_instance
    if _backbone_instance is None:
        _backbone_instance = BackboneLayer()
    return _backbone_instance


__all__ = [
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
    "PRIORITY_CRITICAL",
    "PRIORITY_HIGH",
    "PRIORITY_NORMAL",
    "PRIORITY_LOW",
    "PRIORITY_IDLE",
]
