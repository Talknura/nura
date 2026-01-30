"""
Backbone Layer - Central Async Coordination for Nura Engines.

The backbone layer is the central nervous system that coordinates all engines
with optimal parallelism and async operations for minimum latency.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           BACKBONE LAYER                                    │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │   ┌───────────────────────────────────────────────────────────────────┐    │
    │   │                    CRITICAL PATH (Blocking)                        │    │
    │   │   VAD → STT → Safety+Intent → Temporal → Retrieval → LLM → TTS    │    │
    │   │                        Target: <1500ms                             │    │
    │   └───────────────────────────────────────────────────────────────────┘    │
    │                                                                             │
    │   ┌───────────────────────────────────────────────────────────────────┐    │
    │   │                    ASYNC AFTER RESPONSE                            │    │
    │   │   Memory Write │ Adaptation │ Proactive │ HNSW Update │ Logging   │    │
    │   │                        Non-blocking                                │    │
    │   └───────────────────────────────────────────────────────────────────┘    │
    │                                                                             │
    │   ┌───────────────────────────────────────────────────────────────────┐    │
    │   │                    PREDICTIVE (Idle Time)                          │    │
    │   │   Pre-warm Embeddings │ Pre-fetch Memories │ Cache Context        │    │
    │   └───────────────────────────────────────────────────────────────────┘    │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

Latency Optimization:
    - Critical path only blocks on essential operations
    - All writes happen after response starts
    - Proactive evaluation deferred to after response
    - HNSW updates batched for efficiency
    - Predictive caching during idle time
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
