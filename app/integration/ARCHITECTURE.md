# Backbone Layer Architecture

## Central Async Coordination for Nura Engines

The Backbone Layer is the central nervous system that coordinates all Nura engines with optimal parallelism and async operations for minimum latency.

```
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
```

## Core Components

### 1. BackboneLayer

The central coordinator that manages all engine interactions.

```python
from app.integration import get_backbone

backbone = get_backbone()

# Full pipeline with LLM
result = backbone.process(
    user_input="What did I tell you about my job?",
    user_id=123,
    llm_callable=my_llm_function
)

print(result.llm_output)
print(result.timing)  # {"critical_total": 45.2, "llm": 120.5, ...}
```

### 2. Critical Path Processing

Blocking operations that must complete before response:

| Step | Operation | Target Latency |
|------|-----------|----------------|
| 1 | Safety Check (parallel) | ~1ms |
| 2 | Intent Classification (parallel) | ~5ms |
| 3 | Temporal Parsing | ~1-6ms |
| 4 | Retrieval (if needed) | ~10-50ms |
| **Total** | Critical Path | **~20-60ms** |

```python
# Just critical path (no LLM)
ctx = backbone.process_critical_path(
    user_input="Remember I like mornings",
    user_id=123
)

print(ctx.intent_result)  # {"primary_intent": "EXPLICIT_MEMORY_COMMAND", ...}
print(ctx.temporal_context)  # {"morning": True, ...}
```

### 3. AsyncOperationQueue

Non-blocking operations after response is delivered:

```python
queue = backbone._async_queue

# Queue memory write
task_id = queue.enqueue_memory_write(
    user_id=123,
    role="user",
    text="I prefer mornings",
    session_id="session_1",
    ts=datetime.now(timezone.utc),
    temporal_tags={"morning": True}
)

# Check stats
print(queue.get_stats())
# {"queued": 3, "completed": 10, "failed": 0, "by_category": {...}}
```

### 4. DeferredProactiveQueue

Proactive evaluation happens AFTER response is delivered, results cached for next turn:

```python
# Check if proactive ready from last turn
result = backbone.get_proactive_result(user_id=123)
if result and result.get("should_ask"):
    # Include proactive question in response
    pass

# Proactive is automatically queued after PERSONAL_STATE intents
```

### 5. BatchedHNSWUpdater

HNSW index updates are batched for efficiency:

```python
# Queue individual updates (non-blocking)
backbone.queue_hnsw_update(
    user_id=123,
    tier="FACTS",
    vector=embedding_vector,
    memory_id="mem_123"
)

# Automatic flush every 5 seconds or 10 items
```

### 6. PredictiveCache

Pre-compute likely operations during idle time:

- Common follow-up embeddings ("yes", "no", "tell me more")
- User context caching
- Recent memory prefetch

## Task Priorities

| Priority | Level | Use Case |
|----------|-------|----------|
| PRIORITY_CRITICAL | 0 | Safety checks |
| PRIORITY_HIGH | 1 | Memory writes |
| PRIORITY_NORMAL | 2 | Adaptation updates |
| PRIORITY_LOW | 3 | Proactive evaluation |
| PRIORITY_IDLE | 4 | Predictive caching |

## Latency Optimization Strategy

### Before (Blocking Everything)

```
User Input
    │
    ▼
Safety Check ──────────────────────────────────── 1ms
    │
    ▼
Intent Classification ─────────────────────────── 5ms
    │
    ▼
Temporal Parse ────────────────────────────────── 6ms
    │
    ▼
Memory Write ──────────────────────────────────── 10ms  ← BLOCKING
    │
    ▼
Retrieval ─────────────────────────────────────── 50ms
    │
    ▼
Adaptation Update ─────────────────────────────── 10ms  ← BLOCKING
    │
    ▼
Proactive Evaluation ──────────────────────────── 10ms  ← BLOCKING
    │
    ▼
LLM Response ──────────────────────────────────── 200ms
    │
    ▼
TOTAL: ~292ms before response
```

### After (Backbone Layer)

```
User Input
    │
    ├──────────────┐
    ▼              ▼
Safety Check   Intent Class  ─────────────────── 5ms (parallel)
    │              │
    └──────────────┘
           │
           ▼
    Temporal Parse ────────────────────────────── 6ms
           │
           ▼
    Retrieval (if needed) ─────────────────────── 50ms
           │
           ▼
    LLM Response ──────────────────────────────── 200ms
           │
           ▼
    RESPONSE DELIVERED ────────────────────────── ~260ms
           │
           └─────────────────────────────────────────────────┐
                                                              ▼
                              ┌────────────────────────────────────────┐
                              │         ASYNC (Background)             │
                              │                                        │
                              │  Memory Write ──────────── 10ms        │
                              │  Adaptation Update ─────── 10ms        │
                              │  Proactive Eval ────────── 10ms        │
                              │  HNSW Batch ────────────── batched     │
                              │                                        │
                              └────────────────────────────────────────┘
```

**Result: ~260ms vs ~292ms = 32ms saved + better perceived latency**

## Usage Examples

### Full Pipeline

```python
from app.integration import get_backbone

backbone = get_backbone()

def my_llm(ctx):
    # Build prompt from context
    return generate_response(ctx)

result = backbone.process(
    user_input="I'm feeling stressed about work",
    user_id=123,
    llm_callable=my_llm
)

# Response is delivered immediately
print(result.llm_output)

# Async tasks are running in background
print(result.async_tasks)  # ["async_memory_1", "async_adaptation_1", ...]
```

### Critical Path Only

```python
ctx = backbone.process_critical_path(
    user_input="What did I say about meetings?",
    user_id=123
)

# Use ctx for LLM
llm_output = generate_response(ctx)

# Manually queue async if needed
backbone.queue_async_operations(ctx, llm_output)
```

### Check Proactive

```python
# At start of turn, check for cached proactive
proactive = backbone.get_proactive_result(user_id=123)
if proactive and proactive.get("should_ask"):
    include_proactive = True
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| ASYNC_WORKERS | 3 | Worker threads for async queue |
| CRITICAL_WORKERS | 2 | Worker threads for parallel critical ops |
| HNSW_BATCH_SIZE | 10 | Flush HNSW after this many updates |
| HNSW_FLUSH_INTERVAL | 5.0s | Maximum time before HNSW flush |

## File Structure

```
app/integration/
├── backbone.py              # Central backbone layer
├── async_memory_queue.py    # Legacy async queue
├── parallel_engine_executor.py  # Legacy parallel executor
├── optimized_voice_pipeline.py  # Voice pipeline
├── __init__.py              # Module exports
└── ARCHITECTURE.md          # This file
```

## Engine Integration

The backbone integrates with all Nura engines:

| Engine | Critical Path | Async |
|--------|---------------|-------|
| Safety Layer | Yes (parallel) | - |
| Intent Gate | Yes (parallel) | - |
| Temporal Engine | Yes | - |
| Retrieval Engine | Yes (conditional) | - |
| Memory Engine | - | Yes (write) |
| Adaptation Engine | - | Yes (update) |
| Proactive Engine | - | Yes (deferred) |

## Thread Safety

- All queues use proper locking
- Singleton access is thread-safe
- Background workers use daemon threads
- Graceful shutdown with `backbone.shutdown()`

## Monitoring

```python
stats = backbone.get_stats()
print(stats)
# {
#     "async_queue": {"queued": 2, "completed": 50, "failed": 0, ...},
#     "proactive_cache_size": 5,
#     "predictive_cache_size": 3
# }
```

## Design Principles

1. **Latency First**: Critical path is minimal and fast
2. **Non-Blocking Writes**: All writes happen after response
3. **Deferred Evaluation**: Proactive runs in background
4. **Batched Updates**: HNSW updates are batched for efficiency
5. **Predictive Caching**: Pre-compute during idle time
6. **Graceful Degradation**: Missing engines don't break the pipeline

## Frozen: January 2025

This architecture provides the async optimization layer for all Nura engines.
