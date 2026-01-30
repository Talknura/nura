# Retrieval Engine Architecture

## Multi-Strategy Retrieval for Three-Tier Memory

The Retrieval Engine routes queries to appropriate retrieval strategies based on semantic analysis of the query intent.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              User Query                                     │
│                                  │                                          │
│                                  ▼                                          │
│                    ┌─────────────────────────┐                              │
│                    │   Semantic Query Router  │                              │
│                    │   (Embedding Similarity) │                              │
│                    └───────────┬─────────────┘                              │
│                                │                                            │
│           ┌────────────────────┼────────────────────┐                       │
│           │                    │                    │                       │
│           ▼                    ▼                    ▼                       │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                │
│   │   FACTUAL     │   │   EPISODIC    │   │   MILESTONE   │                │
│   │  "my name?"   │   │  "yesterday?" │   │  "when did?"  │                │
│   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘                │
│           │                   │                   │                         │
│           ▼                   ▼                   ▼                         │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                │
│   │  FACTS TIER   │   │ EPISODES TIER │   │MILESTONES TIER│                │
│   │   (HNSW)      │   │    (HNSW)     │   │    (HNSW)     │                │
│   │   No Decay    │   │ Recency Decay │   │   No Decay    │                │
│   └───────────────┘   └───────────────┘   └───────────────┘                │
│                                                                             │
│   Additional Strategies:                                                    │
│   ┌───────────────┐   ┌───────────────┐                                    │
│   │    HYBRID     │   │   TIMELINE    │                                    │
│   │ "tell me..."  │   │ "this week?"  │                                    │
│   │  All Tiers    │   │ Chronological │                                    │
│   └───────────────┘   └───────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Retrieval Strategies

### 1. FACTUAL Strategy
**Trigger**: Questions about personal attributes
**Examples**: "What's my name?", "Where do I work?", "Who is my wife?"

```
Query → FACTS tier (HNSW) → Direct key-value match → No decay
```

| Property | Value |
|----------|-------|
| Primary Tier | Facts (80% weight) |
| Decay | None |
| Latency | ~5ms (HNSW lookup) |

### 2. EPISODIC Strategy
**Trigger**: Conversation recall queries
**Examples**: "What did we talk about yesterday?", "Do you remember when I told you...?"

```
Query → EPISODES tier → Recency scoring → Time-filtered results
```

| Property | Value |
|----------|-------|
| Primary Tier | Episodes (80% weight) |
| Decay | 60-90 day half-life |
| Special | Disable recency for explicit past references |

### 3. MILESTONE Strategy
**Trigger**: Life event queries
**Examples**: "When did my dad pass away?", "When did I get married?"

```
Query → MILESTONES tier → Event type match → No decay
```

| Property | Value |
|----------|-------|
| Primary Tier | Milestones (80% weight) |
| Decay | None (permanent) |
| Latency | ~5ms (HNSW lookup) |

### 4. HYBRID Strategy
**Trigger**: Ambiguous or broad queries
**Examples**: "Tell me about my family", "What do you know about me?"

```
Query → ALL tiers → Unified scoring → Merged results
```

| Property | Value |
|----------|-------|
| Tiers | Facts (33%) + Milestones (33%) + Episodes (34%) |
| Decay | Per-tier (none for facts/milestones, recency for episodes) |

### 5. TIMELINE Strategy
**Trigger**: Temporal range queries
**Examples**: "What happened this week?", "Summary of this month"

```
Query → Episodes + Milestones → Time filter → Chronological order
```

| Property | Value |
|----------|-------|
| Primary Tier | Episodes (70%) + Milestones (30%) |
| Ordering | Chronological (not by score) |
| Filter | Time window (day/week/month) |

## Query Router

The `SemanticQueryRouter` uses embedding similarity to classify queries:

```python
from app.retrieval.retrieval_strategies import get_query_router

router = get_query_router()
analysis = router.analyze("What's my dog's name?")

# analysis.strategy = RetrievalStrategy.FACTUAL
# analysis.confidence = 0.85
# analysis.tier_weights = {"facts": 0.8, "milestones": 0.1, "episodes": 0.1}
```

### Pattern Groups

| Group | Example Patterns |
|-------|------------------|
| FACTUAL | "what is my name", "where do I live", "who is my" |
| EPISODIC | "what did we talk about", "do you remember when" |
| MILESTONE | "when did my father pass", "when did I get married" |
| TIMELINE | "what happened this week", "summary of today" |
| PAST_REFERENCE | "last year", "years ago", "back when" |

## Ranking & Scoring

### Score Components

```python
score = (
    semantic_similarity * 0.40 +    # How relevant to query
    recency_score * 0.20 +          # How recent (adaptive decay)
    importance * 0.15 +             # Memory importance
    temporal_match * 0.10 +         # Same day/hour context
    emotional_boost * 0.05 +        # Emotional significance
    identity_boost * 0.05           # Core identity relevance
)
```

### Adaptive Decay Rates

| Memory Category | Half-Life | Rationale |
|-----------------|-----------|-----------|
| Identity facts | ∞ (no decay) | "My name is Sam" never expires |
| Family info | ∞ (no decay) | "My wife is Jane" permanent |
| Health conditions | ∞ (no decay) | "I have diabetes" permanent |
| Milestones | ∞ (no decay) | Life events permanent |
| Tasks/Events | 60 days | "Meeting tomorrow" fades |
| Casual mentions | 90 days | "I had pizza" fades |

### Explicit Past References

When query contains past time references ("last year", "back when"), recency penalty is disabled:

```python
# Query: "What did I tell you last Christmas?"
analysis = router.analyze(query)
# analysis.disable_recency = True
# → Old memories not penalized
```

## File Structure

```
app/retrieval/
├── retrieval_engine_v2.py    # Main engine with strategy routing
├── retrieval_strategies.py   # Query router & strategies
├── ranker.py                 # Scoring & ranking
├── query_parser.py           # Temporal parsing
├── filters.py                # Time/type filters
└── ARCHITECTURE.md           # This file
```

## Usage

### Basic Retrieval

```python
from app.retrieval.retrieval_engine_v2 import RetrievalEngineV2, RetrievalState
from datetime import datetime, timezone

engine = RetrievalEngineV2(memory_engine, temporal_engine)

result = engine.retrieve(RetrievalState(
    user_id=1,
    query="What's my wife's name?",
    now=datetime.now(timezone.utc),
    top_k=5
))

# result.strategy_used = RetrievalStrategy.FACTUAL
# result.hits = [{"fact_key": "spouse_name", "fact_value": "Jane", ...}]
# result.facts = {"spouse_name": "Jane", "name": "Sam", ...}
```

### With Temporal Context

```python
result = engine.retrieve(RetrievalState(
    user_id=1,
    query="What did we discuss this morning?",
    now=datetime.now(timezone.utc),
    current_temporal_tags={"day_of_week": "Monday", "hour_of_day": 14},
    top_k=10
))

# result.strategy_used = RetrievalStrategy.EPISODIC
# result.query_analysis.time_window_days = 1
```

## Performance

| Strategy | Avg Latency | Notes |
|----------|-------------|-------|
| FACTUAL | ~5-10ms | HNSW facts lookup |
| MILESTONE | ~5-10ms | HNSW milestones lookup |
| EPISODIC | ~10-20ms | HNSW episodes + scoring |
| TIMELINE | ~15-25ms | Recent fetch + filter |
| HYBRID | ~25-40ms | All tiers + merge |

## Design Principles

1. **Query-Aware Routing**: Different queries need different retrieval approaches
2. **Tier Specialization**: Facts/Milestones are permanent, Episodes decay
3. **HNSW Throughout**: O(log n) retrieval for all tiers
4. **Explicit Past Handling**: "Last year" queries don't penalize old memories
5. **Graceful Fallback**: Keyword matching when embeddings unavailable

## Frozen: January 2025

This architecture is stable and compatible with the Three-Tier Memory Engine.
