# Orchestrator & Intent Gate Architecture

## Central Coordination for Nura Engines

The Orchestrator coordinates all Nura engines based on intent classification. The Intent Gate classifies user input and determines which engines should be activated.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR v2                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            User Input                                       │
│                                │                                            │
│                                ▼                                            │
│                   ┌─────────────────────────┐                               │
│                   │      SAFETY LAYER       │ ◄── Content filtering         │
│                   └────────────┬────────────┘                               │
│                                │ pass                                       │
│                                ▼                                            │
│                   ┌─────────────────────────┐                               │
│                   │      INTENT GATE        │ ◄── Semantic classification   │
│                   │                         │                               │
│                   │  1. Fast Path (~0.1ms)  │                               │
│                   │  2. Semantic (~5ms)     │                               │
│                   │  3. Regex (~0.5ms)      │                               │
│                   └────────────┬────────────┘                               │
│                                │                                            │
│                                ▼                                            │
│                   ┌─────────────────────────┐                               │
│                   │    ROUTING POLICY       │                               │
│                   │                         │                               │
│                   │  • engines_required     │                               │
│                   │  • engines_optional     │                               │
│                   │  • engines_forbidden    │                               │
│                   └────────────┬────────────┘                               │
│                                │                                            │
│        ┌───────────┬───────────┼───────────┬───────────┐                    │
│        ▼           ▼           ▼           ▼           ▼                    │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│   │ TEMPORAL│ │ MEMORY  │ │RETRIEVAL│ │ADAPTATION│ │PROACTIVE│              │
│   │ ENGINE  │ │ ENGINE  │ │ ENGINE  │ │ ENGINE  │ │ ENGINE  │              │
│   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘              │
│        │           │           │           │           │                    │
│        └───────────┴───────────┼───────────┴───────────┘                    │
│                                │                                            │
│                                ▼                                            │
│                   ┌─────────────────────────┐                               │
│                   │    LLM INTERFACE        │                               │
│                   │    (Response Gen)       │                               │
│                   └─────────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Intent Gate

### Intent Categories

| Intent | Description | Example |
|--------|-------------|---------|
| EXPLICIT_MEMORY_COMMAND | Store or delete memory | "Remember this", "Forget that" |
| PAST_SELF_REFERENCE | Ask about past conversations | "What did I say about...", "Last time..." |
| PERSONAL_STATE | Share emotions/feelings | "I feel sad", "I'm worried about..." |
| GENERAL_KNOWLEDGE | Factual questions | "What is...", "How does..." |

### Classification Pipeline

```
User Input
    │
    ▼
┌─────────────────────────────┐
│    FAST PATH CACHE          │ ◄── "remember this" → EXPLICIT_MEMORY (1.0)
│    (~0.1ms)                 │     "i feel" → PERSONAL_STATE (0.85)
└──────────────┬──────────────┘
               │ miss
               ▼
┌─────────────────────────────┐
│    SEMANTIC ANALYZER        │ ◄── ML embeddings (all-MiniLM-L6-v2)
│    (~5ms first, ~1ms)       │     Understands meaning, not keywords
└──────────────┬──────────────┘
               │ low confidence
               ▼
┌─────────────────────────────┐
│    REGEX FALLBACK           │ ◄── Pattern matching for edge cases
│    (~0.5ms)                 │
└──────────────┬──────────────┘
               │
               ▼
         IntentResult
```

### Fast Path Cache

Pre-computed classifications for common phrases:

```python
FAST_PATH_INTENTS = {
    "remember this": (INTENT_EXPLICIT_MEMORY, 1.0, ["remember_directive"]),
    "forget that": (INTENT_EXPLICIT_MEMORY, 1.0, ["forget_directive"]),
    "what did i say": (INTENT_PAST_REFERENCE, 0.95, ["recall_past"]),
    "i feel": (INTENT_PERSONAL_STATE, 0.85, ["emotion_state"]),
    "what is": (INTENT_GENERAL_KNOWLEDGE, 0.80, ["factual_question"]),
}
```

### IntentResult Structure

```python
@dataclass
class IntentResult:
    primary_intent: str = "GENERAL_KNOWLEDGE"
    confidence: float = 0.7
    ambiguity: bool = False
    ambiguity_flags: List[str] = []

    # Routing policy
    engines_required: Set[str] = set()
    engines_optional: Set[str] = set()
    engines_forbidden: Set[str] = set()

    # Detection method
    detection_method: str = "semantic"  # semantic, fast_path, regex
```

## Routing Policy

### Policy by Intent

| Intent | Required | Optional | Forbidden |
|--------|----------|----------|-----------|
| GENERAL_KNOWLEDGE | - | - | All engines |
| EXPLICIT_MEMORY_COMMAND | Memory.ingest | Temporal.tags | Retrieval, Adaptation, Proactive |
| PAST_SELF_REFERENCE | Retrieval.retrieve | Temporal.tags | Memory.ingest, Adaptation, Proactive |
| PERSONAL_STATE | Memory.ingest, Adaptation.update | Retrieval, Temporal, Proactive | - |

### Ambiguity Flags

Flags that modify routing policy:

| Flag | Effect |
|------|--------|
| forget_directive | Disable Temporal for memory deletion |
| learning_reflection | Disable Temporal for past learning queries |
| future_event | Disable Adaptation and Retrieval |
| future_evaluative | Disable Adaptation and Retrieval |
| learning_struggle | Disable Temporal |
| persistent_emotion | Disable Temporal |
| positive_state | Disable Proactive |
| state_update | Disable Temporal and Proactive |

## Orchestrator Flow

### 1. Safety Check

```python
safety_layer = get_safety_layer()
decision = safety_layer.assess(user_id, user_input)
if decision.should_block:
    return refusal_message
```

### 2. Intent Classification

```python
intent_gate = get_intent_gate()
intent_result = intent_gate.classify(user_input)
# IntentResult with primary_intent, confidence, routing policy
```

### 3. Engine Activation

```python
# Check policy for each engine
if engine_name in intent_result.engines_required:
    call_engine()
elif engine_name in intent_result.engines_optional and confidence >= 0.4:
    call_engine()
elif engine_name in intent_result.engines_forbidden:
    skip_engine()
```

### 4. Engine Call Order

1. **Temporal Engine** - Time context and temporal rewrite
2. **Memory Engine** - Ingest user input if appropriate
3. **Retrieval Engine** - Find relevant memories
4. **Adaptation Engine** - Update user communication profile
5. **Proactive Engine** - Evaluate follow-up opportunities

### 5. LLM Response

```python
llm_context = {
    "user_input": user_input,
    "intent": intent_result.primary_intent,
    "memory_results": memory_result,
    "retrieved_items": retrieval_result,
    "temporal_context": temporal_context,
    "adaptation_context": adaptation_context,
    "proactive": proactive_result,
}
response = nura_llm_interface.generate_response(llm_context, user_input)
```

## OrchestratorContext

```python
@dataclass
class OrchestratorContext:
    user_id: int
    user_input: str
    session_id: str
    now: datetime

    # Intent classification
    intent_result: Optional[IntentResult] = None

    # Engine outputs
    temporal_context: Optional[Dict] = None
    temporal_rewrite: Optional[Dict] = None
    memory_result: Optional[Dict] = None
    retrieval_result: Optional[Any] = None
    adaptation_context: Optional[Dict] = None
    proactive_result: Optional[Dict] = None

    # Clarification
    clarification_needed: Optional[Dict] = None

    # Engines called
    engines_called: List[str] = []
    engines_blocked: List[str] = []
```

## Usage Examples

### Basic Usage

```python
from app.orchestrator import get_orchestrator

orchestrator = get_orchestrator()
result = orchestrator.handle_input(
    user_input="What did I tell you about my job interview?",
    user_id=123
)

print(result["llm_output"])
print(result["intent"])  # "PAST_SELF_REFERENCE"
print(result["engines_called"])  # ["RetrievalEngine.retrieve"]
```

### Intent Classification Only

```python
from app.orchestrator import get_intent_gate

gate = get_intent_gate()
result = gate.classify("Remember that I prefer mornings for meetings")

print(result.primary_intent)  # "EXPLICIT_MEMORY_COMMAND"
print(result.confidence)  # 0.95
print(result.engines_required)  # {"MemoryEngine.ingest_event"}
```

### Legacy Interface

```python
from app.orchestrator import classify_intent

# Returns dict for backward compatibility
result = classify_intent("I feel worried about tomorrow")
# {"primary_intent": "PERSONAL_STATE", "confidence": 0.85, ...}
```

## File Structure

```
app/orchestrator/
├── orchestrator.py       # Main coordinator
├── intent_gate.py        # Intent classification
├── runtime_logger.py     # Runtime logging
├── __init__.py           # Module exports
└── ARCHITECTURE.md       # This file

app/semantic/
└── intent_concepts.py    # Semantic intent concepts
```

## Latency Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| Safety Check | ~1ms | Content filtering |
| Intent Gate (fast path) | ~0.1ms | Common phrases |
| Intent Gate (semantic) | ~5ms | First call |
| Intent Gate (semantic cached) | ~1ms | Subsequent calls |
| Intent Gate (regex) | ~0.5ms | Fallback |
| Temporal Engine | ~1-6ms | Depends on complexity |
| Memory Engine | ~5-10ms | Database write |
| Retrieval Engine | ~10-50ms | HNSW search |
| Adaptation Engine | ~5-10ms | Profile update |
| Proactive Engine | ~5-10ms | Follow-up evaluation |
| LLM Response | ~100-500ms | Model inference |

**Total Pipeline**: ~150-600ms (dominated by LLM)

## Design Principles

1. **Intent-Driven Routing**: Engine activation based on semantic intent
2. **Policy Enforcement**: Required/optional/forbidden engines per intent
3. **Latency Optimization**: Fast path cache + cached embeddings
4. **Graceful Fallback**: Regex when semantic unavailable, fallback responses when LLM fails
5. **Safety First**: Content filtering before any engine processing
6. **Logging**: Runtime logs for debugging and monitoring

## Frozen: January 2025

This architecture is stable and integrates all Nura engines.
