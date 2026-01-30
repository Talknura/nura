"""
Intent Gate v2.

Pre-engine routing gate that classifies user input into intent categories.
Uses semantic analysis (ML embeddings) for intelligent classification.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           INTENT GATE v2                                    │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │                            User Input                                       │
    │                                │                                            │
    │                                ▼                                            │
    │                   ┌─────────────────────────┐                               │
    │                   │    FAST PATH CACHE      │ ◄── "remember this", "forget" │
    │                   │    (~0.1ms)             │     Instant classification    │
    │                   └────────────┬────────────┘                               │
    │                                │ miss                                       │
    │                                ▼                                            │
    │                   ┌─────────────────────────┐                               │
    │                   │    SEMANTIC ANALYZER    │ ◄── ML embeddings             │
    │                   │    (~5ms first, ~1ms)   │     (all-MiniLM-L6-v2)        │
    │                   └────────────┬────────────┘                               │
    │                                │ low confidence                             │
    │                                ▼                                            │
    │                   ┌─────────────────────────┐                               │
    │                   │    REGEX FALLBACK       │ ◄── Pattern matching          │
    │                   │    (~0.5ms)             │                               │
    │                   └────────────┬────────────┘                               │
    │                                │                                            │
    │                                ▼                                            │
    │                   ┌─────────────────────────┐                               │
    │                   │    IntentResult         │                               │
    │                   │                         │                               │
    │                   │  • primary_intent       │                               │
    │                   │  • confidence           │                               │
    │                   │  • ambiguity_flags      │                               │
    │                   │  • routing_policy       │                               │
    │                   └─────────────────────────┘                               │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

Intent Categories:
    - EXPLICIT_MEMORY_COMMAND: User wants to store/delete memory
    - PAST_SELF_REFERENCE: User asking about past conversations/memories
    - PERSONAL_STATE: User sharing emotions, feelings, future concerns
    - GENERAL_KNOWLEDGE: User asking factual questions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# =============================================================================
# ENGINE INTEGRATIONS
# =============================================================================

# Semantic Intent Analyzer
try:
    from app.semantic.intent_concepts import (
        get_semantic_intent_analyzer,
        IntentAnalysis,
    )
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("[IntentGate] Semantic analyzer not available, using fallback")


# =============================================================================
# CONSTANTS
# =============================================================================

# Intent categories
INTENT_EXPLICIT_MEMORY = "EXPLICIT_MEMORY_COMMAND"
INTENT_PAST_REFERENCE = "PAST_SELF_REFERENCE"
INTENT_PERSONAL_STATE = "PERSONAL_STATE"
INTENT_GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"

# Intent priority order (earlier takes precedence)
INTENT_PRIORITY = [
    INTENT_PAST_REFERENCE,
    INTENT_EXPLICIT_MEMORY,
    INTENT_PERSONAL_STATE,
    INTENT_GENERAL_KNOWLEDGE,
]

# Confidence thresholds
CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.70
CONFIDENCE_LOW = 0.50


# =============================================================================
# FAST PATH CACHE
# =============================================================================

# Pre-computed intent classifications for common phrases
FAST_PATH_INTENTS = {
    # Explicit memory commands
    "remember this": (INTENT_EXPLICIT_MEMORY, 1.0, ["remember_directive"]),
    "remember that": (INTENT_EXPLICIT_MEMORY, 1.0, ["remember_directive"]),
    "don't forget": (INTENT_EXPLICIT_MEMORY, 1.0, ["remember_directive"]),
    "forget this": (INTENT_EXPLICIT_MEMORY, 1.0, ["forget_directive"]),
    "forget that": (INTENT_EXPLICIT_MEMORY, 1.0, ["forget_directive"]),
    "delete this": (INTENT_EXPLICIT_MEMORY, 1.0, ["forget_directive"]),

    # Past self reference
    "what did i say": (INTENT_PAST_REFERENCE, 0.95, ["recall_past"]),
    "what did i tell you": (INTENT_PAST_REFERENCE, 0.95, ["recall_past"]),
    "do you remember": (INTENT_PAST_REFERENCE, 0.90, ["recall_past"]),
    "last time we talked": (INTENT_PAST_REFERENCE, 0.90, ["past_reference"]),

    # Personal state
    "i feel": (INTENT_PERSONAL_STATE, 0.85, ["emotion_state"]),
    "i'm feeling": (INTENT_PERSONAL_STATE, 0.85, ["emotion_state"]),
    "i'm worried": (INTENT_PERSONAL_STATE, 0.90, ["emotion_state", "negative_emotion"]),
    "i'm scared": (INTENT_PERSONAL_STATE, 0.90, ["emotion_state", "negative_emotion"]),
    "i'm excited": (INTENT_PERSONAL_STATE, 0.85, ["emotion_state", "positive_emotion"]),
    "i'm happy": (INTENT_PERSONAL_STATE, 0.85, ["emotion_state", "positive_emotion"]),

    # General knowledge
    "what is": (INTENT_GENERAL_KNOWLEDGE, 0.80, ["factual_question"]),
    "how does": (INTENT_GENERAL_KNOWLEDGE, 0.80, ["factual_question"]),
    "explain": (INTENT_GENERAL_KNOWLEDGE, 0.80, ["explanation_request"]),
    "tell me about": (INTENT_GENERAL_KNOWLEDGE, 0.75, ["factual_question"]),
}


# =============================================================================
# REGEX FALLBACK PATTERNS
# =============================================================================

_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|ive|me|my|mine|we|our|us)\b", re.IGNORECASE)
_MEMORY_DIRECTIVE = re.compile(
    r"\b(remember|forget|don't forget|dont forget|make a note|note that|keep in mind|"
    r"erase|delete|drop|keep\b.*\bin mind)\b",
    re.IGNORECASE,
)
_FORGET_DIRECTIVE = re.compile(r"\b(forget|erase|delete|drop|don't keep|dont keep)\b", re.IGNORECASE)
_PAST_REFERENCE = re.compile(
    r"\b(what did i say|what i said|what did i mention|what did i tell you|"
    r"remind me what i said|do you remember what i said|you told me|you said|"
    r"last time|before|previously|earlier|did i already|have i already)\b",
    re.IGNORECASE,
)
_RECALL_QUERY = re.compile(
    r"\b(what is my|what's my|whats my|what was my|do you know my|do you remember my|"
    r"tell me my|remind me of my)\b",
    re.IGNORECASE,
)
_EMOTION_STATE = re.compile(
    r"\b(feel|feeling|nervous|anxious|scared|sad|down|depressed|angry|"
    r"overwhelmed|stressed|worried|afraid|excited|happy|confident|hopeless|"
    r"can't do this|can't handle this)\b",
    re.IGNORECASE,
)
_FUTURE_ANCHOR = re.compile(r"\b(tomorrow|next|soon|coming up|upcoming)\b", re.IGNORECASE)
_EVALUATIVE = re.compile(r"\b(important|worried|nervous|big deal|big day|matters)\b", re.IGNORECASE)
_GK_QUERY = re.compile(
    r"\b(what|why|how|explain|define|describe|tell me|walk me through)\b",
    re.IGNORECASE,
)
_EVENT_KEYWORD = re.compile(r"\b(interview|appointment|meeting|exam|presentation)\b", re.IGNORECASE)
_POSITIVE_STATE = re.compile(r"\b(excited|happy|confident|glad|great|amazing)\b", re.IGNORECASE)
_NEGATIVE_STATE = re.compile(r"\b(sad|scared|worried|anxious|nervous|afraid|depressed)\b", re.IGNORECASE)
_STATE_UPDATE = re.compile(r"\b(feel better|feeling better|better now|improved|okay now)\b", re.IGNORECASE)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class IntentResult:
    """Result of intent classification."""
    primary_intent: str = INTENT_GENERAL_KNOWLEDGE
    confidence: float = 0.7
    ambiguity: bool = False
    ambiguity_flags: List[str] = field(default_factory=list)

    # Routing policy hints
    engines_required: Set[str] = field(default_factory=set)
    engines_optional: Set[str] = field(default_factory=set)
    engines_forbidden: Set[str] = field(default_factory=set)

    # Detection method
    detection_method: str = "semantic"  # semantic, fast_path, regex

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_intent": self.primary_intent,
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
            "ambiguity_flags": self.ambiguity_flags,
        }


# =============================================================================
# INTENT GATE v2
# =============================================================================

class IntentGateV2:
    """
    Integrated Intent Gate with semantic classification.

    Features:
        - Semantic classification (ML embeddings)
        - Fast path cache for common phrases
        - Regex fallback for edge cases
        - Routing policy generation
    """

    def __init__(self):
        """Initialize the intent gate."""
        self._semantic_analyzer = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of semantic analyzer."""
        if self._initialized:
            return

        if SEMANTIC_AVAILABLE:
            self._semantic_analyzer = get_semantic_intent_analyzer()

        self._initialized = True

    # -------------------------------------------------------------------------
    # FAST PATH DETECTION
    # -------------------------------------------------------------------------

    def _check_fast_path(self, text: str) -> Optional[IntentResult]:
        """
        Check fast path cache for instant classification.

        Latency: ~0.1ms
        """
        if not text:
            return None

        lowered = text.lower().strip()

        # Exact match
        if lowered in FAST_PATH_INTENTS:
            intent, conf, flags = FAST_PATH_INTENTS[lowered]
            result = IntentResult(
                primary_intent=intent,
                confidence=conf,
                ambiguity=False,
                ambiguity_flags=flags,
                detection_method="fast_path",
            )
            self._apply_routing_policy(result)
            return result

        # Prefix match for common patterns
        for pattern, (intent, conf, flags) in FAST_PATH_INTENTS.items():
            if lowered.startswith(pattern):
                result = IntentResult(
                    primary_intent=intent,
                    confidence=conf * 0.95,  # Slight reduction for prefix match
                    ambiguity=False,
                    ambiguity_flags=flags,
                    detection_method="fast_path",
                )
                self._apply_routing_policy(result)
                return result

        return None

    # -------------------------------------------------------------------------
    # SEMANTIC CLASSIFICATION
    # -------------------------------------------------------------------------

    def _classify_semantic(self, text: str) -> Optional[IntentResult]:
        """
        Classify using semantic analyzer.

        Latency: ~5ms first call, ~1ms cached
        """
        if not SEMANTIC_AVAILABLE or not self._semantic_analyzer:
            return None

        try:
            analysis = self._semantic_analyzer.analyze(text, threshold=0.42)

            result = IntentResult(
                primary_intent=analysis.primary_intent,
                confidence=analysis.confidence,
                ambiguity=analysis.ambiguity,
                ambiguity_flags=analysis.ambiguity_flags,
                detection_method="semantic",
            )
            self._apply_routing_policy(result)
            return result

        except Exception as e:
            print(f"[IntentGate] Semantic analysis failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # REGEX FALLBACK
    # -------------------------------------------------------------------------

    def _classify_regex(self, text: str) -> IntentResult:
        """
        Classify using regex patterns.

        Latency: ~0.5ms
        """
        signals: List[str] = []
        candidates: List[str] = []

        # Pattern detection
        has_first_person = bool(_FIRST_PERSON.search(text))
        has_memory_directive = bool(_MEMORY_DIRECTIVE.search(text))
        has_forget_directive = bool(_FORGET_DIRECTIVE.search(text))
        has_past_reference = bool(_PAST_REFERENCE.search(text))
        has_recall_query = bool(_RECALL_QUERY.search(text))
        has_emotion_state = bool(_EMOTION_STATE.search(text))
        has_future_anchor = bool(_FUTURE_ANCHOR.search(text))
        has_evaluative = bool(_EVALUATIVE.search(text))
        has_gk_query = bool(_GK_QUERY.search(text))
        has_event_keyword = bool(_EVENT_KEYWORD.search(text))
        has_positive_state = bool(_POSITIVE_STATE.search(text))
        has_negative_state = bool(_NEGATIVE_STATE.search(text))
        has_state_update = bool(_STATE_UPDATE.search(text))

        # Build candidates
        if has_memory_directive:
            candidates.append(INTENT_EXPLICIT_MEMORY)
            signals.append("memory_directive")
            if has_forget_directive:
                signals.append("forget_directive")

        if has_past_reference or has_recall_query:
            candidates.append(INTENT_PAST_REFERENCE)
            if has_past_reference:
                signals.append("past_reference")
            if has_recall_query:
                signals.append("recall_query")

        if has_emotion_state:
            candidates.append(INTENT_PERSONAL_STATE)
            signals.append("emotion_state")
            if has_positive_state:
                signals.append("positive_state")
            if has_negative_state:
                signals.append("negative_state")

        if has_future_anchor and (has_event_keyword or has_evaluative):
            candidates.append(INTENT_PERSONAL_STATE)
            signals.append("future_event")
            if has_evaluative:
                signals.append("future_evaluative")

        if has_state_update:
            candidates.append(INTENT_PERSONAL_STATE)
            signals.append("state_update")

        # GK fallback
        base_gk = not has_memory_directive and not has_past_reference and not has_emotion_state
        if base_gk and has_gk_query:
            candidates.append(INTENT_GENERAL_KNOWLEDGE)
            signals.append("general_knowledge_query")
            if has_first_person:
                signals.append("gk_with_personal_context")

        # Default fallback
        if not candidates:
            if has_first_person:
                candidates.append(INTENT_PERSONAL_STATE)
                signals.append("first_person")
            else:
                candidates.append(INTENT_GENERAL_KNOWLEDGE)
                signals.append("fallback_general_knowledge")

        # Resolve by priority
        primary_intent = next(
            (p for p in INTENT_PRIORITY if p in candidates),
            INTENT_GENERAL_KNOWLEDGE
        )

        # Confidence based on intent type
        confidence_map = {
            INTENT_EXPLICIT_MEMORY: 1.0,
            INTENT_PAST_REFERENCE: 0.9,
            INTENT_PERSONAL_STATE: 0.8,
            INTENT_GENERAL_KNOWLEDGE: 0.7,
        }
        confidence = confidence_map.get(primary_intent, 0.7)

        # Ambiguity check
        ambiguity = len(set(candidates)) > 1 or confidence < CONFIDENCE_MEDIUM

        result = IntentResult(
            primary_intent=primary_intent,
            confidence=confidence,
            ambiguity=ambiguity,
            ambiguity_flags=signals,
            detection_method="regex",
        )
        self._apply_routing_policy(result)
        return result

    # -------------------------------------------------------------------------
    # ROUTING POLICY
    # -------------------------------------------------------------------------

    def _apply_routing_policy(self, result: IntentResult) -> None:
        """
        Apply engine routing policy based on intent.

        Determines which engines should be:
        - Required (must run)
        - Optional (may run based on confidence)
        - Forbidden (must not run)
        """
        intent = result.primary_intent
        flags = set(result.ambiguity_flags)

        if intent == INTENT_GENERAL_KNOWLEDGE:
            result.engines_forbidden = {
                "MemoryEngine.ingest_event",
                "RetrievalEngine.retrieve",
                "TemporalEngine.temporal_tags_from_dt",
                "AdaptationEngine.update",
                "ProactiveEngine.evaluate",
            }

        elif intent == INTENT_EXPLICIT_MEMORY:
            result.engines_required = {"MemoryEngine.ingest_event"}
            if "forget_directive" not in flags:
                result.engines_optional = {"TemporalEngine.temporal_tags_from_dt"}
            result.engines_forbidden = {
                "RetrievalEngine.retrieve",
                "AdaptationEngine.update",
                "ProactiveEngine.evaluate",
            }

        elif intent == INTENT_PAST_REFERENCE:
            result.engines_required = {"RetrievalEngine.retrieve"}
            if "learning_reflection" not in flags:
                result.engines_optional = {"TemporalEngine.temporal_tags_from_dt"}
            result.engines_forbidden = {
                "MemoryEngine.ingest_event",
                "AdaptationEngine.update",
                "ProactiveEngine.evaluate",
            }

        elif intent == INTENT_PERSONAL_STATE:
            result.engines_required = {
                "MemoryEngine.ingest_event",
                "AdaptationEngine.update",
            }
            result.engines_optional = {
                "RetrievalEngine.retrieve",
                "TemporalEngine.temporal_tags_from_dt",
                "ProactiveEngine.evaluate",
            }

            # Adjust based on flags
            if "future_event" in flags or "future_evaluative" in flags:
                result.engines_required.discard("AdaptationEngine.update")
                result.engines_forbidden.add("AdaptationEngine.update")
                result.engines_optional.discard("RetrievalEngine.retrieve")
                result.engines_forbidden.add("RetrievalEngine.retrieve")

            if "learning_struggle" in flags or "persistent_emotion" in flags or "state_update" in flags:
                result.engines_optional.discard("TemporalEngine.temporal_tags_from_dt")
                result.engines_forbidden.add("TemporalEngine.temporal_tags_from_dt")

            if "positive_state" in flags or "state_update" in flags:
                result.engines_optional.discard("ProactiveEngine.evaluate")
                result.engines_forbidden.add("ProactiveEngine.evaluate")

    # -------------------------------------------------------------------------
    # MAIN CLASSIFICATION
    # -------------------------------------------------------------------------

    def classify(self, user_text: str) -> IntentResult:
        """
        Classify user input into an intent category.

        Uses:
            1. Fast path cache (~0.1ms)
            2. Semantic analysis (~5ms first, ~1ms cached)
            3. Regex fallback (~0.5ms)
        """
        self._ensure_initialized()

        if not user_text or not user_text.strip():
            return IntentResult()

        # 1. Fast path check
        result = self._check_fast_path(user_text)
        if result:
            self._log_classification(result, user_text)
            return result

        # 2. Semantic classification
        result = self._classify_semantic(user_text)
        if result and result.confidence >= CONFIDENCE_LOW:
            self._log_classification(result, user_text)
            return result

        # 3. Regex fallback
        result = self._classify_regex(user_text)
        self._log_classification(result, user_text)
        return result

    def _log_classification(self, result: IntentResult, text: str) -> None:
        """Log classification decision."""
        preview = text[:50] + "..." if len(text) > 50 else text
        print(
            f"[INTENT_GATE] [{result.detection_method.upper()}] "
            f"intent={result.primary_intent} conf={result.confidence:.2f} "
            f"ambiguity={result.ambiguity} | \"{preview}\""
        )


# =============================================================================
# SINGLETON & LEGACY INTERFACE
# =============================================================================

_gate_instance: Optional[IntentGateV2] = None


def get_intent_gate() -> IntentGateV2:
    """Get or create the singleton intent gate."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = IntentGateV2()
    return _gate_instance


def classify_intent(user_text: str) -> Dict[str, Any]:
    """
    Legacy interface for backward compatibility.

    Returns dict instead of IntentResult.
    """
    gate = get_intent_gate()
    result = gate.classify(user_text)
    return result.to_dict()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "IntentGateV2",
    "IntentResult",
    "get_intent_gate",
    "classify_intent",
    "INTENT_EXPLICIT_MEMORY",
    "INTENT_PAST_REFERENCE",
    "INTENT_PERSONAL_STATE",
    "INTENT_GENERAL_KNOWLEDGE",
]
