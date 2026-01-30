"""
Retrieval Strategies for Three-Tier Memory Architecture.

Different query types require different retrieval approaches:

1. FACTUAL   - Direct fact lookup, no decay
2. EPISODIC  - Conversation recall with recency
3. MILESTONE - Life event lookup, no decay
4. HYBRID    - Multi-tier search, unified ranking
5. TIMELINE  - Temporal-first, chronological

Uses semantic query analysis to route to the best strategy.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

# Try semantic query classifier
try:
    from app.vector.embedder import embed_text
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class RetrievalStrategy(Enum):
    """Query-appropriate retrieval strategies."""
    FACTUAL = "factual"       # "What's my name?" → Facts tier
    EPISODIC = "episodic"     # "What did we talk about?" → Episodes tier
    MILESTONE = "milestone"   # "When did X happen?" → Milestones tier
    HYBRID = "hybrid"         # "Tell me about X" → All tiers
    TIMELINE = "timeline"     # "What happened this week?" → Temporal-first


@dataclass
class QueryAnalysis:
    """Result of analyzing a query for retrieval routing."""
    strategy: RetrievalStrategy
    confidence: float

    # Strategy-specific parameters
    fact_keys: list[str] = field(default_factory=list)      # For FACTUAL
    event_types: list[str] = field(default_factory=list)    # For MILESTONE
    time_window_days: Optional[int] = None                   # For TIMELINE/EPISODIC
    disable_recency: bool = False                            # For explicit past references

    # Tier weights for HYBRID (how much to weight each tier)
    tier_weights: dict[str, float] = field(default_factory=lambda: {
        "facts": 0.33,
        "milestones": 0.33,
        "episodes": 0.34
    })


# =============================================================================
# SEMANTIC QUERY PATTERNS
# =============================================================================

# Patterns that indicate FACTUAL queries (direct fact lookup)
FACTUAL_PATTERNS = [
    "what is my name",
    "what's my name",
    "where do I live",
    "where do I work",
    "what is my job",
    "what's my job",
    "who is my",
    "what's my favorite",
    "what is my favorite",
    "how old am I",
    "when is my birthday",
    "what do I do for work",
    "my dog's name",
    "my cat's name",
    "my pet's name",
]

# Patterns that indicate MILESTONE queries (life events)
MILESTONE_PATTERNS = [
    "when did my father pass",
    "when did my mother pass",
    "when did my dad die",
    "when did my mom die",
    "when did I get married",
    "when did I graduate",
    "when was I born",
    "when did I start my job",
    "when did I move",
    "when did we meet",
    "the day I lost",
    "when I got divorced",
    "when my child was born",
]

# Patterns that indicate EPISODIC queries (conversation recall)
EPISODIC_PATTERNS = [
    "what did we talk about",
    "what did I tell you",
    "what did I say",
    "do you remember when I",
    "remember when I told you",
    "last time we spoke",
    "yesterday we discussed",
    "what did I mention",
    "what was I saying about",
    "our conversation about",
]

# Patterns that indicate TIMELINE queries (temporal range)
TIMELINE_PATTERNS = [
    "what happened this week",
    "what happened today",
    "what happened this month",
    "what's been going on",
    "summary of this week",
    "what did we cover",
    "recap of",
    "what's new",
    "recent conversations",
    "lately we've",
]

# Patterns that indicate explicit past reference (disable recency penalty)
PAST_REFERENCE_PATTERNS = [
    "last year",
    "last month",
    "years ago",
    "months ago",
    "back when",
    "a while ago",
    "long time ago",
    "in the past",
    "previously",
    "before",
]


class SemanticQueryRouter:
    """
    Routes queries to appropriate retrieval strategy using embeddings.

    Compares query embedding against pattern embeddings to determine
    the best retrieval approach.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._pattern_embeddings: dict[str, list[tuple[str, np.ndarray]]] = {}

        if EMBEDDINGS_AVAILABLE:
            self._build_pattern_embeddings()

    def _build_pattern_embeddings(self):
        """Pre-compute embeddings for all patterns."""
        pattern_groups = {
            "factual": FACTUAL_PATTERNS,
            "milestone": MILESTONE_PATTERNS,
            "episodic": EPISODIC_PATTERNS,
            "timeline": TIMELINE_PATTERNS,
            "past_reference": PAST_REFERENCE_PATTERNS,
        }

        for group_name, patterns in pattern_groups.items():
            self._pattern_embeddings[group_name] = []
            for pattern in patterns:
                try:
                    emb = embed_text(pattern)
                    self._pattern_embeddings[group_name].append((pattern, emb))
                except Exception:
                    pass

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _max_similarity(self, query_emb: np.ndarray, group: str) -> tuple[float, str]:
        """Find max similarity between query and patterns in a group."""
        if group not in self._pattern_embeddings:
            return 0.0, ""

        max_sim = 0.0
        best_pattern = ""

        for pattern, pattern_emb in self._pattern_embeddings[group]:
            sim = self._cosine_similarity(query_emb, pattern_emb)
            if sim > max_sim:
                max_sim = sim
                best_pattern = pattern

        return max_sim, best_pattern

    def analyze(self, query: str, threshold: float = 0.50) -> QueryAnalysis:
        """
        Analyze query to determine retrieval strategy.

        Args:
            query: User's query text
            threshold: Minimum similarity to match a pattern

        Returns:
            QueryAnalysis with strategy and parameters
        """
        if not EMBEDDINGS_AVAILABLE or not self._pattern_embeddings:
            return self._fallback_analyze(query)

        try:
            query_emb = embed_text(query)
        except Exception:
            return self._fallback_analyze(query)

        # Compute similarities to each pattern group
        scores = {}
        for group in ["factual", "milestone", "episodic", "timeline"]:
            sim, pattern = self._max_similarity(query_emb, group)
            scores[group] = (sim, pattern)

        # Check for explicit past reference
        past_sim, _ = self._max_similarity(query_emb, "past_reference")
        disable_recency = past_sim >= threshold

        # Find best matching strategy
        best_group = max(scores.keys(), key=lambda g: scores[g][0])
        best_score, best_pattern = scores[best_group]

        # If no strong match, use HYBRID
        if best_score < threshold:
            return QueryAnalysis(
                strategy=RetrievalStrategy.HYBRID,
                confidence=0.5,
                disable_recency=disable_recency
            )

        # Map group to strategy
        strategy_map = {
            "factual": RetrievalStrategy.FACTUAL,
            "milestone": RetrievalStrategy.MILESTONE,
            "episodic": RetrievalStrategy.EPISODIC,
            "timeline": RetrievalStrategy.TIMELINE,
        }

        strategy = strategy_map.get(best_group, RetrievalStrategy.HYBRID)

        # Build analysis result
        analysis = QueryAnalysis(
            strategy=strategy,
            confidence=best_score,
            disable_recency=disable_recency
        )

        # Strategy-specific adjustments
        if strategy == RetrievalStrategy.FACTUAL:
            analysis.tier_weights = {"facts": 0.8, "milestones": 0.1, "episodes": 0.1}
        elif strategy == RetrievalStrategy.MILESTONE:
            analysis.tier_weights = {"facts": 0.1, "milestones": 0.8, "episodes": 0.1}
        elif strategy == RetrievalStrategy.EPISODIC:
            analysis.tier_weights = {"facts": 0.1, "milestones": 0.1, "episodes": 0.8}
        elif strategy == RetrievalStrategy.TIMELINE:
            analysis.tier_weights = {"facts": 0.0, "milestones": 0.3, "episodes": 0.7}
            analysis.time_window_days = self._extract_time_window(query)

        return analysis

    def _extract_time_window(self, query: str) -> Optional[int]:
        """Extract time window from query (e.g., 'this week' → 7 days)."""
        query_lower = query.lower()

        if "today" in query_lower:
            return 1
        elif "yesterday" in query_lower:
            return 2
        elif "this week" in query_lower:
            return 7
        elif "last week" in query_lower:
            return 14
        elif "this month" in query_lower:
            return 30
        elif "last month" in query_lower:
            return 60

        return None

    def _fallback_analyze(self, query: str) -> QueryAnalysis:
        """Keyword-based fallback when embeddings unavailable."""
        query_lower = query.lower()

        # Check patterns with simple keyword matching
        for pattern in FACTUAL_PATTERNS:
            if pattern in query_lower:
                return QueryAnalysis(
                    strategy=RetrievalStrategy.FACTUAL,
                    confidence=0.7,
                    tier_weights={"facts": 0.8, "milestones": 0.1, "episodes": 0.1}
                )

        for pattern in MILESTONE_PATTERNS:
            if pattern in query_lower:
                return QueryAnalysis(
                    strategy=RetrievalStrategy.MILESTONE,
                    confidence=0.7,
                    tier_weights={"facts": 0.1, "milestones": 0.8, "episodes": 0.1}
                )

        for pattern in EPISODIC_PATTERNS:
            if pattern in query_lower:
                return QueryAnalysis(
                    strategy=RetrievalStrategy.EPISODIC,
                    confidence=0.7,
                    tier_weights={"facts": 0.1, "milestones": 0.1, "episodes": 0.8}
                )

        for pattern in TIMELINE_PATTERNS:
            if pattern in query_lower:
                return QueryAnalysis(
                    strategy=RetrievalStrategy.TIMELINE,
                    confidence=0.7,
                    time_window_days=self._extract_time_window(query),
                    tier_weights={"facts": 0.0, "milestones": 0.3, "episodes": 0.7}
                )

        # Check for past reference
        disable_recency = any(p in query_lower for p in PAST_REFERENCE_PATTERNS)

        # Default to HYBRID
        return QueryAnalysis(
            strategy=RetrievalStrategy.HYBRID,
            confidence=0.5,
            disable_recency=disable_recency
        )


# Singleton accessor
_router_instance: Optional[SemanticQueryRouter] = None

def get_query_router() -> SemanticQueryRouter:
    """Get singleton query router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = SemanticQueryRouter()
    return _router_instance
