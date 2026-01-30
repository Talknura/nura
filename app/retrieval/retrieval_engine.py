"""
Retrieval Engine for Nura.

Multi-strategy retrieval for three-tier memory architecture:
  - FACTUAL   → Facts tier (no decay)
  - EPISODIC  → Episodes tier (recency decay)
  - MILESTONE → Milestones tier (no decay)
  - HYBRID    → All tiers (unified ranking)
  - TIMELINE  → Temporal-first (chronological)

Uses semantic query routing to select the best retrieval strategy.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional
from datetime import datetime

from app.memory.memory_engine import MemoryEngine
from app.temporal.temporal_engine import TemporalEngine

# Import new v2 engine
from app.retrieval.retrieval_engine_v2 import (
    RetrievalEngineV2,
    RetrievalState,
    RetrievalResult as RetrievalResultV2,
    RetrievalStrategy
)

# Legacy imports for backward compatibility
from app.retrieval.query_parser import parse_query
from app.retrieval.filters import RetrievalFilters, apply_filters
from app.retrieval.ranker import RankWeights, score_hit, analyze_query_for_retrieval
from config import thresholds


# Re-export RetrievalState for backward compatibility
__all__ = ["RetrievalEngine", "RetrievalState", "RetrievalResult"]


@dataclass
class RetrievalResult:
    """Backward-compatible result format."""
    hits: list[dict]
    facts: dict[str, str]


class RetrievalEngineProtocol(Protocol):
    def retrieve(self, state: RetrievalState) -> RetrievalResult: ...


class RetrievalEngine:
    """
    Retrieval Engine with multi-strategy routing.

    Routes queries to appropriate strategy:
      - FACTUAL:   "What's my name?" → Facts tier
      - EPISODIC:  "What did we discuss?" → Episodes tier
      - MILESTONE: "When did X happen?" → Milestones tier
      - HYBRID:    "Tell me about X" → All tiers
      - TIMELINE:  "What happened this week?" → Chronological

    Uses HNSW indexes for O(log n) retrieval across all tiers.
    """

    def __init__(self, memory: MemoryEngine, temporal: TemporalEngine):
        self.memory = memory
        self.temporal = temporal
        self.weights = RankWeights()

        # Use new v2 engine internally
        self._v2_engine = RetrievalEngineV2(memory, temporal)

    def retrieve(self, state: RetrievalState) -> RetrievalResult:
        """
        Retrieve relevant memories using strategy routing.

        Analyzes query semantically to determine best retrieval approach:
          - Factual queries → Direct fact lookup
          - Episodic queries → Conversation recall with recency
          - Milestone queries → Life event lookup
          - Hybrid queries → Multi-tier search
          - Timeline queries → Temporal-first ordering

        Args:
            state: Retrieval state with query, user_id, etc.

        Returns:
            RetrievalResult with ranked hits and user facts
        """
        # Use v2 engine for actual retrieval
        v2_result = self._v2_engine.retrieve(state)

        # Convert to backward-compatible format
        return RetrievalResult(
            hits=v2_result.hits,
            facts=v2_result.facts
        )

    def retrieve_full(self, state: RetrievalState) -> RetrievalResultV2:
        """
        Retrieve with full result including strategy info.

        Returns:
            RetrievalResultV2 with hits, facts, milestones, strategy_used
        """
        return self._v2_engine.retrieve(state)

    # Legacy method for backward compatibility
    def retrieve_legacy(self, state: RetrievalState) -> RetrievalResult:
        """
        Legacy single-path retrieval (deprecated).

        Use retrieve() instead for strategy-based routing.
        """
        pq = parse_query(state.query, self.temporal, temporal_rewrite=state.temporal_rewrite)
        filters = RetrievalFilters(max_age_days=pq.time_window_days)

        query_characteristics = analyze_query_for_retrieval(state.query)

        top_k = min(state.top_k, thresholds.MAX_MEMORY_HITS)
        candidates = self.memory.search(
            user_id=state.user_id,
            query=state.query,
            k=max(25, top_k * 5),
            max_candidates=250
        )
        candidates = apply_filters(candidates, filters, state.now)
        candidates = [
            c for c in candidates
            if float(c.get("similarity", 1.0)) >= thresholds.MIN_MEMORY_SIMILARITY
        ]

        for h in candidates:
            h["final_score"] = score_hit(
                h,
                now=state.now,
                w=self.weights,
                current_temporal_tags=state.current_temporal_tags or {},
                query_characteristics=query_characteristics
            )
        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        hits = candidates[:top_k]
        facts = self.memory.facts(state.user_id)
        return RetrievalResult(hits=hits, facts=facts)
