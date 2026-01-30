"""
Retrieval Engine v2 for Nura - Three-Tier Architecture.

Routes queries to appropriate retrieval strategy:
  - FACTUAL   → Facts tier (HNSW), no decay
  - EPISODIC  → Episodes tier, recency-weighted
  - MILESTONE → Milestones tier, no decay
  - HYBRID    → All tiers, unified ranking
  - TIMELINE  → Temporal-first, chronological

Leverages HNSW indexes for O(log n) retrieval.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Protocol, Optional, List, Dict, Any

from app.memory.memory_engine import MemoryEngine
from app.temporal.temporal_engine import TemporalEngine
from app.retrieval.retrieval_strategies import (
    get_query_router,
    QueryAnalysis,
    RetrievalStrategy
)
from app.retrieval.ranker import RankWeights, score_hit
from config import thresholds


@dataclass
class RetrievalState:
    """Input state for retrieval."""
    user_id: int
    query: str
    now: datetime
    current_temporal_tags: dict = None
    temporal_rewrite: dict | None = None
    top_k: int = 8
    session_id: str = "default"


@dataclass
class RetrievalResult:
    """Output from retrieval."""
    hits: list[dict]
    facts: dict[str, str]
    milestones: list[dict]
    strategy_used: RetrievalStrategy
    query_analysis: QueryAnalysis


class RetrievalEngineProtocol(Protocol):
    def retrieve(self, state: RetrievalState) -> RetrievalResult: ...


class RetrievalEngineV2:
    """
    Three-tier retrieval engine with strategy routing.

    Analyzes query semantically and routes to best retrieval approach:
      - Factual queries → Direct fact lookup (fast, no decay)
      - Episodic queries → Episode search with recency
      - Milestone queries → Life event search (no decay)
      - Hybrid queries → Multi-tier merge
      - Timeline queries → Temporal-first ordering
    """

    def __init__(self, memory: MemoryEngine, temporal: TemporalEngine):
        self.memory = memory
        self.temporal = temporal
        self.weights = RankWeights()
        self.router = get_query_router()

    def retrieve(self, state: RetrievalState) -> RetrievalResult:
        """
        Retrieve relevant memories using strategy routing.

        Args:
            state: Retrieval state with query, user_id, etc.

        Returns:
            RetrievalResult with hits from appropriate tier(s)
        """
        # Analyze query to determine strategy
        analysis = self.router.analyze(state.query)

        # Route to appropriate retrieval method
        if analysis.strategy == RetrievalStrategy.FACTUAL:
            return self._retrieve_factual(state, analysis)

        elif analysis.strategy == RetrievalStrategy.MILESTONE:
            return self._retrieve_milestone(state, analysis)

        elif analysis.strategy == RetrievalStrategy.EPISODIC:
            return self._retrieve_episodic(state, analysis)

        elif analysis.strategy == RetrievalStrategy.TIMELINE:
            return self._retrieve_timeline(state, analysis)

        else:  # HYBRID
            return self._retrieve_hybrid(state, analysis)

    # =========================================================================
    # STRATEGY: FACTUAL
    # =========================================================================

    def _retrieve_factual(self, state: RetrievalState, analysis: QueryAnalysis) -> RetrievalResult:
        """
        Factual retrieval - prioritize FACTS tier.

        For queries like "What's my name?" or "Where do I work?"
        """
        # Get all facts (fast, key-value lookup)
        all_facts = self.memory.facts(state.user_id)

        # Also search facts semantically for best match
        fact_hits = []
        try:
            from app.memory.memory_store import search_facts
            from app.vector.embedder import embed_text

            query_emb = embed_text(state.query)
            semantic_facts = search_facts(state.user_id, query_emb, k=5)

            for f in semantic_facts:
                fact_hits.append({
                    "id": f"fact_{f['key']}",
                    "content": f"{f['key']}: {f['value']}",
                    "memory_type": "fact",
                    "memory_tier": "fact",
                    "importance": 1.0,
                    "similarity": f["similarity"],
                    "fact_key": f["key"],
                    "fact_value": f["value"],
                    "created_at": None,  # Facts are timeless
                    "final_score": f["similarity"] * 1.2  # Boost facts
                })
        except Exception:
            # Fallback: return all facts as hits
            for key, value in all_facts.items():
                fact_hits.append({
                    "id": f"fact_{key}",
                    "content": f"{key}: {value}",
                    "memory_type": "fact",
                    "memory_tier": "fact",
                    "importance": 1.0,
                    "similarity": 0.8,
                    "fact_key": key,
                    "fact_value": value,
                    "final_score": 0.8
                })

        # Also get some episodes for context
        episode_hits = self.memory.search(
            user_id=state.user_id,
            query=state.query,
            k=3
        )
        for h in episode_hits:
            h["memory_tier"] = "episode"
            h["final_score"] = score_hit(
                h, now=state.now, w=self.weights,
                current_temporal_tags=state.current_temporal_tags or {}
            ) * analysis.tier_weights.get("episodes", 0.1)

        # Merge and sort
        all_hits = fact_hits + episode_hits
        all_hits.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        return RetrievalResult(
            hits=all_hits[:state.top_k],
            facts=all_facts,
            milestones=[],
            strategy_used=RetrievalStrategy.FACTUAL,
            query_analysis=analysis
        )

    # =========================================================================
    # STRATEGY: MILESTONE
    # =========================================================================

    def _retrieve_milestone(self, state: RetrievalState, analysis: QueryAnalysis) -> RetrievalResult:
        """
        Milestone retrieval - prioritize MILESTONES tier.

        For queries like "When did my dad pass away?"
        """
        milestone_hits = []

        try:
            from app.memory.memory_store import search_milestones
            from app.vector.embedder import embed_text

            query_emb = embed_text(state.query)
            semantic_milestones = search_milestones(state.user_id, query_emb, k=5)

            for m in semantic_milestones:
                milestone_hits.append({
                    "id": m["id"],
                    "content": f"{m['event_type']}: {m['description']}",
                    "memory_type": "milestone",
                    "memory_tier": "milestone",
                    "importance": 0.95,
                    "similarity": m["similarity"],
                    "event_type": m["event_type"],
                    "event_date": m["event_date"],
                    "created_at": m.get("event_date"),
                    "final_score": m["similarity"] * 1.2  # Boost milestones
                })
        except Exception:
            pass

        # Also get milestones by type
        all_milestones = self.memory.milestones(state.user_id)

        # Get facts for context
        all_facts = self.memory.facts(state.user_id)

        # Merge and sort
        milestone_hits.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        return RetrievalResult(
            hits=milestone_hits[:state.top_k],
            facts=all_facts,
            milestones=all_milestones,
            strategy_used=RetrievalStrategy.MILESTONE,
            query_analysis=analysis
        )

    # =========================================================================
    # STRATEGY: EPISODIC
    # =========================================================================

    def _retrieve_episodic(self, state: RetrievalState, analysis: QueryAnalysis) -> RetrievalResult:
        """
        Episodic retrieval - prioritize EPISODES tier with recency.

        For queries like "What did we talk about yesterday?"
        """
        # Search episodes
        episode_hits = self.memory.search(
            user_id=state.user_id,
            query=state.query,
            k=state.top_k * 3  # Over-fetch for filtering
        )

        # Apply time window filter if specified
        if analysis.time_window_days:
            cutoff = state.now - timedelta(days=analysis.time_window_days)
            episode_hits = [
                h for h in episode_hits
                if self._is_within_window(h, cutoff)
            ]

        # Score with recency (unless disabled for past references)
        for h in episode_hits:
            h["memory_tier"] = "episode"

            # Build query characteristics
            query_chars = None
            if analysis.disable_recency:
                from app.semantic.retrieval_concepts import RetrievalCharacteristics
                query_chars = RetrievalCharacteristics(
                    half_life_days=0,  # No decay
                    importance_floor=0.5,
                    disable_recency_penalty=True
                )

            h["final_score"] = score_hit(
                h, now=state.now, w=self.weights,
                current_temporal_tags=state.current_temporal_tags or {},
                query_characteristics=query_chars
            )

        # Sort by score
        episode_hits.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        # Get facts for context
        all_facts = self.memory.facts(state.user_id)

        return RetrievalResult(
            hits=episode_hits[:state.top_k],
            facts=all_facts,
            milestones=[],
            strategy_used=RetrievalStrategy.EPISODIC,
            query_analysis=analysis
        )

    # =========================================================================
    # STRATEGY: TIMELINE
    # =========================================================================

    def _retrieve_timeline(self, state: RetrievalState, analysis: QueryAnalysis) -> RetrievalResult:
        """
        Timeline retrieval - temporal-first, chronological ordering.

        For queries like "What happened this week?"
        """
        # Get recent memories within time window
        time_window = analysis.time_window_days or 7

        # Get recent episodes
        recent = self.memory.recent(state.user_id, k=50)

        # Filter by time window
        cutoff = state.now - timedelta(days=time_window)
        timeline_hits = [
            h for h in recent
            if self._is_within_window(h, cutoff)
        ]

        # Mark as episode tier
        for h in timeline_hits:
            h["memory_tier"] = "episode"
            h["final_score"] = 1.0  # Timeline uses chronological, not score

        # Sort chronologically (newest first)
        timeline_hits.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )

        # Also get milestones in time window
        all_milestones = self.memory.milestones(state.user_id)
        recent_milestones = [
            m for m in all_milestones
            if self._milestone_in_window(m, cutoff)
        ]

        # Get facts
        all_facts = self.memory.facts(state.user_id)

        return RetrievalResult(
            hits=timeline_hits[:state.top_k],
            facts=all_facts,
            milestones=recent_milestones,
            strategy_used=RetrievalStrategy.TIMELINE,
            query_analysis=analysis
        )

    # =========================================================================
    # STRATEGY: HYBRID
    # =========================================================================

    def _retrieve_hybrid(self, state: RetrievalState, analysis: QueryAnalysis) -> RetrievalResult:
        """
        Hybrid retrieval - search all tiers, unified ranking.

        For queries like "Tell me about my family"
        """
        all_hits = []
        tier_weights = analysis.tier_weights

        # 1. Search FACTS
        try:
            from app.memory.memory_store import search_facts
            from app.vector.embedder import embed_text

            query_emb = embed_text(state.query)
            fact_results = search_facts(state.user_id, query_emb, k=5)

            for f in fact_results:
                all_hits.append({
                    "id": f"fact_{f['key']}",
                    "content": f"{f['key']}: {f['value']}",
                    "memory_type": "fact",
                    "memory_tier": "fact",
                    "importance": 1.0,
                    "similarity": f["similarity"],
                    "fact_key": f["key"],
                    "final_score": f["similarity"] * tier_weights.get("facts", 0.33) * 1.5
                })
        except Exception:
            pass

        # 2. Search MILESTONES
        try:
            from app.memory.memory_store import search_milestones

            milestone_results = search_milestones(state.user_id, query_emb, k=5)

            for m in milestone_results:
                all_hits.append({
                    "id": m["id"],
                    "content": f"{m['event_type']}: {m['description']}",
                    "memory_type": "milestone",
                    "memory_tier": "milestone",
                    "importance": 0.95,
                    "similarity": m["similarity"],
                    "event_type": m["event_type"],
                    "event_date": m["event_date"],
                    "final_score": m["similarity"] * tier_weights.get("milestones", 0.33) * 1.5
                })
        except Exception:
            pass

        # 3. Search EPISODES
        episode_results = self.memory.search(
            user_id=state.user_id,
            query=state.query,
            k=10
        )

        for h in episode_results:
            h["memory_tier"] = "episode"
            base_score = score_hit(
                h, now=state.now, w=self.weights,
                current_temporal_tags=state.current_temporal_tags or {}
            )
            h["final_score"] = base_score * tier_weights.get("episodes", 0.34)
            all_hits.append(h)

        # Sort by unified score
        all_hits.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        # Get all facts and milestones for context
        all_facts = self.memory.facts(state.user_id)
        all_milestones = self.memory.milestones(state.user_id)

        return RetrievalResult(
            hits=all_hits[:state.top_k],
            facts=all_facts,
            milestones=all_milestones,
            strategy_used=RetrievalStrategy.HYBRID,
            query_analysis=analysis
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _is_within_window(self, hit: dict, cutoff: datetime) -> bool:
        """Check if a memory hit is within the time window."""
        try:
            created = hit.get("created_at")
            if not created:
                return True

            from app.memory.memory_store import iso_to_dt
            t = iso_to_dt(created)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)

            return t >= cutoff
        except Exception:
            return True

    def _milestone_in_window(self, milestone: dict, cutoff: datetime) -> bool:
        """Check if a milestone is within the time window."""
        try:
            event_date = milestone.get("event_date") or milestone.get("created_at")
            if not event_date:
                return True

            from app.memory.memory_store import iso_to_dt
            t = iso_to_dt(event_date)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)

            return t >= cutoff
        except Exception:
            return True


# =============================================================================
# BACKWARD COMPATIBILITY: Alias to original interface
# =============================================================================

class RetrievalEngine(RetrievalEngineV2):
    """Alias for backward compatibility."""

    def retrieve(self, state: RetrievalState) -> "RetrievalResult":
        """
        Override to return simplified result for backward compatibility.

        Returns RetrievalResult with hits and facts (original interface).
        """
        result = super().retrieve(state)

        # Original interface only had hits and facts
        from dataclasses import dataclass as dc

        @dc
        class LegacyResult:
            hits: list[dict]
            facts: dict[str, str]

        return LegacyResult(hits=result.hits, facts=result.facts)
