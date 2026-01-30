"""
Retrieval Ranker for Nura.

Ranks memory hits using weighted scoring across multiple dimensions.

Uses SEMANTIC analysis when available for adaptive decay rates:
    - PERMANENT (no decay): identity, family, health, grief, job, location,
                            education, major life events, trauma, preferences
    - 2-3 months: tasks, events, decisions, daily activities, casual mentions

Falls back to fixed 30-day decay if semantic unavailable.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import math
import json

from app.memory.memory_store import iso_to_dt

# Try to import semantic retrieval analyzer (preferred)
try:
    from app.semantic.retrieval_concepts import (
        get_semantic_retrieval_analyzer,
        RetrievalCharacteristics
    )
    SEMANTIC_RETRIEVAL_AVAILABLE = True
except ImportError:
    SEMANTIC_RETRIEVAL_AVAILABLE = False
    print("[Ranker] Semantic retrieval analyzer not available, using fixed 30-day decay")


@dataclass
class RankWeights:
    semantic: float = 0.40  # Reduced slightly to make room for context boosts
    recency: float = 0.20
    importance: float = 0.15
    type_match: float = 0.05  # Reduced
    temporal_match: float = 0.10
    emotional_boost: float = 0.05  # New: emotional significance
    identity_boost: float = 0.05  # New: core identity facts


def _recency_decay(created_at_iso: str, now: datetime, half_life_days: float = 30.0) -> float:
    """
    Compute recency score with exponential decay.

    Args:
        created_at_iso: ISO timestamp of memory creation
        now: Current datetime
        half_life_days: Days until score decays to 50%

    Returns:
        Recency score (0-1)
    """
    try:
        t = iso_to_dt(created_at_iso)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - t).total_seconds() / 86400.0)
    except Exception:
        age_days = 0.0

    if half_life_days <= 0:
        return 1.0  # No decay

    # exp decay with half-life
    return float(math.exp(-math.log(2) * age_days / half_life_days))

def score_hit(
    hit: dict,
    now: datetime,
    w: RankWeights,
    current_temporal_tags: dict,
    query_characteristics: Optional["RetrievalCharacteristics"] = None
) -> float:
    """
    Score a memory hit using weighted multi-dimensional ranking.

    Uses SEMANTIC analysis when available for:
        - Adaptive decay rates based on memory type
        - Importance floor enforcement
        - Emotional/identity boosts

    Args:
        hit: Memory hit dictionary
        now: Current datetime
        w: Rank weights configuration
        current_temporal_tags: Current time context
        query_characteristics: Optional query analysis (for explicit past references)

    Returns:
        Final ranking score
    """
    content = hit.get("content", "")
    sim = float(hit.get("similarity", 0.0))
    imp = float(hit.get("importance", 0.5))

    # Default values
    half_life = 30.0
    importance_floor = 0.5
    emotional_boost = 0.0
    identity_boost = 0.0
    disable_recency = False

    # =================================================================
    # SEMANTIC ANALYSIS (adaptive decay based on memory content)
    # =================================================================
    if SEMANTIC_RETRIEVAL_AVAILABLE and content:
        try:
            analyzer = get_semantic_retrieval_analyzer()

            # Analyze memory content for characteristics
            mem_chars = analyzer.analyze_memory(content, threshold=0.45)

            # Use adaptive half-life
            half_life = mem_chars.half_life_days
            importance_floor = mem_chars.importance_floor
            emotional_boost = mem_chars.emotional_boost
            identity_boost = mem_chars.identity_boost

            # Check if query explicitly references past (disable recency penalty)
            if query_characteristics and query_characteristics.disable_recency_penalty:
                disable_recency = True

        except Exception as e:
            # Fall through to default decay
            pass

    # Enforce importance floor
    imp = max(imp, importance_floor)

    # Compute recency with adaptive half-life
    if disable_recency:
        rec = 1.0  # No penalty for explicit past references
    else:
        rec = _recency_decay(hit.get("created_at", ""), now, half_life)

    # Temporal context matching (same day of week, same hour)
    temporal_score = 0.0
    hit_tags_raw = hit.get("temporal_tags")
    if hit_tags_raw is not None and hit_tags_raw != "":
        try:
            parsed_tags = json.loads(hit_tags_raw)
            hit_day = parsed_tags.get("day_of_week")
            hit_hour = parsed_tags.get("hour_of_day")
            current_day = current_temporal_tags.get("day_of_week")
            current_hour = current_temporal_tags.get("hour_of_day")
            if hit_day == current_day:
                temporal_score += 0.5
            if hit_hour == current_hour:
                temporal_score += 0.5
            if temporal_score > 1.0:
                temporal_score = 1.0
        except Exception:
            temporal_score = 0.0

    # Compute final score
    score = (
        w.semantic * sim +
        w.recency * rec +
        w.importance * imp +
        w.temporal_match * temporal_score +
        w.emotional_boost * emotional_boost +
        w.identity_boost * identity_boost
    )

    return score


def analyze_query_for_retrieval(query: str) -> Optional["RetrievalCharacteristics"]:
    """
    Analyze query to check for explicit past time references.

    If user asks "What did I tell you last year?", we should NOT
    penalize old memories with recency decay.

    Args:
        query: User's query text

    Returns:
        RetrievalCharacteristics if semantic available, None otherwise
    """
    if not SEMANTIC_RETRIEVAL_AVAILABLE:
        return None

    try:
        analyzer = get_semantic_retrieval_analyzer()
        return analyzer.analyze_query(query, threshold=0.45)
    except Exception:
        return None