import pytest
from datetime import datetime, timezone
from app.retrieval.ranker import score_hit, RankWeights

def test_score_hit_basic():
    hit = {
        "similarity": 0.8,
        "importance": 0.7,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "temporal_tags": "{}"
    }
    now = datetime.now(timezone.utc)
    weights = RankWeights()
    score = score_hit(hit, now, weights, {})
    assert 0.0 <= score <= 1.0
    assert score > 0.3

def test_score_hit_weights():
    hit = {
        "similarity": 1.0,
        "importance": 1.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "temporal_tags": "{}"
    }
    now = datetime.now(timezone.utc)
    weights = RankWeights(semantic=0.5, recency=0.3, importance=0.2, temporal_match=0.0)
    score = score_hit(hit, now, weights, {})
    assert score > 0.9
