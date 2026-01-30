from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, TypedDict


class RetrievalStateInput(TypedDict):
    user_id: int
    query: str
    now: datetime
    current_temporal_tags: Dict[str, Any]
    top_k: int
    session_id: str


class RetrievalEngineOutput(TypedDict):
    hits: List[Dict[str, Any]]
    facts: Dict[str, Any]


RETRIEVAL_ENGINE_FORBIDDEN_KEYS = {"embedding", "vector", "prompt"}
RETRIEVAL_ENGINE_MAX_KB = 64.0
