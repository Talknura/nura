from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from app.temporal.temporal_engine import TemporalEngine

@dataclass
class ParsedQuery:
    raw: str
    time_window_days: int | None = None

def _days_from_rewrite(rewrite: Dict[str, Any]) -> int | None:
    start = rewrite.get("start_ts")
    end = rewrite.get("end_ts")
    if start is None or end is None:
        return None
    try:
        start_dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except Exception:
        return None
    delta_days = (end_dt.date() - start_dt.date()).days + 1
    return max(1, delta_days)


def parse_query(q: str, temporal: TemporalEngine, temporal_rewrite: Optional[Dict[str, Any]] = None) -> ParsedQuery:
    if temporal_rewrite is not None:
        if temporal_rewrite.get("requires_clarification"):
            return ParsedQuery(raw=q, time_window_days=None)
        days = _days_from_rewrite(temporal_rewrite)
        return ParsedQuery(raw=q, time_window_days=days)
    days = temporal.parse_time_window(q)
    return ParsedQuery(raw=q, time_window_days=days)
