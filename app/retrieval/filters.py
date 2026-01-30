from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

@dataclass
class RetrievalFilters:
    memory_type: str | None = None
    max_age_days: int | None = None

def apply_filters(hits: list[dict], filters: RetrievalFilters, now: datetime) -> list[dict]:
    out = hits
    if filters.memory_type:
        out = [h for h in out if h.get("memory_type") == filters.memory_type]
    if filters.max_age_days is not None:
        cutoff = now - timedelta(days=filters.max_age_days)
        def _keep(h):
            try:
                t = datetime.fromisoformat(h["created_at"])
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                return t >= cutoff
            except Exception:
                return True
        out = [h for h in out if _keep(h)]
    return out
