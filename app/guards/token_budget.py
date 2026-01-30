from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_LOG_DIR = r"D:\Nura\Docs\guard_logs"
_LOG_FILE = "token_budget.jsonl"

# Token budgets for Qwen3-4B (4096 context window)
# Total input budget: ~1,500 tokens (leaves room for response)
BUDGETS = {
    "system": 300,       # System prompt + personality
    "memory": 400,       # Memory context (max 5 memories)
    "retrieval": 300,    # Retrieved items
    "temporal": 100,     # Time context
    "adaptability": 100, # Tone/profile
    "user_input": 300,   # User's message
}
RESPONSE_RESERVE = 300   # Max response tokens

# Context window: 4096
# Input budget: 1,500
# Response: 300
# Safety margin: ~2,296 tokens

_SCORE_KEYS = ("final_score", "similarity", "importance", "score")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def _count_tokens(text: str) -> int:
    return len(_tokenize(text))


def _truncate_text(text: str, budget: int) -> str:
    if budget <= 0 or not text:
        return ""
    tokens = _tokenize(text)
    if len(tokens) <= budget:
        return text
    return " ".join(tokens[:budget])


def _truncate_items(items: List[str], budget: int) -> Tuple[List[str], int]:
    if budget <= 0:
        return [], 0
    kept: List[str] = []
    remaining = budget
    for item in items:
        if remaining <= 0:
            break
        item_tokens = _count_tokens(item)
        if item_tokens <= remaining:
            kept.append(item)
            remaining -= item_tokens
        else:
            kept.append(_truncate_text(item, remaining))
            remaining = 0
    used = budget - remaining
    return kept, used


def _sort_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return items
    if not any(any(k in item for k in _SCORE_KEYS) for item in items):
        return items

    def sort_key(item: Dict[str, Any]) -> float:
        for key in _SCORE_KEYS:
            if key in item:
                try:
                    return float(item.get(key) or 0.0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    return sorted(items, key=sort_key, reverse=True)


def _coerce_list(value: Any) -> Optional[List[Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None
    return None


def _log_truncation(payload: Dict[str, Any], log_dir: Optional[str]) -> None:
    out_dir = log_dir or _DEFAULT_LOG_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _LOG_FILE)
    with open(path, "a", encoding="ascii", errors="ignore") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def enforce_token_budgets(
    engine_context: Dict[str, Any],
    system_text: str,
    log_dir: Optional[str] = None
) -> Tuple[Dict[str, Any], str, Dict[str, Dict[str, int]]]:
    """
    Enforce per-section token budgets via truncation only.
    Returns (new_context, new_system_text, report).
    """
    report: Dict[str, Dict[str, int]] = {}
    truncated = False

    memory_summary = str(engine_context.get("memory_summary") or "")
    retrieved_items = engine_context.get("retrieved_items") or ""
    time_context = str(engine_context.get("time_context") or "")
    tone_profile = str(engine_context.get("tone_profile") or "")

    memory_attempted = _count_tokens(memory_summary)
    memory_items = [s.strip() for s in memory_summary.split(";") if s.strip()]
    memory_kept, memory_used = _truncate_items(memory_items, BUDGETS["memory"])
    memory_truncated = memory_attempted > memory_used
    memory_summary_new = "; ".join(memory_kept)

    retrieval_attempted = _count_tokens(str(retrieved_items))
    retrieval_list = _coerce_list(retrieved_items)
    if retrieval_list is None:
        retrieval_text_new = _truncate_text(str(retrieved_items), BUDGETS["retrieval"])
        retrieval_used = _count_tokens(retrieval_text_new)
    else:
        sorted_items = _sort_items([i for i in retrieval_list if isinstance(i, dict)])
        serialized_items = [json.dumps(item, ensure_ascii=True) for item in sorted_items]
        kept_items, retrieval_used = _truncate_items(serialized_items, BUDGETS["retrieval"])
        retrieval_text_new = "[" + ", ".join(kept_items) + "]" if kept_items else "[]"
    retrieval_truncated = retrieval_attempted > retrieval_used

    temporal_attempted = _count_tokens(time_context)
    time_context_new = _truncate_text(time_context, BUDGETS["temporal"])
    temporal_used = _count_tokens(time_context_new)
    temporal_truncated = temporal_attempted > temporal_used

    adaptability_attempted = _count_tokens(tone_profile)
    tone_profile_new = _truncate_text(tone_profile, BUDGETS["adaptability"])
    adaptability_used = _count_tokens(tone_profile_new)
    adaptability_truncated = adaptability_attempted > adaptability_used

    system_attempted = _count_tokens(system_text)
    system_text_new = _truncate_text(system_text, BUDGETS["system"])
    system_used = _count_tokens(system_text_new)
    system_truncated = system_attempted > system_used

    report["memory"] = {"attempted": memory_attempted, "actual": memory_used, "budget": BUDGETS["memory"]}
    report["retrieval"] = {"attempted": retrieval_attempted, "actual": retrieval_used, "budget": BUDGETS["retrieval"]}
    report["temporal"] = {"attempted": temporal_attempted, "actual": temporal_used, "budget": BUDGETS["temporal"]}
    report["adaptability"] = {"attempted": adaptability_attempted, "actual": adaptability_used, "budget": BUDGETS["adaptability"]}
    report["system"] = {"attempted": system_attempted, "actual": system_used, "budget": BUDGETS["system"]}

    truncated = any([
        memory_truncated,
        retrieval_truncated,
        temporal_truncated,
        adaptability_truncated,
        system_truncated,
    ])

    if truncated:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": engine_context.get("user_id") or "unknown",
            "attempted_tokens": {k: v["attempted"] for k, v in report.items()},
            "actual_tokens": {k: v["actual"] for k, v in report.items()},
            "budgets": {k: v["budget"] for k, v in report.items()},
            "response_reserve": RESPONSE_RESERVE,
        }
        _log_truncation(payload, log_dir=log_dir)

    new_context = dict(engine_context)
    new_context["memory_summary"] = memory_summary_new
    new_context["retrieved_items"] = retrieval_text_new
    new_context["time_context"] = time_context_new
    new_context["tone_profile"] = tone_profile_new

    return new_context, system_text_new, report
