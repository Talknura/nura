"""
Orchestrator Module.

Central coordinator for all Nura engines.
Routes user input through Intent Gate, then orchestrates engine calls
based on routing policy.
"""

from app.orchestrator.orchestrator import (
    OrchestratorV2 as Orchestrator,
    OrchestratorContext,
    OrchestratorResult,
    get_orchestrator,
)
from app.orchestrator.intent_gate import (
    IntentGateV2 as IntentGate,
    IntentResult,
    get_intent_gate,
    classify_intent,
    INTENT_EXPLICIT_MEMORY,
    INTENT_PAST_REFERENCE,
    INTENT_PERSONAL_STATE,
    INTENT_GENERAL_KNOWLEDGE,
)

__all__ = [
    # Orchestrator
    "Orchestrator",
    "OrchestratorContext",
    "OrchestratorResult",
    "get_orchestrator",
    # Intent Gate
    "IntentGate",
    "IntentResult",
    "get_intent_gate",
    "classify_intent",
    "INTENT_EXPLICIT_MEMORY",
    "INTENT_PAST_REFERENCE",
    "INTENT_PERSONAL_STATE",
    "INTENT_GENERAL_KNOWLEDGE",
]
