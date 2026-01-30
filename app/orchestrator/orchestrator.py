"""
Orchestrator v2.

Central coordinator for all Nura engines.
Routes user input through Intent Gate, then orchestrates engine calls
based on routing policy.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           ORCHESTRATOR v2                                   │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │                            User Input                                       │
    │                                │                                            │
    │                                ▼                                            │
    │                   ┌─────────────────────────┐                               │
    │                   │      SAFETY LAYER       │ ◄── Content filtering         │
    │                   └────────────┬────────────┘                               │
    │                                │ pass                                       │
    │                                ▼                                            │
    │                   ┌─────────────────────────┐                               │
    │                   │      INTENT GATE        │ ◄── Semantic classification   │
    │                   │                         │     (~0.1-5ms)                │
    │                   │  • EXPLICIT_MEMORY      │                               │
    │                   │  • PAST_SELF_REFERENCE  │                               │
    │                   │  • PERSONAL_STATE       │                               │
    │                   │  • GENERAL_KNOWLEDGE    │                               │
    │                   └────────────┬────────────┘                               │
    │                                │                                            │
    │                                ▼                                            │
    │                   ┌─────────────────────────┐                               │
    │                   │    ROUTING POLICY       │                               │
    │                   │                         │                               │
    │                   │  required / optional /  │                               │
    │                   │  forbidden engines      │                               │
    │                   └────────────┬────────────┘                               │
    │                                │                                            │
    │        ┌───────────┬───────────┼───────────┬───────────┐                    │
    │        ▼           ▼           ▼           ▼           ▼                    │
    │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
    │   │ TEMPORAL│ │ MEMORY  │ │RETRIEVAL│ │ADAPTATION│ │PROACTIVE│              │
    │   │ ENGINE  │ │ ENGINE  │ │ ENGINE  │ │ ENGINE  │ │ ENGINE  │              │
    │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘              │
    │        │           │           │           │           │                    │
    │        └───────────┴───────────┼───────────┴───────────┘                    │
    │                                │                                            │
    │                                ▼                                            │
    │                   ┌─────────────────────────┐                               │
    │                   │    LLM INTERFACE        │                               │
    │                   │    (Response Gen)       │                               │
    │                   └─────────────────────────┘                               │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

# =============================================================================
# ENGINE IMPORTS
# =============================================================================

# Intent Gate
from app.orchestrator.intent_gate import (
    get_intent_gate,
    IntentResult,
    INTENT_GENERAL_KNOWLEDGE,
)

# Runtime Logger
from app.orchestrator.runtime_logger import RuntimeLogger

# Core Engines
try:
    from app.memory import get_memory_store, MemoryStore
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("[Orchestrator] Memory engine not available")

try:
    from app.retrieval import RetrievalEngine, RetrievalState
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False
    print("[Orchestrator] Retrieval engine not available")

try:
    from app.temporal import get_temporal_engine, TemporalEngine
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    print("[Orchestrator] Temporal engine not available")

try:
    from app.adaptation import get_adaptation_engine, AdaptationEngine
    ADAPTATION_AVAILABLE = True
except ImportError:
    ADAPTATION_AVAILABLE = False
    print("[Orchestrator] Adaptation engine not available")

try:
    from app.proactive import get_proactive_engine
    PROACTIVE_AVAILABLE = True
except ImportError:
    PROACTIVE_AVAILABLE = False
    print("[Orchestrator] Proactive engine not available")

# LLM Interface
try:
    from app.services import nura_llm_interface
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[Orchestrator] LLM interface not available")

# Safety Layer
try:
    from app.guards.safety_layer import get_safety_layer
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False
    print("[Orchestrator] Safety layer not available")

# Metrics
try:
    from app.metrics.relationship_metrics import ConversationMetrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OrchestratorContext:
    """Context passed through orchestration pipeline."""
    user_id: int
    user_input: str
    session_id: str
    now: datetime

    # Intent classification
    intent_result: Optional[IntentResult] = None

    # Engine outputs
    temporal_context: Optional[Dict[str, Any]] = None
    temporal_rewrite: Optional[Dict[str, Any]] = None
    memory_result: Optional[Dict[str, Any]] = None
    retrieval_result: Optional[Any] = None
    adaptation_context: Optional[Dict[str, Any]] = None
    proactive_result: Optional[Dict[str, Any]] = None

    # Clarification
    clarification_needed: Optional[Dict[str, Any]] = None

    # Engines called
    engines_called: List[str] = field(default_factory=list)
    engines_blocked: List[str] = field(default_factory=list)


@dataclass
class OrchestratorResult:
    """Result from orchestrator."""
    llm_output: str
    log_path: Optional[str] = None
    safety_blocked: bool = False
    intent: Optional[str] = None
    engines_called: List[str] = field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


# =============================================================================
# ORCHESTRATOR v2
# =============================================================================

class OrchestratorV2:
    """
    Central coordinator for all Nura engines.

    Features:
        - Intent-based routing with semantic classification
        - Policy-driven engine activation
        - Integrated engine calls with proper context passing
        - Runtime logging and contract validation
    """

    def __init__(self):
        """Initialize orchestrator with engine instances."""
        # Intent Gate
        self._intent_gate = get_intent_gate()

        # Engines (lazy loaded)
        self._memory_store = None
        self._retrieval_engine = None
        self._temporal_engine = None
        self._adaptation_engine = None
        self._proactive_engine = None

        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of engines."""
        if self._initialized:
            return

        if MEMORY_AVAILABLE:
            try:
                self._memory_store = get_memory_store()
            except Exception:
                pass

        if TEMPORAL_AVAILABLE:
            try:
                self._temporal_engine = get_temporal_engine()
            except Exception:
                pass

        if ADAPTATION_AVAILABLE:
            try:
                self._adaptation_engine = get_adaptation_engine()
            except Exception:
                pass

        if PROACTIVE_AVAILABLE:
            try:
                self._proactive_engine = get_proactive_engine()
            except Exception:
                pass

        self._initialized = True

    # -------------------------------------------------------------------------
    # SAFETY CHECK
    # -------------------------------------------------------------------------

    def _check_safety(self, user_id: str, user_input: str) -> Optional[str]:
        """
        Check input against safety layer.

        Returns refusal message if blocked, None if safe.
        """
        if not SAFETY_AVAILABLE:
            return None

        try:
            safety_layer = get_safety_layer()
            decision = safety_layer.assess(user_id, user_input)

            if decision.should_block:
                return (
                    "I'm sorry, I can't help with that. "
                    "If you're dealing with something difficult, I can offer general support."
                )
        except Exception:
            pass

        return None

    # -------------------------------------------------------------------------
    # ENGINE CALLS
    # -------------------------------------------------------------------------

    def _should_call_engine(
        self,
        engine_name: str,
        intent_result: IntentResult,
        confidence: float
    ) -> bool:
        """Determine if engine should be called based on policy."""
        if engine_name in intent_result.engines_forbidden:
            return False
        if engine_name in intent_result.engines_required:
            return True
        if engine_name in intent_result.engines_optional:
            return confidence >= 0.4
        return False

    def _call_temporal_engine(
        self,
        ctx: OrchestratorContext,
        logger: RuntimeLogger
    ) -> None:
        """Call temporal engine for time context."""
        if not TEMPORAL_AVAILABLE or not self._temporal_engine:
            return

        engine_name = "TemporalEngine.temporal_tags_from_dt"
        if not self._should_call_engine(engine_name, ctx.intent_result, ctx.intent_result.confidence):
            ctx.engines_blocked.append(engine_name)
            logger.log_engine_blocked(engine_name, {"reason": "policy"})
            return

        try:
            # Generate temporal tags
            ctx.temporal_context = self._temporal_engine.generate_temporal_tags(ctx.now)

            # Parse for temporal rewrite
            result = self._temporal_engine.parse(ctx.user_input, ctx.now)
            ctx.temporal_rewrite = {
                "start_ts": result.start_ts,
                "end_ts": result.end_ts,
                "granularity": result.granularity.value if result.granularity else None,
                "window_days": result.retrieval_window_days,
                "requires_clarification": result.requires_clarification,
            }

            # Check if clarification needed
            if result.requires_clarification:
                ctx.clarification_needed = {
                    "temporal_ambiguity": True,
                    "clarification_prompt": "Could you clarify which specific time you mean?",
                }

            ctx.engines_called.append(engine_name)
            logger.log_engine_call(engine_name, {"now": ctx.now.isoformat()}, ctx.temporal_context)

        except Exception as e:
            logger.log_engine_call(engine_name, {"now": ctx.now.isoformat()}, {"error": str(e)})

    def _call_memory_engine(
        self,
        ctx: OrchestratorContext,
        logger: RuntimeLogger
    ) -> None:
        """Call memory engine to ingest event."""
        if not MEMORY_AVAILABLE or not self._memory_store:
            return

        engine_name = "MemoryEngine.ingest_event"
        if not self._should_call_engine(engine_name, ctx.intent_result, ctx.intent_result.confidence):
            ctx.engines_blocked.append(engine_name)
            logger.log_engine_blocked(engine_name, {"reason": "policy"})
            return

        try:
            # Ingest the user input as memory
            ctx.memory_result = self._memory_store.ingest_event(
                user_id=ctx.user_id,
                role="user",
                text=ctx.user_input,
                session_id=ctx.session_id,
                ts=ctx.now,
                temporal_tags=ctx.temporal_context or {},
                source="orchestrator",
                metadata={},
            )

            ctx.engines_called.append(engine_name)
            logger.log_engine_call(engine_name, {
                "user_id": ctx.user_id,
                "text": ctx.user_input[:100],
            }, ctx.memory_result)

        except Exception as e:
            logger.log_engine_call(engine_name, {"user_id": ctx.user_id}, {"error": str(e)})

    def _call_retrieval_engine(
        self,
        ctx: OrchestratorContext,
        logger: RuntimeLogger
    ) -> None:
        """Call retrieval engine to find relevant memories."""
        if not RETRIEVAL_AVAILABLE:
            return

        engine_name = "RetrievalEngine.retrieve"
        if not self._should_call_engine(engine_name, ctx.intent_result, ctx.intent_result.confidence):
            ctx.engines_blocked.append(engine_name)
            logger.log_engine_blocked(engine_name, {"reason": "policy"})
            return

        # Skip if temporal clarification needed for past reference
        if ctx.clarification_needed and ctx.intent_result.primary_intent == "PAST_SELF_REFERENCE":
            ctx.engines_blocked.append(engine_name)
            logger.log_engine_blocked(engine_name, {"reason": "temporal_ambiguity"})
            return

        try:
            retrieval_state = RetrievalState(
                user_id=ctx.user_id,
                query=ctx.user_input,
                now=ctx.now,
                current_temporal_tags=ctx.temporal_context or {},
                temporal_rewrite=ctx.temporal_rewrite or {},
            )

            # Get retrieval engine instance
            if self._memory_store and self._temporal_engine:
                retrieval_engine = RetrievalEngine(self._memory_store, self._temporal_engine)
                ctx.retrieval_result = retrieval_engine.retrieve(retrieval_state)

            ctx.engines_called.append(engine_name)
            logger.log_engine_call(engine_name, {
                "query": ctx.user_input[:100],
            }, {"hit_count": len(ctx.retrieval_result.hits) if ctx.retrieval_result else 0})

        except Exception as e:
            logger.log_engine_call(engine_name, {"query": ctx.user_input[:100]}, {"error": str(e)})

    def _call_adaptation_engine(
        self,
        ctx: OrchestratorContext,
        logger: RuntimeLogger
    ) -> None:
        """Call adaptation engine to update user profile."""
        if not ADAPTATION_AVAILABLE or not self._adaptation_engine:
            return

        engine_name = "AdaptationEngine.update"
        if not self._should_call_engine(engine_name, ctx.intent_result, ctx.intent_result.confidence):
            ctx.engines_blocked.append(engine_name)
            logger.log_engine_blocked(engine_name, {"reason": "policy"})
            return

        try:
            # Create metrics object
            if METRICS_AVAILABLE:
                metrics = ConversationMetrics.from_turn(ctx.user_input, "")
            else:
                # Fallback to simple object with user_text
                class SimpleMetrics:
                    def __init__(self, text):
                        self.user_text = text
                        self.prefers_direct = False
                metrics = SimpleMetrics(ctx.user_input)

            ctx.adaptation_context = self._adaptation_engine.update(ctx.user_id, metrics)

            ctx.engines_called.append(engine_name)
            logger.log_engine_call(engine_name, {
                "user_id": ctx.user_id,
            }, ctx.adaptation_context)

        except Exception as e:
            logger.log_engine_call(engine_name, {"user_id": ctx.user_id}, {"error": str(e)})

    def _call_proactive_engine(
        self,
        ctx: OrchestratorContext,
        logger: RuntimeLogger
    ) -> None:
        """Call proactive engine for follow-up evaluation."""
        if not PROACTIVE_AVAILABLE or not self._proactive_engine:
            return

        engine_name = "ProactiveEngine.evaluate"
        if not self._should_call_engine(engine_name, ctx.intent_result, ctx.intent_result.confidence):
            ctx.engines_blocked.append(engine_name)
            logger.log_engine_blocked(engine_name, {"reason": "policy"})
            return

        # Skip if ambiguous intent
        if ctx.intent_result.ambiguity:
            ctx.engines_blocked.append(engine_name)
            logger.log_engine_blocked(engine_name, {"reason": "ambiguous_intent"})
            return

        try:
            # Build proactive payload
            recent_memories = []
            if MEMORY_AVAILABLE and self._memory_store:
                try:
                    recent_memories = self._memory_store.recent(ctx.user_id, k=20)
                except Exception:
                    pass

            # Enrich memories for proactive engine
            enriched = []
            for mem in recent_memories:
                enriched.append({
                    "memory_id": mem.get("id"),
                    "content": mem.get("content"),
                    "memory_type": mem.get("memory_type"),
                    "importance": mem.get("importance"),
                    "created_at": mem.get("created_at"),
                    "temporal_tags": mem.get("temporal_tags", {}),
                    "metadata": mem.get("metadata", {}),
                    "due_at": mem.get("metadata", {}).get("due_at"),
                    "focus": mem.get("content", "")[:100],
                    "asked_before": mem.get("metadata", {}).get("asked_before", False),
                })

            payload = {
                "user_id": str(ctx.user_id),
                "now_timestamp": ctx.now.isoformat(),
                "recent_memories": enriched,
                "temporal_tags": list((ctx.temporal_context or {}).keys()),
                "cooldown_state": {
                    "last_asked_at": None,
                    "asks_today": 0,
                }
            }

            result = self._proactive_engine.evaluate(payload)
            ctx.proactive_result = result if isinstance(result, dict) else result.__dict__

            ctx.engines_called.append(engine_name)
            logger.log_engine_call(engine_name, {"user_id": ctx.user_id}, ctx.proactive_result)

        except Exception as e:
            logger.log_engine_call(engine_name, {"user_id": ctx.user_id}, {"error": str(e)})

    # -------------------------------------------------------------------------
    # LLM RESPONSE
    # -------------------------------------------------------------------------

    def _generate_llm_response(
        self,
        ctx: OrchestratorContext,
        logger: RuntimeLogger
    ) -> str:
        """Generate LLM response using engine context."""
        # Build context for LLM
        llm_context = {
            "user_input": ctx.user_input,
            "intent": ctx.intent_result.primary_intent if ctx.intent_result else None,
            "confidence": ctx.intent_result.confidence if ctx.intent_result else 0,
            "memory_results": ctx.memory_result,
            "retrieved_items": ctx.retrieval_result,
            "temporal_context": ctx.temporal_context,
            "temporal_rewrite": ctx.temporal_rewrite,
            "temporal_clarification_needed": ctx.clarification_needed,
            "adaptation_context": ctx.adaptation_context,
            "proactive": ctx.proactive_result,
        }

        if not LLM_AVAILABLE:
            return self._generate_fallback_response(ctx)

        try:
            response = nura_llm_interface.generate_response(llm_context, ctx.user_input)
            logger.log_engine_call("LLMInterface.generate_response", {
                "input_preview": ctx.user_input[:50],
            }, {"response_length": len(response) if response else 0})
            return response

        except Exception as e:
            logger.log_engine_call("LLMInterface.generate_response", {
                "input_preview": ctx.user_input[:50],
            }, {"error": str(e)})
            return self._generate_fallback_response(ctx)

    def _generate_fallback_response(self, ctx: OrchestratorContext) -> str:
        """Generate fallback response when LLM fails."""
        # Temporal clarification
        if ctx.clarification_needed:
            return ctx.clarification_needed.get(
                "clarification_prompt",
                "Could you clarify what you mean?"
            )

        # Retrieval context
        if ctx.retrieval_result:
            hits = getattr(ctx.retrieval_result, 'hits', [])
            if hits:
                return "I found some relevant context from our conversations. What would you like to know?"

        # Generic fallback
        return "I'm here to help. Could you tell me more about what you're thinking?"

    # -------------------------------------------------------------------------
    # MAIN HANDLER
    # -------------------------------------------------------------------------

    def handle_input(
        self,
        user_input: str,
        user_id: int = 1
    ) -> Dict[str, Any]:
        """
        Handle user input through the orchestration pipeline.

        Args:
            user_input: User's message
            user_id: User identifier

        Returns:
            Dict with llm_output, log_path, and metadata
        """
        self._ensure_initialized()
        logger = RuntimeLogger()

        # Build context
        now = datetime.now(timezone.utc)
        session_id = f"orch_{user_id}_{int(now.timestamp())}"

        ctx = OrchestratorContext(
            user_id=user_id,
            user_input=user_input,
            session_id=session_id,
            now=now,
        )

        # 1. Safety check
        refusal = self._check_safety(str(user_id), user_input)
        if refusal:
            return {
                "llm_output": refusal,
                "log_path": None,
                "safety_blocked": True,
            }

        # 2. Intent classification
        ctx.intent_result = self._intent_gate.classify(user_input)
        logger.log_intent_gate(user_input, ctx.intent_result.to_dict())

        # 3. Engine calls based on policy
        self._call_temporal_engine(ctx, logger)
        self._call_memory_engine(ctx, logger)
        self._call_retrieval_engine(ctx, logger)
        self._call_adaptation_engine(ctx, logger)
        self._call_proactive_engine(ctx, logger)

        # 4. Generate LLM response
        llm_output = self._generate_llm_response(ctx, logger)

        # 5. Finalize logging
        engine_context = {
            "intent": ctx.intent_result.to_dict() if ctx.intent_result else None,
            "temporal_context": ctx.temporal_context,
            "temporal_rewrite": ctx.temporal_rewrite,
            "memory_result": ctx.memory_result,
            "retrieval_hits": len(ctx.retrieval_result.hits) if ctx.retrieval_result else 0,
            "adaptation_context": ctx.adaptation_context,
            "proactive": ctx.proactive_result,
            "engines_called": ctx.engines_called,
            "engines_blocked": ctx.engines_blocked,
        }
        log_path = logger.finalize(engine_context)

        return {
            "llm_output": llm_output,
            "log_path": log_path,
            "safety_blocked": False,
            "intent": ctx.intent_result.primary_intent if ctx.intent_result else None,
            "engines_called": ctx.engines_called,
        }


# =============================================================================
# SINGLETON & LEGACY INTERFACE
# =============================================================================

_orchestrator_instance: Optional[OrchestratorV2] = None


def get_orchestrator() -> OrchestratorV2:
    """Get or create the singleton orchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorV2()
    return _orchestrator_instance


# Legacy class for backward compatibility
class Orchestrator(OrchestratorV2):
    """Legacy adapter for backward compatibility."""
    pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "OrchestratorV2",
    "Orchestrator",
    "OrchestratorContext",
    "OrchestratorResult",
    "get_orchestrator",
]
