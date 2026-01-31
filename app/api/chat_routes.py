"""
Chat API Routes - Full Mouth Implementation
Integrates Nura Brain (all engines) with Cartesia Voice (Sonic 3 TTS).
Compatible with legacy UI expectations.
"""

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
import json
import os

from app.memory.memory_engine import MemoryEngine
from app.temporal.temporal_engine import TemporalEngine, get_temporal_engine
from app.adaptation.adaptation_engine import AdaptationEngine
from app.adaptation.breakthrough_detector import detect_breakthrough
from app.retrieval.retrieval_engine import RetrievalEngine, RetrievalState, RetrievalResult
from app.retrieval.retrieval_strategies import RetrievalStrategy, QueryAnalysis
from app.vector.embedding_service import EmbeddingService
from app.metrics.relationship_metrics import ConversationMetrics
from app.services.voice_service import get_voice_service
from app.services.llm_service import get_llm_service
from app.services.ssml_generator import generate_ssml
from app.db.session import get_db_context
from app.core.telemetry import PerformanceTracker
from app.guards.safety_layer import get_safety_layer
from app.guards.memory_hallucination import guard_memories, is_memory_query
from app.orchestrator.intent_gate import classify_intent
from app.proactive.proactive_engine import decide_followup

# OPTIMIZATION: Import async queue and parallel executor
from app.integration.async_memory_queue import get_async_memory_queue
from app.integration.parallel_engine_executor import (
    ParallelEngineExecutor,
    EngineContext,
    create_parallel_executor
)

router = APIRouter()

# OPTIMIZATION: Global parallel executor (initialized lazily)
_parallel_executor: Optional[ParallelEngineExecutor] = None


def _log_engine_context(payload: dict) -> None:
    """Append engine context snapshot as JSONL for auditing."""
    log_dir = "logs"
    log_path = os.path.join(log_dir, "engine_context.jsonl")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="ascii", errors="ignore") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

# Pydantic models (Legacy UI compatible)
class ChatRequest(BaseModel):
    message: str
    mode: str = "spiritual"
    user_id: Optional[str] = "default_user"
    filler: Optional[str] = None  # Frontend sends this (not used)


# Engine singletons
_embedding_service = None
_memory_engine = None
_temporal_engine = None
_adaptation_engine = None
_retrieval_engine = None


def get_engines():
    """Get or create engine singletons."""
    global _embedding_service, _memory_engine, _temporal_engine, _adaptation_engine, _retrieval_engine

    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    if _memory_engine is None:
        _memory_engine = MemoryEngine(_embedding_service)
    if _temporal_engine is None:
        _temporal_engine = TemporalEngine()
    if _adaptation_engine is None:
        _adaptation_engine = AdaptationEngine()
    if _retrieval_engine is None:
        _retrieval_engine = RetrievalEngine(_memory_engine, _temporal_engine)

    return _embedding_service, _memory_engine, _temporal_engine, _adaptation_engine, _retrieval_engine


def get_parallel_executor():
    """Get or create parallel executor singleton (OPTIMIZATION)."""
    global _parallel_executor
    if _parallel_executor is None:
        embedding_service, memory_engine, _, _, _ = get_engines()
        async_queue = get_async_memory_queue()
        _parallel_executor = create_parallel_executor(
            memory_engine=memory_engine,
            embedding_service=embedding_service,
            async_memory_queue=async_queue
        )
    return _parallel_executor


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Main conversation endpoint - THE FULL MOUTH.

    Pipeline:
    1. BRAIN: Nura Engines orchestration
       - Memory: Ingest user message
       - Retrieval: Get relevant memories
       - Adaptation: Update emotional profile
       - Response: Generate contextual text
    2. VOICE: Cartesia Sonic 3 TTS
       - Map profile to voice parameters
       - Synthesize audio using expression labels
    3. RESPONSE: Return audio blob + metadata headers

    Legacy UI Compatibility:
    - Request: {message, mode, user_id}
    - Response: Audio blob (audio/mpeg) or JSON fallback
    - Headers: X-Response-Text, X-Emotion, X-Memory-Count
    """
    try:
        print("=" * 60)
        print(f"[NURA] NEW REQUEST")
        print(f"[NURA] Message: {request.message}")
        print(f"[NURA] Mode: {request.mode}")
        print(f"[NURA] User: {request.user_id}")
        print("=" * 60)

        safety_layer = get_safety_layer()
        safety_decision = safety_layer.assess(str(request.user_id), request.message)
        if safety_decision.should_block:
            refusal_text = (
                "I'm sorry, I can't help with that. "
                "If you're dealing with something difficult, I can offer general support or safer alternatives."
            )
            payload = {
                "text": refusal_text,
                "emotion": "neutral",
                "memory_count": 0,
                "profile": {"warmth": 0.5, "formality": 0.5, "initiative": 0.5},
            }
            return Response(
                content=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
                media_type="application/json",
                headers={
                    "X-Response-Text": refusal_text,
                    "X-Emotion": "neutral",
                    "X-Memory-Count": "0",
                    "X-Nura-Warmth": "0.50",
                    "X-Nura-Formality": "0.50",
                    "X-Nura-Initiative": "0.50",
                    "X-Primary-Emotion": "neutral",
                    "X-Secondary-Emotion": "",
                },
            )

        # Initialize performance tracker
        tracker = PerformanceTracker()

        # INTENT_GATE: pre-engine routing decision
        intent = classify_intent(request.message)
        if intent.get("primary_intent") == "GENERAL_KNOWLEDGE":
            print("[INTENT_GATE] GENERAL_KNOWLEDGE: skipping memory/retrieval/adaptation/proactive")
            # Skip memory/retrieval/adaptation/proactive engines
            llm_service = get_llm_service()
            voice_service = get_voice_service()
            profile = {"warmth": 0.5, "formality": 0.5, "initiative": 0.5}
            temporal_tags = []
            memory_count = 0
            emotion = "neutral"

            # Generation
            tracker.start("generation")
            try:
                # OPTIMIZATION: Removed proactive engine and logging for GENERAL_KNOWLEDGE path
                plain_text_response = llm_service.generate_response(
                    user_message=request.message,
                    mode=request.mode,
                    profile=profile,
                    memories=[],
                    temporal_tags=temporal_tags,
                    emotion=emotion,
                    use_gpt4=False
                )
                response_text = plain_text_response
            except Exception:
                fallback_state = RetrievalResult(hits=[], facts={}, milestones=[], strategy_used=RetrievalStrategy.HYBRID, query_analysis=QueryAnalysis(strategy=RetrievalStrategy.HYBRID, confidence=0.0))
                plain_text_response = _generate_response(
                    user_message=request.message,
                    mode=request.mode,
                    retrieval_result=fallback_state,
                    profile=profile,
                    temporal_tags=temporal_tags,
                    emotion=emotion
                )
                response_text = plain_text_response
            tracker.end("generation")

            # Voice
            tracker.start("voice")
            ssml_text, ssml_emotion = generate_ssml(
                text=plain_text_response,
                profile=profile,
                emotion=emotion,
                temporal_tags=temporal_tags,
            )
            audio_bytes = await voice_service.synthesize(
                text=plain_text_response,
                profile=profile,
                emotion=emotion,
                temporal_tags=temporal_tags,
                text_with_ssml=ssml_text,
                primary_emotion=ssml_emotion,
                secondary_emotion=None
            )
            tracker.end("voice")

            if audio_bytes:
                media_type = "audio/mpeg"
            else:
                audio_bytes = json.dumps({
                    "text": response_text,
                    "emotion": emotion,
                    "memory_count": memory_count,
                    "profile": profile
                }).encode('utf-8')
                media_type = "application/json"

            print(f"[TELEMETRY] Total: {tracker.get_total_time():.2f}ms | Status: {tracker.get_total_status()[1]}")

            return Response(
                content=audio_bytes,
                media_type=media_type,
                headers={
                    "X-Response-Text": plain_text_response,
                    "X-Emotion": emotion,
                    "X-Memory-Count": str(memory_count),
                    "X-Nura-Warmth": f"{profile['warmth']:.2f}",
                    "X-Nura-Formality": f"{profile['formality']:.2f}",
                    "X-Nura-Initiative": f"{profile['initiative']:.2f}",
                    "X-Primary-Emotion": emotion,
                    "X-Secondary-Emotion": ""
                }
            )

        # OPTIMIZATION: Get parallel executor and engines
        parallel_executor = get_parallel_executor()
        _, memory_engine, _, adaptation_engine, _ = get_engines()
        voice_service = get_voice_service()
        llm_service = get_llm_service()

        # Convert user_id to int
        user_id = _parse_user_id(request.user_id)
        session_id = f"session_{user_id}_{int(datetime.now(timezone.utc).timestamp())}"
        now = datetime.now(timezone.utc)

        # ============================================
        # OPTIMIZATION: PARALLEL ENGINE EXECUTION
        # ============================================
        print(f"[BRAIN] Running engines in parallel...")
        tracker.start("engines")

        # Build engine context
        primary_intent = intent.get("primary_intent", "UNKNOWN")
        confidence = float(intent.get("confidence", 0.0))
        ambiguity_flags = set(intent.get("ambiguity_flags") or [])

        engine_ctx = EngineContext(
            user_id=user_id,
            text=request.message,
            now=now,
            session_id=session_id,
            intent=primary_intent,
            confidence=confidence,
            ambiguity_flags=ambiguity_flags
        )

        # Execute engines in parallel (retrieval blocks if needed, memory/adaptation async)
        engine_results, _ = parallel_executor.execute_with_policy(engine_ctx)
        tracker.end("engines")

        # Extract results
        temporal_tags = engine_results.temporal_context
        temporal_rewrite = engine_results.temporal_rewrite
        retrieval_result = engine_results.retrieval_results or RetrievalResult(hits=[], facts={}, milestones=[], strategy_used=RetrievalStrategy.HYBRID, query_analysis=QueryAnalysis(strategy=RetrievalStrategy.HYBRID, confidence=0.0))
        memory_count = len(retrieval_result.hits)

        # Get adaptation profile (from DB, updated async in background)
        profile = adaptation_engine.get_profile(user_id)

        # Assign expression label from text cues
        signals = detect_breakthrough(request.message)
        emotion = _map_signals_to_emotion(signals)
        print(f"[BRAIN] Engines complete: temporal={len(temporal_tags) if temporal_tags else 0} tags, "
              f"retrieval={memory_count} hits, emotion={emotion}")

        # ============================================
        # STEP 4: NURA BRAIN - RESPOND (Local LLM)
        # ============================================
        print(f"[BRAIN] Generating response (local LLM)...")
        tracker.start("generation")

        try:
            # OPTIMIZATION: Removed proactive engine (stubbed cooldown, deferred to Phase 6+)
            # OPTIMIZATION: Removed blocking engine context logging (use async if needed)
            plain_text_response = llm_service.generate_response(
                user_message=request.message,
                mode=request.mode,
                profile=profile,
                memories=retrieval_result.hits[:5],
                temporal_tags=temporal_tags,
                emotion=emotion,
                use_gpt4=False
            )
            response_text = plain_text_response
            primary_emotion = None
            secondary_emotion = None
            print(f"[BRAIN] Local LLM response generated ({len(plain_text_response)} characters)")
        except Exception as e:
            print(f"[BRAIN] WARNING: Local LLM unavailable, using rule-based fallback")
            plain_text_response = _generate_response(
                user_message=request.message,
                mode=request.mode,
                retrieval_result=retrieval_result,
                profile=profile,
                temporal_tags=temporal_tags,
                emotion=emotion
            )
            response_text = plain_text_response
            primary_emotion = None
            secondary_emotion = None
            print(f"[BRAIN] Fallback response generated ({len(plain_text_response)} characters)")

        tracker.end("generation")
        print(f"[BRAIN] Response complete")

        # ============================================
        # OPTIMIZATION: ASYNC ASSISTANT RESPONSE STORAGE
        # ============================================
        # Move assistant response storage to background (doesn't block TTS)
        async_queue = get_async_memory_queue()
        assistant_task_id = async_queue.enqueue_memory_write(
            memory_engine=memory_engine,
            user_id=user_id,
            role="assistant",
            text=plain_text_response,
            session_id=session_id,
            ts=datetime.now(timezone.utc),
            temporal_tags=temporal_tags,
            source="assistant_async",
            metadata={},
            priority=1  # High priority for assistant responses
        )
        print(f"[BRAIN] Assistant response queued for storage (task_id={assistant_task_id})")

        # ============================================
        # VOICE: CARTESIA TTS (Phase 11: LLM-SSML Aware)
        # ============================================
        print(f"[VOICE] Synthesizing audio...")
        print(f"[VOICE] Temporal context: {', '.join(temporal_tags)}")
        tracker.start("voice")
        ssml_text, ssml_emotion = generate_ssml(
            text=plain_text_response,
            profile=profile,
            emotion=emotion,
            temporal_tags=temporal_tags,
        )
        audio_bytes = await voice_service.synthesize(
            text=plain_text_response,  # Fallback plain text
            profile=profile,
            emotion=emotion,
            temporal_tags=temporal_tags,
            text_with_ssml=ssml_text,
            primary_emotion=ssml_emotion,
            secondary_emotion=None
        )
        tracker.end("voice")

        if audio_bytes:
            print(f"[VOICE] Audio generated: {len(audio_bytes)} bytes")
            media_type = "audio/mpeg"
        else:
            # Fallback: Return text as JSON if TTS fails
            print(f"[VOICE] WARNING: TTS unavailable, returning JSON")
            audio_bytes = json.dumps({
                "text": response_text,
                "emotion": emotion,
                "memory_count": memory_count,
                "profile": profile
            }).encode('utf-8')
            media_type = "application/json"

        # ============================================
        # RESPONSE: Audio blob + headers
        # ============================================
        print(f"[NURA] REQUEST COMPLETE")
        print("=" * 60)

        # Display telemetry dashboard (disabled on Windows CMD due to Unicode issues)
        # tracker.display()  # Uncomment when running in Windows Terminal or Linux
        print(f"[TELEMETRY] Total: {tracker.get_total_time():.2f}ms | Status: {tracker.get_total_status()[1]}")

        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-Response-Text": plain_text_response,  # Phase 11: Return plain text (no SSML in headers)
                "X-Emotion": emotion,
                "X-Memory-Count": str(memory_count),
                "X-Nura-Warmth": f"{profile['warmth']:.2f}",
                "X-Nura-Formality": f"{profile['formality']:.2f}",
                "X-Nura-Initiative": f"{profile['initiative']:.2f}",
                "X-Primary-Emotion": primary_emotion if primary_emotion else emotion,  # Phase 11: Cartesia emotion
                "X-Secondary-Emotion": secondary_emotion if secondary_emotion else ""  # Phase 11: Secondary emotion
            }
        )

    except Exception as e:
        # Handle errors (avoid Windows encoding issues with ASCII-only)
        error_msg = str(e)
        try:
            safe_msg = error_msg.encode('ascii', 'replace').decode('ascii')
            print(f"[NURA] ERROR: {safe_msg}")
        except:
            print(f"[NURA] ERROR: {type(e).__name__}")

        import traceback
        try:
            traceback.print_exc()
        except:
            print("[NURA] Traceback unavailable (encoding error)")

        raise HTTPException(status_code=500, detail=error_msg)


def _parse_user_id(user_id_str: str) -> int:
    """Parse user_id string to int."""
    try:
        # Handle "default_user" → 1
        if user_id_str == "default_user":
            return 1
        # Handle "user_123" → 123
        if user_id_str.startswith("user_"):
            return int(user_id_str.replace("user_", ""))
        # Handle "123" → 123
        return int(user_id_str)
    except:
        return 1  # Default


def _generate_placeholder_response(message: str, retrieval_result) -> str:
    """Generate simple placeholder for metrics."""
    if retrieval_result.hits:
        try:
            # Try to get text from first hit
            first_hit = retrieval_result.hits[0]
            if isinstance(first_hit, dict) and 'text' in first_hit:
                return f"I remember you mentioned: {first_hit['text']}"
        except (KeyError, IndexError, TypeError):
            pass
    return "I understand. Tell me more."


def _generate_response(
    user_message: str,
    mode: str,
    retrieval_result,
    profile: dict,
    temporal_tags: list,
    emotion: str
) -> str:
    """
    Generate contextual response based on Nura context.

    Phase 7: Rule-based response generation.
    Phase 8: Local LLM response generation for advanced responses.
    """
    guard_result = guard_memories(retrieval_result.hits if retrieval_result else [])
    if guard_result.should_fallback and is_memory_query(user_message):
        return guard_result.fallback_text

    memory_text = None
    memory_confidence = None
    if guard_result.high_confidence:
        memory_text = guard_result.high_confidence[0]
        memory_confidence = "high"
    elif guard_result.medium_confidence:
        memory_text = guard_result.medium_confidence[0]
        memory_confidence = "medium"
    warmth = profile["warmth"]
    formality = profile["formality"]

    # Get time of day
    time_of_day = _get_time_of_day(temporal_tags)

    # Mode-based base responses (match legacy system)
    mode_responses = {
        "spiritual": "I'm here to support you with faith and understanding.",
        "calm": "Take a deep breath. Let's talk about this calmly.",
        "logical": "Let's think through this step by step.",
        "emotional": "I hear you, and your feelings are valid."
    }

    base = mode_responses.get(mode.lower(), mode_responses["spiritual"])

    # Build response based on context
    if memory_text:
        if memory_confidence == "medium":
            prefix = "I think you mentioned"
        else:
            prefix = "I remember when you shared"

        if warmth > 0.7:
            # High warmth: empathetic, personal
            response = f"{prefix}: '{memory_text}'. {base}"
        elif formality > 0.7:
            # High formality: professional, structured
            if memory_confidence == "medium":
                response = f"I might be mistaken, but I think we discussed '{memory_text}'. {base}"
            else:
                response = f"Based on our previous conversation about '{memory_text}', {base.lower()}"
        else:
            # Normal: balanced
            response = f"{prefix} '{memory_text}' before. {base}"
    else:
        # No memories: fresh conversation
        if warmth > 0.7:
            response = f"{base} What's on your heart right now?"
        elif formality > 0.7:
            response = f"{base} How may I assist you today?"
        else:
            response = base

    # Add temporal context
    if time_of_day == "morning":
        response += " I hope you're having a peaceful morning."
    elif time_of_day == "evening":
        response += " I hope your evening is going well."
    elif time_of_day == "night":
        response += " I hope you can find rest tonight."

    # Add emotion-specific support
    if emotion == "anxious":
        if warmth > 0.6:
            response += " Remember, you're not alone in this. Take it one step at a time."
        else:
            response += " Consider taking a moment to breathe and center yourself."
    elif emotion == "grateful":
        response += " I'm glad to hear that. Gratitude is a beautiful thing."

    return response


def _get_time_of_day(temporal_tags: list) -> str:
    """Extract time of day from temporal tags."""
    if "morning" in temporal_tags:
        return "morning"
    elif "evening" in temporal_tags:
        return "evening"
    elif "night" in temporal_tags:
        return "night"
    else:
        return "day"


def _map_signals_to_emotion(signals) -> str:
    """Map text cues to an expression label (UI-compatible)."""
    if signals.vulnerability:
        return "anxious"
    elif signals.gratitude:
        return "grateful"
    elif signals.prayer:
        return "hopeful"
    else:
        return "neutral"


# ============================================
# LEGACY COMPATIBILITY ENDPOINTS
# ============================================

@router.get("/memories/{user_id}")
async def get_memories(user_id: str):
    """Get user memories (UI-compatible endpoint)."""
    try:
        uid = _parse_user_id(user_id)

        with get_db_context() as conn:
            cursor = conn.execute("""
                SELECT content as text, memory_type, importance, created_at
                FROM memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 20
            """, (uid,))
            rows = cursor.fetchall()

        memories = [
            {
                "content": row["text"],
                "timestamp": row["created_at"],
                "type": row["memory_type"],
                "importance": row["importance"]
            }
            for row in rows
        ]

        return {"user_id": user_id, "memories": memories}

    except Exception as e:
        print(f"[NURA] Error getting memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memories/{user_id}")
async def clear_memories(user_id: str):
    """Clear user memories (UI-compatible endpoint)."""
    try:
        uid = _parse_user_id(user_id)

        with get_db_context() as conn:
            conn.execute("DELETE FROM memories WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM facts WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM temporal_patterns WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM relationship_metrics WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM adaptation_profiles WHERE user_id = ?", (uid,))
            conn.commit()

        return {"status": "success"}

    except Exception as e:
        print(f"[NURA] Error clearing memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
