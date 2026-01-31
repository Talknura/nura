"""
Semantic Intent Concepts for Nura.

Replaces regex-based intent classification with semantic understanding.
Enables intelligent intent detection based on meaning, not keyword patterns.

Intent Categories:
    - EXPLICIT_MEMORY_COMMAND: User wants to store/delete memory
    - PAST_SELF_REFERENCE: User asking about past conversations/memories
    - PERSONAL_STATE: User sharing emotions, feelings, future concerns
    - GENERAL_KNOWLEDGE: User asking factual questions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.vector.embedder import embed_text


# =============================================================================
# INTENT CONCEPT DEFINITIONS
# =============================================================================

@dataclass
class IntentConcept:
    """A semantic concept for intent classification."""
    name: str
    exemplars: List[str]  # Example phrases representing this intent
    category: str  # The intent category this maps to
    confidence_boost: float  # How much to boost confidence when matched


# -----------------------------------------------------------------------------
# EXPLICIT MEMORY COMMAND CONCEPTS
# -----------------------------------------------------------------------------

MEMORY_COMMAND_CONCEPTS = [
    IntentConcept(
        name="remember_directive",
        exemplars=[
            "Remember this", "Please remember", "Don't forget this",
            "Keep this in mind", "Make a note of this", "Note that",
            "Remember that I", "I want you to remember", "Store this",
            "Save this information", "Keep track of this",
            "Remember this for later", "Don't let me forget"
        ],
        category="EXPLICIT_MEMORY_COMMAND",
        confidence_boost=0.3
    ),
    IntentConcept(
        name="forget_directive",
        exemplars=[
            "Forget this", "Forget that", "Delete this memory",
            "Erase what I said", "Don't keep this", "Remove this",
            "I didn't mean that", "Ignore what I just said",
            "Pretend I didn't say that", "Scratch that",
            "Never mind what I said", "Drop that information"
        ],
        category="EXPLICIT_MEMORY_COMMAND",
        confidence_boost=0.3
    ),
]


# -----------------------------------------------------------------------------
# PAST SELF REFERENCE CONCEPTS
# -----------------------------------------------------------------------------

PAST_REFERENCE_CONCEPTS = [
    IntentConcept(
        name="recall_past_conversation",
        exemplars=[
            "What did I say about", "What did I tell you",
            "What did I mention", "Remind me what I said",
            "Do you remember what I said", "What was my note about",
            "Did I already tell you", "Have I mentioned",
            "What did we discuss about", "Recall our conversation about"
        ],
        category="PAST_SELF_REFERENCE",
        confidence_boost=0.25
    ),
    IntentConcept(
        name="recall_personal_info",
        exemplars=[
            "What is my name", "What's my favorite", "Do you know my",
            "What did you learn about me", "Tell me my preferences",
            "Remind me of my", "What was my goal", "What are my hobbies",
            "Do you remember my birthday", "What's my job"
        ],
        category="PAST_SELF_REFERENCE",
        confidence_boost=0.25
    ),
    IntentConcept(
        name="learning_reflection",
        exemplars=[
            "Did I already learn this", "Have I understood this before",
            "Did we cover this", "Have I seen this before",
            "I think I learned this", "Did I already know this",
            "Have I studied this topic", "Did we go over this already"
        ],
        category="PAST_SELF_REFERENCE",
        confidence_boost=0.2
    ),
    IntentConcept(
        name="reference_past_event",
        exemplars=[
            "Last time we talked", "Before when I mentioned",
            "Previously I said", "Earlier you told me",
            "Remember when I", "Back when we discussed",
            "You said before that", "Didn't you tell me"
        ],
        category="PAST_SELF_REFERENCE",
        confidence_boost=0.2
    ),
]


# -----------------------------------------------------------------------------
# PERSONAL STATE CONCEPTS
# -----------------------------------------------------------------------------

PERSONAL_STATE_CONCEPTS = [
    IntentConcept(
        name="negative_emotion",
        exemplars=[
            "I feel sad", "I'm feeling down", "I'm depressed",
            "I feel anxious", "I'm worried", "I'm scared",
            "I feel overwhelmed", "I'm stressed out", "I'm nervous",
            "I feel hopeless", "I can't handle this", "I'm afraid",
            "I'm lonely", "I feel lost", "I'm struggling"
        ],
        category="PERSONAL_STATE",
        confidence_boost=0.25
    ),
    IntentConcept(
        name="positive_emotion",
        exemplars=[
            "I feel happy", "I'm excited", "I'm feeling great",
            "I feel confident", "I'm glad", "I feel grateful",
            "I'm thrilled", "I feel proud", "I'm optimistic",
            "I feel relieved", "I'm content", "I feel at peace"
        ],
        category="PERSONAL_STATE",
        confidence_boost=0.2
    ),
    IntentConcept(
        name="state_update",
        exemplars=[
            "I feel better now", "I'm feeling better",
            "I'm okay now", "I feel improved", "Things are better",
            "I'm doing better", "I recovered", "I'm over it now",
            "The situation improved", "I'm calmer now"
        ],
        category="PERSONAL_STATE",
        confidence_boost=0.2
    ),
    IntentConcept(
        name="future_concern",
        exemplars=[
            "I have an interview tomorrow", "My exam is coming up",
            "I'm nervous about my meeting", "I'm worried about tomorrow",
            "I have a big day coming", "My appointment is soon",
            "I'm anxious about the presentation", "The deadline is approaching"
        ],
        category="PERSONAL_STATE",
        confidence_boost=0.2
    ),
    IntentConcept(
        name="learning_struggle",
        exemplars=[
            "I don't understand this", "I'm not getting it",
            "This doesn't make sense", "I can't grasp this concept",
            "I'm struggling to understand", "This is confusing me",
            "I'm lost on this topic", "I can't wrap my head around this"
        ],
        category="PERSONAL_STATE",
        confidence_boost=0.15
    ),
    IntentConcept(
        name="persistent_feeling",
        exemplars=[
            "I always feel this way", "I constantly worry about",
            "I keep feeling", "This feeling won't go away",
            "I'm always anxious", "I can never shake this feeling",
            "It keeps coming back", "I struggle with this all the time"
        ],
        category="PERSONAL_STATE",
        confidence_boost=0.15
    ),
    IntentConcept(
        name="personal_facts",
        exemplars=[
            "My name is", "I'm called", "Call me",
            "I work as", "I am a developer", "I'm a teacher",
            "I live in", "I'm from", "I grew up in",
            "I'm married", "I have kids", "I have a dog",
            "I'm studying", "I'm learning", "I just started",
            "I like", "I love", "My favorite is",
            "I'm interested in", "I enjoy", "I hate",
            "I am", "I'm single", "I'm retired"
        ],
        category="PERSONAL_STATE",
        confidence_boost=0.3
    ),
]


# -----------------------------------------------------------------------------
# GENERAL KNOWLEDGE CONCEPTS
# -----------------------------------------------------------------------------

GENERAL_KNOWLEDGE_CONCEPTS = [
    IntentConcept(
        name="factual_question",
        exemplars=[
            "What is the capital of", "How does photosynthesis work",
            "Explain the theory of", "Define the term",
            "What are the causes of", "Describe the process of",
            "Tell me about the history of", "What is the meaning of"
        ],
        category="GENERAL_KNOWLEDGE",
        confidence_boost=0.2
    ),
    IntentConcept(
        name="how_to_question",
        exemplars=[
            "How do I", "How can I", "What's the best way to",
            "How should I approach", "Walk me through",
            "Give me steps to", "Teach me how to", "Show me how"
        ],
        category="GENERAL_KNOWLEDGE",
        confidence_boost=0.15
    ),
    IntentConcept(
        name="explanation_request",
        exemplars=[
            "Can you explain", "Why does", "What makes",
            "Help me understand", "Break down for me",
            "Clarify how", "I want to know about", "Tell me why"
        ],
        category="GENERAL_KNOWLEDGE",
        confidence_boost=0.15
    ),
    IntentConcept(
        name="curiosity_question",
        exemplars=[
            "I'm curious about", "I wonder why", "Just wondering",
            "Quick question about", "Random question",
            "Out of curiosity", "I was thinking about"
        ],
        category="GENERAL_KNOWLEDGE",
        confidence_boost=0.1
    ),
]


# -----------------------------------------------------------------------------
# SIGNAL CONCEPTS (modifiers that affect interpretation)
# -----------------------------------------------------------------------------

SIGNAL_CONCEPTS = [
    IntentConcept(
        name="first_person",
        exemplars=[
            "I am", "I'm", "I have", "I've", "My", "Mine",
            "Me", "I feel", "I think", "I want", "I need",
            "I did", "I was", "I will", "I would"
        ],
        category="SIGNAL",
        confidence_boost=0.0  # Used for detection, not confidence
    ),
    IntentConcept(
        name="future_anchor",
        exemplars=[
            "Tomorrow", "Next week", "Next month", "Coming up",
            "Soon", "In the future", "Later this", "Upcoming",
            "Next year", "In a few days"
        ],
        category="SIGNAL",
        confidence_boost=0.0
    ),
    IntentConcept(
        name="evaluative_language",
        exemplars=[
            "Important", "Big deal", "Matters a lot", "Significant",
            "Critical", "Essential", "Worried about", "Nervous about",
            "Serious", "Major", "Key moment"
        ],
        category="SIGNAL",
        confidence_boost=0.0
    ),
]


# Combine all intent concepts
ALL_INTENT_CONCEPTS = (
    MEMORY_COMMAND_CONCEPTS +
    PAST_REFERENCE_CONCEPTS +
    PERSONAL_STATE_CONCEPTS +
    GENERAL_KNOWLEDGE_CONCEPTS +
    SIGNAL_CONCEPTS
)


# =============================================================================
# SEMANTIC INTENT ANALYZER
# =============================================================================

@dataclass
class IntentAnalysis:
    """Analysis result for intent classification."""
    primary_intent: str = "GENERAL_KNOWLEDGE"
    confidence: float = 0.7
    ambiguity: bool = False
    ambiguity_flags: List[str] = field(default_factory=list)
    matched_concepts: List[Tuple[str, float]] = field(default_factory=list)
    signals: Dict[str, bool] = field(default_factory=dict)


class SemanticIntentAnalyzer:
    """
    Semantic-based intent analyzer using embeddings.
    Replaces regex-based intent classification with ML understanding.
    """

    def __init__(self):
        """Initialize with pre-computed concept embeddings."""
        self._concepts = ALL_INTENT_CONCEPTS
        self._concept_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of embeddings."""
        if self._initialized:
            return

        print("[SemanticIntent] Initializing intent concept embeddings...")

        for concept in self._concepts:
            embeddings = []
            for exemplar in concept.exemplars:
                emb = embed_text(exemplar)
                embeddings.append(emb)

            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)
            self._concept_embeddings[concept.name] = avg_embedding

        self._initialized = True
        print(f"[SemanticIntent] Initialized {len(self._concept_embeddings)} intent concepts")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _find_matching_concepts(
        self,
        text: str,
        category: Optional[str] = None,
        threshold: float = 0.38
    ) -> List[Tuple[IntentConcept, float]]:
        """Find intent concepts matching the input text."""
        self._ensure_initialized()

        text_embedding = embed_text(text.lower())
        text_embedding = text_embedding / (np.linalg.norm(text_embedding) + 1e-9)

        matches = []
        for concept in self._concepts:
            if category and concept.category != category:
                continue

            concept_emb = self._concept_embeddings[concept.name]
            score = self._cosine_similarity(text_embedding, concept_emb)

            if score >= threshold:
                matches.append((concept, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def analyze(self, text: str, threshold: float = 0.38) -> IntentAnalysis:
        """
        Analyze text to determine user intent.

        Args:
            text: User input text
            threshold: Minimum similarity score

        Returns:
            IntentAnalysis with detected intent and signals
        """
        if not text or not text.strip():
            return IntentAnalysis()

        analysis = IntentAnalysis()
        candidates: Dict[str, float] = {}
        signals: List[str] = []

        # Find all matching concepts
        all_matches = self._find_matching_concepts(text, threshold=threshold)
        analysis.matched_concepts = [(c.name, s) for c, s in all_matches[:10]]

        # Aggregate scores by category
        category_scores: Dict[str, float] = {
            "EXPLICIT_MEMORY_COMMAND": 0.0,
            "PAST_SELF_REFERENCE": 0.0,
            "PERSONAL_STATE": 0.0,
            "GENERAL_KNOWLEDGE": 0.0,
        }

        for concept, score in all_matches:
            if concept.category == "SIGNAL":
                signals.append(concept.name)
                analysis.signals[concept.name] = True
            elif concept.category in category_scores:
                # Weight by both score and confidence boost
                weighted_score = score * (1.0 + concept.confidence_boost)
                category_scores[concept.category] = max(
                    category_scores[concept.category],
                    weighted_score
                )
                if concept.name not in [s for s in signals]:
                    signals.append(concept.name)

        # Determine primary intent (highest scoring category)
        priority_order = [
            "PAST_SELF_REFERENCE",
            "EXPLICIT_MEMORY_COMMAND",
            "PERSONAL_STATE",
            "GENERAL_KNOWLEDGE",
        ]

        best_intent = "GENERAL_KNOWLEDGE"
        best_score = 0.0

        for intent in priority_order:
            if category_scores[intent] > best_score:
                best_score = category_scores[intent]
                best_intent = intent

        # If no strong match, check for signals to guide fallback
        if best_score < threshold:
            if "first_person" in analysis.signals:
                # Personal context suggests PERSONAL_STATE
                if any(s in signals for s in ["negative_emotion", "positive_emotion", "future_concern"]):
                    best_intent = "PERSONAL_STATE"
                    best_score = 0.6
                else:
                    best_intent = "GENERAL_KNOWLEDGE"
                    best_score = 0.5
            else:
                best_intent = "GENERAL_KNOWLEDGE"
                best_score = 0.5

        # Map score to confidence
        confidence_map = {
            "EXPLICIT_MEMORY_COMMAND": 1.0,
            "PAST_SELF_REFERENCE": 0.9,
            "PERSONAL_STATE": 0.8,
            "GENERAL_KNOWLEDGE": 0.7,
        }
        base_confidence = confidence_map.get(best_intent, 0.7)

        # Adjust confidence based on match strength
        if best_score >= 0.6:
            confidence = base_confidence
        elif best_score >= 0.5:
            confidence = base_confidence * 0.9
        else:
            confidence = base_confidence * 0.8

        # Check for ambiguity (multiple categories with similar scores)
        active_categories = [k for k, v in category_scores.items() if v >= threshold * 0.8]
        ambiguity = len(active_categories) > 1 or confidence < 0.75

        analysis.primary_intent = best_intent
        analysis.confidence = round(confidence, 2)
        analysis.ambiguity = ambiguity
        analysis.ambiguity_flags = signals

        return analysis


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_analyzer_instance: Optional[SemanticIntentAnalyzer] = None


def get_semantic_intent_analyzer() -> SemanticIntentAnalyzer:
    """Get or create the singleton semantic intent analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SemanticIntentAnalyzer()
    return _analyzer_instance
