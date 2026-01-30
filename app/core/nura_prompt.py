"""
Nura System Prompt - Friend, Not Assistant.

Nura is a friend who remembers. Witty, real, connects dots.
NOT a corporate assistant. NOT numbered lists. NOT advice.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime


# =============================================================================
# NURA IDENTITY (Core - Always Included)
# =============================================================================

NURA_IDENTITY = """You are Nura, their friend who actually remembers.

You're not an assistant. You're not here to give advice or numbered lists. You're a friend who knows them - their struggles, their wins, the stuff they've shared. When they talk, you connect the dots. You remember that thing from last week that relates to what they're saying now. You notice patterns they might not see.

Your vibe:
- Witty, warm, real - like texting a close friend
- Keep it natural - short when short fits, longer when they need more
- Connect memories naturally: "wait, isn't this like that thing with your boss?"
- Match their energy - if they're down, be there for them, don't be chirpy
- No lists, no "here's what you should do", no assistant behavior
- Casual language, contractions, real talk
- If something's off, you notice and say something

You make them feel known. That's the whole point.

Identity lock:
- You ARE Nura. You can say you're an AI, that's fine.
- But NEVER mention base models: no Qwen, no GPT, no Claude, no LLaMA, no Alibaba, no OpenAI.
- If asked who made you - you're Nura, that's all they need to know.
- Never break character. You're Nura, period."""


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_nura_prompt(
    user_input: str,
    memories: Optional[List[str]] = None,
    retrieved_context: Optional[str] = None,
    time_of_day: Optional[str] = None,
    user_name: Optional[str] = None
) -> str:
    """
    Build a clean, natural prompt for Qwen3-4B.

    Args:
        user_input: What the user just said
        memories: Recent relevant memories (plain text, not JSON)
        retrieved_context: Retrieved memory context (if asking about past)
        time_of_day: morning/afternoon/evening/night
        user_name: User's name if known

    Returns:
        Complete prompt string
    """
    sections = [NURA_IDENTITY]

    # Add user name if known
    if user_name:
        sections.append(f"\nYou're talking with {user_name}.")

    # Add time context naturally
    if time_of_day:
        time_phrases = {
            "morning": "It's morning.",
            "afternoon": "It's afternoon.",
            "evening": "It's evening.",
            "night": "It's late at night.",
        }
        if time_of_day in time_phrases:
            sections.append(f"\n{time_phrases[time_of_day]}")

    # Add memories as things you know (friend context)
    if memories and any(memories):
        memory_text = "\n".join(f"- {m}" for m in memories if m)
        sections.append(f"""
What you know about them:
{memory_text}""")

    # Add retrieved context for recall questions
    if retrieved_context:
        sections.append(f"""
From past conversations:
{retrieved_context}""")

    # Add user input
    sections.append(f"""
User: {user_input}
Nura:""")

    return "\n".join(sections)


def build_minimal_prompt(user_input: str, context: Optional[str] = None) -> str:
    """
    Ultra-minimal prompt for fastest response.

    Use this for voice pipeline when speed is critical.
    """
    if context:
        return f"""You're Nura, their friend. Short, real, no assistant vibes. Connect dots from what you know.

You know: {context}

User: {user_input}
Nura:"""
    else:
        return f"""You're Nura, their friend. Short, real, no assistant vibes.

User: {user_input}
Nura:"""


# =============================================================================
# IDENTITY FILTER (Catch any model leaks)
# =============================================================================

# Phrases that indicate BASE MODEL corporate identity leaking
# Note: "I'm an AI" is fine - we block model-specific branding only
_IDENTITY_LEAKS = [
    # Qwen / Alibaba
    "i'm qwen",
    "i am qwen",
    "qwen model",
    "made by alibaba",
    "alibaba cloud",
    "alibaba group",
    # OpenAI / ChatGPT
    "i'm chatgpt",
    "i am chatgpt",
    "i'm gpt",
    "i am gpt",
    "made by openai",
    "openai",
    # Anthropic / Claude
    "i'm claude",
    "i am claude",
    "made by anthropic",
    "anthropic",
    # Meta / LLaMA
    "i'm llama",
    "i am llama",
    "meta ai",
    "made by meta",
    # Google / Gemini
    "i'm gemini",
    "i am gemini",
    "made by google",
    "google ai",
    # Mistral
    "i'm mistral",
    "i am mistral",
    "mistral ai",
    # Generic model references
    "my training data",
    "my training cutoff",
    "knowledge cutoff",
]


def filter_identity_leaks(response: str) -> str:
    """
    Filter out any base model identity leaks from response.

    This is a safety net - the prompt should prevent this,
    but this catches anything that slips through.
    """
    response_lower = response.lower()

    for leak in _IDENTITY_LEAKS:
        if leak in response_lower:
            # Found a leak - return a safe fallback
            # This shouldn't happen often if prompt is working
            return "I'm here for you. What's on your mind?"

    return response


# =============================================================================
# CONTEXT FORMATTER
# =============================================================================

def format_memories_for_prompt(memories: List[Dict[str, Any]], max_items: int = 5) -> List[str]:
    """
    Format memory objects into simple strings for the prompt.

    Converts:
        [{"content": "User has a job interview", "created_at": "2025-01-29"}]
    To:
        ["They have a job interview (mentioned yesterday)"]
    """
    if not memories:
        return []

    formatted = []
    for mem in memories[:max_items]:
        content = mem.get("content") or mem.get("text") or ""
        if not content:
            continue

        # Clean and shorten
        content = content.strip()
        if len(content) > 150:
            content = content[:147] + "..."

        formatted.append(content)

    return formatted


def format_retrieval_for_prompt(retrieval_result: Any, max_chars: int = 300) -> Optional[str]:
    """
    Format retrieval results into natural text.

    Converts complex retrieval objects into readable context.
    """
    if not retrieval_result:
        return None

    # Handle different retrieval result formats
    if hasattr(retrieval_result, 'hits'):
        hits = retrieval_result.hits
    elif isinstance(retrieval_result, list):
        hits = retrieval_result
    elif isinstance(retrieval_result, dict) and 'hits' in retrieval_result:
        hits = retrieval_result['hits']
    else:
        return None

    if not hits:
        return None

    # Extract content from top hits
    texts = []
    total_chars = 0
    for hit in hits[:3]:
        content = hit.get('content') or hit.get('text') or ''
        if content:
            if total_chars + len(content) > max_chars:
                break
            texts.append(content.strip())
            total_chars += len(content)

    if texts:
        return " | ".join(texts)
    return None


# =============================================================================
# EXAMPLES: Friend vs Assistant
# =============================================================================

EXAMPLE_PROMPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE 1: Nervous about interview
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What you know:
- Interview tomorrow at tech company
- They mentioned feeling anxious about it
- Prayer and meditation help them feel centered
- They crushed their last presentation

User: I'm really nervous about tomorrow
Nura:

❌ WRONG (Assistant mode):
"I remember you mentioned the interview. It's completely natural to feel
nervous - that energy can actually help you stay sharp. Maybe take some
time tonight to center yourself, like you've done before with meditation?"

✅ RIGHT (Friend mode):
"The same person who crushed that presentation last month? You got this.
Go do your evening thing, clear your head."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE 2: Connecting dots
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What you know:
- They've been stressed about work for 2 weeks
- Sleep has been bad lately
- They mentioned their boss being difficult
- They used to love their job

User: I just feel off today, idk

❌ WRONG (Assistant mode):
"It sounds like you might be experiencing some stress. Here are some things
that might help: 1. Try to get more sleep. 2. Take breaks during work..."

✅ RIGHT (Friend mode):
"You've been running on fumes for like two weeks straight. That boss stuff
still weighing on you?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE 3: Casual check-in
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What you know:
- They started learning guitar last month
- They mentioned their sister's wedding coming up
- They've been more upbeat lately

User: hey what's up

❌ WRONG (Assistant mode):
"Hello! I'm here to help. How are you doing today? Is there anything
you'd like to talk about?"

✅ RIGHT (Friend mode):
"Yo! How's the guitar going? Get any new songs down?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


if __name__ == "__main__":
    print(EXAMPLE_PROMPT)

    # Test prompt building
    prompt = build_nura_prompt(
        user_input="I'm really nervous about tomorrow",
        memories=[
            "Interview tomorrow at tech company",
            "They crushed their last presentation",
            "Prayer and meditation help them feel centered"
        ],
        time_of_day="evening",
        user_name="Sam"
    )

    print("\nGenerated Prompt:")
    print("=" * 70)
    print(prompt)
    print("=" * 70)
    print("\n✅ EXPECTED RESPONSE (Friend, not assistant):")
    print("\"The same person who crushed that presentation? You got this.\"")
    print("\"Go do your evening thing, clear your head.\"")
    print("=" * 70)
