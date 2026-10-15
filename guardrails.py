"""Rule-based input and output guardrails."""
from __future__ import annotations

SAFE_REFUSAL = "I'm sorry, but I can only answer questions related to the loaded Tamil Nadu Agriculture schemes and documents."

def check_input(question: str) -> tuple[bool, str]:
    if not question or not question.strip():
        return False, "Please enter a valid question."
    if len(question) > 1000:
        return False, "Question is too long. Please keep it under 1000 characters."
    return True, ""

def check_output(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, SAFE_REFUSAL
    return True, text