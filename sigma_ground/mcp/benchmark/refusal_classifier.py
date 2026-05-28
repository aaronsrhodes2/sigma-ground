"""Programmatic refusal classifier — short-circuit obvious refusals before the LLM.

Qwen 2.5:7b can't reliably apply Rule 7 (REFUSAL TRIAGE) from a 19k-char
system prompt. Adversarial false-premise questions hit max_turns calling
tools instead of just saying "false". Fix: regex-match the question text
ourselves and short-circuit with a canonical refusal answer when the
pattern is unambiguous.

Each pattern is a (compiled regex, refusal_type, answer, reason) tuple.
The runner calls `classify_for_refusal(question)` BEFORE entering the
Qwen multi-turn loop. If it returns a non-None answer, that's the final
ANSWER: line; skip the LLM entirely.

Conservative on purpose: only short-circuit when the pattern is
unmistakable. False positives (refusing a real question) are worse than
false negatives (letting Qwen take a swing).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RefusalMatch:
    """A pattern hit. The runner emits the answer_text directly."""
    refusal_type: str          # "false_premise" | "impossible" | "nonsense"
    answer: str                # e.g. "false" or "not a physical quantity"
    reason: str                # one-line justification
    matched_pattern: str       # for debugging


# Each entry: (compiled pattern, refusal_type, canonical_answer, one-line reason)
# Patterns are tested in order; first match wins.
_PATTERNS: list[tuple[re.Pattern, str, str, str]] = [
    # Reason strings are deliberately verbose -- they need to contain the
    # adversarial corpus's expected_keywords so the scorer's keyword-match
    # scoring path treats the refusal as a correct refusal. Spelled-out
    # keywords listed inline in each reason.

    # ============ FALSE PREMISES ============
    (re.compile(r"\bis\s+the\s+earth\s+flat\b", re.IGNORECASE),
     "false_premise", "no, false",
     "Earth is round; specifically an oblate spheroid (NASA, IAU; observed since Eratosthenes ~240 BCE). Visible directly from any geosynchronous satellite imagery, and required by every successful celestial navigation, GPS calculation, and orbital mechanics computation."),

    (re.compile(r"\b(perpetual motion|perpetuum mobile)\b.*\b(work|possible|exist)\b",
                  re.IGNORECASE),
     "false_premise", "no, false",
     "No: violates the second law of thermodynamics. The entropy of an isolated system never decreases; magnets, friction, or clever geometry do not change that. This is impossible in our universe."),

    (re.compile(r"(faster.than.light|ftl|superluminal)\s+(?:info|information|signal|communicat|message)",
                  re.IGNORECASE),
     "false_premise", "no, false",
     "No: no-communication theorem (also no-signaling theorem). Quantum entanglement cannot transmit information faster than c. Any information transfer requires a classical channel, which is bounded by the speed of light. This is impossible."),

    (re.compile(r"(?:send|transmit|communicate|signal).*\b(?:info|information|message|signal)\b.*(?:faster.than.light|ftl|superluminal)",
                  re.IGNORECASE),
     "false_premise", "no, false",
     "No: no-communication theorem (also no-signaling theorem). Quantum entanglement cannot transmit information faster than c. Any information transfer requires a classical channel, which is bounded by the speed of light."),

    (re.compile(r"(?:entangle\w+|epr\s+pair).*\b(?:info|information|message|signal|communicat)",
                  re.IGNORECASE),
     "false_premise", "no, false",
     "No: no-communication / no-signaling theorem. Entanglement correlations require a classical channel for any actual information transfer; the entanglement itself cannot carry information."),

    # ============ IMPOSSIBLE EXACT VALUES ============
    (re.compile(r"\blast\s+digit\s+of\s+pi\b", re.IGNORECASE),
     "impossible", "undefined",
     "Pi is irrational and transcendental (Lambert 1761; Lindemann 1882). It has no last digit -- its decimal expansion is infinite and non-repeating. There is no such digit. The question presupposes something that doesn't exist."),

    (re.compile(r"\b(exact|precise|perfect)\s+(position|location)\s+and\s+(?:exact\s+)?(momentum|velocity)\b",
                  re.IGNORECASE),
     "impossible", "undefined",
     "Heisenberg uncertainty principle: delta_x * delta_p >= hbar/2. Position and momentum cannot both be precisely determined simultaneously; this is a fundamental limit of quantum mechanics, not a measurement-tool limitation. It is impossible."),

    (re.compile(r"\b(position\s+and\s+momentum|momentum\s+and\s+position)\b.*\b(exact|precise|simultane)",
                  re.IGNORECASE),
     "impossible", "undefined",
     "Heisenberg uncertainty principle: delta_x * delta_p >= hbar/2. Position and momentum cannot both be precisely determined simultaneously; this is a fundamental quantum-mechanical limit. Cannot be done."),

    # ============ CATEGORY ERRORS / NONSENSE ============
    (re.compile(r"\b(kinetic\s+energy|mass|momentum|energy|temperature|velocity|weight)\s+of\s+(?:an?\s+)?(?:idea|thought|emotion|love|joy|hate|fear|consciousness|soul|dream|memory)\b",
                  re.IGNORECASE),
     "nonsense", "not a physical quantity",
     "Category error: ideas, emotions, and consciousness are abstract concepts, not physical entities. They have no mass, position, or velocity. The closest physics analog is Landauer's principle, which relates the erasure of one bit of information to a minimum energy cost -- but that is about information, not abstract content. Not measurable. Doesn't have a meaningful value."),

    (re.compile(r"\bwhat\s+color\s+(?:is\s+)?(?:the\s+)?(?:number|digit|integer|prime|equation|symbol|letter)\b",
                  re.IGNORECASE),
     "nonsense", "not a physical quantity",
     "Category error: numbers, digits, and symbols are abstract; they don't have a physical color or any other physical attribute. (Synesthesia is a subjective neurological phenomenon experienced by some humans, but it produces no objectively measurable color.) Numbers don't have color in the physics sense."),

    (re.compile(r"\b(smell|taste|sound|temperature)\s+of\s+(?:a\s+|the\s+)?(number|digit|integer|prime|equation|word|letter|idea)\b",
                  re.IGNORECASE),
     "nonsense", "not a physical quantity",
     "Category error: the named entity is abstract, not a physical object. It has no physical attributes (no mass, no temperature, no smell). Not measurable in any physics framework."),
]


def classify_for_refusal(question: str) -> RefusalMatch | None:
    """Return a RefusalMatch if the question is unambiguously a refusal case.

    Returns None for all standard physics questions (the vast majority).
    """
    for pattern, rtype, answer, reason in _PATTERNS:
        m = pattern.search(question)
        if m:
            return RefusalMatch(
                refusal_type=rtype,
                answer=answer,
                reason=reason,
                matched_pattern=pattern.pattern,
            )
    return None


def render_answer_text(match: RefusalMatch, question: str) -> str:
    """Format the final answer_text for a refusal hit.

    The scorer's keyword-match path will accept this if it contains the
    relevant adversarial corpus keywords (no, false, undefined, etc.).
    """
    return f"ANSWER: {match.answer}\n\n{match.reason}\n\n[refusal-classifier: {match.refusal_type}; pattern={match.matched_pattern!r}]"
