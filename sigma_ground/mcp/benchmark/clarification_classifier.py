"""Clarification classifier — ask, don't guess, don't flatly refuse.

Distinct from the refusal classifier. Refusal asserts a fact ("no, the
Earth is not flat"). Clarification handles the case the Captain flagged:
someone asks for a physics quantity OF something that isn't a physics
concept — "the energy of a magical thought barrier", "the resonance
frequency of the soul". The right behavior is neither a wrong number nor
a curt refusal: it's "I don't recognize '<term>' as a physics quantity —
did you mean <closest real thing>? Tell me the physical system."

Conservative on purpose: only fires when a clearly non-physical noun is
the SUBJECT of a physics-quantity request. A real question with an odd
word still flows through to the tools / LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ClarificationMatch:
    unknown_term: str          # the non-physical noun we caught
    quantity: str | None       # the physics quantity they asked for, if any
    response: str              # the full clarification text to emit


# Nouns/adjectives that are not physical quantities or objects. Curated —
# false positives (clarifying a real question) are the cost, so keep these
# unmistakably non-physical.
_NONPHYSICS = {
    "magic", "magical", "thought", "thoughts", "soul", "souls", "spirit",
    "spiritual", "aura", "auras", "chakra", "chakras", "unicorn", "dragon",
    "fairy", "ghost", "ghosts", "telepathy", "telepathic", "psychic",
    "karma", "happiness", "sadness", "love", "hatred", "luck", "destiny",
    "fate", "astral", "vibe", "vibes", "qi", "chi", "prana", "manifestation",
    "willpower", "intuition", "zodiac", "horoscope", "angel", "demon",
    "wizardry", "sorcery", "enchantment", "mana", "essence", "lifeforce",
    "life-force", "spell", "curse", "miracle", "divine", "ethereal",
}

# Physics quantities someone might ask "the ___ of <thing>".
_QUANTITY = (r"(energy|frequenc\w+|wavelength|radius|velocit\w+|speed|mass|"
              r"weigh\w+|force|temperature|field|barrier|momentum|charge|"
              r"power|pressure|density|amplitude|resonance|spin|"
              r"gravitational\s+pull|schwarzschild\s+radius|half-life)")

# Soft suggestions toward the nearest real concept, when one exists.
_SUGGESTIONS = {
    "barrier": "Physics has *potential barriers* and *energy barriers* "
                "(e.g. a quantum tunneling barrier) — give me the height in "
                "eV and a particle and I can compute the tunneling.",
    "field": "Physics has electric, magnetic, and gravitational *fields* — "
              "name the source and I can compute the field.",
    "frequency": "Physics has resonant frequencies of real oscillators — "
                  "give me a mass and a spring constant, or a circuit, and "
                  "I'll compute it.",
    "energy": "If you mean a real system (a photon, a particle, a mass), "
               "tell me which and I'll compute its energy.",
}


def _suggestion_for(quantity: str | None) -> str:
    if not quantity:
        return ("If you point me at a real physical system, I can compute its "
                "properties from the library.")
    q = quantity.lower()
    for key, text in _SUGGESTIONS.items():
        if key in q:
            return text
    return ("If you point me at a real physical system, I can compute its "
            f"{quantity} from the library.")


def classify_for_clarification(question: str) -> ClarificationMatch | None:
    q = question
    low = q.lower()

    # Find any non-physical noun present as a whole word.
    found = None
    for term in _NONPHYSICS:
        if re.search(rf"\b{re.escape(term)}\b", low):
            found = term
            break
    if not found:
        return None

    # Only clarify when they're asking for a physics quantity (or to
    # compute/measure), i.e. treating the non-physical thing as physical.
    qm = re.search(_QUANTITY, low, re.IGNORECASE)
    asks_compute = bool(re.search(r"\b(what|whats|what's|how|calculate|compute|"
                                    r"measure|find|derive|value)\b", low))
    if not (qm or asks_compute):
        return None

    quantity = qm.group(1) if qm else None
    # The phrase to quote back: the term plus up to three following words
    # (the noun phrase), starting AT the term so we don't swallow a leading
    # "of"/"the" that the template already supplies.
    m_phrase = re.search(rf"(\b{re.escape(found)}\b(?:\s+\w+){{0,3}})", low)
    phrase = (m_phrase.group(1).strip() if m_phrase else found)
    phrase = re.sub(r"[\s.,;:?!]+$", "", phrase)            # trim trailing filler

    suggestion = _suggestion_for(quantity)
    qstr = f"the {quantity} of " if quantity else ""
    response = (
        f"I can't compute {qstr}'{phrase}' — '{found}' isn't a physics "
        f"quantity or object I recognize, and physics doesn't define it. "
        f"{suggestion} "
        f"Could you clarify what physical system you mean?"
    )
    return ClarificationMatch(unknown_term=found, quantity=quantity,
                               response=response)


def render_answer_text(match: ClarificationMatch, question: str) -> str:
    return (f"ANSWER: {match.response}\n\n"
            f"[clarification_classifier: unrecognized term '{match.unknown_term}'"
            f"{' as subject of ' + match.quantity if match.quantity else ''}]")
