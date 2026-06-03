"""Groundedness gate — the trust linchpin.

Every answer the switchboard produces is either:
  * GROUNDED  — it came from a deterministic classifier/router (turns==0) or
                from an actual tool call (the library computed it), so it is
                trustworthy and provenance-tagged; or
  * UNGROUNDED — the LLM answered from its own "knowledge" with no tool
                backing it, so it is a GUESS and MUST be flagged.

The gate enforces the product rule: never serve a wrong answer WITHOUT
saying so. An ungrounded answer is wrapped in a visible
[Fitted due to incompetence ...] banner so the user always knows whether a
number is library-verified or a model guess. This is exactly what Wolfram
does when it punts ("Wolfram|Alpha doesn't know how to interpret your
input") — made deterministic.

Conservative by design: when in doubt, flag. A false flag (warning on a
correct answer) is cheap; a silent wrong answer is the failure mode we are
eliminating.
"""

from __future__ import annotations

from dataclasses import dataclass

FLAG = "[Fitted due to incompetence"


@dataclass
class Groundedness:
    grounded: bool
    reason: str
    source: str          # "classifier" | "tool_call" | "ungrounded"


def assess(record: dict) -> Groundedness:
    """Decide whether an answer record is grounded.

    Signals (in the record produced by run_sigma_ground._run_one_question):
      - turns == 0            -> a deterministic classifier/router handled it
      - tool_calls non-empty  -> the LLM invoked a library tool
      - extracted_value       -> a value was actually produced
    """
    turns = record.get("turns", 0) or 0
    tool_calls = record.get("tool_calls") or []
    value = record.get("extracted_value")

    # 0. No value produced at all (error / empty) -> not grounded.
    if value is None and not tool_calls:
        return Groundedness(False, "no value produced (error/empty)", "ungrounded")

    # 0b. The LLM failed to converge (looped to max turns) or errored. Even if
    # it called tools, a non-terminating run means the extracted value came
    # from a messy transcript, not a clean tool result -> NOT trustworthy.
    answer = (record.get("answer_text") or "").lower()
    if "exceeded max turns" in answer or "<error" in answer or "max_turns" in answer:
        return Groundedness(False, "LLM failed to converge (max turns / error)",
                             "ungrounded")

    # 1. Deterministic path: a classifier/router short-circuited (no LLM).
    if turns == 0:
        # find which one, for the provenance note
        hit = next((k for k in record
                     if k.endswith("_hit") or k.endswith("_router_hit")), None)
        return Groundedness(True,
                             f"deterministic dispatch ({record.get(hit, hit) if hit else 'classifier'})",
                             "classifier")

    # 2. LLM path, but it called a real tool and got a value -> grounded.
    if tool_calls and value is not None:
        return Groundedness(True,
                             f"LLM tool-grounded ({len(tool_calls)} tool call(s))",
                             "tool_call")

    # 3. Otherwise the LLM answered from its head -> a guess.
    if not tool_calls:
        return Groundedness(False, "LLM answered with no tool call", "ungrounded")
    return Groundedness(False, "LLM produced no library-backed value", "ungrounded")


def apply_gate(record: dict) -> dict:
    """Stamp the record with `grounded` and, if ungrounded, prepend the
    [Fitted due to incompetence] banner to answer_text. Returns the record
    (mutated in place)."""
    g = assess(record)
    record["grounded"] = g.grounded
    record["groundedness_reason"] = g.reason
    if not g.grounded:
        banner = (f"{FLAG} — no grounded library tool answered this; the "
                   f"following is the model's unverified best guess, NOT "
                   f"computed by sigma-ground. Treat with caution.]\n\n")
        existing = record.get("answer_text", "") or ""
        if FLAG not in existing:
            record["answer_text"] = banner + existing
    return record
