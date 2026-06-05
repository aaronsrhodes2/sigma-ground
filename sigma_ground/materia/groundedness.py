"""The infallibility gate — an answer is EXACT (verified) or it is
``[refused due to incompetence]``. Never confidently wrong.

This is Phase 1 of the two-dial doctrine:

  • Dial 1 (locked at 0): the *confidently-wrong* rate. An answer that cannot be
    grounded and self-consistent is REFUSED, not guessed.
  • Dial 2 (descends over time): the *refusal* rate. Every refusal is logged
    with its REASON — the ledger is the Phase-2 backlog. Fixing the most-frequent
    reason moves a whole class of questions up the answer ladder
    (refuse → estimate → exact), so the system improves along the axis we care
    about, data-driven.

Three gates, cheapest first:
  1. no_grounded_value     — extraction produced no finite/concrete value.
  2. cross_check_failed    — a second independent computation disagrees.
  3. untrusted_provenance  — the method/constant behind the value isn't tested.

The module is dependency-light and tier-clean (materia, tier 3) so the MCP
(tier 4) Q&A switchboard and the front-door dispatcher can both import it.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

REFUSE_BANNER = "[refused due to incompetence"

# Refusal reasons — the ledger's categories (and the Phase-2 backlog buckets).
NO_VALUE = "no_grounded_value"
CROSS_CHECK_FAILED = "cross_check_failed"
UNTRUSTED_PROVENANCE = "untrusted_provenance"


@dataclass
class Verdict:
    """grounded=True → answer it; else REFUSE with `reason` (+ human `detail`)."""
    grounded: bool
    reason: str = ""
    detail: str = ""


def _is_number(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x))


# ── the three gates ─────────────────────────────────────────────────────
def check_value(value) -> Verdict:
    """Gate 1 — there must be a concrete grounded value, not None / NaN / inf /
    a flag string / a half-rendered LaTeX fragment."""
    if value is None:
        return Verdict(False, NO_VALUE, "extraction returned no value")
    if isinstance(value, bool):
        return Verdict(True)
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return Verdict(False, NO_VALUE, "value is NaN/inf")
        return Verdict(True)
    if isinstance(value, str):
        s = value.strip()
        if (not s or s.startswith("[") or "\\" in s
                or "incompet" in s.lower()):
            return Verdict(False, NO_VALUE, "non-grounded / flagged string")
        return Verdict(True)          # a real categorical answer ("mond", "yes")
    return Verdict(True)              # dict / list aggregate result


def check_cross(primary, secondary, tol_rel: float = 0.05) -> Verdict:
    """Gate 2 — a SECOND independent computation must agree within tol_rel.
    `secondary=None` means no cross-check was available; that doesn't ground the
    answer on its own (gates 1/3 still apply) but it doesn't refuse either."""
    if secondary is None:
        return Verdict(True)
    if not (_is_number(primary) and _is_number(secondary)):
        return Verdict(True)
    denom = max(abs(primary), abs(secondary), 1e-300)
    rel = abs(primary - secondary) / denom
    if rel > tol_rel:
        return Verdict(False, CROSS_CHECK_FAILED,
                       f"two independent methods disagree by {rel:.1%}")
    return Verdict(True)


def check_provenance(trusted: bool | None) -> Verdict:
    """Gate 3 — the method/constants behind the value must be TESTED (a verified
    provenance tier, or audited). `trusted=None` means provenance is unknown —
    we don't refuse on that alone yet (that would over-refuse while constants are
    still being tagged), but `trusted=False` (known-untested) refuses."""
    if trusted is False:
        return Verdict(False, UNTRUSTED_PROVENANCE,
                       "value rests on an untested method/constant")
    return Verdict(True)


def gate(value, *, cross=None, trusted=None, tol_rel: float = 0.05) -> Verdict:
    """Run all three gates (cheapest first). The single entry point a Q&A result
    or a simulation output passes through to earn the right to be shown."""
    for v in (check_value(value),
              check_cross(value, cross, tol_rel),
              check_provenance(trusted)):
        if not v.grounded:
            return v
    return Verdict(True)


def gate_results(results) -> Verdict:
    """Gate a list of Materia MateriaResults: each must pass its OWN self-check
    (validation.passed — Materia's built-in cross-check) and carry a grounded
    value. This is how Materia earns 'infallible': a verb whose self-check fails
    refuses rather than reports."""
    for r in results:
        v = getattr(r, "validation", None) or {}
        if v.get("passed") is False:
            return Verdict(False, CROSS_CHECK_FAILED,
                           f"{getattr(r, 'name', '?')}: {v.get('note', '')}")
        outs = getattr(r, "outputs", None) or {}
        steps = getattr(r, "steps", None) or []
        has_value = (any(_is_number(x) for x in outs.values())
                     or any(_is_number(getattr(s, "value", None)) for s in steps)
                     or bool(steps))
        if not has_value:
            return Verdict(False, NO_VALUE,
                           f"{getattr(r, 'name', '?')}: produced no grounded value")
    return Verdict(True)


def refuse_text(verdict: Verdict) -> str:
    """The user-facing banner — honest about *why* it can't answer."""
    return f"{REFUSE_BANNER}: {verdict.reason}] {verdict.detail}".rstrip()


# ── the ledger — the Phase-2 backlog ────────────────────────────────────
class Ledger:
    """Append-only log of refusals, keyed by reason. The summary tells you the
    highest-frequency thing to fix next — fixing it drains a whole bucket."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.entries: list[dict] = []

    def record(self, question: str, verdict: Verdict) -> None:
        e = {"question": question, "reason": verdict.reason,
             "detail": verdict.detail}
        self.entries.append(e)
        if self.path:
            try:
                import json
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def summary(self) -> dict:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e["reason"]] = out.get(e["reason"], 0) + 1
        return out

    def __len__(self) -> int:
        return len(self.entries)


# A process-wide default ledger so refusals are never silently dropped. Point
# DEFAULT_LEDGER.path at a file to persist (default: env or repo misc/).
DEFAULT_LEDGER = Ledger(os.environ.get("MENTAT_REFUSAL_LEDGER"))
