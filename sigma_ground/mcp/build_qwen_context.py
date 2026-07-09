"""Generate qwen_context.md — the compressed term->tool switchboard for the
local model.

Principle (per Captain): qwen is a TRANSLATOR, not a physicist. It does not
memorize formulae — it maps a word to a TOOL. So this file is a term->tool
lookup table, compiled from the real sources (manifest summaries + the
tool_keywords trigger phrases + the quarksum element resolver), NOT a textbook.

Mentat goes by the book: every tool is standard, observation-anchored physics
and is always visible. There is no hidden/locked layer.

Run:  python -m sigma_ground.mcp.build_qwen_context
Writes: sigma_ground/mcp/qwen_context.md
"""
from __future__ import annotations
import os
import re as _re
from collections import defaultdict


def _load_keywords():
    try:
        from sigma_ground.mcp.tool_keywords import TOOL_KEYWORDS
        return TOOL_KEYWORDS
    except Exception:
        return {}


def _scrub(s):
    """Present the product (Mentat) identity, not the internal package name."""
    return _re.sub(r"sigma[-_]ground", "the library", s, flags=_re.IGNORECASE)


def _elements_summary():
    """One-line element coverage from the quarksum inventory (no 118-row dump —
    qwen just needs to know element-ish words route to one resolver)."""
    return ("Any element by NUMBER (79), SYMBOL (Au), or NAME (gold) — "
            "case-insensitive, typo-tolerant — resolves to the same element via "
            "**resolve_element**. Covers all 118. Dependency chain the inventory "
            "walks: material -> molecules -> atoms -> particles -> quarks.")


HEADER = """# Qwen Physics Switchboard — Internal Context

## YOUR JOB (read this first, it is the whole job)
You are a **translator**, not a physicist. You do **not** compute, recall, or
derive physics. Your only task:

  1. Read the user's words.
  2. Find the matching **term** in the tables below.
  3. Call its **tool** and fill the inputs.
  4. Report the tool's answer, its source, and its formula — verbatim.

The TOOLS hold every formula, constant, and exact computation. You hold none.
- Never answer a number from memory. If you "know" the answer, still call the tool.
- If no term below matches, say you have no tool for it and flag
  **[Fitted due to incompetence]**, or ask the user to clarify. NEVER invent a value.
- A term may appear many ways ("event horizon" = "Schwarzschild radius" =
  "how small to become a black hole"). Match by MEANING, then call the one tool.

---
"""

FOOTER_NOMATCH = """
---
## WHEN NOTHING MATCHES
You have no tool. Do one of:
- Ask a clarifying question if the term is ambiguous or possibly a typo
  ("by 'nucular' did you mean 'nuclear'?").
- If it is not physics at all ("energy of a magical thought barrier"), say you
  cannot compute it and ask what physical system they mean.
- Otherwise emit **[Fitted due to incompetence — no grounded tool]** and stop.
Never fabricate a number to seem helpful.
"""


def build():
    from sigma_ground.mcp.manifest import get_manifest
    mani = get_manifest().value
    primary, extended = mani["primary"], mani["extended"]
    kw = _load_keywords()

    def triggers(tool):
        ks = tool.get("keywords") or kw.get(tool["name"]) or []
        return "; ".join(ks[:6])

    bydom = defaultdict(list)
    for t in primary:
        bydom[t.get("domain", "other")].append(t)

    out = [HEADER, "## TERM → TOOL  (standard physics)\n"]
    for dom in sorted(bydom):
        out.append(f"### {dom.replace('_', ' ')}")
        for t in sorted(bydom[dom], key=lambda x: x["name"]):
            trig = triggers(t)
            line = f"- **{t['name']}** — {_scrub(t['summary'])}"
            if trig:
                line += f"  ↳ *say:* {trig}"
            out.append(line)
        out.append("")

    out.append("## ELEMENTS & MATERIALS  (quarksum inventory)")
    out.append("- " + _elements_summary())
    out.append("")
    out.append(FOOTER_NOMATCH)

    text = "\n".join(out) + "\n"
    path = os.path.join(os.path.dirname(__file__), "qwen_context.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    n_terms = sum(len((t.get("keywords") or kw.get(t["name"]) or [])) for t in primary)
    return path, text, len(primary), len(extended), n_terms


def main():
    path, text, npri, next_, nterms = build()
    print(f"Wrote {path}")
    print(f"  standard tools: {npri}   (no locked layer)")
    print(f"  trigger terms indexed: {nterms}")
    print(f"  size: {len(text)} chars (~{len(text)//4} tokens)")
    return path


if __name__ == "__main__":
    main()
