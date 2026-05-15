"""Generate PROVENANCE.md from sigma_ground.field.constants tag comments.

The constants file already carries inline tags on each definition:
  [VERIFIED]            -- measured (CODATA, PDG, IAU, peer-reviewed paper)
  [DERIVED]             -- computed from upstream constants in the file
  [EMPIRICAL-INPUT]     -- a free parameter of SSBM (the one new number it adds)
  [SPECULATIVE-PENDING] -- placeholder, awaiting reference/derivation

This script parses constants.py and emits a structured PROVENANCE.md:
  - Every numeric constant by category
  - Citation / formula / dependencies
  - The "minimal input set" -- the irreducible measured quantities the theory
    takes from observation. Everything else is derived from those.

This is the document that answers Aaron's perfect-library question:
"what fraction of the library is measured, what fraction is derived, what
fraction is still a guess?"
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

_CONSTANTS_PATH = (
    Path(__file__).parent.parent / "sigma_ground" / "field" / "constants.py"
)
_OUTPUT_PATH = (
    Path(__file__).parent.parent / "sigma_ground" / "field" / "PROVENANCE.md"
)


# Recognized provenance tags in inline comments
_TAGS = {
    "VERIFIED":            "measured",
    "DERIVED":             "derived",
    "EMPIRICAL-INPUT":     "free-input",
    "SPECULATIVE-PENDING": "speculative",
}


# Names that LOOK like Python builtins / type hints, not physics constants
_BUILTIN_NAMES = {"math", "True", "False", "None"}


def _parse_constants_file(path: Path) -> list[dict]:
    """Walk constants.py line-by-line and extract definitions with tags."""
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Accumulate preceding-comment context so the citation/note above a def
    # can be attached to that definition.
    preceding_comment: list[str] = []

    # Match  NAME = expression  # optional comment
    # Captures: name, expression, comment_text
    pat = re.compile(
        r"^([A-Z][A-Z0-9_]*(?:_MEV|_KG|_S|_M|_J|_W|_K|_C|_HZ|_T)?)\s*=\s*(.+?)(?:\s*#\s*(.+))?$"
    )

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()

        if stripped.startswith("#"):
            preceding_comment.append(stripped[1:].strip())
            continue

        if not stripped:
            preceding_comment = []
            continue

        m = pat.match(line)
        if not m:
            # Reset preceding-comment buffer if this isn't a constant def
            preceding_comment = []
            continue

        name = m.group(1)
        expr = m.group(2).strip()
        inline_comment = (m.group(3) or "").strip()

        # Identify provenance tag in the inline comment (and possibly preceding)
        full_comment = inline_comment + " | " + " | ".join(preceding_comment)
        tag = None
        for tag_key in _TAGS:
            if f"[{tag_key}]" in full_comment:
                tag = _TAGS[tag_key]
                break

        # If no tag found, try to infer:
        # - bare numeric literal → measured (probably a definitional constant)
        # - math.* expression with NO upstream constants → mathematical (π, φ, etc.)
        # - references other UPPER_SNAKE names → derived
        # - otherwise → unclassified
        if tag is None:
            uses_other_consts = re.search(r"\b[A-Z][A-Z0-9_]+\b", expr) is not None
            uses_math_module  = "math." in expr
            uses_bare_numeric = re.match(r"^[\d.eE+\-*/()\s]+$", expr) is not None
            if uses_bare_numeric:
                tag = "measured"   # bare-number literal
            elif not uses_other_consts and uses_math_module:
                tag = "mathematical"   # e.g. PHI = (1+sqrt(5))/2
            elif uses_other_consts:
                tag = "derived"
            else:
                tag = "unclassified"

        # Identify upstream dependencies (names of other constants on RHS)
        deps = re.findall(r"\b([A-Z][A-Z0-9_]+)\b", expr)
        deps = [d for d in deps if d != name and d not in _BUILTIN_NAMES]

        # Try to find a numeric value at the literal
        rows.append({
            "name":        name,
            "expr":        expr,
            "tag":         tag,
            "comment":     inline_comment,
            "preceding":   " ".join(preceding_comment[-3:]),  # last 3 comment lines
            "deps":        sorted(set(deps)),
            "line_no":     line_no,
        })
        preceding_comment = []

    return rows


def _categorize(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["tag"]].append(r)
    return out


def _format_row(r: dict) -> str:
    name    = r["name"]
    expr    = r["expr"]
    note    = r["comment"]
    deps    = r["deps"]
    if len(expr) > 60:
        expr = expr[:57] + "..."
    if deps:
        deps_str = f" deps: {', '.join(deps)}"
    else:
        deps_str = ""
    note_str = f" -- {note}" if note else ""
    return f"- **`{name}`** = `{expr}`{note_str}{deps_str}"


def write_provenance_md(rows: list[dict], output_path: Path) -> None:
    by_cat = _categorize(rows)
    total = len(rows)
    n_measured     = len(by_cat.get("measured", []))
    n_derived      = len(by_cat.get("derived", []))
    n_free_input   = len(by_cat.get("free-input", []))
    n_speculative  = len(by_cat.get("speculative", []))
    n_mathematical = len(by_cat.get("mathematical", []))
    n_unclassified = len(by_cat.get("unclassified", []))

    lines: list[str] = []
    lines.append("# sigma-ground physics constants -- provenance map")
    lines.append("")
    lines.append(
        "Auto-generated by `scripts/generate_provenance.py`. "
        "Run it after editing constants.py to refresh."
    )
    lines.append("")
    lines.append("## What's measured, what's derived")
    lines.append("")
    lines.append("Aaron's perfect-library ideal: every value in this library should")
    lines.append("trace back to (a) a measurement of reality, (b) a mathematical")
    lines.append("derivation from such measurements, or (c) a definitional standard")
    lines.append("(SI unit definitions). No guesses.")
    lines.append("")
    lines.append(f"**Total quantities: {total}**")
    lines.append("")
    lines.append(f"| Category | Count | Fraction |")
    lines.append(f"|---|---:|---:|")
    lines.append(f"| Measured (`[VERIFIED]`) | {n_measured} | {n_measured/total*100:.1f}% |")
    lines.append(f"| Derived (`[DERIVED]`) | {n_derived} | {n_derived/total*100:.1f}% |")
    lines.append(f"| Free input (`[EMPIRICAL-INPUT]`) | {n_free_input} | {n_free_input/total*100:.1f}% |")
    lines.append(f"| Mathematical (π, φ, e -- pure math) | {n_mathematical} | {n_mathematical/total*100:.1f}% |")
    lines.append(f"| Speculative pending (`[SPECULATIVE-PENDING]`) | {n_speculative} | {n_speculative/total*100:.1f}% |")
    lines.append(f"| Unclassified | {n_unclassified} | {n_unclassified/total*100:.1f}% |")
    lines.append("")
    lines.append("Toward perfect: minimize `free-input`, eliminate `speculative-pending`,")
    lines.append("classify everything in `unclassified`.")
    lines.append("")

    # --- Free inputs (the irreducible SSBM parameters) ---------------------
    lines.append("---")
    lines.append("")
    lines.append("## Free inputs of the theory")
    lines.append("")
    lines.append("These are quantities SSBM takes as INPUT from observation. The whole")
    lines.append("point of the framework is to minimize this list.")
    lines.append("")
    for r in by_cat.get("free-input", []):
        lines.append(_format_row(r))
    if not by_cat.get("free-input"):
        lines.append("- (none currently tagged)")
    lines.append("")

    # --- Speculative ------------------------------------------------------
    if n_speculative > 0:
        lines.append("---")
        lines.append("")
        lines.append("## Speculative / placeholder")
        lines.append("")
        lines.append("Values currently set to `None` or with placeholder coefficients,")
        lines.append("awaiting derivation or measurement extraction. Each is a TODO.")
        lines.append("")
        for r in by_cat.get("speculative", []):
            lines.append(_format_row(r))
        lines.append("")

    # --- Measured -------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## Measured constants")
    lines.append("")
    lines.append("Values from authoritative measurements (CODATA, PDG, IAU,")
    lines.append("peer-reviewed papers cited in constants.py). These are the")
    lines.append("irreducible empirical inputs of the library -- everything in")
    lines.append("the `derived` section computes downstream of these.")
    lines.append("")
    for r in by_cat.get("measured", []):
        lines.append(_format_row(r))
    lines.append("")

    # --- Derived --------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## Derived constants")
    lines.append("")
    lines.append("Computed from upstream measured constants. Each row shows the")
    lines.append("formula and the upstream dependencies. A perfect library has all")
    lines.append("non-input values here.")
    lines.append("")
    for r in by_cat.get("derived", []):
        lines.append(_format_row(r))
    lines.append("")

    # --- Mathematical ---------------------------------------------------
    if n_mathematical > 0:
        lines.append("---")
        lines.append("")
        lines.append("## Mathematical constants")
        lines.append("")
        lines.append("Pure-math values not derived from any physics measurement --")
        lines.append("e.g. π, φ, e. These are definitional and identical in every")
        lines.append("universe; they don't constrain or get constrained by the theory.")
        lines.append("")
        for r in by_cat.get("mathematical", []):
            lines.append(_format_row(r))
        lines.append("")

    # --- Unclassified ---------------------------------------------------
    if n_unclassified > 0:
        lines.append("---")
        lines.append("")
        lines.append("## Unclassified")
        lines.append("")
        lines.append("Values without an explicit `[VERIFIED]` / `[DERIVED]` / etc. tag.")
        lines.append("Each is a candidate for explicit classification -- add the right")
        lines.append("tag to constants.py and re-run this script.")
        lines.append("")
        for r in by_cat.get("unclassified", []):
            lines.append(_format_row(r))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"Parsing {_CONSTANTS_PATH}")
    rows = _parse_constants_file(_CONSTANTS_PATH)
    print(f"Found {len(rows)} numeric constants")

    by_cat = _categorize(rows)
    for cat, items in sorted(by_cat.items()):
        print(f"  {cat:>15s}: {len(items)}")

    write_provenance_md(rows, _OUTPUT_PATH)
    print(f"\nWritten {_OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
