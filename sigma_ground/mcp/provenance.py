"""Structured tool-call results with provenance citations.

Every MCP tool returns a `ToolResult`. The LLM front-end is instructed
to faithfully report the `source` field, so users see e.g.
"6.67430e-11 m^3/(kg s^2), CODATA 2018, rel. uncertainty 2.2e-5"
rather than the bare number.

The structure is intentionally JSON-serializable for transport over the
MCP wire protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """A physics calculation result with full provenance.

    Attributes
    ----------
    value : float | int | str | list | dict
        The computed value. Numeric for most physics queries; can be a
        string (symbolic result, formula), a list (vector), or a dict
        (multi-quantity result).
    units : str
        SI by default unless the user explicitly requested otherwise.
        Empty string for dimensionless quantities.
    source : str
        Citation. Examples: "CODATA 2018", "PDG 2022 best fit",
        "DESI 2024 Union3 (arXiv:2411.08639)", "sigma_ground.field.gr_basics",
        "scipy.constants", "computed from inputs".
    uncertainty : float | None
        1-sigma uncertainty in the same units as `value`, when known.
        `None` means either exact (definitional constants like c) or
        unknown / not propagated.
    formula : str | None
        Symbolic form of the calculation if applicable (e.g. "2 G M / c^2"
        for Schwarzschild radius). Helps the LLM explain derivations.
    provenance_tag : str | None
        Internal sigma-ground tag: VERIFIED, DERIVED, EMPIRICAL-INPUT,
        SPECULATIVE-PENDING, REJECTED. `None` for external-library values.
    notes : str
        Free-text caveats. Examples: "valid for non-relativistic limit",
        "SSBM-specific function; only meaningful at non-zero sigma".
    inputs : dict
        Echo of the input parameters, for the LLM to confirm interpretation.
    """

    value: Any
    units: str = ""
    source: str = ""
    uncertainty: float | None = None
    formula: str | None = None
    provenance_tag: str | None = None
    notes: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for MCP transport."""
        return asdict(self)

    def format_for_llm(self) -> str:
        """Pretty single-string representation for LLM consumption.

        The LLM can include this verbatim in its prose answer, OR parse
        it and reformat -- both work. Format prioritizes citations so
        the LLM is steered toward attribution.
        """
        parts = []
        if isinstance(self.value, (int, float)):
            parts.append(f"value: {self.value:.6g}")
        else:
            parts.append(f"value: {self.value}")
        if self.units:
            parts.append(f"units: {self.units}")
        if self.uncertainty is not None:
            parts.append(f"uncertainty (1-sigma): {self.uncertainty:.3g}")
        if self.source:
            parts.append(f"source: {self.source}")
        if self.provenance_tag:
            parts.append(f"provenance: [{self.provenance_tag}]")
        if self.formula:
            parts.append(f"formula: {self.formula}")
        if self.inputs:
            parts.append(f"inputs: {self.inputs}")
        if self.notes:
            parts.append(f"notes: {self.notes}")
        return " | ".join(parts)
