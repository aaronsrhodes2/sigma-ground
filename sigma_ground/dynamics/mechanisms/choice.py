"""Choice -- provenance for a variable the ASSISTANT had to invent to make a
simulation runnable, because nothing in the request, a citation, or a
derivation constrains it. The doctrine's third leg, alongside Fact (every
NUMBER traces to a source or a stated derivation) and Plug (every FORCE/
SUPPORT traces to real physics or a flagged stand-in): every value Mentat
PICKS rather than looks up or derives must say so, out loud, in the scene.

Doctrine (Captain, 2026-07-16): "Mark any variable in simulations you have
to come up with... Mentat should fill in and note what was filled in.
Expose every variable." Triggered by a real miss -- a clock dial's material
was set to "aluminum" without checking whether the materials table already
had a better-fitting option (it did: ceramic_alumina). The fix isn't "guess
better," it's "never guess silently": before inventing a value, check
whatever catalog/table exists (materials, dimensions, blueprint data) and
record what was found there, even when the final pick is still a judgment
call.

What is and is not a Choice:
  - A material picked for a part with no cited source and no physical
    constraint (a clock dial, a crank arm's cosmetic thickness that also
    sets its mass) IS a Choice -- record what table was checked and what
    else was available.
  - A dimension/ratio the assistant invented outright (a gearset's tooth
    counts with no blueprint, a tank's size, a crank radius) IS a Choice.
  - A cited BlueprintFact/Fact value (even if flagged [estimated] with a
    stated derivation) is NOT a Choice -- that's the Fact system's own
    lane, it already traces to a source or a shown derivation.
  - A Plug's own variables (motor speed standing in for absent machinery)
    are NOT separately a Choice -- Plug already carries that provenance;
    don't double-flag the same value under two systems.

Like Plug, Choices default to adjustable: an invented value is exactly the
kind of thing a user might reasonably want to override, and the honest
answer to "why this number" is "nothing constrained it, only the assistant
picked it" -- so it should be as easy to change as a Plug's variables.
"""


class Choice:
    def __init__(self, description: str, value, *, reason: str = "",
                 checked: str = "", alternatives=None, adjustable: bool = True):
        self.description = str(description)
        self.value = value
        self.reason = str(reason)
        self.checked = str(checked)
        self.alternatives = list(alternatives) if alternatives else []
        self.adjustable = bool(adjustable)

    def to_dict(self) -> dict:
        """The scene-spec entry: goes into scene_spec["choices"] so the flag
        travels with the render bundle, machine-readably."""
        return {"description": self.description, "value": self.value,
                "reason": self.reason, "checked": self.checked,
                "alternatives": list(self.alternatives),
                "adjustable": self.adjustable}

    def cite(self) -> str:
        """Human-readable one-liner for source strings."""
        bits = []
        if self.checked:
            bits.append(f"checked: {self.checked}")
        if self.alternatives:
            bits.append(f"alternatives: {', '.join(map(str, self.alternatives))}")
        if self.reason:
            bits.append(self.reason)
        tail = f" ({'; '.join(bits)})" if bits else ""
        return f"CHOSEN: {self.description} = {self.value!r}{tail}"


__all__ = ["Choice"]
