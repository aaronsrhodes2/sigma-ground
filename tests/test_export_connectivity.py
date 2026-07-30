"""Connectivity ratchet: everything produced must be consumed — or disclosed.

Second standing audit (companion to test_degeneracy_probe.py, same 2026-07-30
mandate: "if one, then all"). The physics-coverage audit found 13 of 21 baked
material keys shipped to the browser and read by nothing — computed, encoded,
transmitted, dead. Reading code finds this class of gap (it is a missing edge
in the call graph), but nothing ENFORCED it, so the dead set could only grow.

Rules, both directions:
  - a key `_bake_material` emits must be referenced somewhere outside the
    exporter (viewer.js or non-test sigma_ground Python), OR carry an entry in
    DEAD_PAYLOAD / PROVENANCE_KEYS saying why it ships anyway;
  - a key listed as dead must actually BE dead — wiring a consumer forces
    deleting its entry, so the list only shrinks through real work.

"Referenced" is a string-level proxy for "consumed" (a producer elsewhere
also counts as a reference). That is deliberately lenient: this test exists
to catch keys with NO edge at all, the provably-dead case, without false
alarms on indirect consumption.
"""
import glob
import os
import re

import pytest

from sigma_ground.radiance.scene_export import _bake_material

_ROOT = os.path.join(os.path.dirname(__file__), "..", "sigma_ground")

# Representative labels spanning every bake tier (metal, semiconductor,
# chromophore glaze, dielectric, insulator, organic, liquid, unknown).
_LABELS = ("copper", "glass", "iron", "water", "ceramic", "silicon",
           "concrete", "cadmium_sulfide", "cobalt_glaze", "wood_oak")

# Provenance/self-documentation strings: shipped so a HUMAN reading the JSON
# knows where each number came from (the same honesty rule as Choice/Fact).
# They are payload BY DESIGN, not dead weight — exempt, with this reason.
PROVENANCE_KEYS = {"mechanism", "note", "mechanics_origin", "matter_origin",
                   "roughness_origin", "density_method"}

# Produced today, consumed by nothing YET — each entry names the queued
# consumer that will delete it. The list may only shrink.
DEAD_PAYLOAD = {
    # _emergent_matter's solver signature (landed 2026-07-30): the consumers
    # are the queued bond-breaking calculation and the label_grid->thermal
    # solver wiring (task #31). tensile/fracture feed the former; the moduli
    # and specific heat feed both.
    "bulk_modulus_gpa": "bond-breaking calc (queued)",
    "shear_modulus_gpa": "bond-breaking calc (queued)",
    "tensile_strength_mpa": "bond-breaking calc (queued)",
    "fracture_toughness_mpa_m05": "bond-breaking calc (queued)",
    "specific_heat_j_kg_k": "per-cell thermal solver (task #31)",
    # _emergent_mechanics' oldest stranded key: E ships since the mechanics
    # bake first landed and nothing has ever read it (the 2026-07-30 physics
    # audit's original headline finding).
    "youngs_modulus_gpa": "no consumer ever; candidate for a material HUD",
}


def _emitted_keys():
    keys = set()
    for label in _LABELS:
        try:
            keys |= set(_bake_material(label).keys())
        except Exception:
            continue
    keys.discard("choice")            # popped by construct_to_scene, never ships
    return keys


def _reference_corpus():
    viewer = open(os.path.join(_ROOT, "radiance", "web", "viewer.js"),
                  encoding="utf-8").read()
    py = []
    for path in glob.glob(os.path.join(_ROOT, "**", "*.py"), recursive=True):
        norm = path.replace(os.sep, "/")
        if norm.endswith("radiance/scene_export.py"):
            continue                   # the producer doesn't count as a consumer
        if "test" in os.path.basename(path):
            continue
        py.append(open(path, encoding="utf-8", errors="replace").read())
    return viewer, "\n".join(py)


def test_every_baked_key_is_consumed_or_disclosed():
    viewer, py = _reference_corpus()
    dead_now = set()
    for key in _emitted_keys():
        if key in PROVENANCE_KEYS:
            continue
        referenced = (key in viewer
                      or re.search(r"[\"']" + re.escape(key) + r"[\"']", py))
        if not referenced:
            dead_now.add(key)
    undisclosed = dead_now - set(DEAD_PAYLOAD)
    assert not undisclosed, (
        f"baked material key(s) with NO consumer and no DEAD_PAYLOAD "
        f"disclosure: {sorted(undisclosed)} — wire a consumer or add an "
        f"entry naming the queued one")
    resurrected = set(DEAD_PAYLOAD) - dead_now
    assert not resurrected, (
        f"key(s) listed dead but now referenced: {sorted(resurrected)} — "
        f"delete their DEAD_PAYLOAD entries so the ratchet tightens")


def test_physics_callback_hooks_have_non_test_producers():
    """A hook nobody can construct a callback for is a dead socket. friction_fn
    sat producer-less from its introduction until 2026-07-30 — every sim ran
    frictionless (mu=0, flagged NOT_PHYSICS in PhysicsScene's own comment)
    because the seam existed and nothing filled it. materia.engine.
    make_material_friction is that producer now; this pins its existence.

    restitution_fn is deliberately NOT required to have one: unlike friction's
    nothing-default, restitution has a real material-derived default path
    (parcels carry material.restitution; collision.py's e = min(restitutions))
    so the hook is a refinement, not the only source.
    """
    _, py = _reference_corpus()
    assert "def make_material_friction" in py, (
        "friction_fn has no non-test producer — make_material_friction "
        "(materia/engine.py) is gone; without it every scene silently "
        "reverts to frictionless")
