"""Robustness: when the model can't shape an object, Deckard never returns
nothing. A known-parts object scaffolds from its composition prior (flagged); an
unknown one falls back to a flagged generic vessel. Both are identified=False and
audit as not-trustworthy. Offline.
"""
from sigma_ground.deckard import research, compile, audit
from sigma_ground.deckard.research import _scaffold_from_composition


def test_scaffold_builds_known_parts_flagged():
    spec = _scaffold_from_composition("scissors")       # a known PartNet decomposition
    assert spec is not None and not spec.identified
    assert {"blade", "handle"} <= {p.name for p in spec.parts}    # the known parts
    c = compile(spec, resolution=40)
    assert c.validation["passed"]                       # a real, validated placeholder
    assert audit(spec, c)["verdict"] in ("estimated", "suspect")  # never "verified"


def test_scaffold_is_none_for_an_unknown_object():
    assert _scaffold_from_composition("zxqwerty gizmo 99") is None


def test_research_never_returns_nothing_for_an_unknown_object():
    # no catalog hit, no composition, no LLM -> a flagged fallback, never None/fake
    spec = research("zxqwerty gizmo 99", allow_llm=False)
    assert not spec.identified
    assert compile(spec, resolution=40).validation["passed"]
