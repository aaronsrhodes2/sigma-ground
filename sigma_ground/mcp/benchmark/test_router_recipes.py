"""Router v2 recipes: each one must teach something true.

Static checks need no server: every template carries every slot its plan uses (and no
unknown slot), plans have one to three calls and only refer backwards, a rendered slot
agrees with its label. The last test runs a few recipes against the real MCP server,
re-executes each gold plan independently, and checks every tool name and parameter against
the live catalogue.
"""
import asyncio
import json
import os
import random
import re
import sys

import pytest

from sigma_ground.mcp import param_names as PN
from sigma_ground.mcp.benchmark import router_dataset as D
from sigma_ground.mcp.benchmark import router_plan as RP
from sigma_ground.mcp.benchmark import router_recipes as RR
from sigma_ground.mcp.benchmark import router_recipes_v3 as RR3  # noqa: F401  (registers the v3 recipes)
from sigma_ground.mcp.benchmark import router_recipes_v31 as RR31  # noqa: F401  (registers the v3.1 recipes)
from sigma_ground.mcp.benchmark import router_recipes_sky as RRS  # noqa: F401  (registers the sky recipes)


def test_recipe_ids_are_unique_and_there_are_enough():
    ids = [r.id for r in RR.RECIPES]
    assert len(ids) == len(set(ids)) and len(ids) >= 45
    fams = {r.family for r in RR.RECIPES}
    assert {"chain", "percent", "convert", "atomic", "astro"} <= fams


@pytest.mark.parametrize("recipe", RR.RECIPES, ids=lambda r: r.id)
def test_templates_and_plans_are_consistent(recipe):
    assert recipe.templates, "a recipe needs question templates"
    assert 1 <= len(recipe.steps) <= RP.MAX_STEPS
    for t in recipe.templates:
        holes = set(re.findall(r"\{(\w+)\}", t))
        assert holes <= set(recipe.slots), (t, holes - set(recipe.slots))
        assert recipe.used_slots() <= holes, f"template lacks a slot the plan uses: {t}"
    for slot in recipe.slots:
        assert slot in recipe.used_slots() or slot in recipe.distractors, \
            f"slot {slot!r} is sampled but never used by the plan"
    assert set(recipe.distractors) <= set(recipe.slots)
    for t in recipe.templates:
        assert set(recipe.distractors) <= set(re.findall(r"\{(\w+)\}", t)), \
            f"a distractor is missing from a template of {recipe.id}: {t}"
    for i, (tool, spec) in enumerate(recipe.steps, start=1):
        for kind, x in spec.values():
            assert kind in ("s", "r", "rf", "k", "l")
            if kind == "r":
                assert x < i, f"{recipe.id}: step {i} refers to step {x}"
            if kind == "rf":
                assert x[0] < i, f"{recipe.id}: step {i} refers to step {x[0]}"
    for step_no in recipe.bounds:
        assert 1 <= step_no <= len(recipe.steps)
    if len(recipe.steps) > 1:
        assert any(kind in ("r", "rf") for _, spec in recipe.steps[1:] for kind, _ in spec.values()), \
            "a multi-step recipe must carry a value forward"


@pytest.mark.parametrize("recipe", RR.RECIPES, ids=lambda r: r.id)
def test_a_sampled_question_and_plan_agree(recipe):
    """The rendered text of every quantity is the label times its unit, and the plan
    carries exactly the labels (no float noise, "$n" for carried values)."""
    for seed in range(25):
        rng = random.Random(seed)
        args = {n: RR._sample_slot(n, spec, rng) for n, spec in recipe.slots.items()}
        for t in recipe.templates:
            q = RR.render(recipe, args, t)
            assert "{" not in q and "  " not in q
            for a in args.values():
                assert a.text in q
        steps = RR.build_plan(recipe, args)
        plan = RP.make_plan(steps)
        assert RP.canonical(plan) is not None
        RP.check_refs(RP.canonical(plan)["steps"])
        text = RP.dumps_plan(plan)
        assert not re.search(r"[eE][-+]?0\d", text), text
        for n, a in args.items():
            spec = recipe.slots[n]
            if isinstance(spec, RR.Q):
                num, _, disp = a.text.partition(" ")
                factor = dict(spec.unit.renderings).get(disp, 1.0)
                if num and disp != "°C":
                    assert a.value == pytest.approx(float(num) * factor, rel=1e-11)


def test_solar_mass_drills_pin_the_library_constant():
    r = next(x for x in RR.RECIPES if x.id == "solar_isco_radius")
    seen = set()
    for seed in range(200):
        a = RR._sample_slot("M", r.slots["M"], random.Random(seed))
        num = float(a.text.split(" ")[0])
        assert a.text.endswith("solar masses")
        assert a.value == pytest.approx(num * RR.M_SUN_KG, rel=1e-11)
        seen.add(num)
    assert len(seen) > 100                      # non-round masses, not a handful of values


def test_the_tnt_unit_slot_uses_the_names_the_tools_accept():
    r = next(x for x in RR.RECIPES if x.id == "tnt_to_mass")
    assert {v for _, v in r.slots["u"].options} == {"kt", "MT", "ton"}


def test_shared_pairs_are_declared_and_exist_as_recipes():
    pairs = {tuple(t for t, _ in r.steps) for r in RR.RECIPES if len(r.steps) > 1}
    assert RR.PAIRS_SHARED_WITH_BENCHMARK <= pairs


@pytest.mark.skipif(os.environ.get("SKIP_SERVER_TESTS") == "1", reason="needs the MCP server")
def test_recipes_against_the_real_server(tmp_path):
    """A handful of recipes (one of each kind) generate, re-execute, and match the live catalogue."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    wanted = {"orbit_kinetic_energy", "mass_to_tnt", "hydrogen_photon_ev", "percent_of_power",
              "convert_kmh_ms", "ionization_energy", "solar_hawking_temperature", "tnt_to_mass"}

    async def run():
        async with stdio_client(D.server_params(0)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                catalogue = {t.name: set((t.inputSchema or {}).get("properties", {}))
                             for t in (await session.list_tools()).tools}
                examples, report = await RR.generate(session, 4, 0, D.BenchmarkGuard.from_repo(), wanted)
                redone = []
                for e in examples:
                    c = RP.canonical(e["plan"])
                    values = []
                    for s in c["steps"]:
                        assert s["tool"] in catalogue, s["tool"]
                        # the recipes are written with the parameter names of the functions; the schema shows the canonical ones (param_names.py)
                        # and still accepts the old ones, so a recipe name must be one of either
                        accepted = catalogue[s["tool"]] | PN.accepted_old_names(s["tool"], catalogue[s["tool"]])
                        assert set(s["args"]) <= accepted, (s["tool"], set(s["args"]) - accepted)
                        res = await D.execute(session, s["tool"], RP.resolve_args(s["args"], values),
                                              e["question"], timeout_s=300)
                        assert res is not None, (s, e["question"])
                        values.append(res["value"])
                    redone.append((e, values))
                return examples, report, redone

    examples, report, redone = asyncio.run(run())
    assert {e["meta"]["recipe"] for e in examples} == wanted, report
    for e, values in redone:
        recorded = [v["value"] for v in e["meta"]["verified"]]
        assert values == pytest.approx(recorded, rel=1e-9)
        assert D._norm(e["question"]) and e["group"].startswith("recipe:")
    assert any(len(e["plan"].get("steps", [])) >= 2 for e in examples)


def test_the_guard_counts_the_hard_set_as_test_data_unless_told_otherwise():
    """router_hardval.json sits in the benchmark folder, so training data is kept away from it;
    the hard set's own benchmark check must leave it out or every question matches itself."""
    import json
    from pathlib import Path
    spec = json.loads((Path(D.__file__).parent / "router_hardval.json").read_text(encoding="utf-8"))
    q = spec[0]["question"]
    assert D.BenchmarkGuard.from_repo().check(q)                                    # a hard question is test data
    assert D.BenchmarkGuard.from_repo(exclude=("router_hardval.json",)).check(q) is None   # but not benchmark proper
