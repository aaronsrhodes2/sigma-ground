"""PDG drift gate — the adoption contract for particle masses.

inventory/data/particle_masses.json is the vendored PDG snapshot (written by
tools/distill_pdg.py, which never touches code). This gate holds every Python
root to that snapshot, so adopting a new PDG edition is a DELIBERATE, loud,
single-commit event:

    1. Dataset Minder notices a new 'pdg' PyPI release (new edition)
    2. tools/distill_pdg.py re-distills particle_masses.json
    3. THIS FILE FAILS on every drifted value
    4. a human updates inventory/core/constants.py (+ field/constants.py)
       and the failures name each number that moved

History: the pre-2026 hand-transcribed values ("PDG 2024") drifted for two
editions unnoticed (down 4.67 vs 4.70, top 172500 vs 172603.6 MeV, ...)
because nothing compared code to a citable snapshot. This gate is the fix —
and the reason there is exactly ONE Python root (CONSTANTS) instead of the
four independent copies the reconciliation audit found.
"""
import json
import pathlib

import pytest

from sigma_ground.inventory.core.constants import CONSTANTS

_JSON = (pathlib.Path(__file__).resolve().parents[1] / "sigma_ground"
         / "inventory" / "data" / "particle_masses.json")


def _pdg():
    return json.loads(_JSON.read_text(encoding="utf-8"))["particles"]


# ── quark MS-bar masses: CONSTANTS must equal the snapshot exactly ──────────

@pytest.mark.parametrize("flavor,attr", [
    ("up", "m_up_mev"), ("down", "m_down_mev"), ("strange", "m_strange_mev"),
    ("charm", "m_charm_mev"), ("bottom", "m_bottom_mev"), ("top", "m_top_mev"),
])
def test_quark_mev_matches_snapshot(flavor, attr):
    assert getattr(CONSTANTS, attr) == pytest.approx(
        _pdg()[flavor]["mass_mev"], rel=1e-12), (
        f"{attr} drifted from particle_masses.json ({flavor}) — new PDG "
        f"edition? update constants.py deliberately")


# ── lepton/boson kg values: converted via the same MeV→kg the code uses ────

@pytest.mark.parametrize("name,attr", [
    ("muon", "m_muon"), ("tau", "m_tau"),
    ("W_boson", "m_W"), ("Z_boson", "m_Z"), ("Higgs", "m_higgs"),
])
def test_lepton_boson_kg_matches_snapshot(name, attr):
    expected_kg = _pdg()[name]["mass_mev"] * CONSTANTS.MeV_to_kg
    assert getattr(CONSTANTS, attr) == pytest.approx(expected_kg, rel=1e-9)


# ── nucleons: AME2020 kg root must agree with PDG to measurement precision ──

@pytest.mark.parametrize("name,attr", [("proton", "m_p"), ("neutron", "m_n"),
                                       ("electron", "m_e")])
def test_nucleon_electron_agree_with_pdg(name, attr):
    # primary roots stay AME2020/CODATA (finer provenance); PDG must agree
    # within combined quoted precision — a real disagreement here means an
    # upstream evaluation shifted and BOTH roots need review
    expected_kg = _pdg()[name]["mass_mev"] * CONSTANTS.MeV_to_kg
    assert getattr(CONSTANTS, attr) == pytest.approx(expected_kg, rel=5e-9)


# ── single-root discipline: every other copy references CONSTANTS ──────────

def test_quark_factories_are_the_constants_root():
    from sigma_ground.inventory.models.quark import Quark
    assert Quark.up().bare_mass_mev == CONSTANTS.m_up_mev
    assert Quark.down().bare_mass_mev == CONSTANTS.m_down_mev
    assert Quark.strange().bare_mass_mev == CONSTANTS.m_strange_mev
    assert Quark.charm().bare_mass_mev == CONSTANTS.m_charm_mev
    assert Quark.bottom().bare_mass_mev == CONSTANTS.m_bottom_mev
    assert Quark.top().bare_mass_mev == CONSTANTS.m_top_mev
    # top has no dressed state — constituent is tied to bare by construction
    assert Quark.top().constituent_mass_mev == CONSTANTS.m_top_mev


def test_field_constants_match_inventory_root():
    from sigma_ground.field import constants as fc
    assert fc.M_UP_MEV == CONSTANTS.m_up_mev
    assert fc.M_DOWN_MEV == CONSTANTS.m_down_mev


def test_qcd_binding_is_derived_not_transcribed():
    """The former frozen literals (929.282088 / 928.065421) are now computed
    from the constants root at import — verify against the defining formula
    AND confirm they moved off the stale pre-2026 numbers."""
    from sigma_ground.inventory.models.particle import Proton, Neutron
    m_p_mev = CONSTANTS.m_p / CONSTANTS.MeV_to_kg
    m_n_mev = CONSTANTS.m_n / CONSTANTS.MeV_to_kg
    p, n = Proton.create(), Neutron.create()
    assert p.qcd_binding_energy_mev == pytest.approx(
        m_p_mev - (2 * CONSTANTS.m_up_mev + CONSTANTS.m_down_mev), rel=1e-12)
    assert n.qcd_binding_energy_mev == pytest.approx(
        m_n_mev - (CONSTANTS.m_up_mev + 2 * CONSTANTS.m_down_mev), rel=1e-12)
    assert p.qcd_binding_energy_mev != pytest.approx(929.282088, abs=1e-4)
    assert n.qcd_binding_energy_mev != pytest.approx(928.065421, abs=1e-4)


def test_snapshot_edition_is_recorded():
    meta = json.loads(_JSON.read_text(encoding="utf-8"))
    assert meta["_edition"] == "2026", (
        "particle_masses.json edition changed — re-review every gate above")
