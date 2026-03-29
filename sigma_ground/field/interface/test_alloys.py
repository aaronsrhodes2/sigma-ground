"""Tests for alloy property prediction from atomic-fraction compositions.

Validates:
  - Pure element recovery (alloy of one element = that element's McMillan T_c)
  - Known compound comparison (Nb3Sn stoichiometry vs known T_c)
  - Non-superconducting alloys (CuZn, CuSn predict T_c = 0)
  - Model consistency (all models agree for pure elements)
  - σ-field dependence (T_c unchanged at σ=0, decreases for σ>0)
  - Composition validation (bad fractions, missing elements raise ValueError)
  - Composition sweep (monotonic for simple systems)
  - Nordheim resistivity (pure element = 0, alloy > 0)
  - Database integrity (all ALLOYS_PREDICTABLE entries produce valid results)
"""

import math
import pytest

from sigma_ground.field.interface.alloys import (
    alloy_properties,
    predict_alloy_Tc,
    alloy_Tc_all_models,
    sigma_alloy_Tc,
    alloy_Nordheim_resistivity,
    composition_sweep,
    predict_all,
    list_alloys,
    ALLOYS,
    ALLOYS_PREDICTABLE,
)
from sigma_ground.field.interface.superconductivity import (
    SUPERCONDUCTORS,
    mcmillan_Tc,
)
from sigma_ground.field.constants import SIGMA_HERE


# ── Pure Element Recovery ─────────────────────────────────────────

class TestPureElementRecovery:
    """A pure-element 'alloy' must reproduce that element's McMillan T_c."""

    ELEMENTS_WITH_MCMILLAN = [
        key for key, data in SUPERCONDUCTORS.items()
        if (data.get('lambda_ep') is not None
            and data.get('mu_star') is not None
            and data.get('theta_D_K') is not None)
    ]

    @pytest.mark.parametrize("element", ELEMENTS_WITH_MCMILLAN)
    def test_pure_element_linear(self, element):
        """Pure element via linear model = direct McMillan."""
        data = SUPERCONDUCTORS[element]
        expected = mcmillan_Tc(
            data['theta_D_K'], data['lambda_ep'], data['mu_star'])
        predicted = predict_alloy_Tc({element: 1.0}, model='linear')
        assert abs(predicted - round(expected, 3)) < 0.01, (
            f"{element}: expected {expected:.3f}, got {predicted:.3f}")

    @pytest.mark.parametrize("element", ELEMENTS_WITH_MCMILLAN)
    def test_pure_element_dos(self, element):
        """Pure element via DOS model = direct McMillan (same as linear)."""
        data = SUPERCONDUCTORS[element]
        expected = mcmillan_Tc(
            data['theta_D_K'], data['lambda_ep'], data['mu_star'])
        predicted = predict_alloy_Tc({element: 1.0}, model='dos_weighted')
        assert abs(predicted - round(expected, 3)) < 0.01

    @pytest.mark.parametrize("element", ELEMENTS_WITH_MCMILLAN)
    def test_models_agree_pure(self, element):
        """All models must agree for a pure element."""
        result = alloy_Tc_all_models({element: 1.0})
        assert result['summary']['spread_K'] < 0.01


# ── Known Compound Comparison ────────────────────────────────────

class TestKnownCompounds:
    """Compare predictions against known measured T_c values."""

    def test_Nb3Sn_stoichiometry(self):
        """Nb3Sn: measured T_c = 18.3K. Solid-solution prediction will
        underestimate because A15 crystal structure enhances λ.
        But should be in the right order of magnitude."""
        Tc = predict_alloy_Tc({'niobium': 0.75, 'tin': 0.25})
        # Solid-solution underestimates: expect 10-18 K range
        assert Tc > 5, f"Nb3Sn stoich too low: {Tc}"
        assert Tc < 25, f"Nb3Sn stoich too high: {Tc}"

    def test_NbTi_wire(self):
        """NbTi wire: measured T_c = 9.2-9.8K.
        McMillan overestimates for strong coupling, so expect 8-15K."""
        Tc = predict_alloy_Tc({'niobium': 0.53, 'titanium': 0.47})
        assert Tc > 5, f"NbTi too low: {Tc}"
        assert Tc < 20, f"NbTi too high: {Tc}"


# ── Non-Superconducting Alloys ───────────────────────────────────

class TestNonSuperconductors:
    """Alloys of weak-coupling metals should predict T_c ≈ 0."""

    def test_CuZn_brass(self):
        Tc = predict_alloy_Tc({'copper': 0.70, 'zinc': 0.30})
        assert Tc == 0.0, f"Brass should be non-SC, got T_c = {Tc}"

    def test_CuAg(self):
        Tc = predict_alloy_Tc({'copper': 0.50, 'silver': 0.50})
        assert Tc == 0.0, f"CuAg should be non-SC, got T_c = {Tc}"

    def test_CuSn_bronze(self):
        """Bronze: 88% Cu, 12% Sn. Cu dominates → T_c near zero."""
        Tc = predict_alloy_Tc({'copper': 0.88, 'tin': 0.12})
        assert Tc < 0.5, f"Bronze T_c too high: {Tc}"

    def test_AuAg(self):
        Tc = predict_alloy_Tc({'gold': 0.50, 'silver': 0.50})
        assert Tc == 0.0


# ── Composition Validation ───────────────────────────────────────

class TestValidation:
    """Composition validation: bad inputs raise ValueError."""

    def test_empty_composition(self):
        with pytest.raises(ValueError, match="Empty"):
            predict_alloy_Tc({})

    def test_fractions_not_summing(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            predict_alloy_Tc({'niobium': 0.5, 'titanium': 0.3})

    def test_unknown_element(self):
        with pytest.raises(ValueError, match="Unknown"):
            predict_alloy_Tc({'unobtanium': 0.5, 'niobium': 0.5})

    def test_negative_fraction(self):
        with pytest.raises(ValueError, match="Negative"):
            predict_alloy_Tc({'niobium': 1.5, 'titanium': -0.5})

    def test_missing_mcmillan_data(self):
        """Elements without λ_ep should fail validation."""
        with pytest.raises(ValueError, match="McMillan"):
            predict_alloy_Tc({'lithium': 0.5, 'niobium': 0.5})


# ── Model Properties ─────────────────────────────────────────────

class TestModelProperties:

    def test_all_models_return_dict(self):
        result = alloy_Tc_all_models({'niobium': 0.53, 'titanium': 0.47})
        assert 'models' in result
        assert 'summary' in result
        assert 'linear' in result['models']
        assert 'dos_weighted' in result['models']
        assert result['summary']['T_c_min_K'] <= result['summary']['T_c_max_K']

    def test_properties_contain_all_fields(self):
        props = alloy_properties({'niobium': 0.53, 'titanium': 0.47})
        for field in ['theta_D_K', 'lambda_ep', 'mu_star', 'n_e_m3',
                      'v_F_m_s', 'T_c_predicted_K', 'model', 'warnings']:
            assert field in props, f"Missing field: {field}"

    def test_intermetallic_warning(self):
        """A3B stoichiometry should trigger intermetallic warning."""
        result = alloy_Tc_all_models({'niobium': 0.75, 'tin': 0.25})
        warnings = result['warnings']
        assert any('A₃B' in w or 'intermetallic' in w.lower()
                   for w in warnings), f"Expected A3B warning, got: {warnings}"

    def test_ferromagnet_warning(self):
        """Alloys with >10% ferromagnetic element should warn."""
        props = alloy_properties({'niobium': 0.50, 'iron_ambient': 0.50})
        assert any('ferromagnet' in w.lower() for w in props['warnings'])


# ── σ-Field Dependence ───────────────────────────────────────────

class TestSigmaField:

    def test_sigma_zero_unchanged(self):
        """T_c at σ = σ_here (Earth) equals standard prediction."""
        comp = {'niobium': 0.53, 'titanium': 0.47}
        Tc_std = predict_alloy_Tc(comp)
        Tc_sigma = sigma_alloy_Tc(comp, SIGMA_HERE)
        assert abs(Tc_sigma - Tc_std) < 0.01

    def test_sigma_positive_decreases(self):
        """T_c decreases monotonically with increasing σ."""
        comp = {'niobium': 0.53, 'titanium': 0.47}
        Tc_0 = sigma_alloy_Tc(comp, 0.0)
        Tc_mid = sigma_alloy_Tc(comp, 0.1)
        Tc_high = sigma_alloy_Tc(comp, 1.0)
        assert Tc_mid < Tc_0
        assert Tc_high < Tc_mid


# ── Composition Sweep ────────────────────────────────────────────

class TestCompositionSweep:

    def test_sweep_endpoints(self):
        """Sweep endpoints should match pure element predictions."""
        sweep = composition_sweep('niobium', 'titanium', steps=11)
        # First point: pure Nb
        assert sweep[0]['x_B'] == 0.0
        pure_Nb = predict_alloy_Tc({'niobium': 1.0})
        assert abs(sweep[0]['T_c_linear_K'] - pure_Nb) < 0.01
        # Last point: pure Ti
        assert sweep[-1]['x_B'] == 1.0
        pure_Ti = predict_alloy_Tc({'titanium': 1.0})
        assert abs(sweep[-1]['T_c_linear_K'] - pure_Ti) < 0.01

    def test_sweep_correct_length(self):
        sweep = composition_sweep('aluminum', 'zinc', steps=21)
        assert len(sweep) == 21


# ── Nordheim Resistivity ─────────────────────────────────────────

class TestNordheim:

    def test_pure_element_zero(self):
        result = alloy_Nordheim_resistivity({'niobium': 1.0})
        assert result['rho_residual_uOhm_cm'] == 0.0

    def test_binary_alloy_positive(self):
        result = alloy_Nordheim_resistivity({'niobium': 0.5, 'titanium': 0.5})
        assert result['rho_residual_uOhm_cm'] > 0

    def test_50_50_maximum(self):
        """Nordheim: maximum disorder at 50:50."""
        r50 = alloy_Nordheim_resistivity({'niobium': 0.5, 'titanium': 0.5})
        r80 = alloy_Nordheim_resistivity({'niobium': 0.8, 'titanium': 0.2})
        assert r50['rho_residual_uOhm_cm'] > r80['rho_residual_uOhm_cm']


# ── Database Integrity ───────────────────────────────────────────

class TestDatabaseIntegrity:
    """Every alloy in ALLOYS_PREDICTABLE must produce a valid result."""

    @pytest.mark.parametrize("name", list(ALLOYS_PREDICTABLE.keys()))
    def test_alloy_produces_result(self, name):
        comp = ALLOYS_PREDICTABLE[name]
        result = alloy_Tc_all_models(comp)
        assert result['summary']['T_c_mean_K'] >= 0
        assert result['summary']['confidence'] in ('high', 'moderate', 'low')

    def test_predict_all_runs(self):
        results = predict_all()
        assert len(results) == len(ALLOYS_PREDICTABLE)

    def test_list_alloys(self):
        alloys = list_alloys()
        assert isinstance(alloys, dict)
        assert len(alloys) > 20
