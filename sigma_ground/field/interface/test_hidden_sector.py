"""
Tests for sigma_ground.field.interface.hidden_sector — Phase XII.d.3.
"""

import pytest

from sigma_ground.field.constants import (
    AXION_MASS_ULDM_MIN_EV,
    AXION_MASS_ULDM_MAX_EV,
    ALP_PHOTON_COUPLING_MAX_INVGEV,
    XI_KOBAKHIDZE,
)
from sigma_ground.field.interface.hidden_sector import (
    axion_mass_regime,
    alp_coupling_allowed,
    dark_axion_portal_available,
    kobakhidze_gauge_group_ratio,
)


# ── Axion mass regime ─────────────────────────────────────────────────

def test_sub_ultralight():
    assert axion_mass_regime(1e-24) == 'sub-ultralight'


def test_ultralight_dm():
    assert axion_mass_regime(1e-15) == 'ultralight-DM'


def test_qcd_axion_window():
    assert axion_mass_regime(1e-5) == 'QCD-axion-window'


def test_heavy_axion():
    assert axion_mass_regime(1.0) == 'heavy'


def test_axion_mass_rejects_nonpositive():
    with pytest.raises(ValueError):
        axion_mass_regime(0.0)
    with pytest.raises(ValueError):
        axion_mass_regime(-1e-10)


# ── ALP-photon coupling ───────────────────────────────────────────────

def test_alp_coupling_below_cast_bound():
    assert alp_coupling_allowed(1e-12) is True


def test_alp_coupling_above_cast_bound():
    assert alp_coupling_allowed(1e-9) is False


def test_alp_coupling_rejects_negative():
    with pytest.raises(ValueError):
        alp_coupling_allowed(-1e-10)


# ── Dark axion portal (stub) ──────────────────────────────────────────

def test_dark_axion_portal_unavailable():
    """DR3 Z-factory numerics not yet lifted."""
    assert dark_axion_portal_available() is False


# ── Kobakhidze N=8 ratio ──────────────────────────────────────────────

def test_kobakhidze_n8_ratio():
    """N=8 gives Ω_DM/Ω_B = 5.36 via XI_KOBAKHIDZE = 0.1572."""
    ratio = kobakhidze_gauge_group_ratio(N=8)
    expected = (1.0 - XI_KOBAKHIDZE) / XI_KOBAKHIDZE
    assert ratio == pytest.approx(expected, rel=1e-12)
    assert 5.3 < ratio < 5.4  # Published value ≈ 5.36


def test_kobakhidze_other_N_not_implemented():
    with pytest.raises(NotImplementedError):
        kobakhidze_gauge_group_ratio(N=5)
