"""Phase 4 gates — the InvoluteGear analytic shape (kernel/gear.py).

The flank-distance math here has a specific history worth encoding in
gates: the first implementation passed every "obvious" self-test (signs at
center/far-outside/tooth-centerline, volume convergence) while carrying a
frame-mapping bug that made flank distance MAGNITUDES wrong by up to ~1
module-unit — the gear frame maps to the involute's own frame by a
REFLECTION (phi_inv = theta_rb - phi_folded), not a rotation, because the
flank curls clockwise away from the tooth centerline while the canonical
involute curls counterclockwise. Only an adversarial comparison against an
independent dense-polygon ground truth caught it. The on-flank gate below
(distance ~0 for points constructed ON the flank via the independent
theta(rho) identity) is the regression test for exactly that bug: it fails
at ~0.6 for the rotation convention and passes at ~1e-12 for the reflection.
"""
import math

import pytest

from sigma_ground.kernel.gear import InvoluteGear


def _gear(**kw):
    args = dict(module=1.0, teeth=20, pressure_angle=math.radians(20.0),
               addendum_coeff=1.0, dedendum_coeff=1.25, fillet_coeff=0.35,
               face_width=0.5)
    args.update(kw)
    return InvoluteGear(**args)


def test_radii_and_theta_identities():
    g = _gear()
    assert g.r_p == pytest.approx(10.0)
    assert g.r_b == pytest.approx(10.0 * math.cos(math.radians(20.0)))
    assert g.r_a == pytest.approx(11.0)
    assert g.r_f == pytest.approx(8.75)
    # theta(r_p) == psi exactly, and theta strictly decreases r_b -> r_a
    assert g._theta(g.r_p) == pytest.approx(g.psi, abs=1e-12)
    rhos = [g.r_b + k * (g.r_a - g.r_b) / 20.0 for k in range(21)]
    thetas = [g._theta(r) for r in rhos]
    assert all(a > b for a, b in zip(thetas, thetas[1:]))


def test_on_flank_points_have_near_zero_distance():
    """THE regression gate for the reflection-vs-rotation frame bug (see
    module docstring): points constructed exactly ON the flank via the
    independent theta(rho) identity must read ~0 from the full 2D SDF."""
    g = _gear()
    for frac in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
        rho = g.r_b + frac * (g.r_a - g.r_b)
        phi_flank = g._theta(rho)             # the flank's own angular position
        # place the point on EVERY tooth's flank, not just tooth 0 —
        # the angular fold must land them all identically
        for tooth in (0, 3, 11):
            phi = phi_flank + tooth * (2.0 * math.pi / g.teeth)
            x, y = rho * math.cos(phi), rho * math.sin(phi)
            d = g._gear_sdf_2d(x, y)
            assert abs(d) < 1e-9, (
                f"on-flank point (frac={frac}, tooth={tooth}) reads d={d}")


def test_surface_distance_signs_at_known_points():
    g = _gear()
    # center: deeply inside; nearest surface is the end cap (face_width/2)
    assert g.surface_distance(0.0, 0.0, 0.0) == pytest.approx(-0.25, abs=1e-9)
    # far outside radially: distance ~ rho - r_a
    assert g.surface_distance(55.0, 0.0, 0.0) == pytest.approx(44.0, rel=1e-6)
    # on the tooth centerline between r_f and r_a: inside
    assert g.surface_distance(9.875, 0.0, 0.0) < 0.0
    # in the gap between teeth at pitch radius: outside the material
    beta = math.pi / g.teeth
    x, y = 10.0 * math.cos(beta), 10.0 * math.sin(beta)   # gap centerline
    assert g.surface_distance(x, y, 0.0) > 0.0
    # beyond the face width: outside
    assert g.surface_distance(0.0, 0.0, 0.3) > 0.0


def test_volume_and_inertia_bounded_by_root_and_tip_cylinders():
    """A gear is material between the root and tip cylinders — its volume
    and axial inertia must land strictly between those two closed forms."""
    g = _gear()
    vol = g.volume()
    iz = g.inertia_factor('z')
    h = 0.5
    vol_rf = math.pi * g.r_f ** 2 * h
    vol_ra = math.pi * g.r_a ** 2 * h
    assert vol_rf < vol < vol_ra
    assert 0.5 * g.r_f ** 2 < iz < 0.5 * g.r_a ** 2


def test_non_involute_tooth_form_raises():
    with pytest.raises(NotImplementedError):
        _gear(tooth_form="cycloidal")


def test_scene_export_round_trip():
    """"Gear" leaves must survive _shape_to_dict -> _shape_from_dict with
    surface_distance intact — the contract the viewer's in-page self-check
    (sdf_samples) depends on."""
    from sigma_ground.radiance.scene_export import (_shape_to_dict,
                                                    _shape_from_dict,
                                                    SUPPORTED_SHAPE_TYPES)
    assert "Gear" in SUPPORTED_SHAPE_TYPES
    g = _gear()
    g.source = "test citation"
    d = _shape_to_dict(g)
    assert d["type"] == "Gear"
    assert d["source"] == "test citation"
    g2 = _shape_from_dict(d)
    for p in ((0.0, 0.0, 0.0), (9.9, 0.3, 0.1), (11.5, 0.0, 0.0),
             (9.875, 0.0, 0.0)):
        assert g2.surface_distance(*p) == pytest.approx(
            g.surface_distance(*p), abs=1e-12)
    assert g2.source == "test citation"
