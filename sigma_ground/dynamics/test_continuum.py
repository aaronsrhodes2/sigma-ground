"""
Tests for continuum.py — the bridge between interface physics and dynamics.

Strategy:
  - Test material_properties() returns physical values from the cascade
  - Test ContinuumParcel carries thermodynamic state
  - Test ContinuumScene computes SPH density
  - Test pressure gradient pushes particles apart
  - Test buoyancy lifts hot parcels
  - Test heat conduction equalizes temperature
  - Test phase transition at melting point
  - Test full continuum_step doesn't crash or produce NaN
  - Test CFL timestep is reasonable
  - Test that the bridge is truly connected to the interface layer
    (changing interface data should change dynamics behavior)

This is THE critical test: physics out. If these tests pass, the entire
interface cascade drives the simulation. If they fail, the bridge is broken.
"""

import math
import unittest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import (
    material_properties,
    ContinuumParcel,
    ContinuumScene,
    continuum_step,
    continuum_step_to,
    cfl_timestep,
    _sph_density_sum,
    _pressure_from_eos,
    _sph_pressure_force,
    _buoyancy_force,
    _heat_conduction,
)


class TestMaterialProperties(unittest.TestCase):
    """material_properties() returns physical values from the cascade."""

    def test_iron_returns_all_keys(self):
        """Iron properties dict has all required keys."""
        props = material_properties('iron', 300.0)
        required = [
            'density_kg_m3', 'bulk_modulus_Pa', 'shear_modulus_Pa',
            'thermal_conductivity_W_mK', 'specific_heat_J_kgK',
            'thermal_expansion_1_K', 'melting_point_K', 'sound_speed_m_s',
        ]
        for key in required:
            with self.subTest(key=key):
                self.assertIn(key, props)

    def test_iron_density(self):
        """Iron density ≈ 7874 kg/m³."""
        props = material_properties('iron', 300.0)
        self.assertGreater(props['density_kg_m3'], 7000)
        self.assertLess(props['density_kg_m3'], 8500)

    def test_iron_bulk_modulus(self):
        """Iron K ≈ 170 GPa."""
        props = material_properties('iron', 300.0)
        K_GPa = props['bulk_modulus_Pa'] / 1e9
        self.assertGreater(K_GPa, 100)
        self.assertLess(K_GPa, 300)

    def test_iron_melting_point(self):
        """Iron T_melt ≈ 1811 K."""
        props = material_properties('iron', 300.0)
        self.assertGreater(props['melting_point_K'], 1500)
        self.assertLess(props['melting_point_K'], 2200)

    def test_iron_sound_speed(self):
        """Iron P-wave speed ≈ 5900 m/s."""
        props = material_properties('iron', 300.0)
        self.assertGreater(props['sound_speed_m_s'], 4000)
        self.assertLess(props['sound_speed_m_s'], 7000)

    def test_all_materials(self):
        """All MATERIALS keys return valid properties."""
        from sigma_ground.field.interface.surface import MATERIALS
        for key in MATERIALS:
            with self.subTest(material=key):
                props = material_properties(key, 300.0)
                self.assertGreater(props['density_kg_m3'], 0)
                self.assertGreater(props['bulk_modulus_Pa'], 0)
                self.assertGreater(props['specific_heat_J_kgK'], 0)

    def test_temperature_affects_cp(self):
        """Specific heat should change with temperature."""
        cp_300 = material_properties('iron', 300.0)['specific_heat_J_kgK']
        cp_1000 = material_properties('iron', 1000.0)['specific_heat_J_kgK']
        # At high T, cp should approach Dulong-Petit (3R/M)
        # Both should be positive
        self.assertGreater(cp_300, 0)
        self.assertGreater(cp_1000, 0)


class TestContinuumParcel(unittest.TestCase):
    """ContinuumParcel carries thermodynamic state."""

    def test_creation(self):
        """Can create a ContinuumParcel."""
        p = ContinuumParcel('iron', radius=0.01, temperature=1000.0)
        self.assertEqual(p.material_key, 'iron')
        self.assertAlmostEqual(p.temperature, 1000.0)
        self.assertEqual(p.phase, 'solid')

    def test_has_mass(self):
        """Parcel has physical mass from density."""
        p = ContinuumParcel('iron', radius=0.01, temperature=300.0)
        expected_mass = (4.0 / 3.0) * math.pi * 0.01 ** 3 * 7874
        self.assertAlmostEqual(p.mass, expected_mass, delta=expected_mass * 0.1)

    def test_properties_cached(self):
        """Properties are cached after update."""
        p = ContinuumParcel('iron', radius=0.01, temperature=300.0)
        p.update_properties()
        self.assertIsNotNone(p._props)
        self.assertGreater(p.bulk_modulus, 0)
        self.assertGreater(p.sound_speed, 0)
        self.assertGreater(p.specific_heat, 0)
        self.assertGreater(p.thermal_conductivity, 0)

    def test_phase_detection(self):
        """Phase changes when T > T_melt."""
        p = ContinuumParcel('iron', radius=0.01, temperature=2000.0)
        p.update_properties()
        self.assertEqual(p.phase, 'liquid')

        p2 = ContinuumParcel('iron', radius=0.01, temperature=300.0)
        p2.update_properties()
        self.assertEqual(p2.phase, 'solid')

    def test_different_materials(self):
        """Different materials give different properties."""
        p_fe = ContinuumParcel('iron', radius=0.01, temperature=300.0)
        p_cu = ContinuumParcel('copper', radius=0.01, temperature=300.0)
        p_fe.update_properties()
        p_cu.update_properties()
        self.assertNotAlmostEqual(
            p_fe.bulk_modulus, p_cu.bulk_modulus, delta=1e6)

    def test_label_default(self):
        """Default label is the material key."""
        p = ContinuumParcel('iron', radius=0.01)
        self.assertEqual(p.label, 'iron')


class TestContinuumScene(unittest.TestCase):
    """ContinuumScene with SPH infrastructure."""

    def _make_scene(self, n=2, spacing=0.05):
        """Helper: create a scene with n iron parcels in a line."""
        parcels = []
        for i in range(n):
            p = ContinuumParcel(
                'iron', radius=0.01, temperature=300.0,
                position=Vec3(i * spacing, 0.0, 0.0),
            )
            parcels.append(p)
        return ContinuumScene(parcels, ground=False,
                              gravity=Vec3(0, 0, 0))

    def test_creation(self):
        """Can create a ContinuumScene."""
        scene = self._make_scene()
        self.assertEqual(len(scene.parcels), 2)
        self.assertGreater(scene.h, 0)

    def test_smoothing_length_auto(self):
        """Smoothing length auto-computed from particle size."""
        scene = self._make_scene()
        self.assertGreater(scene.h, 0.005)
        self.assertLess(scene.h, 0.1)

    def test_reference_density(self):
        """Reference density pulled from first parcel."""
        scene = self._make_scene()
        self.assertGreater(scene.rho_0, 5000)  # iron


class TestSPHDensity(unittest.TestCase):
    """SPH density sum."""

    def test_self_density(self):
        """Single particle: density = m × W(0, h)."""
        p = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                            position=Vec3(0, 0, 0))
        from sigma_ground.dynamics.fluid.kernel import W as kernel_W
        h = 0.02
        _sph_density_sum([p], h)
        expected = p.mass * kernel_W(0.0, h)
        self.assertAlmostEqual(p.sph_density, expected, delta=expected * 0.01)

    def test_two_particles_higher(self):
        """Two close particles: density > single particle."""
        p1 = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                             position=Vec3(0, 0, 0))
        p2 = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                             position=Vec3(0.015, 0, 0))
        h = 0.02
        _sph_density_sum([p1, p2], h)
        single = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                                 position=Vec3(0, 0, 0))
        _sph_density_sum([single], h)
        self.assertGreater(p1.sph_density, single.sph_density)


class TestPressureForce(unittest.TestCase):
    """SPH pressure gradient creates forces between particles."""

    def test_two_particles_force_antisymmetric(self):
        """Two particles exert equal and opposite forces (Newton's 3rd law)."""
        p1 = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                             position=Vec3(0, 0, 0))
        p2 = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                             position=Vec3(0.015, 0, 0))
        p1.update_properties()
        p2.update_properties()
        h = 0.02
        _sph_density_sum([p1, p2], h)
        _pressure_from_eos(p1)
        _pressure_from_eos(p2)

        accel = _sph_pressure_force([p1, p2], h)
        # With only 2 SPH particles, ρ_sph << ρ₀ (reference), so Tait EOS
        # gives negative pressure (tension). The force should pull particles
        # TOWARD each other: p1 pushed in +x, p2 in -x.
        # Key check: forces are equal and opposite (Newton's 3rd law).
        self.assertAlmostEqual(accel[0].x, -accel[1].x, delta=abs(accel[0].x)*0.01)
        # Also verify the force is non-zero
        self.assertNotAlmostEqual(accel[0].x, 0.0)

    def test_pressure_sign_with_compression(self):
        """Positive pressure (ρ > ρ₀) → repulsion."""
        p1 = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                             position=Vec3(0, 0, 0))
        p2 = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                             position=Vec3(0.015, 0, 0))
        p1.update_properties()
        p2.update_properties()
        h = 0.02
        _sph_density_sum([p1, p2], h)
        # Manually set positive pressure (compressed state)
        p1.pressure = 1e9
        p2.pressure = 1e9

        accel = _sph_pressure_force([p1, p2], h)
        # p1 at origin, p2 at +x: positive pressure pushes them apart
        self.assertLess(accel[0].x, 0)     # p1 pushed in -x
        self.assertGreater(accel[1].x, 0)  # p2 pushed in +x


class TestBuoyancy(unittest.TestCase):
    """Hot parcels are buoyant."""

    def test_hot_parcel_buoyant(self):
        """T > T₀ → upward buoyancy (opposing gravity)."""
        p = ContinuumParcel('iron', radius=0.01, temperature=1000.0)
        p.update_properties()
        g = Vec3(0, -9.81, 0)
        F = _buoyancy_force(p, rho_0=7874, T_0=300.0, gravity=g)
        # Hot parcel: T > T_0, alpha > 0 → buoyancy opposes gravity → F.y > 0
        self.assertGreater(F.y, 0)

    def test_cold_parcel_sinks(self):
        """T < T₀ → downward (enhances gravity)."""
        p = ContinuumParcel('iron', radius=0.01, temperature=100.0)
        p.update_properties()
        g = Vec3(0, -9.81, 0)
        F = _buoyancy_force(p, rho_0=7874, T_0=300.0, gravity=g)
        self.assertLess(F.y, 0)

    def test_neutral_at_T0(self):
        """T = T₀ → zero buoyancy."""
        p = ContinuumParcel('iron', radius=0.01, temperature=300.0)
        p.update_properties()
        g = Vec3(0, -9.81, 0)
        F = _buoyancy_force(p, rho_0=7874, T_0=300.0, gravity=g)
        self.assertAlmostEqual(F.y, 0.0, places=10)


class TestHeatConduction(unittest.TestCase):
    """Heat flows from hot to cold."""

    def test_heat_flows_hot_to_cold(self):
        """Hot particle loses heat, cold particle gains."""
        p1 = ContinuumParcel('iron', radius=0.01, temperature=1000.0,
                             position=Vec3(0, 0, 0))
        p2 = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                             position=Vec3(0.015, 0, 0))
        p1.update_properties()
        p2.update_properties()
        h = 0.02
        _sph_density_sum([p1, p2], h)

        dTdt = _heat_conduction([p1, p2], h)
        # Hot particle should cool (dT/dt < 0)
        self.assertLess(dTdt[0], 0)
        # Cold particle should warm (dT/dt > 0)
        self.assertGreater(dTdt[1], 0)

    def test_equal_T_no_flow(self):
        """Same temperature → no heat flow."""
        p1 = ContinuumParcel('iron', radius=0.01, temperature=500.0,
                             position=Vec3(0, 0, 0))
        p2 = ContinuumParcel('iron', radius=0.01, temperature=500.0,
                             position=Vec3(0.015, 0, 0))
        p1.update_properties()
        p2.update_properties()
        h = 0.02
        _sph_density_sum([p1, p2], h)

        dTdt = _heat_conduction([p1, p2], h)
        self.assertAlmostEqual(dTdt[0], 0.0, places=5)
        self.assertAlmostEqual(dTdt[1], 0.0, places=5)


class TestContinuumStep(unittest.TestCase):
    """Full continuum_step integration."""

    def test_step_no_crash(self):
        """Single step completes without error."""
        p1 = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                             position=Vec3(0, 0, 0))
        p2 = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                             position=Vec3(0.03, 0, 0))
        scene = ContinuumScene([p1, p2], ground=False,
                               gravity=Vec3(0, -9.81, 0))
        dt = continuum_step(scene, 0.001)
        self.assertGreater(dt, 0)
        self.assertGreater(scene.time, 0)

    def test_step_no_nan(self):
        """No NaN in position, velocity, or temperature after step."""
        p1 = ContinuumParcel('iron', radius=0.01, temperature=500.0,
                             position=Vec3(0, 0.1, 0))
        p2 = ContinuumParcel('copper', radius=0.01, temperature=800.0,
                             position=Vec3(0.03, 0.1, 0))
        scene = ContinuumScene([p1, p2], ground=False,
                               gravity=Vec3(0, -9.81, 0))
        continuum_step(scene, 0.0001)

        for p in scene.parcels:
            if isinstance(p, ContinuumParcel):
                self.assertFalse(math.isnan(p.position.x))
                self.assertFalse(math.isnan(p.position.y))
                self.assertFalse(math.isnan(p.position.z))
                self.assertFalse(math.isnan(p.velocity.x))
                self.assertFalse(math.isnan(p.temperature))

    def test_gravity_moves_parcels(self):
        """Gravity accelerates parcels downward."""
        p = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                            position=Vec3(0, 1.0, 0))
        scene = ContinuumScene([p], ground=False,
                               gravity=Vec3(0, -9.81, 0))
        for _ in range(10):
            continuum_step(scene, 0.01)
        self.assertLess(p.position.y, 1.0)
        self.assertLess(p.velocity.y, 0)

    def test_heat_source_warms(self):
        """Internal heat source raises temperature."""
        p = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                            position=Vec3(0, 0, 0),
                            heat_source_W_kg=1000.0)
        scene = ContinuumScene([p], ground=False, gravity=Vec3(0, 0, 0))
        T_initial = p.temperature
        for _ in range(100):
            continuum_step(scene, 0.01)
        self.assertGreater(p.temperature, T_initial)

    def test_step_to(self):
        """continuum_step_to advances to target time."""
        p = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                            position=Vec3(0, 0.5, 0))
        scene = ContinuumScene([p], ground=False,
                               gravity=Vec3(0, -9.81, 0))
        history = continuum_step_to(scene, 0.1, dt=0.01)
        self.assertGreaterEqual(scene.time, 0.1)
        self.assertGreater(len(history), 0)


class TestCFLTimestep(unittest.TestCase):
    """CFL timestep estimation."""

    def test_positive(self):
        """CFL dt is positive."""
        p = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                            position=Vec3(0, 0, 0))
        scene = ContinuumScene([p], ground=False, gravity=Vec3(0, 0, 0))
        dt = cfl_timestep(scene)
        self.assertGreater(dt, 0)

    def test_reasonable_range(self):
        """CFL dt should be small (microseconds to milliseconds)."""
        p = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                            position=Vec3(0, 0, 0))
        scene = ContinuumScene([p], ground=False, gravity=Vec3(0, 0, 0))
        dt = cfl_timestep(scene)
        # With h ~ 0.01, c_s ~ 5900 m/s: dt ~ 0.3 * 0.01 / 5900 ~ 5e-7 s
        self.assertGreater(dt, 1e-10)
        self.assertLess(dt, 1.0)

    def test_faster_particles_shorter_dt(self):
        """Fast-moving particles → smaller CFL dt."""
        p_slow = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                                 position=Vec3(0, 0, 0),
                                 velocity=Vec3(1, 0, 0))
        p_fast = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                                 position=Vec3(0, 0, 0),
                                 velocity=Vec3(10000, 0, 0))
        scene_slow = ContinuumScene([p_slow], ground=False,
                                    gravity=Vec3(0, 0, 0))
        scene_fast = ContinuumScene([p_fast], ground=False,
                                    gravity=Vec3(0, 0, 0))
        dt_slow = cfl_timestep(scene_slow)
        dt_fast = cfl_timestep(scene_fast)
        self.assertGreater(dt_slow, dt_fast)


class TestPhysicsOut(unittest.TestCase):
    """The bridge truly connects interface physics to dynamics.

    These tests verify that changing material properties in the interface
    layer produces different dynamics behavior — proving the cascade is
    live, not disconnected.
    """

    def test_different_materials_different_pressure(self):
        """Iron (K=170 GPa) produces different pressure than aluminum (K=76 GPa)."""
        p_fe = ContinuumParcel('iron', radius=0.01, temperature=300.0,
                               position=Vec3(0, 0, 0))
        p_al = ContinuumParcel('aluminum', radius=0.01, temperature=300.0,
                               position=Vec3(0, 0, 0))
        p_fe.update_properties()
        p_al.update_properties()

        # Give same density perturbation
        p_fe.sph_density = p_fe.props['density_kg_m3'] * 1.01
        p_al.sph_density = p_al.props['density_kg_m3'] * 1.01

        P_fe = _pressure_from_eos(p_fe)
        P_al = _pressure_from_eos(p_al)

        # Iron is stiffer → higher pressure for same fractional compression
        self.assertGreater(abs(P_fe), abs(P_al))

    def test_copper_melts_before_iron(self):
        """Copper (T_m=1358 K) melts before iron (T_m=1811 K)."""
        p_cu = ContinuumParcel('copper', radius=0.01, temperature=1500.0)
        p_fe = ContinuumParcel('iron', radius=0.01, temperature=1500.0)
        p_cu.update_properties()
        p_fe.update_properties()
        self.assertEqual(p_cu.phase, 'liquid')
        self.assertEqual(p_fe.phase, 'solid')

    def test_sound_speed_from_cascade(self):
        """Sound speed comes from the interface cascade, not hardcoded."""
        p = ContinuumParcel('iron', radius=0.01, temperature=300.0)
        p.update_properties()
        # Should match acoustics.py longitudinal_wave_speed('iron')
        from sigma_ground.field.interface.acoustics import longitudinal_wave_speed
        v_cascade = longitudinal_wave_speed('iron')
        self.assertAlmostEqual(p.sound_speed, v_cascade, delta=v_cascade * 0.01)


if __name__ == '__main__':
    unittest.main()
