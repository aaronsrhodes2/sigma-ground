"""Phase E seeding: one deterministic seed, two engines, identical matter."""
import math
import unittest

from sigma_ground.dynamics.seeding import (
    G_CODATA, seed_ring_annulus, seed_rotating_cloud, to_nbody, to_continuum,
)

RING = dict(central_mass_kg=5.0e15, central_radius_m=800.0,
            r_inner_m=6000.0, r_outer_m=12000.0, ring_mass_kg=1.0e9,
            particle_density_kg_m3=920.0)          # water ice — ring matter
CLOUD = dict(cloud_mass_kg=1.0e12, cloud_radius_m=5000.0,
             omega_rad_s=1.0e-4, particle_density_kg_m3=920.0)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_bodies(self):
        a = seed_ring_annulus(50, seed=7, **RING)
        b = seed_ring_annulus(50, seed=7, **RING)
        self.assertEqual(a, b)                    # frozen dataclass equality

    def test_different_seed_different_bodies(self):
        a = seed_ring_annulus(50, seed=7, **RING)
        b = seed_ring_annulus(50, seed=8, **RING)
        self.assertNotEqual(a, b)


class TestRingPhysics(unittest.TestCase):
    def setUp(self):
        self.bodies = seed_ring_annulus(200, seed=1, **RING)

    def test_masses_and_counts(self):
        self.assertEqual(len(self.bodies), 201)
        self.assertEqual(self.bodies[0].label, "central")
        ring = self.bodies[1:]
        self.assertAlmostEqual(sum(b.mass_kg for b in ring),
                               RING["ring_mass_kg"], delta=1.0)

    def test_particles_inside_annulus(self):
        for b in self.bodies[1:]:
            r = math.hypot(b.position_m[0], b.position_m[1])
            self.assertGreaterEqual(r, RING["r_inner_m"] * 0.999)
            self.assertLessEqual(r, RING["r_outer_m"] * 1.001)

    def test_keplerian_circular_speed(self):
        """Zero dispersion → every particle's speed matches sqrt(G·M/r)
        (up to the tiny barycentric-frame shift)."""
        for b in self.bodies[1:]:
            r = math.hypot(b.position_m[0], b.position_m[1])
            v = math.hypot(b.velocity_m_s[0], b.velocity_m_s[1])
            v_kep = math.sqrt(G_CODATA * RING["central_mass_kg"] / r)
            self.assertAlmostEqual(v, v_kep, delta=v_kep * 0.01)

    def test_zero_total_momentum(self):
        px = sum(b.mass_kg * b.velocity_m_s[0] for b in self.bodies)
        py = sum(b.mass_kg * b.velocity_m_s[1] for b in self.bodies)
        pz = sum(b.mass_kg * b.velocity_m_s[2] for b in self.bodies)
        scale = self.bodies[0].mass_kg
        for p in (px, py, pz):
            self.assertLess(abs(p), scale * 1e-9)

    def test_bad_annulus_rejected(self):
        with self.assertRaises(ValueError):
            seed_ring_annulus(10, seed=0, **{**RING, "r_inner_m": 20000.0})


class TestCloudPhysics(unittest.TestCase):
    def setUp(self):
        self.bodies = seed_rotating_cloud(300, seed=2, **CLOUD)

    def test_inside_sphere_and_solid_body_rotation(self):
        """v = ω ẑ × r plus one CONSTANT barycentric-frame shift — the
        zero-momentum correction is O(1/sqrt(N))·ωR for a finite sample,
        identical for every body, so the residual after subtracting the
        mean must be ~machine zero."""
        w = CLOUD["omega_rad_s"]
        n = len(self.bodies)
        sx = sum(b.velocity_m_s[0] + w * b.position_m[1]
                 for b in self.bodies) / n
        sy = sum(b.velocity_m_s[1] - w * b.position_m[0]
                 for b in self.bodies) / n
        tol = abs(w * CLOUD["cloud_radius_m"]) * 1e-9
        for b in self.bodies:
            r = math.sqrt(sum(c * c for c in b.position_m))
            self.assertLessEqual(r, CLOUD["cloud_radius_m"] * 1.001)
            self.assertAlmostEqual(b.velocity_m_s[0],
                                   -w * b.position_m[1] + sx, delta=tol)
            self.assertAlmostEqual(b.velocity_m_s[1],
                                   w * b.position_m[0] + sy, delta=tol)

    def test_fills_the_volume_not_the_shell(self):
        """Uniform-in-volume: the median radius of a uniform ball is
        (1/2)^(1/3)·R ≈ 0.794·R — a shell would put it at ~R, a center-
        crowded sampler well below."""
        radii = sorted(math.sqrt(sum(c * c for c in b.position_m))
                       for b in self.bodies)
        median = radii[len(radii) // 2]
        self.assertGreater(median, 0.70 * CLOUD["cloud_radius_m"])
        self.assertLess(median, 0.88 * CLOUD["cloud_radius_m"])


class TestAdapters(unittest.TestCase):
    def test_both_engines_receive_identical_masses_and_state(self):
        bodies = seed_ring_annulus(20, seed=3, **RING)
        sys_nb = to_nbody(bodies, softening_m=10.0)
        scene = to_continuum(bodies, material_key='iron', G=G_CODATA)
        self.assertEqual(len(sys_nb.bodies), len(bodies))
        self.assertEqual(len(scene.parcels), len(bodies))
        for sb, cb, p in zip(bodies, sys_nb.bodies, scene.parcels):
            self.assertAlmostEqual(cb.mass_kg, sb.mass_kg)
            # SPH mass re-derived from radius x material density must land
            # on the SAME mass the n-body side carries
            self.assertAlmostEqual(p.mass, sb.mass_kg,
                                   delta=sb.mass_kg * 1e-6)
            self.assertEqual(tuple(cb.position_m), sb.position_m)
            self.assertEqual((p.position.x, p.position.y, p.position.z),
                             sb.position_m)
            self.assertEqual((p.velocity.x, p.velocity.y, p.velocity.z),
                             sb.velocity_m_s)
        self.assertEqual(scene.self_gravity_G, G_CODATA)

    def test_ring_particles_are_outside_kernel_reach_of_central(self):
        """The documented global-h limit: kernel terms between the central
        parcel and ring particles must vanish (separation > 2h)."""
        bodies = seed_ring_annulus(20, seed=3, **RING)
        scene = to_continuum(bodies, material_key='iron')
        central = scene.parcels[0]
        for p in scene.parcels[1:]:
            d = math.sqrt((p.position.x - central.position.x) ** 2
                          + (p.position.y - central.position.y) ** 2
                          + (p.position.z - central.position.z) ** 2)
            self.assertGreater(d, 2.0 * scene.h)


if __name__ == "__main__":
    unittest.main()
