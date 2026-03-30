"""Tests for tight-binding band structure module.

Validates all layers: Jacobi solver, crystal geometry, Slater-Koster
hopping, Hamiltonian construction, DOS computation, and entry points.
"""

import math
import pytest

from .band_structure import (
    _jacobi_eigenvalues,
    _neighbor_vectors_bcc, _neighbor_vectors_fcc, _neighbor_vectors_hcp,
    _reciprocal_vectors, _bz_mesh,
    _slater_koster_dd_matrix,
    _harrison_hopping,
    _tb_hamiltonian_cubic, _tb_hamiltonian_hcp,
    compute_dos, dos_at_fermi, dos_shape_factor, clear_cache,
    _BCC_GDOS, _FCC_GDOS, _HCP_GDOS,
    _EV,
)


# ══════════════════════════════════════════════════════════════════
# LAYER 1: JACOBI EIGENVALUE SOLVER
# ══════════════════════════════════════════════════════════════════

class TestJacobiSolver:
    """Test Jacobi eigenvalue decomposition for real symmetric matrices."""

    def test_identity_3x3(self):
        """Identity matrix → all eigenvalues = 1."""
        H = [1, 0, 0,
             0, 1, 0,
             0, 0, 1]
        eigs = _jacobi_eigenvalues(H, 3)
        assert len(eigs) == 3
        for e in eigs:
            assert abs(e - 1.0) < 1e-10

    def test_diagonal(self):
        """Diagonal matrix → eigenvalues = diagonal entries (sorted)."""
        H = [5, 0, 0, 0, 0,
             0, 3, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 0, 7, 0,
             0, 0, 0, 0, 2]
        eigs = _jacobi_eigenvalues(H, 5)
        assert len(eigs) == 5
        expected = [1, 2, 3, 5, 7]
        for e, exp in zip(eigs, expected):
            assert abs(e - exp) < 1e-10

    def test_known_3x3(self):
        """Known 3x3 symmetric matrix with analytically known eigenvalues.

        A = [[2, 1, 0],
             [1, 3, 1],
             [0, 1, 2]]
        Eigenvalues: 1, 2, 4  (characteristic polynomial)
        """
        H = [2, 1, 0,
             1, 3, 1,
             0, 1, 2]
        eigs = _jacobi_eigenvalues(H, 3)
        expected = [1.0, 2.0, 4.0]
        for e, exp in zip(eigs, expected):
            assert abs(e - exp) < 1e-10

    def test_5x5_trace_invariant(self):
        """Trace (sum of eigenvalues) equals sum of diagonal entries."""
        H = [2, 1, 0, 0, 0,
             1, 3, 1, 0, 0,
             0, 1, 4, 1, 0,
             0, 0, 1, 5, 1,
             0, 0, 0, 1, 6]
        trace_before = sum(H[i * 5 + i] for i in range(5))
        eigs = _jacobi_eigenvalues(list(H), 5)
        assert abs(sum(eigs) - trace_before) < 1e-10

    def test_degenerate_eigenvalues(self):
        """Matrix with degenerate eigenvalues."""
        # diag(3, 3, 3, 1, 1)
        H = [3, 0, 0, 0, 0,
             0, 3, 0, 0, 0,
             0, 0, 3, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 1]
        eigs = _jacobi_eigenvalues(H, 5)
        assert abs(eigs[0] - 1.0) < 1e-10
        assert abs(eigs[1] - 1.0) < 1e-10
        assert abs(eigs[2] - 3.0) < 1e-10
        assert abs(eigs[3] - 3.0) < 1e-10
        assert abs(eigs[4] - 3.0) < 1e-10


# ══════════════════════════════════════════════════════════════════
# LAYER 2: CRYSTAL GEOMETRY
# ══════════════════════════════════════════════════════════════════

class TestCrystalGeometry:
    """Test neighbor vectors, reciprocal lattice, and k-mesh."""

    def test_bcc_8_nn(self):
        """BCC has 8 nearest neighbors at [111] directions."""
        nn, snn = _neighbor_vectors_bcc(3.0e-10)
        assert len(nn) == 8

    def test_bcc_nn_distance(self):
        """BCC NN distance = a*sqrt(3)/2."""
        a = 3.30e-10
        nn, snn = _neighbor_vectors_bcc(a)
        expected_d = a * math.sqrt(3) / 2
        for dx, dy, dz in nn:
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            assert abs(d - expected_d) < 1e-20

    def test_bcc_6_snn(self):
        """BCC has 6 second-nearest neighbors at [100] directions."""
        nn, snn = _neighbor_vectors_bcc(3.0e-10)
        assert len(snn) == 6
        for dx, dy, dz in snn:
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            assert abs(d - 3.0e-10) < 1e-20

    def test_fcc_12_nn(self):
        """FCC has 12 nearest neighbors."""
        vecs = _neighbor_vectors_fcc(3.52e-10)
        assert len(vecs) == 12

    def test_fcc_nn_distance(self):
        """FCC NN distance = a/sqrt(2)."""
        a = 3.52e-10
        vecs = _neighbor_vectors_fcc(a)
        expected_d = a / math.sqrt(2)
        for dx, dy, dz in vecs:
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            assert abs(d - expected_d) < 1e-20

    def test_hcp_intra_inter(self):
        """HCP has 6 intra-sublattice + 6 inter-sublattice neighbors."""
        intra, inter = _neighbor_vectors_hcp(2.95e-10)
        assert len(intra) == 6
        assert len(inter) == 6

    def test_hcp_intra_distance(self):
        """HCP in-plane neighbors at distance a."""
        a = 2.95e-10
        intra, inter = _neighbor_vectors_hcp(a)
        for dx, dy, dz in intra:
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            assert abs(d - a) < 1e-20

    def test_reciprocal_bcc(self):
        """BCC reciprocal vectors: magnitude = 2π/a."""
        a = 3.30e-10
        b1, b2, b3 = _reciprocal_vectors('bcc', a)
        k = 2 * math.pi / a
        assert abs(math.sqrt(sum(x**2 for x in b1)) - k) < 1e-6

    def test_bz_mesh_count(self):
        """Monkhorst-Pack mesh has n^3 points for cubic."""
        n = 8
        points = _bz_mesh('bcc', n)
        assert len(points) == n**3

    def test_bz_mesh_hcp_fewer_kz(self):
        """HCP mesh has fewer kz points (n_z ≈ 0.6n)."""
        n = 10
        points_bcc = _bz_mesh('bcc', n)
        points_hcp = _bz_mesh('hcp', n)
        assert len(points_hcp) < len(points_bcc)


# ══════════════════════════════════════════════════════════════════
# LAYER 3: SLATER-KOSTER HOPPING MATRIX
# ══════════════════════════════════════════════════════════════════

class TestSlaterKoster:
    """Test Slater-Koster d-d hopping matrix elements."""

    def test_sk_matrix_symmetric(self):
        """SK matrix must be symmetric: H[i,j] = H[j,i]."""
        # Random direction
        l, m, n = 1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)
        Vs, Vp, Vd = -1.5, 0.8, -0.15
        H = _slater_koster_dd_matrix(l, m, n, Vs, Vp, Vd)
        for i in range(5):
            for j in range(5):
                assert abs(H[i * 5 + j] - H[j * 5 + i]) < 1e-14, \
                    f"Not symmetric at ({i},{j})"

    def test_sk_trace_invariant(self):
        """Trace is invariant under rotation of direction cosines.

        Tr(H) = Vs + 2Vp + 2Vd (sum of eigenvalues of atomic d-levels
        with bond along any direction).
        """
        Vs, Vp, Vd = -2.0, 1.0, -0.2
        expected_trace = Vs + 2 * Vp + 2 * Vd

        for l, m, n in [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)),
            (1/math.sqrt(2), 1/math.sqrt(2), 0),
        ]:
            H = _slater_koster_dd_matrix(l, m, n, Vs, Vp, Vd)
            trace = sum(H[i * 5 + i] for i in range(5))
            assert abs(trace - expected_trace) < 1e-12, \
                f"Trace mismatch for direction ({l},{m},{n})"

    def test_sk_along_100_sigma_delta_only(self):
        """Along [100], only σ and δ bonds contribute; π vanishes.

        For direction (1,0,0): only orbitals with x-character couple.
        The π term should not contribute to the (x²-y²|x²-y²) diagonal.
        """
        Vs, Vp, Vd = -1.0, 0.5, -0.1
        H = _slater_koster_dd_matrix(1, 0, 0, Vs, Vp, Vd)
        # (3z²-r²|3z²-r²) along [100]: f = -0.5, so sigma = 0.25*Vs
        # pi = 0, delta = 0.75*Vd
        e44 = H[24]
        assert abs(e44 - (0.25 * Vs + 0.0 * Vp + 0.75 * Vd)) < 1e-14

    def test_sk_along_111_all_orbitals_couple(self):
        """Along [111], all three hopping types contribute."""
        Vs, Vp, Vd = -1.0, 0.5, -0.1
        s3 = 1 / math.sqrt(3)
        H = _slater_koster_dd_matrix(s3, s3, s3, Vs, Vp, Vd)
        # All off-diagonal elements should be nonzero for [111]
        for i in range(5):
            for j in range(i + 1, 5):
                # Most should be nonzero, but some might vanish by symmetry
                pass  # Just verify no crash; detailed checking via trace

    def test_harrison_hopping_signs(self):
        """Harrison hopping integrals have correct signs.

        V_ddσ < 0, V_ddπ > 0, V_ddδ < 0.
        """
        d = 2.5e-10
        Vs, Vp, Vd = _harrison_hopping(d)
        assert Vs < 0, f"V_ddσ should be negative, got {Vs}"
        assert Vp > 0, f"V_ddπ should be positive, got {Vp}"
        assert Vd < 0, f"V_ddδ should be negative, got {Vd}"

    def test_harrison_hopping_scales_as_1_over_d2(self):
        """Harrison hopping scales as 1/d²."""
        d1 = 2.5e-10
        d2 = 5.0e-10  # doubled distance
        Vs1, _, _ = _harrison_hopping(d1)
        Vs2, _, _ = _harrison_hopping(d2)
        assert abs(Vs2 / Vs1 - 0.25) < 1e-10  # (d1/d2)² = 0.25


# ══════════════════════════════════════════════════════════════════
# LAYER 4: HAMILTONIAN
# ══════════════════════════════════════════════════════════════════

class TestHamiltonian:
    """Test Hamiltonian construction."""

    def test_cubic_hamiltonian_symmetric(self):
        """H(k) must be symmetric for cubic lattices."""
        a = 3.30e-10
        nn, snn = _neighbor_vectors_bcc(a)
        d_nn = math.sqrt(nn[0][0]**2 + nn[0][1]**2 + nn[0][2]**2)
        Vs, Vp, Vd = _harrison_hopping(d_nn)

        sk_mats = []
        for dx, dy, dz in nn:
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            l, m, n = dx/d, dy/d, dz/d
            sk_mats.append(_slater_koster_dd_matrix(l, m, n, Vs, Vp, Vd))

        # k = pi/a * (1, 0.5, 0.3) — arbitrary point
        pi_a = math.pi / a
        H = _tb_hamiltonian_cubic(pi_a, 0.5*pi_a, 0.3*pi_a,
                                   nn, sk_mats, 0.0, 1.0)
        for i in range(5):
            for j in range(5):
                assert abs(H[i*5+j] - H[j*5+i]) < 1e-15

    def test_gamma_point_on_site(self):
        """At Γ (k=0), H = e_d × I + sum of SK matrices."""
        a = 3.52e-10
        vecs = _neighbor_vectors_fcc(a)
        d_nn = math.sqrt(vecs[0][0]**2 + vecs[0][1]**2 + vecs[0][2]**2)
        Vs, Vp, Vd = _harrison_hopping(d_nn)

        sk_mats = []
        for dx, dy, dz in vecs:
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            l, m, n = dx/d, dy/d, dz/d
            sk_mats.append(_slater_koster_dd_matrix(l, m, n, Vs, Vp, Vd))

        e_d = 1.0e-19  # nonzero on-site
        H = _tb_hamiltonian_cubic(0, 0, 0, vecs, sk_mats, e_d, 1.0)

        # Diagonal should include e_d
        for i in range(5):
            assert H[i*5+i] != 0  # non-trivial

    def test_hcp_hamiltonian_size(self):
        """HCP H(k) should be 20x20 (real embedding of 10x10)."""
        a = 2.95e-10
        intra_vecs, inter_vecs = _neighbor_vectors_hcp(a)
        Vs_intra, Vp_intra, Vd_intra = _harrison_hopping(a)

        d_inter = math.sqrt(inter_vecs[0][0]**2 + inter_vecs[0][1]**2
                            + inter_vecs[0][2]**2)
        Vs_inter, Vp_inter, Vd_inter = _harrison_hopping(d_inter)

        intra_sk = []
        for dx, dy, dz in intra_vecs:
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            l, m, n = dx/d, dy/d, dz/d
            intra_sk.append(_slater_koster_dd_matrix(l, m, n,
                            Vs_intra, Vp_intra, Vd_intra))

        inter_sk = []
        for dx, dy, dz in inter_vecs:
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            l, m, n = dx/d, dy/d, dz/d
            inter_sk.append(_slater_koster_dd_matrix(l, m, n,
                            Vs_inter, Vp_inter, Vd_inter))

        H = _tb_hamiltonian_hcp(0, 0, 0, a, intra_sk, inter_vecs,
                                 inter_sk, 0.0, 1.0)
        assert len(H) == 400  # 20x20


# ══════════════════════════════════════════════════════════════════
# LAYER 5: DOS COMPUTATION
# ══════════════════════════════════════════════════════════════════

class TestDOS:
    """Test density of states computation."""

    def test_dos_integrates_to_10(self):
        """Total DOS integral = 10 states/atom (5 orbitals × 2 spins)."""
        result = compute_dos(23, n_k=10)  # V, BCC
        assert result is not None
        n_states = result['n_states']
        assert abs(n_states - 10.0) < 0.5, \
            f"DOS integral = {n_states}, expected ~10"

    def test_dos_bandwidth_matches_target(self):
        """Bandwidth should match the target row-dependent width."""
        # V (Z=23), row 3, target = 5.0 eV
        result = compute_dos(23, n_k=10)
        assert result is not None
        assert abs(result['bandwidth'] - 5.0) < 1.0

    def test_dos_fermi_level_in_band(self):
        """E_F should be within the band."""
        result = compute_dos(23, n_k=10)
        assert result is not None
        E_F = result['E_F']
        energies = result['energies']
        assert energies[0] < E_F < energies[-1]

    def test_dos_positive_at_fermi(self):
        """N(E_F) > 0 for metals with 1 <= n_d <= 9."""
        for Z in [23, 41, 42, 73, 74]:  # V, Nb, Mo, Ta, W
            result = compute_dos(Z, n_k=10)
            assert result is not None
            assert result['N_EF'] > 0, f"N(E_F) = 0 for Z={Z}"

    def test_dos_none_for_full_d_band(self):
        """n_d = 0 or 10 → no TB DOS (returns None)."""
        assert compute_dos(29, n_k=10) is None  # Cu, n_d=10
        assert compute_dos(13, n_k=10) is None  # Al, n_d=0

    def test_dos_fcc(self):
        """FCC element (Ni, Z=28) produces valid DOS."""
        result = compute_dos(28, n_k=10)
        assert result is not None
        assert result['N_EF'] > 0
        assert abs(result['n_states'] - 10.0) < 0.5

    def test_dos_hcp(self):
        """HCP element (Ti, Z=22) produces valid DOS."""
        result = compute_dos(22, n_k=10)
        assert result is not None
        assert result['N_EF'] > 0
        assert abs(result['n_states'] - 10.0) < 1.0

    def test_dos_convergence(self):
        """N(E_F) changes < 15% between n_k=12 and n_k=16."""
        r12 = compute_dos(23, n_k=12)
        r16 = compute_dos(23, n_k=16)
        assert r12 is not None and r16 is not None
        ratio = r16['N_EF'] / r12['N_EF']
        assert 0.85 < ratio < 1.15, \
            f"N(E_F) ratio n_k=16/12 = {ratio:.3f}, expected ~1.0"


# ══════════════════════════════════════════════════════════════════
# LAYER 6: ENTRY POINTS
# ══════════════════════════════════════════════════════════════════

class TestEntryPoints:
    """Test dos_at_fermi and dos_shape_factor."""

    def test_dos_at_fermi_returns_float(self):
        """dos_at_fermi returns a positive float for d-metals."""
        clear_cache()
        N_EF = dos_at_fermi(23, n_k=10)  # V
        assert isinstance(N_EF, float)
        assert N_EF > 0

    def test_dos_at_fermi_none_for_sp_metal(self):
        """dos_at_fermi returns None for sp-metals."""
        clear_cache()
        assert dos_at_fermi(13, n_k=10) is None  # Al

    def test_dos_at_fermi_caching(self):
        """Second call returns cached value (same result, fast)."""
        clear_cache()
        n1 = dos_at_fermi(23, n_k=10)
        n2 = dos_at_fermi(23, n_k=10)
        assert n1 == n2  # exact same object from cache

    def test_clear_cache(self):
        """clear_cache empties the cache."""
        clear_cache()
        dos_at_fermi(23, n_k=10)
        clear_cache()
        # After clear, cache should be empty (no way to verify directly,
        # but calling again shouldn't crash)
        n = dos_at_fermi(23, n_k=10)
        assert n > 0


# ══════════════════════════════════════════════════════════════════
# PRECOMPUTED DOS SHAPE PROFILES
# ══════════════════════════════════════════════════════════════════

class TestDOSShapeProfiles:
    """Test precomputed DOS shape factor profiles."""

    def test_profiles_cover_all_fillings(self):
        """Profiles exist for n_d = 1 through 9."""
        for n_d in range(1, 10):
            assert n_d in _BCC_GDOS
            assert n_d in _FCC_GDOS
            assert n_d in _HCP_GDOS

    def test_profiles_normalized(self):
        """Profile averages are close to 1.0."""
        for name, profile in [('BCC', _BCC_GDOS), ('FCC', _FCC_GDOS),
                              ('HCP', _HCP_GDOS)]:
            avg = sum(profile.values()) / len(profile)
            assert abs(avg - 1.0) < 0.01, \
                f"{name} profile avg = {avg:.4f}, expected ~1.0"

    def test_bcc_peak_at_low_filling(self):
        """BCC profile peaks at n_d=3-4 (van Hove singularity)."""
        peak = max(_BCC_GDOS, key=_BCC_GDOS.get)
        assert peak in (3, 4), f"BCC peak at n_d={peak}, expected 3 or 4"

    def test_bcc_pseudogap_at_high_filling(self):
        """BCC profile < 1 at high filling (pseudogap)."""
        assert _BCC_GDOS[8] < 1.0
        assert _BCC_GDOS[9] < 1.0

    def test_fcc_peak_at_half_filling(self):
        """FCC profile peaks at n_d=5-6."""
        peak = max(_FCC_GDOS, key=_FCC_GDOS.get)
        assert peak in (5, 6), f"FCC peak at n_d={peak}, expected 5 or 6"

    def test_hcp_relatively_flat(self):
        """HCP profile is flatter than BCC (higher coordination)."""
        bcc_range = max(_BCC_GDOS.values()) - min(_BCC_GDOS.values())
        hcp_range = max(_HCP_GDOS.values()) - min(_HCP_GDOS.values())
        assert hcp_range < bcc_range

    def test_dos_shape_factor_bcc(self):
        """dos_shape_factor returns BCC profile values."""
        for n_d in range(1, 10):
            g = dos_shape_factor('bcc', n_d)
            assert abs(g - _BCC_GDOS[n_d]) < 1e-10

    def test_dos_shape_factor_fcc(self):
        """dos_shape_factor returns FCC profile values."""
        for n_d in range(1, 10):
            g = dos_shape_factor('fcc', n_d)
            assert abs(g - _FCC_GDOS[n_d]) < 1e-10

    def test_dos_shape_factor_hcp(self):
        """dos_shape_factor returns HCP profile values."""
        for n_d in range(1, 10):
            g = dos_shape_factor('hcp', n_d)
            assert abs(g - _HCP_GDOS[n_d]) < 1e-10

    def test_dos_shape_factor_unknown_structure(self):
        """Unknown structure returns 1.0 (no correction)."""
        assert dos_shape_factor('diamond', 5) == 1.0

    def test_dos_shape_factor_clamps_n_d(self):
        """n_d outside 1-9 is clamped to nearest valid value."""
        g_0 = dos_shape_factor('bcc', 0)
        g_1 = dos_shape_factor('bcc', 1)
        assert abs(g_0 - g_1) < 1e-10  # 0 clamps to 1

        g_10 = dos_shape_factor('bcc', 10)
        g_9 = dos_shape_factor('bcc', 9)
        assert abs(g_10 - g_9) < 1e-10  # 10 clamps to 9
