"""Tight-binding band structure for d-band metals.

Computes the density of states (DOS) from a Slater-Koster tight-binding
Hamiltonian, replacing the rectangular d-band approximation with real
van Hove singularities and pseudogaps derived from first principles.

The pipeline:
    Z -> crystal structure -> neighbor vectors -> Slater-Koster hopping
    -> H(k) at each k-point -> Jacobi eigenvalues -> DOS histogram
    -> Gaussian broadening -> N(E_F)

Physics:
    - 5 d-orbitals per atom: xy, yz, zx, x²-y², 3z²-r²
    - Harrison universal hopping integrals: V_ddσ, V_ddπ, V_ddδ
    - Bandwidth scaled to match measured row-dependent d-band widths
    - BCC/FCC: 5×5 real symmetric H(k) (centrosymmetric → cos phases)
    - HCP: 10×10 Hermitian H(k) (2 atoms/cell → real embedding)

Ref: Slater & Koster, Phys. Rev. 94, 1498 (1954)
     Harrison, Electronic Structure (1980)
     Papaconstantopoulos, Handbook of Band Structure (1986)

FIRST_PRINCIPLES: DOS shape from tight-binding eigenvalues.
APPROXIMATION: nearest-neighbor d-d hopping only; no s-d hybridization.
"""

import math

from ...field.constants import HBAR, M_ELECTRON_KG, E_CHARGE

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

_EV = E_CHARGE  # 1 eV in Joules

# Harrison universal d-d hopping coefficients: V = eta * hbar^2 / (m_e * d^2)
_ETA_DD_SIGMA = -16.2
_ETA_DD_PI = 8.75
_ETA_DD_DELTA = -1.62

# Ideal c/a ratio for HCP
_HCP_CA = math.sqrt(8.0 / 3.0)  # 1.6330...


# ══════════════════════════════════════════════════════════════════
# LAYER 1: JACOBI EIGENVALUE SOLVER
# ══════════════════════════════════════════════════════════════════

def _jacobi_eigenvalues(H, n):
    """Eigenvalues of n x n real symmetric matrix via cyclic Jacobi.

    Performs cyclic sweeps over all off-diagonal pairs, applying Givens
    rotations to zero each element. Converges quadratically; typically
    ~5 sweeps for 5x5.

    Args:
        H: flat list of length n*n (row-major), modified in place
        n: matrix dimension

    Returns:
        list of n eigenvalues, sorted ascending
    """
    MAX_SWEEPS = 50
    tol = 1e-12

    for sweep in range(MAX_SWEEPS):
        # Check convergence: sum of squares of off-diagonal elements
        off_diag = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off_diag += H[i * n + j] ** 2
        if off_diag < tol:
            break

        for p in range(n):
            for q in range(p + 1, n):
                apq = H[p * n + q]
                if abs(apq) < 1e-15:
                    continue

                app = H[p * n + p]
                aqq = H[q * n + q]
                tau = (aqq - app) / (2.0 * apq)

                # Stable computation of t = sign(tau) / (|tau| + sqrt(1+tau^2))
                if tau >= 0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))

                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                # Update diagonal elements
                H[p * n + p] = app - t * apq
                H[q * n + q] = aqq + t * apq
                H[p * n + q] = 0.0
                H[q * n + p] = 0.0

                # Update off-diagonal elements
                for r in range(n):
                    if r == p or r == q:
                        continue
                    hrp = H[r * n + p]
                    hrq = H[r * n + q]
                    H[r * n + p] = c * hrp - s * hrq
                    H[p * n + r] = H[r * n + p]
                    H[r * n + q] = s * hrp + c * hrq
                    H[q * n + r] = H[r * n + q]

    eigs = [H[i * n + i] for i in range(n)]
    eigs.sort()
    return eigs


# ══════════════════════════════════════════════════════════════════
# LAYER 2: CRYSTAL GEOMETRY
# ══════════════════════════════════════════════════════════════════

def _neighbor_vectors_bcc(a):
    """BCC neighbors: 8 NN at [111] + 6 SNN at [100].

    Second-neighbor hopping breaks particle-hole symmetry, shifting
    the van Hove singularity from half-filling to ~35% filling.
    Without SNN: peak at n_d=5 (Mo). With SNN: peak at n_d~4 (Nb).

    Returns:
        (nn_vectors, snn_vectors) — nearest and second-nearest
    """
    h = a / 2.0
    nn = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                nn.append((sx * h, sy * h, sz * h))

    # 6 second neighbors at (+-a, 0, 0), (0, +-a, 0), (0, 0, +-a)
    snn = [
        (a, 0.0, 0.0), (-a, 0.0, 0.0),
        (0.0, a, 0.0), (0.0, -a, 0.0),
        (0.0, 0.0, a), (0.0, 0.0, -a),
    ]

    return nn, snn


def _neighbor_vectors_fcc(a):
    """12 nearest neighbors for FCC: permutations of (+-a/2, +-a/2, 0)."""
    h = a / 2.0
    vecs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            vecs.append((sx * h, sy * h, 0.0))
            vecs.append((sx * h, 0.0, sy * h))
            vecs.append((0.0, sx * h, sy * h))
    return vecs


def _neighbor_vectors_hcp(a):
    """HCP neighbors: 6 in-plane (intra-sublattice) + 6 inter-sublattice.

    Lattice: a1 = (a, 0, 0), a2 = (a/2, a*sqrt(3)/2, 0), c = (0, 0, c*a).
    Sublattice B offset: (a/2, a*sqrt(3)/6, c/2).

    Returns:
        (intra_vectors, inter_vectors)
        intra: 6 in-plane AA neighbors
        inter: 6 AB neighbors (3 above + 3 below)
    """
    c = _HCP_CA * a
    s3 = math.sqrt(3.0)

    # 6 in-plane AA neighbors (hexagonal ring at distance a)
    intra = [
        (a, 0.0, 0.0),
        (-a, 0.0, 0.0),
        (a / 2.0, a * s3 / 2.0, 0.0),
        (-a / 2.0, a * s3 / 2.0, 0.0),
        (a / 2.0, -a * s3 / 2.0, 0.0),
        (-a / 2.0, -a * s3 / 2.0, 0.0),
    ]

    # 6 inter-sublattice AB neighbors (3 above + 3 below)
    # B site at (a/2, a*sqrt(3)/6, c/2) relative to A
    # The 3 nearest B neighbors from an A site:
    dx0 = a / 2.0
    dy0 = a * s3 / 6.0
    dz = c / 2.0

    # Three AB vectors (above)
    inter = [
        (dx0, dy0, dz),
        (dx0 - a, dy0, dz),
        (0.0, dy0 - a * s3 / 2.0, dz),
    ]
    # Three BA vectors (below) = negatives
    for i in range(3):
        x, y, z = inter[i]
        inter.append((-x, -y, -z))

    return intra, inter


def _reciprocal_vectors(structure, a):
    """Reciprocal lattice vectors (b1, b2, b3) as 3-tuples.

    Returns vectors such that a_i . b_j = 2*pi * delta_ij.
    """
    twopi = 2.0 * math.pi

    if structure == 'bcc':
        # Real-space: a1=(a,0,0), a2=(0,a,0), a3=(0,0,a) for conventional cell
        # Simple cubic reciprocal: b_i = 2*pi/a * e_i
        k = twopi / a
        return (k, 0.0, 0.0), (0.0, k, 0.0), (0.0, 0.0, k)

    elif structure == 'fcc':
        k = twopi / a
        return (k, 0.0, 0.0), (0.0, k, 0.0), (0.0, 0.0, k)

    elif structure == 'hcp':
        c = _HCP_CA * a
        s3 = math.sqrt(3.0)
        # Hexagonal reciprocal lattice
        # b1 = 2*pi/a * (1, -1/sqrt(3), 0)
        # b2 = 2*pi/a * (0, 2/sqrt(3), 0)
        # b3 = 2*pi/c * (0, 0, 1)
        k = twopi / a
        return (
            (k, -k / s3, 0.0),
            (0.0, 2.0 * k / s3, 0.0),
            (0.0, 0.0, twopi / c),
        )

    raise ValueError(f"Unknown structure: {structure}")


def _bz_mesh(structure, n):
    """Monkhorst-Pack k-point mesh in Cartesian coordinates (1/m).

    Args:
        structure: 'bcc', 'fcc', or 'hcp'
        n: mesh density (n x n x n for cubic, n x n x n_z for hcp)

    Returns:
        list of (kx, ky, kz) tuples
    """
    b1, b2, b3 = _reciprocal_vectors(structure, 1.0)  # unit-cell normalized

    nz = max(1, int(n * 0.6)) if structure == 'hcp' else n

    points = []
    for i in range(n):
        fi = (2 * i - n + 1) / (2.0 * n)
        for j in range(n):
            fj = (2 * j - n + 1) / (2.0 * n)
            for k in range(nz):
                fk = (2 * k - nz + 1) / (2.0 * nz)
                kx = fi * b1[0] + fj * b2[0] + fk * b3[0]
                ky = fi * b1[1] + fj * b2[1] + fk * b3[1]
                kz = fi * b1[2] + fj * b2[2] + fk * b3[2]
                points.append((kx, ky, kz))

    return points


# ══════════════════════════════════════════════════════════════════
# LAYER 3: SLATER-KOSTER d-d HOPPING MATRIX
# ══════════════════════════════════════════════════════════════════

def _slater_koster_dd_matrix(l, m, n, Vs, Vp, Vd):
    """5x5 d-d Slater-Koster hopping matrix.

    Orbital ordering: 0=xy, 1=yz, 2=zx, 3=x²-y², 4=3z²-r²

    All 15 independent matrix elements from Slater & Koster (1954) Table I,
    verified against Papaconstantopoulos (1986) Appendix A.

    Args:
        l, m, n: direction cosines of bond vector R/|R|
        Vs, Vp, Vd: V_ddσ, V_ddπ, V_ddδ hopping integrals

    Returns:
        flat list of 25 elements (5x5, row-major), real symmetric
    """
    l2, m2, n2 = l * l, m * m, n * n
    s3 = math.sqrt(3.0)

    H = [0.0] * 25

    # ── Diagonal ──

    # (xy|xy)
    H[0] = 3.0 * l2 * m2 * Vs + (l2 + m2 - 4.0 * l2 * m2) * Vp + (n2 + l2 * m2) * Vd

    # (yz|yz)
    H[6] = 3.0 * m2 * n2 * Vs + (m2 + n2 - 4.0 * m2 * n2) * Vp + (l2 + m2 * n2) * Vd

    # (zx|zx)
    H[12] = 3.0 * n2 * l2 * Vs + (n2 + l2 - 4.0 * n2 * l2) * Vp + (m2 + n2 * l2) * Vd

    # (x²-y²|x²-y²)
    lm2 = (l2 - m2)
    H[18] = (0.75 * lm2 * lm2 * Vs
             + (l2 + m2 - lm2 * lm2) * Vp
             + (n2 + 0.25 * lm2 * lm2) * Vd)

    # (3z²-r²|3z²-r²)
    f = n2 - 0.5 * (l2 + m2)
    H[24] = f * f * Vs + 3.0 * n2 * (l2 + m2) * Vp + 0.75 * (l2 + m2) ** 2 * Vd

    # ── Off-diagonal (upper triangle, then mirror) ──

    # (xy|yz) [0,1]
    v01 = 3.0 * l * m2 * n * Vs + l * n * (1.0 - 4.0 * m2) * Vp + l * n * (m2 - 1.0) * Vd
    H[1] = v01; H[5] = v01

    # (xy|zx) [0,2]
    v02 = 3.0 * l2 * m * n * Vs + m * n * (1.0 - 4.0 * l2) * Vp + m * n * (l2 - 1.0) * Vd
    H[2] = v02; H[10] = v02

    # (xy|x²-y²) [0,3]
    v03 = 1.5 * l * m * lm2 * Vs + 2.0 * l * m * (-lm2) * Vp + 0.5 * l * m * lm2 * Vd
    H[3] = v03; H[15] = v03

    # (xy|3z²-r²) [0,4]
    v04 = s3 * l * m * f * Vs - 2.0 * s3 * l * m * n2 * Vp + 0.5 * s3 * l * m * (1.0 + n2) * Vd
    H[4] = v04; H[20] = v04

    # (yz|zx) [1,2]
    v12 = 3.0 * l * m * n2 * Vs + l * m * (1.0 - 4.0 * n2) * Vp + l * m * (n2 - 1.0) * Vd
    H[7] = v12; H[11] = v12

    # (yz|x²-y²) [1,3]
    v13 = 1.5 * m * n * lm2 * Vs - m * n * (1.0 + 2.0 * lm2) * Vp + m * n * (1.0 + 0.5 * lm2) * Vd
    H[8] = v13; H[16] = v13

    # (yz|3z²-r²) [1,4]
    v14 = s3 * m * n * f * Vs + s3 * m * n * (l2 + m2 - n2) * Vp - 0.5 * s3 * m * n * (l2 + m2) * Vd
    H[9] = v14; H[21] = v14

    # (zx|x²-y²) [2,3]
    v23 = 1.5 * n * l * lm2 * Vs + n * l * (1.0 - 2.0 * lm2) * Vp - n * l * (1.0 - 0.5 * lm2) * Vd
    H[13] = v23; H[17] = v23

    # (zx|3z²-r²) [2,4]
    v24 = s3 * l * n * f * Vs + s3 * l * n * (m2 + n2 - n2) * Vp - 0.5 * s3 * l * n * (l2 + m2) * Vd
    # Wait: the pi term for (zx|3z2-r2). Let me reconsider.
    # From SK table: <zx|3z2-r2> = sqrt(3)*ln*[n2-0.5(l2+m2)]*sigma
    #   + sqrt(3)*ln*[(l2+m2) - n2]*pi       (NOT m2+n2-n2)
    #   - 0.5*sqrt(3)*ln*(l2+m2)*delta
    # That's the same as (yz|3z2-r2) with m<->l. Let me recalculate:
    # Actually for (yz|3z2-r2): sigma term = sqrt(3)*mn*[n2-0.5(l2+m2)]
    #   pi = sqrt(3)*mn*(l2+m2-n2), delta = -0.5*sqrt(3)*mn*(l2+m2)
    # And (zx|3z2-r2): same with m->l, but that gives:
    #   sigma = sqrt(3)*ln*[n2-0.5(l2+m2)]
    #   pi = sqrt(3)*ln*(l2+m2-n2)  -- WAIT, should it be (m2+l2-n2)?
    # Yes, it should be the same: (l2+m2-n2). The formulas for yz and zx
    # with 3z2-r2 differ only in the prefactor (mn vs ln).
    v24 = s3 * l * n * f * Vs + s3 * l * n * (l2 + m2 - n2) * Vp - 0.5 * s3 * l * n * (l2 + m2) * Vd
    H[14] = v24; H[22] = v24

    # (x²-y²|3z²-r²) [3,4]
    v34 = (0.5 * s3 * lm2 * f * Vs
           + s3 * n2 * (-lm2) * Vp   # = -sqrt(3)*n2*(l2-m2)*pi
           + 0.25 * s3 * (1.0 + n2) * lm2 * Vd)
    H[19] = v34; H[23] = v34

    return H


# ══════════════════════════════════════════════════════════════════
# LAYER 4: HAMILTONIAN CONSTRUCTION
# ══════════════════════════════════════════════════════════════════

def _harrison_hopping(d_nn):
    """Harrison universal d-d hopping integrals for NN distance d_nn (meters).

    Returns (V_ddsigma, V_ddpi, V_dddelta) in Joules.
    """
    prefactor = HBAR * HBAR / (M_ELECTRON_KG * d_nn * d_nn)
    return (
        _ETA_DD_SIGMA * prefactor,
        _ETA_DD_PI * prefactor,
        _ETA_DD_DELTA * prefactor,
    )


def _tb_hamiltonian_cubic(kx, ky, kz, neighbors, sk_matrices, e_d, scale):
    """5x5 real symmetric H(k) for BCC or FCC.

    For centrosymmetric lattices, neighbors come in inversion pairs.
    Phase factors combine: exp(ik.R) + exp(-ik.R) = 2*cos(k.R).
    Result is real symmetric.

    Args:
        kx, ky, kz: k-point in 1/m (but we work in fractional * 2pi/a)
        neighbors: list of neighbor vectors [(dx,dy,dz), ...]
        sk_matrices: list of 5x5 SK matrices (flat, len 25) for each neighbor
        e_d: on-site d-level energy (J)
        scale: bandwidth scaling factor

    Returns:
        flat list of 25 elements (5x5, row-major)
    """
    H = [0.0] * 25

    # On-site energy
    for i in range(5):
        H[i * 5 + i] = e_d

    # Sum over neighbors: H += SK_R * cos(k.R)  (pairs give 2*cos, but
    # we iterate all neighbors individually so each gets cos once)
    for idx, (dx, dy, dz) in enumerate(neighbors):
        phase = math.cos(kx * dx + ky * dy + kz * dz)
        sk = sk_matrices[idx]
        for i in range(25):
            H[i] += scale * sk[i] * phase

    return H


def _tb_hamiltonian_hcp(kx, ky, kz, a, intra_sk, inter_vecs, inter_sk,
                         e_d, scale):
    """10x10 -> 20x20 real symmetric H(k) for HCP via real embedding.

    HCP has 2 atoms per unit cell. The Hamiltonian is:
        H = [[H_AA, H_AB],
             [H_BA, H_BB]]

    H_AA = H_BB (intra-sublattice, real symmetric from in-plane neighbors)
    H_AB = sum_R SK_R * exp(ik.R) (complex for inter-sublattice)

    Real embedding: a+bi -> [[a, -b], [b, a]] for each element,
    giving a 20x20 real symmetric matrix. Each eigenvalue doubles.

    Returns:
        flat list of 400 elements (20x20, row-major)
    """
    # Build 5x5 H_AA (real symmetric, intra-sublattice)
    H_AA = [0.0] * 25
    for i in range(5):
        H_AA[i * 5 + i] = e_d

    # In-plane neighbors (AA): all have dz=0 in ideal HCP
    # These come in pairs (R, -R) so cos phases, real symmetric
    intra_vecs = _neighbor_vectors_hcp(a)[0]
    for idx, (dx, dy, dz) in enumerate(intra_vecs):
        phase = math.cos(kx * dx + ky * dy + kz * dz)
        sk = intra_sk[idx]
        for i in range(25):
            H_AA[i] += scale * sk[i] * phase

    # Build 5x5 H_AB (complex: Re + i*Im)
    H_AB_re = [0.0] * 25
    H_AB_im = [0.0] * 25
    for idx, (dx, dy, dz) in enumerate(inter_vecs):
        kr = kx * dx + ky * dy + kz * dz
        cos_kr = math.cos(kr)
        sin_kr = math.sin(kr)
        sk = inter_sk[idx]
        for i in range(25):
            H_AB_re[i] += scale * sk[i] * cos_kr
            H_AB_im[i] += scale * sk[i] * sin_kr

    # Assemble 20x20 real symmetric embedding
    # Block structure (each 5x5 block becomes 10x10 via real embedding):
    #   [[H_AA,    0  , Re(H_AB), -Im(H_AB)],
    #    [  0,   H_AA , Im(H_AB),  Re(H_AB)],
    #    [Re(H_AB)^T, Im(H_AB)^T, H_AA, 0  ],
    #    [-Im(H_AB)^T, Re(H_AB)^T, 0, H_AA  ]]
    #
    # Actually for Hermitian H: H_BA = H_AB^dagger = Re(H_AB)^T - i*Im(H_AB)^T
    # The real embedding of the 10x10 Hermitian is a 20x20 real symmetric:
    #   Row/col indices: A_re(0-4), A_im(5-9), B_re(10-14), B_im(15-19)

    N = 20
    H = [0.0] * (N * N)

    # H_AA blocks: (A_re, A_re) and (A_im, A_im) and (B_re, B_re) and (B_im, B_im)
    for i in range(5):
        for j in range(5):
            val = H_AA[i * 5 + j]
            # A_re, A_re
            H[(i) * N + (j)] = val
            # A_im, A_im
            H[(i + 5) * N + (j + 5)] = val
            # B_re, B_re
            H[(i + 10) * N + (j + 10)] = val
            # B_im, B_im
            H[(i + 15) * N + (j + 15)] = val

    # H_AB blocks: Re and Im parts
    for i in range(5):
        for j in range(5):
            re_val = H_AB_re[i * 5 + j]
            im_val = H_AB_im[i * 5 + j]

            # (A_re, B_re) = Re(H_AB)
            H[(i) * N + (j + 10)] = re_val
            H[(j + 10) * N + (i)] = re_val  # symmetric

            # (A_re, B_im) = -Im(H_AB)
            H[(i) * N + (j + 15)] = -im_val
            H[(j + 15) * N + (i)] = -im_val

            # (A_im, B_re) = Im(H_AB)
            H[(i + 5) * N + (j + 10)] = im_val
            H[(j + 10) * N + (i + 5)] = im_val

            # (A_im, B_im) = Re(H_AB)
            H[(i + 5) * N + (j + 15)] = re_val
            H[(j + 15) * N + (i + 5)] = re_val

    return H


# ══════════════════════════════════════════════════════════════════
# LAYER 5: DOS COMPUTATION
# ══════════════════════════════════════════════════════════════════

def compute_dos(Z, n_k=20, n_bins=200):
    """Compute d-band DOS from tight-binding band structure.

    Full pipeline: Z -> structure -> neighbors -> hopping -> k-mesh ->
    diagonalize at each k -> Gaussian broadening -> DOS(E).

    Args:
        Z: atomic number
        n_k: k-mesh density (n_k x n_k x n_k for cubic)
        n_bins: number of energy bins for DOS histogram

    Returns:
        dict with keys:
            energies: list of energy bin centers (eV)
            dos: list of DOS values (states/eV/atom, both spins)
            E_F: Fermi energy (eV)
            N_EF: N(E_F) in states/eV/atom (both spins)
            bandwidth: total d-band width (eV)
            n_states: total integrated states (should be ~10)
    """
    from .element import (predict_crystal_structure, predict_lattice_parameter_m,
                          d_electron_count, d_row)

    n_d = d_electron_count(Z)
    if n_d == 0 or n_d >= 10:
        return None

    structure = predict_crystal_structure(Z)
    a = predict_lattice_parameter_m(Z)
    row = d_row(Z)

    # Row-dependent d-band width (eV)
    _D_BAND_WIDTH_EV = {3: 5.0, 4: 7.5, 5: 10.0}
    W_target_eV = _D_BAND_WIDTH_EV.get(row, 7.0)

    # On-site d-level: center of band (set to 0 for now; we only care
    # about DOS shape and relative E_F position)
    e_d = 0.0

    if structure in ('bcc', 'fcc'):
        return _compute_dos_cubic(Z, structure, a, n_d, W_target_eV,
                                  e_d, n_k, n_bins)
    elif structure == 'hcp':
        return _compute_dos_hcp(Z, a, n_d, W_target_eV, e_d, n_k, n_bins)
    else:
        # Diamond / unknown: return None (fallback to rectangular)
        return None


def _compute_dos_cubic(Z, structure, a, n_d, W_target_eV, e_d, n_k, n_bins):
    """DOS for BCC or FCC (5x5 real symmetric Hamiltonian).

    For BCC, includes second-neighbor hopping to break particle-hole
    symmetry and shift the van Hove peak from half-filling to ~35%.
    """
    # Collect all neighbor shells with their hopping integrals
    all_neighbors = []  # list of (dx, dy, dz)
    all_sk = []         # corresponding SK matrices

    if structure == 'bcc':
        nn_vecs, snn_vecs = _neighbor_vectors_bcc(a)

        # NN hopping (8 neighbors at [111])
        d_nn = math.sqrt(nn_vecs[0][0]**2 + nn_vecs[0][1]**2 + nn_vecs[0][2]**2)
        Vs_nn, Vp_nn, Vd_nn = _harrison_hopping(d_nn)
        for (dx, dy, dz) in nn_vecs:
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            l, m, n = dx/d, dy/d, dz/d
            all_neighbors.append((dx, dy, dz))
            all_sk.append(_slater_koster_dd_matrix(l, m, n, Vs_nn, Vp_nn, Vd_nn))

        # SNN hopping (6 neighbors at [100], scaled by (d_nn/d_snn)^2)
        d_snn = a  # second neighbor distance in BCC
        Vs_snn, Vp_snn, Vd_snn = _harrison_hopping(d_snn)
        for (dx, dy, dz) in snn_vecs:
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            l, m, n = dx/d, dy/d, dz/d
            all_neighbors.append((dx, dy, dz))
            all_sk.append(_slater_koster_dd_matrix(l, m, n, Vs_snn, Vp_snn, Vd_snn))

    else:
        # FCC: nearest neighbors only
        neighbors = _neighbor_vectors_fcc(a)
        d_nn = math.sqrt(neighbors[0][0]**2 + neighbors[0][1]**2 + neighbors[0][2]**2)
        Vs, Vp, Vd = _harrison_hopping(d_nn)
        for (dx, dy, dz) in neighbors:
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            l, m, n = dx/d, dy/d, dz/d
            all_neighbors.append((dx, dy, dz))
            all_sk.append(_slater_koster_dd_matrix(l, m, n, Vs, Vp, Vd))

    # Determine bandwidth scaling from high-symmetry points
    H_gamma = _tb_hamiltonian_cubic(0, 0, 0, all_neighbors, all_sk, e_d, 1.0)
    eigs_gamma = _jacobi_eigenvalues(list(H_gamma), 5)
    e_min = min(eigs_gamma)
    e_max = max(eigs_gamma)

    pi_a = math.pi / a
    if structure == 'bcc':
        sym_points = [(pi_a, 0, 0), (pi_a, pi_a, 0), (pi_a, pi_a, pi_a)]
    else:
        sym_points = [(2*pi_a, 0, 0), (pi_a, pi_a, pi_a), (2*pi_a, pi_a, 0)]

    for kpt in sym_points:
        H_k = _tb_hamiltonian_cubic(kpt[0], kpt[1], kpt[2],
                                     all_neighbors, all_sk, e_d, 1.0)
        eigs_k = _jacobi_eigenvalues(list(H_k), 5)
        e_min = min(e_min, min(eigs_k))
        e_max = max(e_max, max(eigs_k))

    W_raw_J = e_max - e_min
    if W_raw_J / _EV < 1e-10:
        return None

    scale = (W_target_eV * _EV) / W_raw_J

    # Generate k-mesh and collect all eigenvalues
    k_mesh = _bz_mesh(structure, n_k)
    scale_k = 1.0 / a

    all_eigs = []
    for (kx, ky, kz) in k_mesh:
        kx_s = kx * scale_k
        ky_s = ky * scale_k
        kz_s = kz * scale_k
        H_k = _tb_hamiltonian_cubic(kx_s, ky_s, kz_s,
                                     all_neighbors, all_sk, e_d, scale)
        eigs = _jacobi_eigenvalues(list(H_k), 5)
        all_eigs.extend(eigs)

    return _dos_from_eigenvalues(all_eigs, n_d, W_target_eV, n_bins, len(k_mesh))


def _compute_dos_hcp(Z, a, n_d, W_target_eV, e_d, n_k, n_bins):
    """DOS for HCP (10x10 Hermitian -> 20x20 real symmetric)."""

    intra_vecs, inter_vecs = _neighbor_vectors_hcp(a)

    # Hopping integrals for intra-sublattice (distance = a)
    Vs_intra, Vp_intra, Vd_intra = _harrison_hopping(a)

    # Hopping integrals for inter-sublattice
    d_inter = math.sqrt(inter_vecs[0][0]**2 + inter_vecs[0][1]**2 + inter_vecs[0][2]**2)
    Vs_inter, Vp_inter, Vd_inter = _harrison_hopping(d_inter)

    # SK matrices for intra-sublattice neighbors
    intra_sk = []
    for (dx, dy, dz) in intra_vecs:
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        l, m, n = dx/d, dy/d, dz/d
        intra_sk.append(_slater_koster_dd_matrix(l, m, n,
                        Vs_intra, Vp_intra, Vd_intra))

    # SK matrices for inter-sublattice neighbors
    inter_sk = []
    for (dx, dy, dz) in inter_vecs:
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        l, m, n = dx/d, dy/d, dz/d
        inter_sk.append(_slater_koster_dd_matrix(l, m, n,
                        Vs_inter, Vp_inter, Vd_inter))

    # Bandwidth scaling: sample Gamma point
    H_gamma = _tb_hamiltonian_hcp(0, 0, 0, a, intra_sk, inter_vecs, inter_sk,
                                   e_d, 1.0)
    eigs_gamma = _jacobi_eigenvalues(list(H_gamma), 20)
    # Remove duplicates (real embedding doubles eigenvalues)
    eigs_gamma_unique = eigs_gamma[::2]  # take every other

    e_min = min(eigs_gamma_unique)
    e_max = max(eigs_gamma_unique)

    # Sample more k-points for bandwidth
    pi_a = math.pi / a
    c = _HCP_CA * a
    pi_c = math.pi / c
    for kpt in [(pi_a, 0, 0), (0, 0, pi_c), (pi_a, pi_a/math.sqrt(3), 0)]:
        H_k = _tb_hamiltonian_hcp(kpt[0], kpt[1], kpt[2], a,
                                   intra_sk, inter_vecs, inter_sk, e_d, 1.0)
        eigs_k = _jacobi_eigenvalues(list(H_k), 20)
        eigs_unique = eigs_k[::2]
        e_min = min(e_min, min(eigs_unique))
        e_max = max(e_max, max(eigs_unique))

    W_raw_J = e_max - e_min
    W_raw_eV = W_raw_J / _EV
    if W_raw_eV < 1e-10:
        return None

    scale = (W_target_eV * _EV) / W_raw_J

    # k-mesh and eigenvalues
    k_mesh = _bz_mesh('hcp', n_k)
    scale_k = 1.0 / a

    all_eigs = []
    for (kx, ky, kz) in k_mesh:
        kx_s = kx * scale_k
        ky_s = ky * scale_k
        kz_s = kz * scale_k
        H_k = _tb_hamiltonian_hcp(kx_s, ky_s, kz_s, a,
                                   intra_sk, inter_vecs, inter_sk, e_d, scale)
        eigs = _jacobi_eigenvalues(list(H_k), 20)
        # Take unique eigenvalues (every other from sorted list)
        all_eigs.extend(eigs[::2])

    return _dos_from_eigenvalues(all_eigs, n_d, W_target_eV, n_bins,
                                  len(k_mesh))


def _dos_from_eigenvalues(all_eigs_J, n_d, W_target_eV, n_bins, n_kpts):
    """Convert eigenvalue list to DOS histogram with Gaussian broadening.

    Args:
        all_eigs_J: all eigenvalues in Joules
        n_d: d-electron count (determines E_F)
        W_target_eV: target bandwidth in eV
        n_bins: number of energy bins
        n_kpts: number of k-points (for normalization)

    Returns:
        dict with energies, dos, E_F, N_EF, bandwidth, n_states
    """
    # Convert to eV
    all_eigs = [e / _EV for e in all_eigs_J]
    all_eigs.sort()

    n_total = len(all_eigs)
    e_min = all_eigs[0]
    e_max = all_eigs[-1]
    bandwidth = e_max - e_min

    # Determine E_F by filling n_d/10 of total states
    # (each eigenvalue holds 2 electrons due to spin)
    # Total electrons per k-point: n_d (per atom, both spins)
    # States per k-point: 5 orbitals (or 10 for HCP with 2 atoms)
    # Fraction filled: n_d / 10
    fill_fraction = n_d / 10.0
    n_filled = int(fill_fraction * n_total)
    if n_filled >= n_total:
        n_filled = n_total - 1
    if n_filled < 1:
        n_filled = 1

    E_F = 0.5 * (all_eigs[n_filled - 1] + all_eigs[n_filled])

    # Gaussian broadening — physically motivated width.
    # The d-only TB model gives sharp van Hove peaks that are broadened
    # in reality by: (1) s-d hybridization, (2) electron-electron
    # scattering (quasiparticle lifetime), (3) phonon renormalization.
    # Using sigma = W/20 captures this effective broadening.
    sigma_broad = bandwidth / 20.0
    if sigma_broad < 0.01:
        sigma_broad = 0.01

    margin = 3.0 * sigma_broad
    e_lo = e_min - margin
    e_hi = e_max + margin
    de = (e_hi - e_lo) / n_bins
    energies = [e_lo + (i + 0.5) * de for i in range(n_bins)]
    dos = [0.0] * n_bins

    # Normalization: total integral of DOS over energy = 10 states/atom
    # (5 orbitals x 2 spins). Each eigenvalue contributes a Gaussian
    # normalized to 1. Total area = n_total Gaussians.
    # DOS per atom = (2 / n_kpts) * sum of Gaussians
    # (factor 2 for spin, divide by n_kpts for per-k normalization)
    # Wait: n_total = 5 * n_kpts for cubic, 10 * n_kpts for HCP (but
    # we took unique, so 5 * n_kpts for HCP too).
    # Actually for HCP: 10 eigenvalues per k-point, but we take every other
    # (real embedding), giving 10 unique eigenvalues per k-point
    # (2 atoms * 5 orbitals). So n_total = 10 * n_kpts for HCP,
    # and integral should be 10 states/atom (for 1 atom).
    # Hmm, for HCP per atom: 5 orbitals * 2 spins = 10 states.
    # With 2 atoms/cell: 10 bands, each holds 2 electrons.
    # DOS per atom = (1/n_kpts) * (1/n_atoms_per_cell) * sum of Gaussians * 2(spin)
    #
    # Let's keep it simple: bands_per_atom = 5 (always).
    # n_total eigenvalues / n_kpts = bands_per_kpoint.
    # For cubic: 5. For HCP: 10 (but per 2 atoms, so 5/atom).
    # Normalization: integral of DOS = 10 (5 orbitals * 2 spins).
    # So norm = 10.0 / (n_total * 1.0) per eigenvalue, times n_total = 10.
    # Each Gaussian has unit area. Total area = n_total.
    # We want integral = 10 states/atom.
    # For cubic: n_total = 5 * n_kpts. Want: (coeff * n_total * 1) = 10.
    #   coeff = 10 / (5 * n_kpts) = 2 / n_kpts. That's the spin factor.
    # For HCP: n_total = 10 * n_kpts (5 per atom * 2 atoms).
    #   Want: integral per atom = 10.
    #   coeff = 10 / (10 * n_kpts) = 1 / n_kpts.
    # So: bands_per_atom = n_total / n_kpts. Either 5 (cubic) or 10 (HCP).
    # Actually for HCP we want per-atom DOS. With 2 atoms/cell:
    # coeff = 2 / n_kpts for cubic (2 = spin), 1 / n_kpts for HCP (spin/n_atoms).
    # Generically: coeff = 2 / (n_kpts * n_atoms_per_cell)
    # where n_atoms_per_cell = n_total / (5 * n_kpts) ... hmm circular.
    # Let me just use: norm_factor = 10.0 / n_total (ensures integral = 10)

    norm = 10.0 / n_total  # states/atom including spin
    gauss_prefactor = norm / (sigma_broad * math.sqrt(2.0 * math.pi))

    for e_val in all_eigs:
        # Find which bins this Gaussian contributes to
        i_center = int((e_val - e_lo) / de)
        i_lo = max(0, i_center - int(4.0 * sigma_broad / de) - 1)
        i_hi = min(n_bins - 1, i_center + int(4.0 * sigma_broad / de) + 1)
        for i in range(i_lo, i_hi + 1):
            x = (energies[i] - e_val) / sigma_broad
            dos[i] += gauss_prefactor * math.exp(-0.5 * x * x)

    # Extract N(E_F)
    i_F = int((E_F - e_lo) / de)
    i_F = max(0, min(n_bins - 1, i_F))
    N_EF = dos[i_F]

    # Verify integrated states
    n_states = sum(dos) * de

    return {
        'energies': energies,
        'dos': dos,
        'E_F': E_F,
        'N_EF': N_EF,
        'bandwidth': bandwidth,
        'n_states': n_states,
    }


# ══════════════════════════════════════════════════════════════════
# LAYER 6: ENTRY POINT
# ══════════════════════════════════════════════════════════════════

_DOS_CACHE = {}


def dos_at_fermi(Z, n_k=20):
    """N(E_F) in states/eV/atom from tight-binding band structure.

    Cached per Z. Returns None for non-d-metals (n_d == 0 or 10)
    or for structures where TB is not implemented.

    Args:
        Z: atomic number
        n_k: k-mesh density (default 20 -> 8000 k-points for cubic)

    Returns:
        float: N(E_F) in states/eV/atom (both spins), or None
    """
    if Z in _DOS_CACHE:
        return _DOS_CACHE[Z]

    result = compute_dos(Z, n_k=n_k)
    if result is None:
        _DOS_CACHE[Z] = None
        return None

    N_EF = result['N_EF']
    _DOS_CACHE[Z] = N_EF
    return N_EF


def clear_cache():
    """Clear the DOS cache (useful after parameter changes)."""
    _DOS_CACHE.clear()


# ══════════════════════════════════════════════════════════════════
# PRECOMPUTED DOS SHAPE PROFILES
# ══════════════════════════════════════════════════════════════════
#
# Normalized g_dos(n_d) profiles computed from the tight-binding
# Hamiltonian above with n_k=16. Each profile is normalized so that
# the average over n_d=1..9 is 1.0, making it a pure shape correction
# that doesn't alter the absolute DOS scale.
#
# Usage: multiply rectangular N(E_F) = 10/W_d by g_dos(n_d) to get
# the filling-dependent correction from van Hove singularities.

_BCC_GDOS = {1: 0.815, 2: 1.183, 3: 1.346, 4: 1.321, 5: 1.192,
             6: 1.053, 7: 0.894, 8: 0.670, 9: 0.526}

_FCC_GDOS = {1: 0.412, 2: 0.603, 3: 0.879, 4: 1.302, 5: 1.467,
             6: 1.474, 7: 1.324, 8: 0.949, 9: 0.590}

_HCP_GDOS = {1: 0.829, 2: 1.034, 3: 1.004, 4: 0.953, 5: 1.029,
             6: 1.134, 7: 1.197, 8: 1.106, 9: 0.713}


def dos_shape_factor(structure, n_d):
    """Filling-dependent DOS shape correction from tight-binding.

    Returns g_dos: the ratio of TB DOS at E_F to the rectangular DOS,
    normalized so the average over all fillings is 1.0. This captures
    van Hove singularities (g_dos > 1) and pseudogaps (g_dos < 1)
    without changing the absolute DOS scale.

    Args:
        structure: 'bcc', 'fcc', or 'hcp'
        n_d: d-electron count (1-9)

    Returns:
        float: g_dos shape factor (1.0 = featureless, >1 = peak, <1 = gap)
    """
    n_d_int = max(1, min(9, round(n_d)))
    if structure == 'bcc':
        return _BCC_GDOS.get(n_d_int, 1.0)
    elif structure == 'fcc':
        return _FCC_GDOS.get(n_d_int, 1.0)
    elif structure == 'hcp':
        return _HCP_GDOS.get(n_d_int, 1.0)
    return 1.0
