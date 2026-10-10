"""U7 — initial-condition masses for the SPH lattices: the bulk keeps the uniform bulk-tuned mass m_bulk (summation
density = rho0 in an infinite lattice); only parcels within `depth_max` layers of a free surface get adjusted,
non-negative masses (bounded least squares on the engine's own kernel W) so the undisturbed lattice's summation
density is ~ rho0 everywhere and the reference run is near-static. A disclosed IC Choice; the engine is unmodified."""
import json, math, sys
import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from sigma_ground.dynamics.fluid.kernel import W
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"

def kernel_matrix(P, h):
    n = len(P)
    tree = cKDTree(P)
    pairs = tree.query_pairs(2 * h - 1e-12, output_type="ndarray")
    w = np.array([W(float(np.linalg.norm(P[a] - P[b])), h) for a, b in pairs])
    return sp.coo_matrix((np.r_[w, w, np.full(n, W(0.0, h))], (np.r_[pairs[:, 0], pairs[:, 1], np.arange(n)],
                                                             np.r_[pairs[:, 1], pairs[:, 0], np.arange(n)])), shape=(n, n)).tocsr()

def solve(nx, ny, nz, rho0, s=0.01, depth_max=2):
    h = 1.5 * s
    ijk = np.array([(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)])
    P = ijk * s
    A = kernel_matrix(P.astype(float), h)
    m_bulk = rho0 / (A.sum(1).max())
    depth = np.minimum.reduce([ijk[:, 0], nx - 1 - ijk[:, 0], ijk[:, 1], ny - 1 - ijk[:, 1], ijk[:, 2], nz - 1 - ijk[:, 2]])
    free = depth <= depth_max
    m = np.full(len(P), m_bulk)
    res0 = A @ m / rho0 - 1
    Af = A[:, free] / rho0
    b = 1.0 - A[:, ~free] @ m[~free] / rho0
    r = lsq_linear(Af * m_bulk, b, bounds=(0, np.inf), method="trf", tol=1e-12, max_iter=5000, lsmr_tol="auto")
    m[free] = r.x * m_bulk
    res = A @ m / rho0 - 1
    stats = {int(dp): dict(n=int((depth == dp).sum()), m_min=float(m[depth == dp].min() / m_bulk),
                           m_max=float(m[depth == dp].max() / m_bulk), res_max=float(np.abs(res[depth == dp]).max()),
                           res0_max=float(np.abs(res0[depth == dp]).max()))
             for dp in np.unique(depth)}
    return m, res, res0, m_bulk, stats, r.status

if __name__ == "__main__":
    nx, ny, nz, rho0, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    dmax = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    m, res, res0, mb, stats, status = solve(nx, ny, nz, rho0, depth_max=dmax)
    np.save(f"{OUT}/masses_{tag}.npy", m)
    info = dict(dims=[nx, ny, nz], rho0=rho0, m_bulk=mb, depth_max=dmax, lsq_status=int(status),
                res_max=float(np.abs(res).max()), res_rms=float(np.sqrt((res ** 2).mean())),
                uniform_mass_res_max=float(np.abs(res0).max()), uniform_mass_res_rms=float(np.sqrt((res0 ** 2).mean())),
                by_depth=stats)
    json.dump(info, open(f"{OUT}/masses_{tag}.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
