"""U2 common setup: Mentat leaves (PLATINUM 21), the one seed, both engines, the echo machinery.
Nothing here modifies the repo; it only imports sigma_ground engines (which are Mentat)."""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"D:/Aaron/development/sigma-ground"
sys.path.insert(0, REPO + "/tools")
LEAVES = os.path.join(HERE, "mentat_leaves.json")

# ---------------------------------------------------------------- Mentat leaves
def build_leaves():
    from mentat_client import mentat
    import sympy
    L = {}
    def rec(key, tool, args):
        v, s = mentat(tool, args)
        L[key] = {"value": v, "source": s, "tool": tool, "args": args}
        return v
    G = rec("G", "lookup_constant", {"name": "G"})
    rec("R_gas", "lookup_constant", {"name": "R"})
    kB = rec("k_B", "lookup_constant", {"name": "k_B"})
    rec("rho_water_ice", "density", {"material": "water_ice"})
    rec("T_melt_water_ice", "melting_point", {"material": "water_ice"})
    M, R = 1e12, 5000.0
    expr = f"sqrt(3*pi/(32*{G!r}*(3*{M!r}/(4*pi*{R!r}**3))))"
    v = rec("t_ff_expr", "simplify_expr", {"expression": expr})
    L["t_ff"] = {"value": float(sympy.sympify(v).evalf(17)), "source": "Mentat simplify_expr (sympy.simplify) of "
                 "t_ff = sqrt(3 pi / (32 G rho_mean)), evaluated with sympy.evalf", "tool": "simplify_expr"}
    E3 = rec("E_equipartition_f3_300K", "thermal_energy_per_molecule",
             {"temperature_k": 300.0, "degrees_of_freedom": 3})
    L["gamma_monatomic"] = {"value": 1.0 + kB * 300.0 / E3,
                            "source": "1 + k_B T / E(f=3): Mentat lookup_constant(k_B) + thermal_energy_per_molecule(f=3)"}
    with open(LEAVES, "w", encoding="utf-8") as f:
        json.dump(L, f, indent=1)
    return L

def leaves():
    if not os.path.exists(LEAVES):
        return build_leaves()
    with open(LEAVES, encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------- the one cloud
def setup(N=64):
    L = leaves()
    G = L["G"]["value"]; tff = L["t_ff"]["value"]
    M, R = 1e12, 5000.0
    from sigma_ground.dynamics.seeding import seed_rotating_cloud
    bodies = seed_rotating_cloud(N, cloud_mass_kg=M, cloud_radius_m=R, omega_rad_s=1e-5,
                                 particle_density_kg_m3=L["rho_water_ice"]["value"],
                                 jitter_frac=0.02, seed=3)
    cs = 0.5 * math.sqrt(G * M / R)                      # Choice: isothermal dispersion closure
    h = 1.5 * (4.0 / 3.0 * math.pi * R ** 3 / N) ** (1.0 / 3.0)   # Choice: smoothing length
    eps = 0.02 * R                                          # Choice: Plummer softening (both engines)
    return dict(N=N, M=M, R=R, G=G, tff=tff, cs=cs, h=h, eps=eps, bodies=bodies, leaves=L)

def rms_rel(a, b, R):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()) / R)

def unit_vectors(n, seed=12345):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3))
    return u / np.linalg.norm(u, axis=1)[:, None]

def u_soft(ms, ps, G, eps):
    ps = np.asarray(ps, float); ms = np.asarray(ms, float)
    d = ps[:, None, :] - ps[None, :, :]
    r = np.sqrt((d ** 2).sum(-1) + eps * eps)
    iu = np.triu_indices(len(ms), 1)
    return float(-(G * ms[:, None] * ms[None, :] / r)[iu].sum())
