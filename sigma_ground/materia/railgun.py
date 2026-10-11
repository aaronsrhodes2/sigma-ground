"""Railgun shot physics -- the coupled process the library had the terms for.

The library holds the pieces (capacitor energy, Bloch-Gruneisen resistivity,
Debye specific heat, melting points, a parallel-rail inductance per metre, the
fixed-step RK4 integrator) but nothing integrated them together.  This module
is that integration, and ONLY that: it composes library calls into one ODE and
steps it with ``sg_solve_ivp(method='rk4')``.  It owns no material number.

State  y = [V_c, I, x, v, T, W_circuit, W_rails, W_armature]

    C dV_c/dt   = -I                                   capacitor discharging
    d(L I)/dt   = V_c - R_tot I   with  L(x) = L0 + L' x
                  =>  L dI/dt = V_c - R_tot I - L' v I   (back-EMF  L' v I)
    m dv/dt     = F(I) = 1/2 L' I^2                    Lorentz force on armature
    dx/dt       = v
    m c_p(T) dT/dt = I^2 R_arm(T)                      Joule heating of armature
    dW_k/dt     = I^2 R_k                              Joule ledgers (circuit,
                                                       rails, armature)

The 1/2 in F = 1/2 L' I^2 is the semi-infinite-rail geometry fact: only the
rails BEHIND the armature carry current, so the field at the armature is half
the field deep inside the pair (the energy-derivative F = dE_mag/dx at constant
I gives the same 1/2 L' I^2).  The energy ledger closes only with this factor
and the back-EMF term together; mutate either and the ledger fails.

Termination (event located by bisecting the last step, deterministic):
    end_of_rails   x reaches rail_len_m  -> a muzzle exit (if the circuit is
                   spent first the armature coasts out at constant v, exactly)
    stalled        circuit spent while the armature is at rest (refused)
    melt           armature T reaches the library melting point (onset of
                   melting; latent heat is NOT integrated -- the shot is over)
    t_max          never left the rails inside t_max_s (stalled) -> refused

Everything not modelled is listed in ``NOT_MODELLED`` and in
misc/RAILGUN_SCENARIO.md.  Missing owner data raises ``NeedsData``.
"""
from __future__ import annotations

import math

import numpy as np

from ..field.interface.adapters.integrator import sg_solve_ivp

# Materials the three library tables all cover (resistivity(T) + specific heat +
# melting point).  Derived at import from the tables themselves, not typed.
def _supported_metals() -> list:
    from ..field.interface.electronics import METAL_TRANSPORT
    from ..field.interface.surface import MATERIALS
    from ..field.interface.phase_transition import PHASE_DATA
    return sorted(k for k, d in METAL_TRANSPORT.items()
                  if d["rho_300"] < 1.0 and k in MATERIALS and k in PHASE_DATA)


# PHYSICS_GAP: bulk Joule heating only; skin effect, contact/arc, plasma armature, erosion,
# flat-rail L', latent heat are absent -- see KNOWN_GAPS.md and misc/RAILGUN_SCENARIO.md
NOT_MODELLED = (
    "skin effect / current diffusion (armature and rails conduct uniformly; real "
    "pulses of ~ms crowd the current into a skin, which heats a thin layer far "
    "more than the bulk number reported here -- bulk Joule heating is a LOWER bound)",
    "sliding-contact resistance, contact arcing, transition to a plasma armature "
    "(contact/bank/switch resistance is only what the caller enters as circuit_R_ohm)",
    "rail erosion, gouging, rail heating (rails are held at T0; rail resistance "
    "is DC, 2*rho(T0)*x/(pi r^2))",
    "rail inductance-gradient detail: L' is the library's round-wire pair formula "
    "(or an owner-entered override); flat-rail geometry, proximity effect, "
    "frequency-dependent L', end effects, and the rail L0 splice are not modelled",
    "magnetic saturation (the rails and armature are non-magnetic; no iron core)",
    "barrel friction, bore pressure/air column ahead of the armature, gravity, "
    "rail recoil, armature or rail elastic deformation",
    "latent heat of fusion, liquid armature, vaporization (the shot stops at the "
    "onset of melting)",
    "switch and diode behaviour: no crowbar, so the bank may ring and reverse "
    "(reported as an event, not prevented)",
)

_SPENT_FRACTION = 1e-10   # circuit energy fraction below which the armature coasts


# Law hooks.  Tests substitute a wrong law here and require the self-checks to
# fail ("mutate a law and the test must fail").
def _force_law(L_prime: float, I: float) -> float:
    """Lorentz force on the armature, F = 1/2 L' I^2 (N)."""
    return 0.5 * L_prime * I * I


def _back_emf_voltage(L_prime: float, v: float, I: float) -> float:
    """Motional EMF term of d(L I)/dt, V = L' v I (V)."""
    return L_prime * v * I


class NeedsData(ValueError):
    """A number the shot needs that no library table holds: the owner must
    enter it with a cited source.  Never filled with an invented default."""


def _need(name: str, value, why: str, missing: list):
    if value is None:
        missing.append(f"{name}: {why}")


def simulate_rail_shot(cap_F=None, volts=None, rail_len_m=None,
                       rail_radius_m=None, rail_spacing_m=None,
                       armature_material="aluminum", armature_mass_kg=None,
                       rail_material="copper", circuit_R_ohm=None,
                       circuit_L0_H=None, L_per_m=None, T0_K=300.0,
                       tolerance=1e-3, steps_per_tau=200, t_max_s=None,
                       max_steps=400_000) -> dict:
    """Integrate one capacitor-driven railgun shot; return scalars + history.

    Required, no library source (NeedsData if absent): cap_F, volts, rail_len_m,
    rail_radius_m, rail_spacing_m, armature_mass_kg (design inputs) and
    circuit_R_ohm, circuit_L0_H (the bank/busbar/switch/contact resistance and
    inductance: they come from the owner's datasheet or measurement).
    ``L_per_m`` overrides the library round-wire inductance gradient with an
    owner-entered (e.g. measured or field-solved flat-rail) value.
    """
    from ..field.interface import electronics as ele
    from ..field.interface import mobius
    from ..field.interface import thermal as thm
    from ..field.interface import phase_transition as pt
    from ..field.interface.atmosphere import speed_of_sound

    missing: list = []
    _need("cap_F", cap_F, "capacitor bank capacitance (F)", missing)
    _need("volts", volts, "initial bank voltage (V)", missing)
    _need("rail_len_m", rail_len_m, "rail length the armature travels (m)", missing)
    _need("rail_radius_m", rail_radius_m,
          "rail conductor radius (m): feeds the library round-wire L' and the rail "
          "resistance", missing)
    _need("rail_spacing_m", rail_spacing_m, "rail centre-to-centre spacing (m)",
          missing)
    _need("armature_mass_kg", armature_mass_kg, "armature mass (kg)", missing)
    _need("circuit_R_ohm", circuit_R_ohm,
          "bank ESR + busbar + switch + contact resistance (ohm): no library "
          "source, enter from the datasheet/measurement with a citation", missing)
    _need("circuit_L0_H", circuit_L0_H,
          "bank ESL + busbar + switch inductance (H): no library source, enter "
          "from the datasheet/measurement with a citation", missing)
    if missing:
        raise NeedsData("rail_shot needs data: " + "; ".join(missing))

    metals = _supported_metals()
    for role, key in (("armature_material", armature_material),
                      ("rail_material", rail_material)):
        if key not in metals:
            raise NeedsData(
                f"{role} {key!r} lacks resistivity(T)/specific-heat/melting data in "
                f"the library; supported: {metals}. Add a cited entry to "
                f"field/interface/electronics.METAL_TRANSPORT to extend.")

    for name, val in (("cap_F", cap_F), ("volts", volts), ("rail_len_m", rail_len_m),
                      ("rail_radius_m", rail_radius_m),
                      ("rail_spacing_m", rail_spacing_m),
                      ("armature_mass_kg", armature_mass_kg),
                      ("circuit_L0_H", circuit_L0_H), ("T0_K", T0_K)):
        if not (isinstance(val, (int, float)) and math.isfinite(val) and val > 0):
            raise ValueError(f"{name} must be a positive finite number (got {val!r})")
    if not (circuit_R_ohm >= 0 and math.isfinite(circuit_R_ohm)):
        raise ValueError(f"circuit_R_ohm must be >= 0 (got {circuit_R_ohm!r})")
    gap = rail_spacing_m - 2.0 * rail_radius_m
    if gap <= 0:
        raise ValueError(f"rails overlap: spacing {rail_spacing_m} m <= 2 x radius "
                         f"{rail_radius_m} m (no gap for an armature)")

    # ---- library inputs (every one cited in the step list) -------------------
    L_lib = mobius.parallel_pair_inductance_per_m(rail_radius_m, rail_spacing_m)
    if L_per_m is None:
        L_prime, L_src = L_lib, "library"
    else:
        if not (isinstance(L_per_m, (int, float)) and L_per_m > 0):
            raise ValueError(f"L_per_m must be positive (got {L_per_m!r})")
        L_prime, L_src = float(L_per_m), "owner-entered override"
    if L_prime <= 0:
        raise ValueError("library returned zero inductance gradient (rail spacing "
                         "must exceed the rail radius)")

    E0 = ele.energy_stored(cap_F, volts)
    rho_dens = ele.METAL_TRANSPORT[armature_material]["rho_kg_m3"]
    # The armature is a bar filling the rail gap: current path length = gap,
    # cross-section A = m / (density * gap).  (assumption, flagged)
    A_arm = armature_mass_kg / (rho_dens * gap)
    A_rail = math.pi * rail_radius_m ** 2
    rho_rail = ele.resistivity(rail_material, T0_K)
    R_rail_per_m = 2.0 * rho_rail / A_rail
    T_melt = pt.PHASE_DATA[armature_material]["T_melt_K"]
    if T0_K >= T_melt:
        raise ValueError(f"T0_K={T0_K} is at/above the {armature_material} melting "
                         f"point {T_melt} K")

    # Library temperature tables sampled on a fixed grid (deterministic); the ODE
    # reads them by linear interpolation.  Same library functions, no new data.
    Tgrid = np.linspace(T0_K, T_melt, 96)
    rho_tab = np.array([ele.resistivity(armature_material, float(T)) for T in Tgrid])
    cp_tab = np.array([thm.specific_heat_j_kg_K(armature_material, float(T))
                       for T in Tgrid])

    def rho_arm(T):
        return float(np.interp(T, Tgrid, rho_tab))

    def cp_arm(T):
        return float(np.interp(T, Tgrid, cp_tab))

    R_circ = float(circuit_R_ohm)
    m = float(armature_mass_kg)
    L0 = float(circuit_L0_H)
    Cf = float(cap_F)

    def rhs(t, y):
        Vc, I, x, v, theta, _Wc, _Wr, _Wa = y
        T = T0_K + theta          # state carries dT (no cancellation at tiny heating)
        Rr = R_rail_per_m * x
        Ra = rho_arm(T) * gap / A_arm
        Rt = R_circ + Rr + Ra
        L = L0 + L_prime * x
        dI = (Vc - Rt * I - _back_emf_voltage(L_prime, v, I)) / L
        F = _force_law(L_prime, I)
        return [-I / Cf, dI, v, F / m, I * I * Ra / (m * cp_arm(T)),
                I * I * R_circ, I * I * Rr, I * I * Ra]

    # ---- step sizing: resolve the bank's own time scale ----------------------
    tau = math.sqrt(L0 * Cf)                      # LC time scale of the bank
    if R_circ > 0:
        tau = min(tau, L0 / R_circ)               # and its L/R time scale
    dt = tau / float(steps_per_tau)
    t_cap = float(t_max_s) if t_max_s is not None else 100.0 * math.sqrt(L0 * Cf)

    def one_step(t, y, h):
        sol = sg_solve_ivp(rhs, (t, t + h), np.asarray(y, float), method="rk4", dt=h)
        return sol.y[:, -1]

    def gfun(y):          # event function: >= 0 once x hits the end or T melts
        return max((y[2] - rail_len_m) / rail_len_m, (T0_K + y[4] - T_melt) / T_melt)

    y = np.array([volts, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], float)   # y[4] = T - T0
    ts, ys = [0.0], [y.copy()]
    t = 0.0
    reason = "t_max"
    coast_s = 0.0
    steps = 0
    while t < t_cap and steps < max_steps:
        y_new = one_step(t, y, dt)
        steps += 1
        if not np.all(np.isfinite(y_new)):
            raise ArithmeticError("rail_shot integration went non-finite")
        if gfun(y_new) >= 0.0:
            lo, hi = 0.0, dt                      # bisect the step to the event
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if gfun(one_step(t, y, mid)) >= 0.0:
                    hi = mid
                else:
                    lo = mid
            y_new = one_step(t, y, hi)
            t += hi
            ts.append(t)
            ys.append(y_new.copy())
            gx = (y_new[2] - rail_len_m) / rail_len_m
            gT = (T0_K + y_new[4] - T_melt) / T_melt
            reason = "end_of_rails" if gx >= gT else "melt"
            y = y_new
            break
        t += dt
        y = y_new
        ts.append(t)
        ys.append(y.copy())
        # Circuit spent: nothing left to push with.  The remaining energy bounds
        # any further kinetic-energy gain to <= 1e-10 E0, so the armature coasts
        # out at constant v exactly (no friction modelled) -- no stepping needed.
        e_left = 0.5 * Cf * y[0] ** 2 + 0.5 * (L0 + L_prime * y[2]) * y[1] ** 2
        if e_left <= _SPENT_FRACTION * E0:
            if y[3] > 0.0:
                t_coast = (rail_len_m - y[2]) / y[3]
                coast_s = t_coast
                t += t_coast
                y = y.copy()
                y[2] = float(rail_len_m)
                ts.append(t)
                ys.append(y.copy())
                reason = "end_of_rails"
            else:
                reason = "stalled"
            break
    H = np.array(ys)
    t_arr = np.array(ts)
    Vc, I, x, v, theta, Wc, Wr, Wa = (H[:, k] for k in range(8))
    T = T0_K + theta

    # ---- scalars --------------------------------------------------------------
    KE = 0.5 * m * v ** 2
    Ecap = 0.5 * Cf * Vc ** 2
    Emag = 0.5 * (L0 + L_prime * x) * I ** 2
    ledger = E0 - (Ecap + Emag + KE + Wc + Wr + Wa)         # J (should be ~0)
    ledger_rel = float(np.max(np.abs(ledger)) / E0)
    # second, independent heat ledger: armature Joule = m * integral c_p dT
    cpT = cp_tab_at(T, Tgrid, cp_tab)
    Q_from_T = m * float(np.sum(0.5 * (cpT[1:] + cpT[:-1]) * np.diff(theta)))
    heat_rel = abs(Q_from_T - Wa[-1]) / Wa[-1] if Wa[-1] > 0 else 0.0

    F_hist = np.array([_force_law(L_prime, float(i)) for i in I])
    k_pk = int(np.argmax(F_hist))
    sound = speed_of_sound(288.15)

    return {
        "t": t_arr, "Vc": Vc, "I": I, "x": x, "v": v, "T": T, "theta": theta,
        "W_circuit": Wc, "W_rails": Wr, "W_armature": Wa,
        "F": F_hist, "E_cap": Ecap, "E_mag": Emag, "KE": KE,
        "ledger_J": ledger,
        "scalars": {
            "reason": reason, "exited_rails": reason == "end_of_rails",
            "t_end_s": float(t_arr[-1]), "x_end_m": float(x[-1]),
            "speed_end_m_s": float(v[-1]), "T_end_K": float(T[-1]),
            "peak_current_A": float(np.max(np.abs(I))),
            "peak_force_N": float(F_hist[k_pk]), "t_peak_force_s": float(t_arr[k_pk]),
            "I_at_peak_force_A": float(I[k_pk]),
            "E_cap0_J": float(E0), "KE_end_J": float(KE[-1]),
            "E_cap_residual_J": float(Ecap[-1]), "E_mag_end_J": float(Emag[-1]),
            "joule_circuit_J": float(Wc[-1]), "joule_rails_J": float(Wr[-1]),
            "joule_armature_J": float(Wa[-1]),
            "efficiency": float(KE[-1] / E0),
            "ledger_rel": ledger_rel, "heat_ledger_rel": float(heat_rel),
            "min_Vc_V": float(np.min(Vc)),
            "bank_reversed": bool(np.min(Vc) < -1e-9 * volts),
            "steps": steps, "dt_s": dt, "coast_s": float(coast_s),
        },
        "model": {
            "L_prime_H_m": L_prime, "L_prime_source": L_src,
            "L_prime_library_H_m": L_lib, "gap_m": gap,
            "A_armature_m2": A_arm, "A_rail_m2": A_rail,
            "R_rail_per_m_ohm": R_rail_per_m, "rho_rail_ohm_m": rho_rail,
            "T_melt_K": T_melt, "armature_material": armature_material,
            "rail_material": rail_material, "armature_mass_kg": m,
            "cap_F": Cf, "volts": float(volts), "rail_len_m": float(rail_len_m),
            "circuit_R_ohm": R_circ, "circuit_L0_H": L0, "T0_K": float(T0_K),
            "rail_radius_m": float(rail_radius_m),
            "rail_spacing_m": float(rail_spacing_m),
            "speed_of_sound_m_s": float(sound), "tolerance": float(tolerance),
            "armature_density_kg_m3": rho_dens,
            "rho_armature_T0_ohm_m": rho_arm(T0_K),
            "cp_armature_T0_J_kgK": cp_arm(T0_K),
        },
        "tables": {"T_K": Tgrid, "cp": cp_tab},
    }


def cp_tab_at(T, Tgrid, cp_tab):
    """c_p at each recorded armature temperature (the same interpolation the
    ODE used)."""
    return np.interp(T, Tgrid, cp_tab)


def check_rail_shot(sim: dict) -> dict:
    """Run the self-checks on a simulated shot.  Returns the validation dict;
    ``passed`` is False if ANY check fails (refusal on failure)."""
    from ..mcp.tools.plasma_em import electromagnetic_force_analysis

    s, mo = sim["scalars"], sim["model"]
    tol = mo["tolerance"]
    checks: dict = {}

    # 1. energy ledger: E_cap0 = E_cap + E_mag + KE + Joule(circuit+rails+armature)
    checks["energy_ledger"] = {
        "passed": s["ledger_rel"] <= tol,
        "worst_residual_over_E0": s["ledger_rel"], "tolerance": tol,
        "note": "E0 = C V^2/2 = E_cap + L I^2/2 + m v^2/2 + sum(int I^2 R dt); "
                "worst residual over the whole shot"}
    # 1b. armature heat: Joule in the armature = m * int c_p dT (independent
    # of the Joule accumulator: it reads the integrated T history)
    checks["armature_heat_balance"] = {
        "passed": s["heat_ledger_rel"] <= max(tol, 1e-2),
        "residual": s["heat_ledger_rel"],
        "note": "int I^2 R_arm dt vs m * int c_p(T) dT over the recorded T(t)"}
    # 2. muzzle speed below the all-energy-to-KE ceiling sqrt(2E/m)
    ceiling = math.sqrt(2.0 * s["E_cap0_J"] / mo["armature_mass_kg"])
    checks["speed_below_energy_ceiling"] = {
        "passed": s["speed_end_m_s"] <= ceiling,
        "speed_m_s": s["speed_end_m_s"], "ceiling_m_s": ceiling,
        "note": "v <= sqrt(2 E_cap / m): all stored energy as kinetic energy"}
    # 3. peak force vs the library's Lorentz force (F = q v x B with q*v = I*gap,
    # B = half the rail-pair field, see module docstring)
    gap, Ipk = mo["gap_m"], s["I_at_peak_force_A"]
    B_at_armature = 0.5 * mo["L_prime_H_m"] * Ipk / gap
    lib = electromagnetic_force_analysis(charge_c=Ipk * gap, velocity_m_s=1.0,
                                         b_field_t=B_at_armature)
    F_lib = lib["value"]["magnetic_force_N"]
    rel = abs(F_lib - s["peak_force_N"]) / max(s["peak_force_N"], 1e-300)
    checks["peak_force_vs_library_lorentz"] = {
        "passed": rel <= 1e-6, "F_model_N": s["peak_force_N"], "F_library_N": F_lib,
        "relative_difference": rel,
        "note": "electromagnetic_force_analysis magnetic_force_N with q*v = I*gap "
                "and B = L' I / (2 gap); checks the F = L' I^2 / 2 law and units"}
    # 4. physical sanity of the trajectory
    dx = np.diff(sim["x"])
    dT = np.diff(sim["T"])
    dv = np.diff(sim["v"])
    checks["x_monotone"] = {"passed": bool(np.all(dx >= 0.0)),
                            "note": "armature never moves backward"}
    checks["no_negative_temperature"] = {
        "passed": bool(np.min(sim["T"]) > 0.0 and np.all(dT >= -1e-9)),
        "T_min_K": float(np.min(sim["T"])),
        "note": "T > 0 and non-decreasing (Joule heating only, no cooling modelled)"}
    checks["speed_monotone"] = {
        "passed": bool(np.all(dv >= -1e-12 * max(1.0, s["speed_end_m_s"]))),
        "note": "F = L' I^2 / 2 >= 0 with no friction, so v never decreases"}
    checks["no_loss_gain"] = {
        "passed": bool(s["E_cap_residual_J"] >= 0 and s["joule_circuit_J"] >= 0
                       and s["joule_rails_J"] >= 0 and s["joule_armature_J"] >= 0),
        "note": "no ledger term is negative"}
    # 5. the shot reached a legitimate end
    checks["shot_terminated"] = {
        "passed": s["reason"] in ("end_of_rails", "melt"),
        "reason": s["reason"],
        "note": "'t_max' / 'stalled' mean the armature never left the rails"}

    failed = [k for k, c in checks.items() if not c["passed"]]
    return {
        "passed": not failed, "failed_checks": failed, "checks": checks,
        "energy_residual": s["ledger_rel"],
        "note": ("all %d self-checks passed (energy ledger worst residual %.2e of E0)"
                 % (len(checks), s["ledger_rel"])) if not failed else
                ("SELF-CHECK FAILED: " + ", ".join(failed)),
    }


def rail_shot_impact(outputs: dict, layer_material: str, thickness_m: float,
                     radius_m: float, penetrator_material: str = "tungsten") -> dict:
    """Muzzle state -> one armour layer, via games.weapon_vs_armor kinetic.interact
    (Poncelet / Birkhoff, empirical multipliers).  Needs the repo root importable.
    Raises KeyError loudly if the shot never exited the rails."""
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[2])   # repo root holds games/
    if root not in sys.path:
        sys.path.append(root)
    from games.weapon_vs_armor.physics import kinetic
    weapon = {"type": "kinetic",
              "mass_kg": outputs["projectile_mass_kg"],
              "velocity_ms": outputs["muzzle_speed_m_s"],
              "radius_m": radius_m, "material": penetrator_material}
    return kinetic.interact(weapon, layer_material, thickness_m)
