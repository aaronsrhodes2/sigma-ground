"""Independent grader for a Mentat in-game autotest run.

`stage/gmod/addon/lua/mentat/autotest.lua` records RAW engine values while GMod replays a
scenario and then hands the bodies to Havok: Source-world positions (HU), the entity's
forward/right/up vectors, velocities, timestamps. This script judges them against the
original bundle using its OWN frame math (nothing imported from the Lua or from
export_gmod's conversions), so a mistake in the Lua mapping cannot grade itself.

    python stage/gmod/verify_autotest.py fall_skillet
    python stage/gmod/verify_autotest.py fall_skillet --data "D:/Steam/steamapps/common/GarrysMod/garrysmod/data/mentat"

Checks
  REPLAY    position error (HU) and axis error (degrees) at each sample's own pose time
  RELEASE   Havok's free fall vs the analytic vacuum fall from the handed-over state, and
            vs the bundle (which includes drag): the gap is the finding, not a failure
  LANDING   resting height of the pivot above the floor and that the body stayed upright
Exit status is 0 only if every hard check passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sigma_ground.radiance.export_gmod import load_bundle  # noqa: E402  (data loading only)
from sigma_ground.radiance.trajectory import poses_at  # noqa: E402

HU_PER_M = 1.0 / 0.01905                     # Source: 1 HU = 0.75 in
DEFAULT_DATA = Path("D:/Steam/steamapps/common/GarrysMod/garrysmod/data/mentat")

POS_TOL_HU = 0.05          # float32 GetPos at ~hundreds of HU
AXIS_TOL_DEG = 0.1
LAND_TOL_HU = 1.5          # polygonal hull + contact offset
FALL_TOL_FRAC = 0.03       # Havok vs analytic vacuum fall, as a fraction of fall distance


# -- own frame math (deliberately not shared with the Lua) --------------------------------
def to_source(up, v):
    """Sigma-world vector -> Source-world vector (proper rotation)."""
    x, y, z = v
    return (x, -z, y) if up == "y" else (x, y, z)


def rotate(q, v):
    """Active rotation of v by unit quaternion q=[x,y,z,w] via t = 2 q_vec x v."""
    qx, qy, qz, qw = q
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    vx, vy, vz = v
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    return (vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx))


def expected_pos(up, p, anchor, floor_m):
    sx, sy, sz = to_source(up, p)
    return (anchor[0] + sx * HU_PER_M, anchor[1] + sy * HU_PER_M, anchor[2] + (sz - floor_m) * HU_PER_M)


def expected_axes(up, q):
    """(forward, right, up) Source vectors of an entity whose sigma-body orientation is q.
    GMod's Entity:GetRight() is the entity's RIGHT, i.e. minus its local +Y (left)."""
    ex = to_source(up, rotate(q, (1, 0, 0)))
    ey = to_source(up, rotate(q, (0, 1, 0)))
    ez = to_source(up, rotate(q, (0, 0, 1)))
    return ex, tuple(-c for c in ey), ez


def angle_deg(a, b):
    na, nb = math.sqrt(sum(c * c for c in a)), math.sqrt(sum(c * c for c in b))
    d = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))
    return math.degrees(math.acos(d))


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))



# -- angular velocity: expected from the bundle vs what the engine did after release ------
def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def expected_omega_sigma(traj, t):
    """World-frame angular velocity (rad/s, sigma frame) of body 0 at sim time t: the rotation
    vector between the bracketing recorded frames over their time gap, shortest arc. Before
    the first / after the last frame the first / last segment is used (as the stage does)."""
    fr = traj["frames"]
    lo = 0
    while lo + 1 < len(fr) and fr[lo + 1]["t_sim"] <= t:
        lo += 1
    lo = min(lo, len(fr) - 2)
    qa, qb = fr[lo]["bodies"][0]["quat"], fr[lo + 1]["bodies"][0]["quat"]
    dt = fr[lo + 1]["t_sim"] - fr[lo]["t_sim"]
    na = math.sqrt(sum(c * c for c in qa))
    nb = math.sqrt(sum(c * c for c in qb))
    qa = [c / na for c in qa]
    qb = [c / nb for c in qb]
    dq = _qmul(qb, (-qa[0], -qa[1], -qa[2], qa[3]))
    if dq[3] < 0:
        dq = tuple(-c for c in dq)
    sv = math.sqrt(dq[0] ** 2 + dq[1] ** 2 + dq[2] ** 2)
    if sv < 1e-12 or dt <= 0:
        return (0.0, 0.0, 0.0)
    ang = 2 * math.atan2(sv, dq[3])
    k = ang / sv / dt
    return (dq[0] * k, dq[1] * k, dq[2] * k)


def _mat(f, r, u):
    left = [-c for c in r]                      # GetRight is the entity's right = minus left
    return [[f[0], left[0], u[0]], [f[1], left[1], u[1]], [f[2], left[2], u[2]]]


def _mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _tr(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def engine_rotation(samples):
    """Mean world-frame (rate rad/s, unit axis) the engine applied across consecutive samples."""
    rates, axes = [], []
    for a, b in zip(samples, samples[1:]):
        dt = b["ct"] - a["ct"]
        if dt <= 0:
            continue
        R = _mm(_mat(b["fwd"], b["right"], b["up"]), _tr(_mat(a["fwd"], a["right"], a["up"])))
        c = max(-1.0, min(1.0, (R[0][0] + R[1][1] + R[2][2] - 1) / 2))
        ang = math.acos(c)
        if ang < 1e-9:
            rates.append(0.0)
            continue
        s2 = 2 * math.sin(ang)
        ax = ((R[2][1] - R[1][2]) / s2, (R[0][2] - R[2][0]) / s2, (R[1][0] - R[0][1]) / s2)
        n = math.sqrt(sum(x * x for x in ax))
        rates.append(ang / dt)
        axes.append(tuple(x / n for x in ax))
    if not rates:
        return 0.0, (0.0, 0.0, 1.0)
    rate = sum(rates) / len(rates)
    if not axes:
        return rate, (0.0, 0.0, 1.0)
    m = tuple(sum(a[i] for a in axes) / len(axes) for i in range(3))
    n = math.sqrt(sum(x * x for x in m)) or 1.0
    return rate, tuple(x / n for x in m)


RATE_TOL_FRAC = 0.05
OMEGA_AXIS_TOL_DEG = 2.0
OMEGA_MIN = 0.5            # rad/s: below this the angular check is not meaningful


def check_angular(run, up, traj, rows, hard_fail, rest_z):
    """Did the spin survive the handoff? Uses the first few airborne ticks after release."""
    released = [s for s in run["samples"] if s["released"]]
    if len(released) < 4:
        return
    want = to_source(up, expected_omega_sigma(traj, run["released_sim_t"]))
    want_rate = math.sqrt(sum(c * c for c in want))
    air = [s for s in released[1:] if s["pos"][2] > rest_z + 3.0][:6]
    if len(air) < 3:
        return
    rate, axis = engine_rotation(air)
    if want_rate < OMEGA_MIN:
        rows.append(("spin after release", f"{rate:.3f} rad/s", f"bundle ~{want_rate:.3f} rad/s: no meaningful spin to compare"))
        if rate > 0.5:
            hard_fail.append("body spins after release but the bundle does not")
        return
    err = abs(rate - want_rate) / want_rate
    wax = tuple(c / want_rate for c in want)
    dotp = max(-1.0, min(1.0, sum(a * b for a, b in zip(axis, wax))))
    aerr = math.degrees(math.acos(dotp))
    rows.append(("spin rate after release", f"{rate:.3f} rad/s vs bundle {want_rate:.3f}",
                 f"err {err * 100:.2f}% , tol {RATE_TOL_FRAC * 100:.0f}%"))
    rows.append(("spin axis after release", f"{aerr:.3f} deg off the bundle axis", f"tol {OMEGA_AXIS_TOL_DEG}"))
    if err > RATE_TOL_FRAC:
        hard_fail.append("spin rate after release differs from the bundle")
    if aerr > OMEGA_AXIS_TOL_DEG:
        hard_fail.append("spin axis after release differs from the bundle")

# -- grading ------------------------------------------------------------------------------
def grade(run: dict, bundle: dict, pivot_m: float, landing: str = "cylinder"):
    rows, hard_fail = [], []
    up = (bundle["scene"].get("camera") or {}).get("up")
    up = "y" if up == [0.0, 1.0, 0.0] else "z"
    # floor of the bundle: top of the unbound slab (re-derived here, not imported)
    floor_m = 0.0
    ui = 1 if up == "y" else 2
    tops = []
    for leaf in bundle["scene"]["csg_leaves"]:
        sh = leaf["shape"]
        if leaf.get("body") is None and sh["type"] == "Box":
            dims = [sh["x"], sh["y"], sh["z"]]
            if dims[ui] == min(dims):
                tops.append(sh["center"][ui] + dims[ui] / 2)
    if tops:
        floor_m = min(tops)

    anchor = run["anchor"]
    traj = bundle["trajectory"]
    samples = run["samples"]
    replay = [s for s in samples if not s["released"] and s.get("pose_t") is not None]
    released = [s for s in samples if s["released"]]

    # REPLAY -------------------------------------------------------------------------
    pos_err, ax_err = [], []
    for s in replay:
        pose = poses_at(traj, s["pose_t"])[0]
        pos_err.append(dist(s["pos"], expected_pos(up, pose["pos"], anchor, floor_m)))
        f, r, u = expected_axes(up, pose["quat"])
        ax_err.append(max(angle_deg(s["fwd"], f), angle_deg(s["right"], r), angle_deg(s["up"], u)))
    if replay:
        rows.append(("REPLAY samples", f"{len(replay)}", ""))
        rows.append(("REPLAY max position error", f"{max(pos_err):.2e} HU", f"tol {POS_TOL_HU} (float32 noise is ~1e-6)"))
        rows.append(("REPLAY max axis error", f"{max(ax_err):.2e} deg", f"tol {AXIS_TOL_DEG}"))
        if max(pos_err) > POS_TOL_HU:
            hard_fail.append("replay position outside tolerance")
        if max(ax_err) > AXIS_TOL_DEG:
            hard_fail.append("replay orientation outside tolerance")
        first = replay[0]
        rows.append(("cylinder axis in Source (first sample)", f"up={[round(c, 4) for c in first['up']]}",
                     "expect [0,0,1]: the skillet stands upright"))
    else:
        hard_fail.append("no replay samples recorded")

    # RELEASE / FALL -----------------------------------------------------------------
    g_hu = -run["gravity"][2]
    if released:
        h0 = released[0]
        t_rel_ct = run["released_ct"]
        rows.append(("released at sim t", f"{run['released_sim_t']:.4f} s", f"requested {run.get('release_at')}"))
        rows.append(("gravity in game", f"{g_hu:.2f} HU/s^2 = {g_hu / HU_PER_M:.4f} m/s^2",
                     "bundle assumes 9.80665"))
        # analytic vacuum fall from the FIRST released sample's recorded state
        fall = []
        for s in released:
            dt = s["ct"] - h0["ct"]
            if dt <= 0:
                continue
            z_pred = h0["pos"][2] + h0["vel"][2] * dt - 0.5 * g_hu * dt * dt
            fall.append((dt, s["pos"][2], z_pred, s["vel"][2]))
        # Only the airborne part: both the engine and the prediction are still clear of the
        # resting height (contact, bounce and settling are not free fall).
        rest_z = released[-1]["pos"][2]
        airborne = [r for r in fall if r[1] > rest_z + 1.0 and r[2] > rest_z + 1.0]
        if airborne:
            drop = h0["pos"][2] - min(r[1] for r in airborne)
            worst = max(abs(r[1] - r[2]) for r in airborne)
            rows.append(("release speed (engine)", f"{h0['vel'][2]:.1f} HU/s = {h0['vel'][2] / HU_PER_M:.3f} m/s", ""))
            vacuum = run.get("air_density") == 0 and run.get("drag_enabled") is False
            if vacuum:
                rows.append(("Havok vs analytic vacuum fall", f"worst {worst:.3f} HU over {drop:.1f} HU drop",
                             f"air off: tol {FALL_TOL_FRAC * 100:.0f}% of drop"))
                if worst > FALL_TOL_FRAC * max(drop, 1.0):
                    hard_fail.append("Havok free fall deviates from analytic vacuum fall with air resistance off")
            else:
                # Source applies its own air resistance to physics objects by default; it is
                # much stronger than the real drag baked into the bundle. Informational only.
                rows.append(("Havok vs vacuum fall (engine drag ON)", f"worst {worst:.3f} HU over {drop:.1f} HU drop",
                             "Source default air resistance; not a handoff error"))
        # bundle (with drag) vs Havok at equal sim time, while the bundle still has data
        gaps = []
        for s in released:
            tsim = run["released_sim_t"] + (s["ct"] - t_rel_ct)
            if tsim <= traj["t_end_s"]:
                pose = poses_at(traj, tsim)[0]
                gaps.append((tsim, s["pos"][2] - expected_pos(up, pose["pos"], anchor, floor_m)[2]))
        if gaps:
            rows.append(("Havok minus bundle height", f"{gaps[-1][1]:+.3f} HU at sim t={gaps[-1][0]:.3f}",
                         "bundle includes air drag, Havok has none (informational)"))

        # LANDING -------------------------------------------------------------------
        last = released[-1]
        rest_hu = last["pos"][2] - anchor[2]
        want_hu = pivot_m * HU_PER_M
        rows.append(("settled", str(run.get("settled")), ""))
        if not run.get("settled"):
            hard_fail.append("body never settled")
        if landing == "cylinder":
            rows.append(("resting pivot height", f"{rest_hu:.3f} HU", f"expect {want_hu:.3f} HU (base on floor), tol {LAND_TOL_HU}"))
            rows.append(("upright at rest", f"up.z = {last['up'][2]:.4f}", "expect > 0.99"))
            if abs(rest_hu - want_hu) > LAND_TOL_HU:
                hard_fail.append("resting height does not match the cylinder resting on its base")
            if last["up"][2] < 0.99:
                hard_fail.append("body is not upright at rest")
        else:
            rows.append(("resting pivot height", f"{rest_hu:.3f} HU", "no expectation for this shape (informational)"))
        check_angular(run, up, traj, rows, hard_fail, last["pos"][2])
    else:
        hard_fail.append("no released samples recorded (release never happened)")
    return rows, hard_fail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="GMod garrysmod/data/mentat directory")
    ap.add_argument("--landing", choices=["cylinder", "none"],
                    help="resting-pose expectation (default: cylinder for fall_skillet, none otherwise)")
    args = ap.parse_args(argv)

    path = Path(args.data) / f"autotest_{args.scenario}.json"
    if not path.exists():
        print(f"no result file: {path}\nThe game has not written it yet, or it never reached the autotest.")
        return 2
    run = json.loads(path.read_text(encoding="utf-8"))
    if not run.get("ok", False) or run.get("error"):
        print(f"autotest reported an error: {run.get('error')}")
        return 1
    bundle, _, _, _ = load_bundle(args.scenario)
    pivot_m = (bundle["scene"].get("bodies") or [{}])[0].get("pivot", [0, 0, 0])[2]
    landing = args.landing or ("cylinder" if args.scenario == "fall_skillet" else "none")
    rows, fails = grade(run, bundle, pivot_m, landing)

    print(f"scenario {run['scenario']}  map {run.get('map')}  mentat {run.get('version')}  "
          f"tick {run.get('tick_interval'):.5f}s  samples {len(run['samples'])}")
    w = max(len(r[0]) for r in rows)
    for name, val, note in rows:
        print(f"  {name.ljust(w)}  {val}" + (f"   [{note}]" if note else ""))
    if fails:
        print("\nFAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nPASS: every hard check held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
