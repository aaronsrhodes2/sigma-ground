"""U7 — analyse the SPH runs: arrival times of the causal response dv = v_pert - v_ref (x component) on the x axis.

Statistics per distance d (in lattice spacings s): first crossing of f x (first-pulse peak) for f = 0.1, 0.25, 0.5
(linear interpolation), the first-pulse peak time (parabolic sub-step interpolation), and the literal
threshold 1e-3 x (peak |dv| of the source parcel). Fits over a window of d: log-log slope p (pre-registered
statistic), linear d = c t + d0 (speed), and the two-model discriminator t = t0 + d/c (hyperbolic) vs
t = t0 + d^2/D (parabolic) by RMS residual.
"""
import json, math, sys
import numpy as np
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"

def ols_loglog(x, y):
    A = np.vstack([np.log(x), np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, np.log(y), rcond=None)[0][0])

def first_cross(t, y, thr):
    idx = np.nonzero(y >= thr)[0]
    if len(idx) == 0:
        return None
    j = idx[0]
    if j == 0:
        return float(t[0])
    return float(t[j - 1] + (thr - y[j - 1]) * (t[j] - t[j - 1]) / (y[j] - y[j - 1]))

def first_peak(y):
    """first local maximum followed by a drop below 0.8 of it (the first pulse; later reflections excluded)."""
    for j in range(1, len(y) - 1):
        if y[j] > 0 and y[j] >= y[j - 1] and y[j] >= y[j + 1]:
            k = j + 1
            while k < len(y) and y[k] <= y[j]:
                if y[k] < 0.8 * y[j]:
                    return j
                k += 1
    return int(np.argmax(y))

def model_compare(d, t):
    """RMS residual (in t) of t = t0 + d/c and of t = t0 + d^2/D; both 2-parameter linear least squares."""
    out = {}
    for name, x in (("hyperbolic_t0+d/c", d), ("parabolic_t0+d2/D", d * d)):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        out[name] = dict(rms=float(np.sqrt(np.mean((A @ coef - t) ** 2))), slope=float(coef[0]), t0=float(coef[1]))
    out["rms_ratio_parabolic_over_hyperbolic"] = out["parabolic_t0+d2/D"]["rms"] / out["hyperbolic_t0+d/c"]["rms"]
    return out

def analyse(ref_tag, pert_tag, dmin=4, dmax=10, sign=+1):
    r = np.load(f"{OUT}/sph_{ref_tag}.npz"); p = np.load(f"{OUT}/sph_{pert_tag}.npz")
    meta = json.load(open(f"{OUT}/sph_{pert_tag}.json"))
    ijk = [tuple(int(v) for v in x) for x in p["ijk"]]; c = tuple(int(v) for v in p["centre"])
    dt, s = meta["dt"], meta["s"]
    t = np.arange(p["V"].shape[0]) * dt
    dV = p["V"] - r["V"]
    v0 = float(np.linalg.norm(dV[:, ijk.index(c)], axis=1).max())
    rows = {}
    for d in range(1, dmax + 1):
        k = ijk.index((c[0] + sign * d, c[1], c[2]))
        y = np.abs(dV[:, k, 0]); ya = np.linalg.norm(dV[:, k], axis=1)
        j = first_peak(y)
        if 0 < j < len(y) - 1:                      # parabolic sub-step peak time
            a_, b_, c_ = y[j - 1], y[j], y[j + 1]
            den = a_ - 2 * b_ + c_
            tp = t[j] + (0.5 * (a_ - c_) / den * dt if den != 0 else 0.0)
        else:
            tp = t[j]
        row = dict(peak_step=int(j), t_peak=float(tp), peak=float(y[j]),
                   t_thr_1e3=first_cross(t, ya, 1e-3 * v0),
                   ref_speed_max_to_peak=float(np.linalg.norm(r["V"][: j + 1, k], axis=1).max()))
        for f in (0.1, 0.25, 0.5):
            row[f"t_{f:g}"] = first_cross(t[: j + 1], y[: j + 1], f * y[j])
        rows[d] = row
    sel = list(range(dmin, dmax + 1)); dd = np.array(sel, float) * s
    fits = {}
    for key in ("t_0.1", "t_0.25", "t_0.5", "t_peak", "t_thr_1e3"):
        tt = np.array([rows[d][key] for d in sel])
        lin = np.polyfit(tt, dd, 1)
        fits[key] = dict(p_loglog=ols_loglog(dd, tt), speed=float(lin[0]), offset_d0_over_s=float(lin[1] / s),
                         speed_over_c_tait=float(lin[0] / meta["c_tait"]), speed_over_c_L=float(lin[0] / meta["c_L_cascade"]),
                         local_speed_over_c_tait=(np.diff(dd) / np.diff(tt) / meta["c_tait"]).tolist(),
                         models=model_compare(dd, tt))
    ref_axis = max(np.linalg.norm(r["V"][:, ijk.index((c[0] + sign * d, c[1], c[2]))], axis=1).max() for d in range(0, dmax + 1))
    return dict(ref=ref_tag, pert=pert_tag, sign=sign, dt=dt, s=s, c_tait=meta["c_tait"], c_L=meta["c_L_cascade"],
                source=meta.get("source", "point"), alpha_visc=meta["alpha_visc"], dt_factor=meta["dt_factor"],
                material=meta["material"], v0_source_peak=v0, window=[dmin, dmax], rows=rows, fits=fits,
                ref_max_speed_axis=float(ref_axis), ref_max_speed_any=float(np.linalg.norm(r["V"], axis=2).max()))

def linearity(ref_tag, full_tag, half_tag, dmax=10):
    r = np.load(f"{OUT}/sph_{ref_tag}.npz"); p = np.load(f"{OUT}/sph_{full_tag}.npz"); q = np.load(f"{OUT}/sph_{half_tag}.npz")
    ijk = [tuple(int(v) for v in x) for x in p["ijk"]]; c = tuple(int(v) for v in p["centre"])
    out = {}
    for d in range(1, dmax + 1):
        k = ijk.index((c[0] + d, c[1], c[2]))
        y = p["V"][:, k, 0] - r["V"][:, k, 0]; yh = q["V"][:, k, 0] - r["V"][:, k, 0]
        j = first_peak(np.abs(y))
        out[d] = float(y[j] / yh[j])
    return out

if __name__ == "__main__":
    o = analyse(sys.argv[1], sys.argv[2], sign=int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    print(json.dumps({k: {kk: (vv["p_loglog"], vv["speed_over_c_tait"], vv["offset_d0_over_s"], vv["models"]["rms_ratio_parabolic_over_hyperbolic"]) for kk, vv in v.items()} if k == "fits" else v
                      for k, v in o.items() if k not in ("rows",)}, indent=1, default=float))

def track_spatial_peak(ref_tag, pert_tag, dlo=4, dhi=10, sign=+1):
    """Post-hoc diagnostic (added after the temporal statistics were seen): position of the spatial maximum of the
    leading lobe of dv_x along the axis at each step (parabolic sub-cell interpolation), kept while it lies in
    [dlo, dhi]; a linear fit x_peak(t) gives the pulse-centre (group) speed."""
    r = np.load(f"{OUT}/sph_{ref_tag}.npz"); p = np.load(f"{OUT}/sph_{pert_tag}.npz")
    meta = json.load(open(f"{OUT}/sph_{pert_tag}.json"))
    ijk = [tuple(int(v) for v in x) for x in p["ijk"]]; c = tuple(int(v) for v in p["centre"])
    dt, s = meta["dt"], meta["s"]
    ks = [ijk.index((c[0] + sign * d, c[1], c[2])) for d in range(0, dhi + 3)]
    dvx = sign * (p["V"][:, ks, 0] - r["V"][:, ks, 0])
    pts = []
    for n in range(1, dvx.shape[0]):
        row = dvx[n]
        seg = row[dlo - 1: dhi + 2]
        j = int(np.argmax(seg)) + dlo - 1
        if not (dlo <= j <= dhi) or row[j] <= 0:
            continue
        a_, b_, c_ = row[j - 1], row[j], row[j + 1]
        den = a_ - 2 * b_ + c_
        xj = j + (0.5 * (a_ - c_) / den if den != 0 else 0.0)
        pts.append((n * dt, xj * s, float(row[j])))
    pts = np.array(pts)
    # keep the monotone advancing branch (the outgoing pulse)
    keep = [0]
    for i in range(1, len(pts)):
        if pts[i, 1] >= pts[keep[-1], 1] - 0.25 * s:
            keep.append(i)
    pts = pts[keep]
    fit = np.polyfit(pts[:, 0], pts[:, 1], 1)
    return dict(points=pts.tolist(), speed=float(fit[0]), speed_over_c_tait=float(fit[0] / meta["c_tait"]),
                speed_over_c_L=float(fit[0] / meta["c_L_cascade"]), n_points=len(pts), x_range_over_s=[float(pts[0, 1] / s), float(pts[-1, 1] / s)])
