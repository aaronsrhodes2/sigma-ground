"""verify_svo_adversarial.py — R4 refutation instrument (adversarial twin of
tools/verify_svo_parity.py).

The shipped parity tool sweeps the committed baseline cameras and a golden-
ratio fuzz around each body. This tool attacks the SAME claim ("svoDist is a
drop-in leaf distance — hit/miss/material parity with the dense path at
matched LOD") where it is most likely to break:

  A1  grazing-silhouette ray fans — silhouette pixels found by a coarse dense
      pass, then a dense sub-pixel fan around each; every outcome flip is
      field-verified (one-sided field divergence = HARD FAIL).
  A2  rays STARTING INSIDE bodies — svoDist's interior sign/magnitude and the
      exit-hit parity (the shipped tool only marches from the camera).
  A3  LOD pins (cap = L-1, L-2): overshoot svo(cap) − dense measured against
      the documented seam bound (~1 cell at cap) — including the uniform-
      fallback floor `guar = band0·2^-cap − 0.87·vs_cap`, which scales the
      FINE-level classification guarantee by 2^(L-cap) and is therefore
      suspected of exceeding true distance at pinned LOD.
  A4  brick-border/root-face probes — seam planes at ±1e-9·span, the
      0.99999994 clamp corner, exact faces/corners, plus a recording
      traversal that asserts every atlas read stays inside the brick's 18^3
      stored block (the CLAMP_TO_EDGE question, answered numerically).
  A5  degenerate synthetic bodies fed through the REAL builder
      (tools/dump_svo.node.js): flush-face grids (pad=0 — the "beyond-cube
      is empty" builder assumption is false there), tiny/odd dims, extreme
      aspect rods, offset spheres with one flush face, 1e6/1e-6 voxel sizes,
      payload-shared twin bodies (dedupe + material identity).

Exit 0 = no HARD finding; 1 = at least one HARD finding; 2 = infrastructure.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DATA = _ROOT / "sigma_ground" / "radiance" / "web" / "data"
_MANIFEST = _ROOT / "sigma_ground" / "radiance" / "web" / "baselines" / "manifest.json"

_spec = importlib.util.spec_from_file_location(
    "vsp", str(_ROOT / "tools" / "verify_svo_parity.py"))
vsp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vsp)

STORED = 18
HARD = []          # accumulated hard findings
NOTES = []         # measured-and-documented (not hard)


def note(msg):
    NOTES.append(msg)
    print("  [note] " + msg)


def hard(msg):
    HARD.append(msg)
    print("  [HARD] " + msg)


# ── recording SVO: same traversal, asserts atlas reads stay in-brick ───────
class RecSVO(vsp.SVO):
    def __init__(self, outdir):
        super().__init__(outdir)
        self.oob = 0                     # reads outside the brick's 18^3 block

    def d(self, row, q, cap=None):
        # verbatim vsp.SVO.d with an instrumented `at`
        f32 = np.float32
        NO_BRICK = 0xFFFFFFFF
        B = self.bodies[row]
        lo, span, band0, L = B["lo"], B["rootSpan"], B["band0"], B["L"]
        cap = L if cap is None else cap
        u = [float(f32((q[0] - lo[0]) / span)), float(f32((q[1] - lo[1]) / span)),
             float(f32((q[2] - lo[2]) / span))]
        CL = 0.99999994
        cl = [min(max(v, 0.0), CL) for v in u]
        node, brick, blvl, last_cls = B["rootNode"], NO_BRICK, 0, 2
        for l in range(12):
            child, bptr, b = int(self.pool[node, 0]), int(self.pool[node, 1]), int(self.pool[node, 2])
            if bptr != NO_BRICK:
                brick, blvl = bptr, l
            last_cls = (b >> 16) & 3
            if last_cls != 2 or l >= cap or child == 0:
                break
            s = float(f32(2.0 ** (l + 1)))
            cx = int(math.floor(float(f32(cl[0] * s)))) & 1
            cy = int(math.floor(float(f32(cl[1] * s)))) & 1
            cz = int(math.floor(float(f32(cl[2] * s)))) & 1
            node = child + (cx | (cy << 1) | (cz << 2))
        side = 2.0 ** blvl
        dim, bpa = self.dim, self.bpa
        ox = STORED * (brick % bpa)
        oy = STORED * ((brick // bpa) % bpa)
        oz = STORED * (brick // (bpa * bpa))
        fr = [float(f32(cl[0] * side)) % 1.0, float(f32(cl[1] * side)) % 1.0,
              float(f32(cl[2] * side)) % 1.0]
        sx, sy, sz = 1 + fr[0] * 16, 1 + fr[1] * 16, 1 + fr[2] * 16
        ix = int(math.floor(sx - 0.5)); wx = sx - 0.5 - ix
        iy = int(math.floor(sy - 0.5)); wy = sy - 0.5 - iy
        iz = int(math.floor(sz - 0.5)); wz = sz - 0.5 - iz

        def at(x, y, z):
            if not (0 <= x <= 17 and 0 <= y <= 17 and 0 <= z <= 17):
                self.oob += 1
                x = min(max(x, 0), 17); y = min(max(y, 0), 17); z = min(max(z, 0), 17)
            return float(self.atlas[((oz + z) * dim + (oy + y)) * dim + (ox + x)])

        v = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    v += (at(ix + dx, iy + dy, iz + dz)
                          * (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
                          * (wz if dz else 1 - wz))
        band_l = band0 * 2.0 ** (-blvl)
        d = (v - 128.0) / 127.0 * band_l
        vsl = span / (16.0 * 2.0 ** blvl)
        if last_cls != 2:
            # FINEST-level floor (twin of the fixed jsSVODist/glslSVO/parity)
            vsf = span / (16.0 * 2.0 ** L)
            guar = band0 * 2.0 ** (-L) - 0.87 * vsf
            d = min(d + vsl, -guar) if last_cls == 1 else max(d - vsl, guar)
        if cl[0] != u[0] or cl[1] != u[1] or cl[2] != u[2]:
            hi = [lo[0] + span, lo[1] + span, lo[2] + span]
            e = [max(max(lo[i] - q[i], q[i] - hi[i]), 0.0) for i in range(3)]
            d = max(d, 0.0) + math.hypot(*e)
        return d


# ── scene pack loader (run_scene's setup, reused as data) ──────────────────
def load_pack(scene_path, svo_cls=RecSVO):
    o = json.loads(pathlib.Path(scene_path).read_text(encoding="utf-8"))
    sc = o.get("scene") if o.get("kind") == "trajectory" else o
    tr = o.get("trajectory") if o.get("kind") == "trajectory" else None
    td = tempfile.mkdtemp()
    out = pathlib.Path(td)
    r = subprocess.run(["node", str(_ROOT / "tools" / "dump_svo.node.js"),
                        str(scene_path), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"INFRASTRUCTURE: dump_svo failed: {r.stderr[:400]}", file=sys.stderr)
        sys.exit(2)
    svo = svo_cls(out)
    denses = []
    for e in svo.meta["eligible"]:
        leaf = sc["csg_leaves"][e["leaf"]]
        sh = dict(e)
        sh["_sdf_b64"] = leaf["shape"].get("sdf_b64") or \
            sc["csg_leaves"][leaf["shape"]["sdf_ref"]]["shape"]["sdf_b64"]
        denses.append(vsp.Dense(sh))
    poses = {}
    if tr and tr.get("frames"):
        for k, b in enumerate(tr["frames"][0]["bodies"]):
            poses[k] = (b["pos"], b.get("quat") or [0, 0, 0, 1])
    for e in svo.meta["eligible"]:
        poses.setdefault(e["body"], ([0, 0, 0], [0, 0, 0, 1]))
    return sc, svo, denses, poses


def make_sdfs(svo, denses, poses):
    def local(row, p):
        e = svo.meta["eligible"][row]
        pos, quat = poses[e["body"]]
        q = vsp.qrot_inv(quat, [p[0] - pos[0], p[1] - pos[1], p[2] - pos[2]])
        return q, e["pivot"]

    def sdf_dense(p):
        best, mat = 1e30, -1
        for row, dn in enumerate(denses):
            q, piv = local(row, p)
            d = dn.d([q[0] + piv[0], q[1] + piv[1], q[2] + piv[2]])
            if d < best:
                best, mat = d, row
        return best, mat

    def sdf_svo(p, cap=None):
        best, mat = 1e30, -1
        for row in range(len(svo.bodies)):
            q, _ = local(row, p)
            d = svo.d(row, q, cap)
            if d < best:
                best, mat = d, row
        return best, mat
    return sdf_dense, sdf_svo, local


def field_scan(fld, eye, rd, t_ref, fine_vs, eps):
    lo_t, hi_t = max(eps, t_ref - 40 * fine_vs), t_ref + 40 * fine_vs
    prev, crossed, mn = None, False, 1e30
    for s in range(801):
        t = lo_t + (hi_t - lo_t) * s / 800
        p = [eye[0] + rd[0] * t, eye[1] + rd[1] * t, eye[2] + rd[2] * t]
        d, _ = fld(p)
        mn = min(mn, abs(d))
        if prev is not None and (d < 0) != (prev < 0):
            crossed = True
        prev = d
    return crossed, mn


# ── A1: grazing-silhouette fans ────────────────────────────────────────────
def attack_silhouette(cam_id, slug, nfan=7):
    print(f"[A1 {cam_id}] grazing-silhouette fans")
    sc, svo, denses, poses = load_pack(_DATA / f"{slug}.json")
    sdf_dense, sdf_svo, _ = make_sdfs(svo, denses, poses)
    man = json.loads(_MANIFEST.read_text())
    cam = man["baselines"][cam_id]["cam"]
    up = (sc.get("camera") or {}).get("up") or [0, 0, 1]
    fov = (sc.get("camera") or {}).get("fov_deg") or 40
    nx, ny = 64, 36
    eye, rays, mkray = vsp.rays_of(cam, up, fov, nx, ny)
    R = max(cam[2], 1e-9)
    eps = max(5e-7, R * 0.0015)
    max_dist = cam[2] * 4 + R * 3
    fine_vs = min(d.vs for d in denses)
    quant_tol = 2.0 * denses[0].band / 255.0

    outcome = []
    for j in range(ny):
        row = []
        for i in range(nx):
            t, _ = vsp.march(sdf_dense, eye, rays[j * nx + i], eps, max_dist)
            row.append(t is not None)
        outcome.append(row)
    sil = []
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx and outcome[j][i] != outcome[j][i + 1]:
                sil.append((i, j))
            elif j + 1 < ny and outcome[j][i] != outcome[j + 1][i]:
                sil.append((i, j))
    sil = sil[:40]
    tested = flips = hard_n = 0
    for (i, j) in sil:
        for a in range(nfan):
            for b in range(nfan):
                dx = (a + 0.5) / nfan - 0.5
                dy = (b + 0.5) / nfan - 0.5
                rd = mkray(i, j, dx, dy)
                t_d, m_d = vsp.march(sdf_dense, eye, rd, eps, max_dist)
                t_s, m_s = vsp.march(lambda p: sdf_svo(p), eye, rd, eps, max_dist)
                tested += 1
                if (t_d is None) != (t_s is None):
                    flips += 1
                    t_ref = t_d if t_d is not None else t_s
                    hit_fld = sdf_dense if t_d is not None else (lambda p: sdf_svo(p))
                    miss_fld = (lambda p: sdf_svo(p)) if t_d is not None else sdf_dense
                    hx, hmn = field_scan(hit_fld, eye, rd, t_ref, fine_vs, eps)
                    mx, mmn = field_scan(miss_fld, eye, rd, t_ref, fine_vs, eps)
                    if abs(hmn - mmn) <= quant_tol or (hx and (mx or mmn <= quant_tol)):
                        pass                          # marcher-phase artifact, fields agree
                    else:
                        hard_n += 1
                        hard(f"{cam_id} silhouette ray ({i},{j})+({dx:.2f},{dy:.2f}): "
                             f"one-sided field divergence (hit min|d| {hmn:.3e}, "
                             f"miss min|d| {mmn:.3e}, tol {quant_tol:.3e})")
                elif t_d is not None and m_d != m_s:
                    hard_n += 1
                    hard(f"{cam_id} silhouette ray ({i},{j}): material {m_d} vs {m_s}")
    print(f"  {len(sil)} silhouette px, {tested} fan rays, {flips} outcome flips "
          f"(all field-verified), {hard_n} hard")


# ── A2: rays starting inside bodies ────────────────────────────────────────
def attack_inside(cam_id, slug, ndirs=48):
    print(f"[A2 {cam_id}] inside-body rays")
    sc, svo, denses, poses = load_pack(_DATA / f"{slug}.json")
    sdf_dense, sdf_svo, local = make_sdfs(svo, denses, poses)
    fine_vs = min(d.vs for d in denses)
    quant_tol = 2.0 * denses[0].band / 255.0
    hard_n = interior_checked = 0
    worst_sign = 0.0
    for row, dn in enumerate(denses):
        e = svo.meta["eligible"][row]
        pos, quat = poses[e["body"]]
        # find an interior point: scan lattice for the most-negative dense cell
        piv = e["pivot"]
        best_q, best_d = None, 1e30
        for s in range(600):
            q = [e["center"][i] + (((s * f) % 1.0) - 0.5) * e["dims"][i] * e["voxel_size"] * 0.8
                 for i, f in enumerate((0.6180339887, 0.7548776662, 0.5698402910))]
            d = dn.d(q)
            if d < best_d:
                best_d, best_q = d, q
        if best_d >= 0:
            note(f"{cam_id} body {e['body']}: no interior lattice point found (thin body)")
            continue
        # interior sign parity of svoDist itself
        ds = svo.d(row, [best_q[0] - piv[0], best_q[1] - piv[1], best_q[2] - piv[2]])
        interior_checked += 1
        if ds >= 0:
            hard(f"{cam_id} body {e['body']}: interior point dense {best_d:.3e} "
                 f"but svoDist {ds:.3e} (sign flip)")
        worst_sign = max(worst_sign, abs(ds - best_d))
        # world-space interior origin (invert the local transform: p = rot(qloc)+pos)
        def qrot(qt, v):
            ux, uy, uz, w = qt
            tx, ty, tz = 2 * (uy * v[2] - uz * v[1]), 2 * (uz * v[0] - ux * v[2]), 2 * (ux * v[1] - uy * v[0])
            return [v[0] + w * tx + (uy * tz - uz * ty),
                    v[1] + w * ty + (uz * tx - ux * tz),
                    v[2] + w * tz + (ux * ty - uy * tx)]
        ro = [c + o for c, o in zip(qrot(quat, best_q), pos)]
        eps = max(5e-7, fine_vs * 0.5)
        max_dist = max(e["dims"][i] * e["voxel_size"] for i in range(3)) * 4
        for k in range(ndirs):
            th = math.acos(1 - 2 * ((k * 0.6180339887) % 1.0))
            ph = 2 * math.pi * ((k * 0.7548776662) % 1.0)
            rd = [math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)]
            t_d, m_d = vsp.march(sdf_dense, ro, rd, eps, max_dist)
            t_s, m_s = vsp.march(lambda p: sdf_svo(p), ro, rd, eps, max_dist)
            if (t_d is None) != (t_s is None):
                t_ref = t_d if t_d is not None else t_s
                hit_fld = sdf_dense if t_d is not None else (lambda p: sdf_svo(p))
                miss_fld = (lambda p: sdf_svo(p)) if t_d is not None else sdf_dense
                hx, hmn = field_scan(hit_fld, ro, rd, t_ref, fine_vs, eps)
                mx, mmn = field_scan(miss_fld, ro, rd, t_ref, fine_vs, eps)
                if abs(hmn - mmn) <= quant_tol or (hx and (mx or mmn <= quant_tol)):
                    pass
                else:
                    hard_n += 1
                    hard(f"{cam_id} body {e['body']} inside-out dir {k}: one-sided "
                         f"field divergence")
            elif t_d is not None and abs(t_d - t_s) > 2 * fine_vs:
                note(f"{cam_id} body {e['body']} inside-out dir {k}: exit-t delta "
                     f"{abs(t_d - t_s):.3e} m (> 2 fine cells)")
    print(f"  {interior_checked} interior points checked, worst interior |svo-dense| "
          f"{worst_sign:.3e} m, {hard_n} hard")


# ── A3: LOD pins — overshoot vs the documented ~1-cell-at-cap bound ───────
def attack_lod(cam_id, slug, nfuzz=6000):
    print(f"[A3 {cam_id}] LOD pins: conservative bound at capped depth")
    sc, svo, denses, poses = load_pack(_DATA / f"{slug}.json")
    for row, dn in enumerate(denses):
        e = svo.meta["eligible"][row]
        piv = e["pivot"]
        L = svo.bodies[row]["L"]
        span = svo.bodies[row]["rootSpan"]
        ext = [e["dims"][i] * e["voxel_size"] for i in range(3)]
        for cap in range(max(0, L - 2), L):
            vs_cap = span / (16.0 * 2 ** cap)
            worst, worst_q, nbad = 0.0, None, 0
            for s in range(nfuzz):
                q = [e["center"][i] + (((s * f) % 1.0) - 0.5) * ext[i] * 1.3
                     for i, f in enumerate((0.6180339887, 0.7548776662, 0.5698402910))]
                dd = dn.d(q)
                ds = svo.d(row, [q[0] - piv[0], q[1] - piv[1], q[2] - piv[2]], cap)
                over = ds - dd
                if over > vs_cap * 1.05:
                    nbad += 1
                    if over > worst:
                        worst, worst_q = over, q
            tag = f"{cam_id} body {e['body']} cap={cap} (L={L})"
            if nbad:
                hard(f"{tag}: {nbad}/{nfuzz} pts exceed dense by > 1.05 cell-at-cap "
                     f"(worst overshoot {worst:.3e} m = {worst / vs_cap:.2f} cells-at-cap) "
                     f"— sphere-trace overshoot at pinned LOD; suspect the uniform-"
                     f"fallback floor guar=band0*2^-cap-0.87*vs_cap (fine-level "
                     f"guarantee scaled by 2^(L-cap))")
            else:
                print(f"  {tag}: overshoot bound holds over {nfuzz} pts (cell {vs_cap:.3e} m)")


# ── A4: brick borders, root faces, clamp corner, in-brick read assert ──────
def attack_borders(cam_id, slug):
    print(f"[A4 {cam_id}] brick borders / root faces / clamp corner")
    sc, svo, denses, poses = load_pack(_DATA / f"{slug}.json")
    worst_seam = 0.0
    for row, dn in enumerate(denses):
        B = svo.bodies[row]
        span = B["rootSpan"]
        L = B["L"]
        nseams = 16 * 2 ** L // 16                    # brick-boundary planes per axis
        for axis in range(3):
            for k in range(1, min(nseams, 8)):
                plane = span * k / min(nseams, 8)
                for s in range(150):
                    q = [B["lo"][i] + ((s * f) % 1.0) * span
                         for i, f in enumerate((0.6180339887, 0.7548776662, 0.5698402910))]
                    q[axis] = B["lo"][axis] + plane - 1e-9 * span
                    a = svo.d(row, q)
                    q[axis] = B["lo"][axis] + plane + 1e-9 * span
                    b = svo.d(row, q)
                    worst_seam = max(worst_seam, abs(a - b))
        # root faces + corners + clamp corner probes
        for s in range(400):
            base = [B["lo"][i] + ((s * f) % 1.0) * span
                    for i, f in enumerate((0.6180339887, 0.7548776662, 0.5698402910))]
            for axis in range(3):
                for off in (0.0, 1e-9 * span, -1e-9 * span, span, span - 1e-9 * span,
                            span + 1e-9 * span, span * 0.99999994):
                    q = list(base)
                    q[axis] = B["lo"][axis] + off
                    svo.d(row, q)                     # RecSVO asserts in-brick reads
        e = svo.meta["eligible"][row]
        piv = e["pivot"]
        # C0 across the root faces vs dense (outside-branch handoff)
        worst_face = 0.0
        for s in range(300):
            base = [B["lo"][i] + ((s * f) % 1.0) * span
                    for i, f in enumerate((0.6180339887, 0.7548776662, 0.5698402910))]
            for axis in range(3):
                for side_ in (0, 1):
                    q = list(base)
                    q[axis] = B["lo"][axis] + (span if side_ else 0.0)
                    qa = list(q); qa[axis] -= 1e-7 * span
                    qb = list(q); qb[axis] += 1e-7 * span
                    worst_face = max(worst_face, abs(svo.d(row, qa) - svo.d(row, qb)))
    if svo.oob:
        hard(f"{cam_id}: {svo.oob} atlas reads OUTSIDE the 18^3 stored brick "
             f"(CLAMP_TO_EDGE would bleed into a neighboring brick on GPU)")
    else:
        print(f"  all atlas reads in-brick (CLAMP_TO_EDGE never bites); "
              f"worst brick-seam step {worst_seam:.3e} m; worst root-face step {worst_face:.3e} m")


# ── A5: degenerate synthetic bodies through the real builder ───────────────
def synth_shape(dims, vs, fn, center=(0.0, 0.0, 0.0)):
    nx, ny, nz = dims
    band = 6.0 * vs
    i = np.arange(nx); j = np.arange(ny); k = np.arange(nz)
    X = center[0] + (i - (nx - 1) / 2) * vs
    Y = center[1] + (j - (ny - 1) / 2) * vs
    Z = center[2] + (k - (nz - 1) / 2) * vs
    gx, gy, gz = np.meshgrid(X, Y, Z, indexing="ij")     # [i][j][k]
    d = fn(gx, gy, gz)
    q = np.clip(np.round(d / band * 127), -127, 127).astype(np.int8)
    # pack x-fastest: flat[(k*ny + j)*nx + i] — transpose to [k][j][i]
    flat = np.ascontiguousarray(q.transpose(2, 1, 0)).reshape(-1)
    return {"type": "Voxel", "dims": list(dims), "voxel_size": vs, "band": band,
            "center": list(center), "sdf_b64": base64.b64encode(flat.tobytes()).decode()}


def synth_scene(shapes_mats):
    leaves, bodies = [], []
    for n, (sh, mat) in enumerate(shapes_mats):
        leaves.append({"shape": sh, "material": mat, "op": "add", "body": n})
        bodies.append({"pivot": [0, 0, 0]})
    return {"csg_leaves": leaves, "bodies": bodies,
            "materials": {m: {} for _, m in shapes_mats}}


def attack_synthetic():
    print("[A5] degenerate synthetic bodies (real builder via dump_svo)")
    sph = lambda r: (lambda x, y, z: np.sqrt(x * x + y * y + z * z) - r)
    off_sph = lambda r, c: (lambda x, y, z: np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) - r)
    box = lambda h: (lambda x, y, z: np.maximum(np.maximum(np.abs(x) - h[0], np.abs(y) - h[1]), np.abs(z) - h[2]))
    cases = [
        ("flush16_sphere", synth_shape((16, 16, 16), 0.05, sph(0.30))),        # pad=0, band clipped at faces
        ("tiny_2x3x2", synth_shape((2, 3, 2), 0.05, sph(0.04))),
        ("odd17_sphere", synth_shape((17, 17, 17), 0.05, sph(0.30))),          # L=1, asymmetric pad
        ("flushleft63", synth_shape((63, 63, 63), 0.02, off_sph(0.25, (-0.4, 0, 0)))),  # pad 0 left / 1 right
        ("rod64x4x4", synth_shape((64, 4, 4), 0.05, box((1.5, 0.08, 0.08)))),  # x flush both ends
        ("huge_vs", synth_shape((24, 24, 24), 1.0e6, sph(8.0e6))),
        ("tiny_vs", synth_shape((24, 24, 24), 1.0e-6, sph(8.0e-6))),
    ]
    any_hard = 0
    for name, sh in cases:
        with tempfile.TemporaryDirectory() as td:
            sp = pathlib.Path(td) / "scene.json"
            sp.write_text(json.dumps(synth_scene([(sh, "m0")])), encoding="utf-8")
            sc, svo, denses, poses = load_pack(sp)
        dn = denses[0]
        e = svo.meta["eligible"][0]
        vs = e["voxel_size"]; band = e["band"]
        ext = [e["dims"][i] * vs for i in range(3)]
        quant_tol = 2.0 * band / 255.0
        # NEAR region (<= 1.1x extent, the shipped drop-in territory): any
        # conservative violation is HARD. FAR region (out to 1.5x): the two
        # kernels use DIFFERENT outside-box continuations (dense: grid-clamped
        # cell + direct Euclidean; svo: cube-face field + cube Euclidean — a
        # triangle-inequality detour, provably svo >= dense there, so never a
        # phantom hit) — measured + documented, march parity cross-checked
        # separately (see the rod fan below).
        worst_in, near_bad, near_worst = 0.0, 0, 0.0
        far_bad, far_worst, under_far = 0, 0.0, 0
        for s in range(20000):
            mult = 1.1 if s % 2 == 0 else 1.5
            q = [e["center"][i] + (((s * f) % 1.0) - 0.5) * ext[i] * mult
                 for i, f in enumerate((0.6180339887, 0.7548776662, 0.5698402910))]
            dd = dn.d(q)
            ds = svo.d(0, q)
            if abs(dd) < band:
                worst_in = max(worst_in, abs(ds - dd))
            over = ds - dd
            near = all(abs(q[i] - e["center"][i]) <= ext[i] * 0.55 for i in range(3))
            if over > vs * 1.05:
                if near:
                    near_bad += 1; near_worst = max(near_worst, over)
                else:
                    far_bad += 1; far_worst = max(far_worst, over)
            if not near and over < -vs * 1.05:
                under_far += 1                        # svo SHORTER than dense: the
                #                                       SAFE direction (both fields
                #                                       > 0 far outside — no zero
                #                                       crossing, march just steps
                #                                       shorter); measured, not hard
        print(f"  {name}: in-band max|svo-dense| {worst_in:.3e} "
              f"({worst_in / (band / 127):.1f} LSB), NEAR violations {near_bad} "
              f"(worst {near_worst / vs:.2f} cells), FAR overshoot {far_bad} pts "
              f"(worst {far_worst / vs:.2f} cells), FAR undershoot {under_far} pts (safe dir)")
        if near_bad:
            any_hard += 1
            hard(f"A5 {name}: {near_bad} NEAR-region pts exceed dense by > 1.05 fine "
                 f"cells at matched LOD (worst {near_worst / vs:.2f} cells) — drop-in "
                 f"claim broken for this payload class")
        if far_bad:
            note(f"A5 {name}: far-outside continuation detour, {far_bad} pts up to "
                 f"{far_worst / vs:.2f} cells above dense (cube-vs-grid box "
                 f"decomposition; march-outcome parity cross-checked by the rod fan)")
        if svo.oob:
            hard(f"A5 {name}: {svo.oob} out-of-brick atlas reads")
    # payload-shared twin bodies (dedupe + material identity)
    sh = synth_shape((24, 24, 24), 0.05, sph(0.4))
    sc2 = synth_scene([(sh, "matA"), (json.loads(json.dumps(sh)), "matB")])
    sc2["csg_leaves"][1]["shape"] = {"type": "Voxel", "sdf_ref": 0,
                                     **{k: sh[k] for k in ("dims", "voxel_size", "band", "center")}}
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "scene.json"
        sp.write_text(json.dumps(sc2), encoding="utf-8")
        sc, svo, denses, poses = load_pack(sp)
    st = svo.meta["stats"]
    if st["trees"] != 1:
        hard(f"A5 twins: dedupe failed — {st['trees']} trees for identical payloads")
    if svo.bodies[0]["matId"] == svo.bodies[1]["matId"]:
        hard("A5 twins: distinct materials collapsed to one palette id")
    else:
        print(f"  twins: 1 tree, 2 bodies, palette ids "
              f"{svo.bodies[0]['matId']}/{svo.bodies[1]['matId']} — dedupe + material identity OK")
    # rod march-outcome fan: does the far-outside detour break hit/miss parity?
    box = lambda h: (lambda x, y, z: np.maximum(np.maximum(np.abs(x) - h[0], np.abs(y) - h[1]), np.abs(z) - h[2]))
    sh = synth_shape((64, 4, 4), 0.05, box((1.5, 0.08, 0.08)))
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "scene.json"
        sp.write_text(json.dumps(synth_scene([(sh, "m0")])), encoding="utf-8")
        sc, svo, denses, poses = load_pack(sp)
    sdf_dense, sdf_svo, _ = make_sdfs(svo, denses, poses)
    dn = denses[0]
    fine_vs = dn.vs
    quant_tol = 2.0 * dn.band / 255.0
    eps = fine_vs * 0.5
    tested = flips = hard_n = 0
    for a in range(24):
        th = 2 * math.pi * a / 24
        eye = [4.0 * math.cos(th), 4.0 * math.sin(th), 0.9]
        for tx in range(40):
            tgt = [-1.7 + 3.4 * tx / 39, 0.081 * math.sin(7 * tx), 0.079 * math.cos(5 * tx)]
            Lr = math.dist(eye, tgt)
            rd = [(tgt[i] - eye[i]) / Lr for i in range(3)]
            t_d, m_d = vsp.march(sdf_dense, eye, rd, eps, 12.0)
            t_s, m_s = vsp.march(lambda p: sdf_svo(p), eye, rd, eps, 12.0)
            tested += 1
            if (t_d is None) != (t_s is None):
                flips += 1
                t_ref = t_d if t_d is not None else t_s
                hf = sdf_dense if t_d is not None else (lambda p: sdf_svo(p))
                mf = (lambda p: sdf_svo(p)) if t_d is not None else sdf_dense
                hx, hmn = field_scan(hf, eye, rd, t_ref, fine_vs, eps)
                mx, mmn = field_scan(mf, eye, rd, t_ref, fine_vs, eps)
                if not (abs(hmn - mmn) <= quant_tol or (hx and (mx or mmn <= quant_tol))):
                    hard_n += 1
                    hard(f"A5 rod fan: one-sided field divergence on grazing ray "
                         f"eye={eye} tgt={tgt}")
    print(f"  rod grazing/end-on fan: {tested} rays, {flips} outcome flips "
          f"(field-verified), {hard_n} hard — far detour does not break march parity"
          if hard_n == 0 else
          f"  rod grazing/end-on fan: {tested} rays, {flips} flips, {hard_n} HARD")
    return any_hard


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # cp1252 consoles choke on Δ/—
    except Exception:
        pass
    for cam_id, slug in (("vox_chair", "vox_chair"),
                         ("solar_earth", "solar_system_theater")):
        attack_silhouette(cam_id, slug)
        attack_inside(cam_id, slug)
        attack_lod(cam_id, slug)
        attack_borders(cam_id, slug)
    attack_synthetic()
    print()
    if HARD:
        print(f"ADVERSARIAL RESULT: {len(HARD)} HARD finding(s)")
        for h in HARD:
            print("  - " + h)
        return 1
    print("ADVERSARIAL RESULT: no hard findings "
          f"({len(NOTES)} documented notes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
