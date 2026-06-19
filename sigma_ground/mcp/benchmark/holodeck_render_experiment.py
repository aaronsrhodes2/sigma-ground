"""Holodeck render experiment — physics scenarios → Materia → entangler renders,
plus a scorecard of how well the local model (qwen2.5:7b) routes each one.

For every motion scenario (PIRA Mechanics family + our renderable verbs):
  1. Route it TWO ways:
       - deterministic translator (use_qwen=False) → the reliable spec that
         actually drives the render;
       - the local model, forced (translator._qwen_translate) → the system under
         test, scored against the expected verb.
  2. Run the deterministic spec → outputs + render_handle.
  3. Bridge the handle → entangler still frames: the sphere in its true cold
     material colour, and — when the scenario's own physics says it gets hot —
     glowing at that computed peak temperature.
  4. Tally: did the local model route correctly? did it render? did it glow?

Writes a dark-theme gallery.html (every still + its physics caption) and a
scorecard (markdown + json). Run:

    python -m sigma_ground.mcp.benchmark.holodeck_render_experiment
"""
from __future__ import annotations

import html
import json
import os
import time

from sigma_ground.materia import translate, run_spec, translator as _t
from sigma_ground.radiance.entangler_scene import render_handle_gallery

# ── Scenario catalog ─────────────────────────────────────────────────────────
# Each: (id, question, expected_primary_verb, tag). Phrasings hit the
# deterministic triggers so the render is reliable; the local model is scored
# on whether it picks the same primary scenario. PIRA Mechanics areas covered:
# 1D-30 free fall, 1D-40 air resistance / terminal velocity, 1E-30 reentry,
# 1D-52 projectile, 1C-20 launch-and-return.
CATALOG = [
    # — plain drops: the emergent COLD material-colour showcase (true n+k) —
    ("fall_copper", "drop a copper ball from 10 km — how fast does it hit?", "terminal_velocity_drop", "free-fall"),
    ("fall_gold",   "drop a gold ball from 5 km — how fast does it land?", "terminal_velocity_drop", "free-fall"),
    ("fall_alum",   "drop an aluminum ball from 8 km — impact speed?", "terminal_velocity_drop", "free-fall"),
    ("fall_lead",   "drop a lead sphere from 3 km — terminal velocity?", "terminal_velocity_drop", "terminal-v"),
    ("fall_silver", "drop a silver ball from 6 km — how fast at the ground?", "terminal_velocity_drop", "free-fall"),
    ("fall_titan",  "drop a titanium ball from 12 km — impact speed?", "terminal_velocity_drop", "free-fall"),
    ("fall_plat",   "drop a platinum sphere from 7 km — how fast does it hit?", "terminal_velocity_drop", "free-fall"),
    ("fall_alum2",  "drop an aluminum ball from 20 km — terminal velocity?", "terminal_velocity_drop", "terminal-v"),

    # — drag-heating drops: the GLOW showcase (sphere glows if the physics says so) —
    ("heat_iron40", "does an iron sphere get hot falling from 40 km?", "drag_heating_drop", "drag-heat"),
    ("heat_tung60", "does a tungsten ball heat up dropping from 60 km?", "drag_heating_drop", "drag-heat"),
    ("heat_nick50", "does a nickel sphere get hot falling from 50 km?", "drag_heating_drop", "drag-heat"),
    ("heat_lead30", "does a lead ball get hot dropping from 30 km?", "drag_heating_drop", "drag-heat"),
    ("heat_iron70", "does an iron sphere heat up falling from 70 km?", "drag_heating_drop", "drag-heat"),
    ("heat_copper", "does a copper ball get hot falling from 30 km?", "drag_heating_drop", "drag-heat"),

    # — high-altitude descent / reentry —
    ("descent_sky", "a skydiver jumps from 38 km — does he slow down after going supersonic?", "high_altitude_descent", "reentry"),
    ("descent_80",  "drop a payload from 80 km — does it decelerate as the air thickens?", "high_altitude_descent", "reentry"),
    ("descent_120", "a capsule reenters from 120 km — does it slow after going supersonic?", "high_altitude_descent", "reentry"),

    # — supersonic projectile (transonic deceleration) —
    ("slug_m25", "how far does a Mach 2.5 bullet travel before it goes subsonic?", "supersonic_projectile", "supersonic"),
    ("slug_m4",  "a Mach 4 tungsten slug — how far to subsonic?", "supersonic_projectile", "supersonic"),
    ("slug_m3",  "how far does a Mach 3 steel slug travel before slowing to subsonic?", "supersonic_projectile", "supersonic"),
    ("slug_m5",  "a Mach 5 tungsten slug — distance to subsonic?", "supersonic_projectile", "supersonic"),

    # — vertical launch (and launch→fall→heat chains) —
    ("launch_steel", "throw a steel ball straight up at 300 m/s — how high does it go?", "vertical_launch", "launch"),
    ("launch_iron",  "hurl an iron ball up at 500 m/s — how hot is it when it comes down?", "vertical_launch", "launch"),
    ("launch_copper","throw a copper ball straight up at 200 m/s — how high?", "vertical_launch", "launch"),
    ("launch_gold",  "fling a gold ball up at 150 m/s — what apex altitude?", "vertical_launch", "launch"),

    # — more drops / heating: extra material + glow variety —
    ("heat_tung80",  "does a tungsten ball heat up dropping from 80 km?", "drag_heating_drop", "drag-heat"),
    ("heat_iron100", "does an iron sphere get hot falling from 100 km?", "drag_heating_drop", "drag-heat"),
    ("fall_nickel",  "drop a nickel ball from 9 km — impact speed?", "terminal_velocity_drop", "free-fall"),
    ("fall_copper2", "drop a copper ball from 25 km — terminal velocity?", "terminal_velocity_drop", "terminal-v"),
    ("launch_alum",  "throw an aluminum ball up at 250 m/s — how high?", "vertical_launch", "launch"),
    ("slug_m6",      "a Mach 6 tungsten slug — distance to subsonic?", "supersonic_projectile", "supersonic"),

    # — named objects: now RENDERED via the Deckard→entangler CSG converter —
    # (real CSG shape sampled off its SDF surface, with per-part materials) —
    ("obj_anvil",    "drop an anvil from 1 km — what's its terminal speed?", "drop_object", "named-object"),
    ("obj_hammer",   "drop a hammer from 100 m — how fast does it land?", "drop_object", "named-object"),
    ("obj_dumbbell", "drop a dumbbell from 200 m — terminal velocity?", "drop_object", "named-object"),
    ("obj_skillet",  "drop a cast iron skillet from 50 m — impact speed?", "drop_object", "named-object"),
    ("obj_cup",      "drop a coffee cup from 10 m — how fast does it land?", "drop_object", "named-object"),
    ("obj_kettle",   "drop an electric kettle from 30 m — terminal speed?", "drop_object", "named-object"),
    ("obj_feather",  "drop a feather from 2 m — how fast does it land?", "drop_object", "named-object"),
]


def _merge_outputs(results):
    out = {}
    for r in results:
        out.update(r.outputs or {})
    return out


def _det_route_and_run(q):
    """Deterministic route + run → (verbs, outputs, handle, summary). Reliable."""
    spec = translate(q, use_qwen=False)
    verbs = [s.verb for s in spec.steps]
    if not verbs:
        return [], {}, None, "", "deterministic"
    results = run_spec(spec)
    out = _merge_outputs(results)
    handle = next((r.outputs.get("render_handle") for r in results
                   if r.outputs.get("can_render") and r.outputs.get("render_handle")),
                  None)
    summary = " ".join(r.summary for r in results if getattr(r, "summary", None))
    return verbs, out, handle, summary, spec.source


def _qwen_route(q):
    """Force the local model to route (bypasses the deterministic shortcut).
    Returns the model's primary verb, or 'none'/'error'."""
    try:
        spec = _t._qwen_translate(q)
        if spec is None or not spec.steps:
            return "none"
        return spec.steps[0].verb
    except Exception as e:
        return f"error:{type(e).__name__}"


def run_experiment(out_dir=None, px=220):
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")     # nice glyphs on Windows too
    except Exception:
        pass
    here = os.path.dirname(__file__)
    out_dir = out_dir or os.path.join(here, "holodeck_out")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    print(f"Holodeck render experiment - {len(CATALOG)} scenarios\n" + "=" * 64)
    for sid, q, expected, tag in CATALOG:
        t0 = time.time()
        verbs, outputs, handle, summary, source = _det_route_and_run(q)
        det_primary = verbs[0] if verbs else "none"

        gallery = {"slug": sid, "renderable": False, "stills": [], "note": "no render_handle"}
        if handle:
            gallery = render_handle_gallery(handle, outputs, {"T": 288.15},
                                            img_dir, sid, px=px)

        qwen_primary = _qwen_route(q)
        qwen_ok = (qwen_primary == expected)
        det_ok = (det_primary == expected)
        peak_T = outputs.get("peak_T_K")
        glow = any(s.get("glowing") for s in gallery.get("stills", []))

        row = {
            "id": sid, "question": q, "tag": tag, "expected": expected,
            "det_primary": det_primary, "det_ok": det_ok,
            "qwen_primary": qwen_primary, "qwen_ok": qwen_ok,
            "renderable": gallery.get("renderable", False),
            "n_stills": len(gallery.get("stills", [])),
            "stills": gallery.get("stills", []),
            "material": gallery.get("material") or (handle or {}).get("material_key"),
            "peak_T_K": peak_T, "glow": glow,
            "impact_speed_m_s": outputs.get("impact_speed_m_s") or outputs.get("max_speed_m_s"),
            "summary": summary, "render_note": gallery.get("note"),
            "elapsed_s": round(time.time() - t0, 2),
        }
        rows.append(row)
        flag = "OK " if qwen_ok else "XX "
        rkind = (f"{row['n_stills']} stills" + (" (GLOW)" if glow else "")
                 if row["renderable"] else "-")
        print(f"  {sid:14s} det={det_primary:24s} qwen={qwen_primary:24s} "
              f"{flag} render={rkind}")

    _write_scorecard(rows, out_dir)
    _write_gallery(rows, out_dir)
    return rows, out_dir


def _write_scorecard(rows, out_dir):
    n = len(rows)
    det_ok = sum(r["det_ok"] for r in rows)
    qwen_ok = sum(r["qwen_ok"] for r in rows)
    rendered = sum(r["renderable"] for r in rows)
    glowing = sum(r["glow"] for r in rows)
    n_stills = sum(r["n_stills"] for r in rows)

    lines = [
        "# Holodeck Render Experiment — Scorecard", "",
        f"- Scenarios: **{n}**",
        f"- Local-model (qwen2.5:7b) routing correct: **{qwen_ok}/{n} = {100*qwen_ok/n:.0f}%**",
        f"- Deterministic routing correct (baseline): **{det_ok}/{n} = {100*det_ok/n:.0f}%**",
        f"- Rendered by the entangler: **{rendered}/{n}** ({n_stills} stills)",
        f"- Scenarios that glow (peak ≥ Draper 700 K): **{glowing}**", "",
        "| id | tag | expected | qwen routed | ✓ | render | peak T (K) | glow |",
        "|----|-----|----------|-------------|---|--------|-----------|------|",
    ]
    for r in rows:
        pk = f"{r['peak_T_K']:.0f}" if isinstance(r["peak_T_K"], (int, float)) else "—"
        rend = (f"{r['n_stills']}" if r["renderable"] else "gap")
        lines.append(
            f"| {r['id']} | {r['tag']} | {r['expected']} | {r['qwen_primary']} | "
            f"{'✓' if r['qwen_ok'] else '✗'} | {rend} | {pk} | "
            f"{'🔥' if r['glow'] else ''} |")
    with open(os.path.join(out_dir, "scorecard.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "scorecard.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _write_gallery(rows, out_dir):
    cards = []
    for r in rows:
        imgs = "".join(
            f'<figure><img src="images/{html.escape(s["file"])}" alt="{html.escape(s.get("label") or "")}">'
            f'<figcaption>{html.escape(s.get("label") or "")} · {s["T_K"]} K'
            f'{" · 🔥" if s.get("glowing") else ""}</figcaption></figure>'
            for s in r["stills"])
        if not imgs:
            imgs = (f'<div class="gap">no entangler render — {html.escape(str(r.get("render_note") or ""))}</div>')
        pk = f'{r["peak_T_K"]:.0f} K' if isinstance(r["peak_T_K"], (int, float)) else "—"
        spd = (f'{r["impact_speed_m_s"]:.0f} m/s'
               if isinstance(r["impact_speed_m_s"], (int, float)) else "—")
        badge = ("ok" if r["qwen_ok"] else "bad")
        cards.append(f"""
    <section class="card">
      <h2>{html.escape(r['question'])}</h2>
      <div class="imgs">{imgs}</div>
      <div class="meta">
        <span class="tag">{html.escape(r['tag'])}</span>
        <span>material: <b>{html.escape(str(r['material']))}</b></span>
        <span>peak&nbsp;T: <b>{pk}</b></span>
        <span>impact: <b>{spd}</b></span>
        <span>expected: <code>{html.escape(r['expected'])}</code></span>
        <span class="route {badge}">qwen → <code>{html.escape(r['qwen_primary'])}</code> {'✓' if r['qwen_ok'] else '✗'}</span>
      </div>
      <p class="sum">{html.escape(r['summary'][:240])}</p>
    </section>""")

    n = len(rows)
    qwen_ok = sum(r["qwen_ok"] for r in rows)
    glowing = sum(r["glow"] for r in rows)
    rendered = sum(r["renderable"] for r in rows)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Holodeck Render Gallery</title><style>
  body {{ background:#0a0a0c; color:#e8e8ea; font:15px/1.5 system-ui,sans-serif; margin:0; padding:24px; }}
  h1 {{ font-weight:650; }}
  .summary {{ color:#9aa; margin-bottom:20px; }}
  .summary b {{ color:#6ec1ff; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:18px; }}
  .card {{ background:#141418; border:1px solid #222; border-radius:12px; padding:14px; }}
  .card h2 {{ font-size:15px; font-weight:550; margin:0 0 10px; color:#fff; }}
  .imgs {{ display:flex; gap:6px; flex-wrap:wrap; }}
  figure {{ margin:0; text-align:center; }}
  img {{ width:120px; height:120px; border-radius:8px; background:#000; image-rendering:auto; }}
  figcaption {{ font-size:11px; color:#99a; margin-top:3px; }}
  .gap {{ color:#c97; font-size:13px; padding:24px 8px; border:1px dashed #533; border-radius:8px; }}
  .meta {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; font-size:12px; color:#aab; }}
  .meta b {{ color:#dde; }} code {{ color:#8fd; }}
  .tag {{ background:#1d2733; color:#6ec1ff; padding:1px 8px; border-radius:10px; }}
  .route.ok {{ color:#7ee787; }} .route.bad {{ color:#ff7b72; }}
  .sum {{ color:#778; font-size:12px; margin:8px 0 0; }}
</style></head><body>
  <h1>Holodeck Render Gallery</h1>
  <p class="summary">{n} physics scenarios → Materia simulation → <b>entangler</b> render
  (direct physics → pixel). Local model <b>qwen2.5:7b</b> routed
  <b>{qwen_ok}/{n}</b> correctly · <b>{rendered}/{n}</b> rendered · <b>{glowing}</b> glow
  because the physics said they were hot.</p>
  <div class="grid">{''.join(cards)}</div>
</body></html>"""
    with open(os.path.join(out_dir, "gallery.html"), "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    rows, out_dir = run_experiment()
    n = len(rows)
    print("=" * 64)
    print(f"qwen routing: {sum(r['qwen_ok'] for r in rows)}/{n} | "
          f"rendered: {sum(r['renderable'] for r in rows)}/{n} | "
          f"glowing: {sum(r['glow'] for r in rows)}")
    print(f"gallery → {os.path.join(out_dir, 'gallery.html')}")
