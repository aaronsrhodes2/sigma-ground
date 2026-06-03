"""Language -> Physics interpreter v2 demo.

Semantic router (nomic-embed-text, local) maps a natural-language question
to a library tool by MEANING, dispatches it, attaches the tool's formula +
source as a physics-style micro-bibliography, runs the groundedness gate,
and renders the whole thing to a dark Skippy-themed HTML page.

Run:  python -m sigma_ground.mcp.benchmark.interpreter_demo
Writes: misc/physics_interpreter_demo.html  (+ prints a console summary)
"""
from __future__ import annotations
import math, re, json, html
import requests

OLLAMA = "http://localhost:11434"

# ── body data (kg, m) ───────────────────────────────────────────────────
_BODIES = {
    "sun": (1.98892e30, 6.957e8), "earth": (5.972e24, 6.371e6),
    "moon": (7.342e22, 1.7374e6), "jupiter": (1.898e27, 6.9911e7),
    "mars": (6.39e23, 3.3895e6), "venus": (4.867e24, 6.0518e6),
}

# ── intent index: (tool, embedding-doc, dispatch) ───────────────────────
def _mass_from(q):
    m = re.search(r"([\d.]+)\s*solar\s*mass", q, re.I)
    if m: return float(m.group(1)) * _BODIES["sun"][0]
    if re.search(r"\b(one|a|single)\s+solar\s*mass", q, re.I):
        return _BODIES["sun"][0]
    for b,(mass,_) in _BODIES.items():
        if re.search(rf"\b{b}\b", q, re.I): return mass
    m = re.search(r"([\d.eE+]+)\s*kg", q); return float(m.group(1)) if m else None

def _radius_from(q):
    import sigma_ground.field.constants as C
    if re.search(r"\b(planck|tiniest|smallest possible)\b", q, re.I): return C.L_PLANCK
    for pat,sc in [(r"([\d.]+)\s*fm",1e-15),(r"([\d.]+)\s*nm",1e-9),
                    (r"([\d.]+)\s*km",1e3),(r"([\d.]+)\s*m\b",1.0)]:
        m=re.search(pat,q,re.I)
        if m: return float(m.group(1))*sc
    return None

def _disp_schwarzschild(q):
    from sigma_ground.mcp.tools.gr import schwarzschild_radius
    M=_mass_from(q); return schwarzschild_radius(M) if M else None
def _disp_bekenstein(q):
    from sigma_ground.mcp.tools.frontier import bekenstein_hawking_entropy
    M=_mass_from(q); return bekenstein_hawking_entropy(M) if M else None
def _disp_bubble(q):
    from sigma_ground.mcp.tools.frontier import entanglements_to_pop_bubble
    R=_radius_from(q); return entanglements_to_pop_bubble(R) if R else None
def _disp_matchmass(q):
    from sigma_ground.mcp.tools.frontier import holographic_matching_mass
    return holographic_matching_mass()
def _disp_debroglie(q):
    from sigma_ground.mcp.tools.atomic import de_broglie_from_kinetic_energy
    m=re.search(r"([\d.]+)\s*(ev|kev|mev|gev)",q,re.I)
    part=next((p for p in ["electron","proton","neutron"] if p in q.lower()),"electron")
    if not m: return None
    sc={"ev":1,"kev":1e3,"mev":1e6,"gev":1e9}[m.group(2).lower()]
    return de_broglie_from_kinetic_energy(float(m.group(1))*sc, part)
def _disp_orbital(q):
    from sigma_ground.mcp.tools.orbital import orbital_period
    m=re.search(r"([\d.]+)\s*au",q,re.I); return orbital_period(semimajor_axis_au=float(m.group(1))) if m else None
def _disp_entangle(q):
    from sigma_ground.mcp.tools.frontier import entanglement_channel
    return entanglement_channel(q)
def _disp_unruh(q):
    from sigma_ground.mcp.tools.frontier import unruh_temperature
    m=re.search(r"([\d.]+)\s*m/s",q,re.I); return unruh_temperature(float(m.group(1))) if m else None
def _disp_escape(q):
    from sigma_ground.mcp.tools.kinematics import escape_velocity_classical
    for b,(mass,rad) in _BODIES.items():
        if re.search(rf"\b{b}\b",q,re.I): return escape_velocity_classical(mass,rad)
    return None
def _disp_photon(q):
    from sigma_ground.mcp.tools.atomic import photon_energy_from_wavelength
    for pat,sc in [(r"([\d.]+)\s*nm",1e-9),(r"([\d.]+)\s*(?:micro|µ|u)m",1e-6),
                    (r"([\d.]+)\s*angstrom",1e-10),(r"([\d.]+)\s*pm",1e-12)]:
        m=re.search(pat,q,re.I)
        if m: return photon_energy_from_wavelength(float(m.group(1))*sc)
    return None
def _disp_lightspeed(q):
    from sigma_ground.mcp.tools.constants import lookup_constant
    return lookup_constant("c")
def _disp_bodymass(q):
    from sigma_ground.mcp.provenance import ToolResult
    for b,(mass,_) in _BODIES.items():
        if re.search(rf"\b{b}\b",q,re.I):
            return ToolResult(value=mass,units="kg",source="NASA/IAU planetary fact sheet",
                              formula="M (measured)",inputs={"body":b})
    return None
def _disp_orbvel(q):
    from sigma_ground.mcp.tools.orbital import orbital_velocity
    m=re.search(r"([\d,]+)\s*km",q,re.I)
    if m and re.search(r"earth",q,re.I):
        return orbital_velocity(central_body="earth",altitude_km=float(m.group(1).replace(",","")))
    return None
def _disp_surfgrav(q):
    from sigma_ground.mcp.tools.orbital import surface_gravity
    for b,(mass,rad) in _BODIES.items():
        if re.search(rf"\b{b}\b",q,re.I): return surface_gravity(mass,rad)
    return None
def _disp_percent(q):
    from sigma_ground.mcp.tools.mathx import percent_of
    mp=re.search(r"([\d.]+)\s*(?:percent|%)",q,re.I)
    if not mp: return None
    pct=float(mp.group(1))
    mv=re.search(r"([\d.]+)\s*(?:watt|W)\b",q)
    if mv: return percent_of(pct,float(mv.group(1)))
    nums=[float(x) for x in re.findall(r"\b([\d.]+)\b",q) if float(x)!=pct]
    return percent_of(pct,nums[0]) if nums else None

INTENTS = [
 ("schwarzschild_radius","Schwarzschild radius, event horizon size of a black hole, "
   "how small an object becomes if it collapses into a black hole, radius where escape "
   "velocity equals the speed of light", _disp_schwarzschild, None),
 ("bekenstein_hawking_entropy","Bekenstein-Hawking entropy of a black hole, how much "
   "entropy or information a black hole horizon holds, thread count", _disp_bekenstein,"entropy_k_B"),
 ("entanglements_to_pop_bubble","number of entanglement threads needed to pop or "
   "saturate a bubble, holographic thread count, quantum of cavitation", _disp_bubble, None),
 # NOTE: holographic_matching_mass is intentionally NOT in the semantic index --
 # it takes no args, so it ALWAYS "dispatches", making it a dangerous catch-all
 # that false-matched 11 main-corpus questions. It stays regex-only (frontier
 # classifier requires explicit 'matching mass' / 'crossover' keywords).
 ("de_broglie_from_kinetic_energy","de Broglie matter wavelength of a particle with a "
   "given kinetic energy, the wave nature wavelength of an electron", _disp_debroglie, None),
 ("orbital_period","orbital period, how long an object takes to orbit or circle the Sun "
   "at a given distance, Kepler's third law", _disp_orbital, None),
 ("entanglement_channel","can entangled particles communicate or send a message faster "
   "than light, quantum entanglement signaling, EPR, spooky action", _disp_entangle,"verdict"),
 ("unruh_temperature","Unruh temperature seen by an accelerating observer", _disp_unruh, None),
 ("escape_velocity_classical","escape velocity, the speed a rocket needs to leave a "
   "planet or moon's gravity forever, velocity required to escape Earth", _disp_escape, None),
 ("photon_energy_from_wavelength","energy of a single photon of light at a given "
   "wavelength, how much energy a green or blue photon of light carries", _disp_photon, None),
 ("lookup_constant_c","the speed of light in vacuum, how fast light travels through "
   "empty space", _disp_lightspeed, None),
 ("body_mass","the mass of a planet or the Sun, how heavy or massive a celestial body "
   "is, the weight of the Earth", _disp_bodymass, None),
 ("orbital_velocity","orbital velocity or speed of a satellite or space station "
   "orbiting the Earth at a given altitude in km, how fast a spacecraft moves in orbit",
   _disp_orbvel, None),
 # NOTE: surface_gravity / percent_of intents were REMOVED — as single-quantity
 # semantic intents they over-matched (any planet-mention -> "surface gravity"),
 # causing 3 main-corpus regressions for 1 rescue. The TOOLS remain
 # (orbital.surface_gravity, mathx.percent_of); the lesson is that fuzzy
 # single-tool router patches regress. Multi-step PROCEDURES (procedures.py),
 # recognized by their specific named workflow, are the right abstraction.
]

# ── semantic router ─────────────────────────────────────────────────────
def _embed(text, kind="query"):
    pre = "search_query: " if kind=="query" else "search_document: "
    return requests.post(f"{OLLAMA}/api/embeddings",
        json={"model":"nomic-embed-text","prompt":pre+text},timeout=30).json()["embedding"]
def _cos(a,b):
    return sum(x*y for x,y in zip(a,b))/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b)))

_INDEX=None
def _build_index():
    global _INDEX
    if _INDEX is None:
        _INDEX=[(tool,_embed(doc,"doc"),disp,field) for tool,doc,disp,field in INTENTS]
    return _INDEX

# ── typo / "did you mean" normalizer ────────────────────────────────────
_VOCAB = ["earth","light","velocity","vacuum","energy","wavelength","electron",
  "proton","photon","mass","escape","orbital","gravity","jupiter","venus","mars",
  "nuclear","momentum","frequency","radius","entropy","speed","horizon","moon",
  "entanglement","solar","planck","schwarzschild","temperature","charge","force"]
_COMMON = set(("the a an of to in is are how what whats does do can it at on for "
  "and or with that this into out fast far big small much many take takes circle "
  "leave become would size hold long heavy carry single one two go goes move").split())

def _lev(a,b):
    if abs(len(a)-len(b))>2: return 3
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(prev[j]+1,cur[j-1]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def _normalize_typos(q):
    out=[]
    for tok in re.findall(r"[A-Za-z]+|[^A-Za-z]+", q):
        low=tok.lower()
        if tok.isalpha() and len(tok)>3 and low not in _COMMON and low not in _VOCAB:
            best=min(_VOCAB,key=lambda v:_lev(low,v)); d=_lev(low,best)
            if d<=2 and d<len(tok)/2:          # close enough, not a wild guess
                out.append(best); continue
        out.append(tok)
    return "".join(out)


def route(question, threshold=0.55):
    from sigma_ground.mcp.benchmark.clarification_classifier import classify_for_clarification
    qn=_normalize_typos(question)
    clar=classify_for_clarification(qn)
    if clar is not None:
        return {"kind":"clarify","tool":"clarification","score":1.0,"response":clar.response,
                "normalized":qn}
    qv=_embed(qn,"query")
    ranked=sorted(((_cos(qv,dv),tool,disp,field) for tool,dv,disp,field in _build_index()),reverse=True)
    score,tool,disp,field=ranked[0]
    return {"kind":"tool" if score>=threshold else "lowconf","tool":tool,"score":score,
            "disp":disp,"field":field,"normalized":qn,"runner_up":(ranked[1][1],ranked[1][0])}


# ── answer assembly (route -> dispatch -> show-work -> gate) ─────────────
def _fmt(v):
    if isinstance(v,(int,float)):
        a=abs(v)
        return f"{v:.4g}" if (a>=1e4 or (a<1e-3 and a>0)) else f"{v:.4g}"
    return str(v)

def answer(question):
    r=route(question)
    base={"question":question,"value":None,"units":"","formula":None,"source":None,
          "inputs":{},"notes":"","grounded":False,"badge":"FLAGGED","tool":r.get("tool")}
    if r["kind"]=="clarify":
        base.update(routing="unrecognized term -> clarification gate",value=r["response"],
                    source="clarification gate (no grounded tool)",grounded=True,badge="CLARIFY")
        return base
    if r["kind"]=="lowconf":
        base.update(routing=f"no confident match (best: {r['tool']} @ {r['score']:.2f})")
        return base
    res=r["disp"](r.get("normalized",question))
    if res is None or getattr(res,"value",None) is None:
        base.update(routing=f"semantic -> {r['tool']} ({r['score']:.2f}); could not extract inputs")
        return base
    val=res.value
    if isinstance(val,dict) and r.get("field"): val=val.get(r["field"])
    base.update(routing=f"interpreted by meaning -> {r['tool']} (similarity {r['score']:.2f})",
                value=val,units=res.units or "",formula=res.formula,source=res.source,
                inputs=res.inputs or {},notes=res.notes or "",grounded=True,badge="GROUNDED")
    return base


# ── production cascade interface ─────────────────────────────────────────
def semantic_answer(question, threshold=0.64, margin=0.0):
    """Run the semantic router as a cascade short-circuit.

    Conservative by design (it runs before the LLM, so a wrong confident match
    is a regression): requires similarity >= threshold AND a margin over the
    runner-up, AND that the chosen tool actually extracts its inputs. Otherwise
    returns None -> the LLM handles it.

    threshold=0.64, margin=0.0 chosen by a (threshold, margin) sweep over the
    full 150-question main corpus + 16 robustness paraphrases:
        0 main-corpus regressions  AND  15/16 paraphrases correct
    (baseline regex+LLM scored ~7/16 on those paraphrases; the no-arg
    holographic_matching_mass catch-all is excluded from the index, see above).
    """
    r = route(question, threshold=threshold)
    if r.get("kind") != "tool":
        return None
    ru = r.get("runner_up")
    if ru and (r["score"] - ru[1]) < margin:
        return None                       # too ambiguous (close 2nd) -> LLM
    res = r["disp"](r.get("normalized", question))
    if res is None or getattr(res, "value", None) is None:
        return None                       # selected a tool but couldn't fill args
    val = res.value
    if isinstance(val, dict) and r.get("field"):
        val = val.get(r["field"])
    if val is None:
        return None
    show = _fmt(val) if isinstance(val, (int, float)) else str(val)
    atext = (f"ANSWER: {show} {res.units or ''}\n\n"
             f"[semantic interpreter -> {r['tool']} (similarity {r['score']:.2f})]\n"
             f"Work: {res.formula}\nSource: {res.source}")
    return {
        "answer_text":            atext,
        "extracted_value":        val,
        "extracted_units":        res.units or "",
        "tool_calls":             [r["tool"]],
        "turns":                  0,
        "elapsed_s":              0.0,
        "extracted_via_fallback": False,
        "nudges_sent":            0,
        "semantic_router_hit":    r["tool"],
    }


# ── HTML render (Skippy palette + KaTeX micro-bibliography) ──────────────
def _tex(f):
    if not f: return ""
    repl={"²":"^{2}","³":"^{3}","π":"\\pi ","ħ":"\\hbar ","ε":"\\epsilon ","×":"\\times ",
          "≈":"\\approx ","√":"\\sqrt","·":"\\cdot ","Λ":"\\Lambda ","σ":"\\sigma ","∫":"\\int "}
    for k,v in repl.items(): f=f.replace(k,v)
    return f

def render_html(results):
    cards=[]
    for i,a in enumerate(results):
        bc={"GROUNDED":"#00FF88","CLARIFY":"#FFAA00","FLAGGED":"#FF4444"}[a["badge"]]
        bl={"GROUNDED":"✓ GROUNDED","CLARIFY":"⚠ NEEDS CLARIFICATION","FLAGGED":"⚠ UNVERIFIED"}[a["badge"]]
        val=a["value"]
        ans = (f'<span class="val">{html.escape(_fmt(val))}</span> '
               f'<span class="unit">{html.escape(a["units"])}</span>') if val is not None else \
              '<span class="val muted">—</span>'
        if isinstance(val,str): ans=f'<span class="prose">{html.escape(val)}</span>'
        work=""
        if a["formula"]:
            work+=f'<div class="work"><span class="lab">Work</span><span class="ktx">$$ {html.escape(_tex(a["formula"]))} $$</span></div>'
        if a["inputs"]:
            ins=", ".join(f"{html.escape(str(k))}={html.escape(_fmt(v))}" for k,v in a["inputs"].items() if v is not None)
            if ins: work+=f'<div class="inputs"><span class="lab">Inputs</span> {ins}</div>'
        bib=""
        if a["source"]:
            bib=(f'<div class="bib"><span class="lab">Sources</span>'
                 f'<ol><li>{html.escape(a["source"])}'
                 +(f' <span class="note">— {html.escape(a["notes"])}</span>' if a["notes"] else "")
                 +'</li></ol></div>')
        cards.append(f'''
    <div class="card">
      <div class="route">{html.escape(a["routing"])}</div>
      <div class="q">{html.escape(a["question"])}</div>
      <div class="a">{ans}</div>
      {work}{bib}
      <div class="badge" style="color:{bc};border-color:{bc}">{bl}</div>
    </div>''')
    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>Skippy · Physics for All</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}}],throwOnError:false}})"></script>
<style>
 body{{background:#000;color:#fff;font-family:'Segoe UI',system-ui,sans-serif;max-width:880px;margin:0 auto;padding:32px}}
 h1{{color:#00FF88;font-weight:600;letter-spacing:.5px}} h1 .s{{color:#00CCFF}}
 .sub{{color:#006644;margin:-8px 0 28px;font-size:14px}}
 .card{{background:#000;border:1px solid #003322;border-radius:12px;padding:20px 22px;margin:16px 0}}
 .route{{color:#00CCFF;font-size:12px;font-family:ui-monospace,monospace;margin-bottom:10px;opacity:.85}}
 .q{{color:#fff;font-size:18px;font-weight:600;margin-bottom:14px}}
 .a{{margin:6px 0 14px}} .val{{color:#00FF88;font-size:30px;font-weight:700}}
 .unit{{color:#FFAA00;font-size:18px}} .prose{{color:#00FF88;font-size:16px;line-height:1.5}}
 .muted{{color:#003322}}
 .work,.inputs,.bib{{border-top:1px solid #003322;padding-top:10px;margin-top:10px;font-size:14px;color:#cfe}}
 .lab{{display:inline-block;color:#818CF8;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-right:8px}}
 .ktx{{color:#fff}} .inputs{{color:#FFAA00;font-family:ui-monospace,monospace;font-size:13px}}
 .bib ol{{margin:6px 0 0;padding-left:22px;color:#9fb}} .bib li{{margin:3px 0}}
 .note{{color:#006644}}
 .badge{{display:inline-block;margin-top:14px;padding:4px 12px;border:1px solid;border-radius:20px;
   font-size:12px;font-weight:600;letter-spacing:.5px}}
</style></head><body>
<h1>Skippy <span class="s">·</span> Physics for All</h1>
<div class="sub">local · free · every answer cites its work, or says it can't</div>
{''.join(cards)}
</body></html>'''


def main():
    Qs=[
     "If the Sun collapsed into a black hole, how small would it become?",
     "How much entropy does a black hole the mass of the Sun hold?",
     "How many entanglement threads does it take to pop the tiniest possible bubble?",
     "At what mass do a black hole's quarks and its surface pixels match up?",
     "What's the wave-nature wavelength of an electron carrying 1 keV of energy?",
     "How long does something 5 AU out take to circle the Sun?",
     "Can two entangled particles send a message faster than light?",
     "What's the energy of a magical thought barrier?",
    ]
    results=[answer(q) for q in Qs]
    import os
    out=os.path.join(os.path.dirname(__file__),"..","..","..","misc","physics_interpreter_demo.html")
    out=os.path.abspath(out)
    open(out,"w",encoding="utf-8").write(render_html(results))
    print(f"Wrote {out}\n")
    for a in results:
        v=_fmt(a["value"]) if a["value"] is not None else "—"
        print(f"[{a['badge']:9s}] {a['question'][:50]}")
        print(f"            -> {v[:60]} {a['units']}   ({a['routing'][:54]})")
    return results

if __name__=="__main__":
    main()
