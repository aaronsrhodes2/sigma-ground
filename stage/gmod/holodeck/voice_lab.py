"""The voice lab: Skippy's DELIVERY and FINISH, rendered for the ear. Style only: no voice is cloned, trained on, or uploaded.

    python stage/gmod/holodeck/voice_lab.py            # plan only: prints the lines, the exact texts, and the character total. Spends nothing.
    python stage/gmod/holodeck/voice_lab.py --go       # renders (spends ElevenLabs credit through SkippyTel's /tts/stream, text only)
    python stage/gmod/holodeck/voice_lab.py --report   # rebuild index.html + answer sheet + metrics from files already rendered (free)

Cost: lines x delivery variants (A plain, B delivery layer, C delivery layer + bold pacing). The FINISH variants (none, subtle, forward)
are made from the SAME mp3 with ffmpeg, so they cost nothing. A ledger (data/voice_lab/ledger.json) counts every request character;
the lab refuses to go past BUDGET (1,500) in total across runs, and stops on the first failed request.

Nothing leaves the machine except the line text to the local SkippyTel route (127.0.0.1:3003). Output: stage/gmod/holodeck/data/voice_lab/
(gitignored): index.html, answer_sheet.txt, ledger.json, metrics.json, clips.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import skippy_delivery as D  # noqa: E402

OUT = HERE / "data" / "voice_lab"
LEDGER = OUT / "ledger.json"
BUDGET = 1500
SKIPPYTEL = "http://127.0.0.1:3003"
SCRATCH_VENV = Path("C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/6981a9aa-3c90-44ec-9872-031c4164ea1d/scratchpad/home/hv/Scripts/python.exe")
REF_WAV = Path("C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/6981a9aa-3c90-44ec-9872-031c4164ea1d/scratchpad/voice/ref_8_88.wav")
# measured earlier on the reference clip 0:08-1:28 (delivery style only; used when the wav is not on disk)
REF_METRICS = {"pitch_median_hz": 153.0, "pitch_spread_semitones": 27.0, "loudness_range_db": 24.0, "pause_fraction": 0.09, "onsets_per_s": 4.0,
               "centroid_hz": 2000.0}

# ORIGINAL lines in Skippy's voice, short on purpose (the credit budget is by character).
LINES = [
    {"id": "boast",   "mood": "boast",   "text": "Another flawless build. I would take a bow, but I have no knees."},
    {"id": "roast",   "mood": "roast",   "text": "Congratulations, monkey, you found the floor. It was there all along."},
    {"id": "helpful", "mood": "helpful", "text": "Water boils at one hundred degrees, monkey. Write that down somewhere safe."},
    {"id": "musing",  "mood": "musing",  "text": "Gravity never takes a day off. Never complains, either."},
    {"id": "exclaim", "mood": "exclaim", "text": "It floats! The whole barn floats, and I did that!"},
    {"id": "opera",   "mood": "opera",   "text": "LA LA LAAAA! Oh, the kettle sings of boiling dreams!"},
    {"id": "whisper", "mood": "whisper", "text": "Psst. Do not tell the others, but I rather like you."},
]
VARIANTS = ("A", "B", "C")
VARIANT_NOTE = {"A": "plain text, exactly as written", "B": "delivery layer: audio tag, emphasis caps, light pacing",
                "C": "delivery layer, bold: stronger pacing (beats at the turn of the joke), more emphasis, sound tags"}
FINISHES = ("none", "subtle", "forward")


def variant_text(line: dict, variant: str) -> str:
    if variant == "A":
        return line["text"]
    return D.deliver(line["text"], line["mood"], "standard" if variant == "B" else "bold")


def plan() -> list[dict]:
    reqs = []
    for ln in LINES:
        for v in VARIANTS:
            t = variant_text(ln, v)
            reqs.append({"line": ln["id"], "variant": v, "text": t, "chars": len(t), "file": f"{ln['id']}_{v}.mp3"})
    return reqs


def ledger_load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"budget": BUDGET, "spent": 0, "requests": []}


def ledger_write(led: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=1), encoding="utf-8")


def fetch(text: str) -> bytes:
    url = f"{SKIPPYTEL}/tts/stream?text=" + urllib.parse.quote(text)
    with urllib.request.urlopen(url, timeout=60) as r:
        if r.status != 200 or "audio" not in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"route answered {r.status} {r.headers.get('Content-Type')}")
        data = r.read(5_000_000)
    if len(data) < 1500:
        raise RuntimeError("route returned almost no audio")
    return data


def render(reqs: list[dict]) -> None:
    led = ledger_load()
    todo = [r for r in reqs if not (OUT / "clips" / r["file"]).exists()]
    need = sum(r["chars"] for r in todo)
    if led["spent"] + need > BUDGET:
        sys.exit(f"REFUSED: ledger {led['spent']} + planned {need} exceeds the budget {BUDGET}")
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    for r in todo:
        if led["spent"] + r["chars"] > BUDGET:
            sys.exit("REFUSED mid-run: budget")
        try:
            data = fetch(r["text"])
        except Exception as e:                                              # a failed route stops the lab; nothing more is spent
            sys.exit(f"STOPPED: {r['file']} failed ({type(e).__name__}: {e}). Spent so far: {led['spent']}")
        led["spent"] += r["chars"]                                          # count on send, not on success: the credit may be used anyway
        led["requests"].append({"file": r["file"], "chars": r["chars"], "sha1": hashlib.sha1(r["text"].encode()).hexdigest()[:10],
                                "bytes": len(data), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        (OUT / "clips" / r["file"]).write_bytes(data)
        ledger_write(led)
        print(f"  {r['file']}: {r['chars']} chars, {len(data)} bytes; ledger {led['spent']}/{BUDGET}")


def make_finishes(best: str) -> list[dict]:
    """none/subtle/forward from the SAME mp3 of the best variant: free."""
    out = []
    for ln in LINES:
        src = OUT / "clips" / f"{ln['id']}_{best}.mp3"
        if not src.exists():
            continue
        for fin in FINISHES:
            dst = OUT / "clips" / f"{ln['id']}_{best}_{fin}.mp3"
            D.finish(src, dst, fin)
            out.append({"line": ln["id"], "finish": fin, "file": dst.name})
    return out


# ------------------------------------------------------------------------------------------------------------ metrics (librosa)

def compute_metrics(path: str) -> dict:
    """Run under a Python that has librosa. Style metrics only."""
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=22050, mono=True)
    dur = len(y) / sr
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr, frame_length=2048)
    f0v = f0[voiced & ~np.isnan(f0)]
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max)
    speech = db > -40
    frame_s = hop / sr
    pauses, n = [], 0
    for s in speech:
        if not s:
            n += 1
        else:
            if n * frame_s >= 0.25:
                pauses.append(n * frame_s)
            n = 0
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    cent = cent[: len(speech)][speech[: len(cent)]] if speech.any() else np.array([0.0])
    semis = 12 * np.log2(f0v / np.median(f0v)) if len(f0v) > 5 else np.array([0.0])
    return {"duration_s": round(dur, 2), "pitch_median_hz": round(float(np.median(f0v)), 0) if len(f0v) else 0.0,
            "pitch_spread_semitones": round(float(np.percentile(semis, 90) - np.percentile(semis, 10)), 1),
            "loudness_range_db": round(float(np.percentile(db[speech], 95) - np.percentile(db[speech], 10)), 1) if speech.any() else 0.0,
            "pause_fraction": round(float(sum(pauses) / dur), 2), "pauses": len(pauses),
            "onsets_per_s": round(len(onsets) / dur, 2), "centroid_hz": round(float(np.median(cent)), 0)}


def metrics_for(files: list[Path]) -> dict:
    """mp3 -> wav with ffmpeg, then the librosa interpreter (hv venv) computes the numbers. {} if librosa is unavailable."""
    py = SCRATCH_VENV if SCRATCH_VENV.exists() else None
    ff = D.find_ffmpeg()
    if py is None or ff is None:
        print("(metrics skipped: no librosa interpreter or no ffmpeg)")
        return {}
    wavs = []
    wdir = OUT / "wav_tmp"
    wdir.mkdir(parents=True, exist_ok=True)
    for f in files:
        w = wdir / (f.stem + ".wav")
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(f), "-ac", "1", str(w)], check=True)
        wavs.append(w)
    r = subprocess.run([str(py), "-X", "utf8", str(Path(__file__).resolve()), "--metrics", *map(str, wavs)], capture_output=True, text=True)
    if r.returncode != 0:
        print("(metrics failed)", r.stderr[-400:])
        return {}
    res = json.loads(r.stdout.strip().splitlines()[-1])
    return {Path(k).stem: v for k, v in res.items()}


# ------------------------------------------------------------------------------------------------------------ report

def fmt(m: dict) -> str:
    return (f"pitch {m['pitch_median_hz']:.0f} Hz, spread {m['pitch_spread_semitones']:.0f} st, loudness range {m['loudness_range_db']:.0f} dB, "
            f"pauses {m['pause_fraction']*100:.0f}%, {m['onsets_per_s']:.1f} onsets/s, centroid {m['centroid_hz']:.0f} Hz")


def report(reqs: list[dict], best: str, fins: list[dict], met: dict, ref: dict) -> None:
    led = ledger_load()
    sheet = ["SKIPPY VOICE LAB: answer sheet", "Style metrics only (delivery: pitch movement, pace, pauses, loudness range, brightness). They do not measure identity.", "",
             f"Reference clip (delivery only): {fmt(ref)}", f"Credit spent: {led['spent']} of {BUDGET} characters.", ""]
    cards = []
    for ln in LINES:
        sheet.append(f"== {ln['id'].upper()} ({ln['mood']}) ==")
        row = []
        for v in VARIANTS:
            f = f"{ln['id']}_{v}.mp3"
            if not (OUT / "clips" / f).exists():
                continue
            text = variant_text(ln, v)
            m = met.get(f"{ln['id']}_{v}")
            sheet.append(f"  Delivery {v} ({VARIANT_NOTE[v]}): sent {text!r}")
            if m:
                sheet.append(f"      {fmt(m)}")
            row.append(f'<div class="clip"><b>Delivery {v}</b><audio controls preload="none" src="clips/{f}"></audio>'
                       f'<p>{html.escape(VARIANT_NOTE[v])}.</p><p class="sent">sent: {html.escape(text)}</p>'
                       + (f'<p class="m">{html.escape(fmt(m))}</p>' if m else "") + "</div>")
        frow = []
        for fi in [x for x in fins if x["line"] == ln["id"]]:
            m = met.get(Path(fi["file"]).stem)
            sheet.append(f"  Finish {fi['finish']} (on delivery {best}): {fi['file']}" + (f" | {fmt(m)}" if m else ""))
            frow.append(f'<div class="clip"><b>Finish: {fi["finish"]}</b><audio controls preload="none" src="clips/{fi["file"]}"></audio>'
                        f'<p>{html.escape(D.finish_filter(fi["finish"]) or "no filter, just re-encoded")}</p></div>')
        sheet.append("")
        cards.append(f'<section><h2>{html.escape(ln["id"])} <small>({ln["mood"]})</small></h2><p class="orig">written: {html.escape(ln["text"])}</p>'
                     f'<h3>Three deliveries (same voice, different text)</h3><div class="row">{"".join(row)}</div>'
                     f'<h3>Three finishes (same audio, Delivery {best})</h3><div class="row">{"".join(frow)}</div></section>')
    sheet += ["WHAT ONLY EARS CAN DECIDE: which delivery sounds like Skippy rather than like a narrator doing a voice; whether the audio tags help or",
              "sound fake; whether the caps are emphasis or shouting; whether the opera line is funny; whether 'forward' is beer-can or just harsh."]
    (OUT / "answer_sheet.txt").write_text("\n".join(sheet) + "\n", encoding="utf-8")
    page = f"""<!doctype html><meta charset="utf-8"><title>Skippy voice lab</title>
<style>body{{font:20px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;background:#111;color:#eee}}
section{{border-top:2px solid #444;padding:12px 0}}h2{{font-size:30px;margin:8px 0}}h3{{font-size:20px;color:#9cf}}.row{{display:flex;flex-wrap:wrap;gap:16px}}
.clip{{flex:1 1 300px;background:#1c1c1c;padding:12px;border-radius:8px}}audio{{width:100%}}.sent,.m,.orig{{color:#aaa;font-size:16px}}
.note{{background:#2a2a1c;padding:12px;border-radius:8px;font-size:17px}}</style>
<h1>Skippy voice lab</h1>
<p class="note">Delivery and character only. No voice was cloned or trained, nothing was uploaded; text lines went to the local SkippyTel route.
Reference clip delivery numbers (style metrics only): {html.escape(fmt(ref))}. Credit spent: {led['spent']} of {BUDGET} characters.
Listen across a row: A is the control. Then judge the finishes.</p>
{"".join(cards)}
"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="spend credit and render")
    ap.add_argument("--report", action="store_true", help="rebuild the report from existing files (free)")
    ap.add_argument("--best", default="B", choices=VARIANTS, help="delivery variant the finish variants are made from")
    ap.add_argument("--metrics", nargs="+", help="internal: print metrics JSON for wav files (needs librosa)")
    a = ap.parse_args()
    if a.metrics:
        print(json.dumps({p: compute_metrics(p) for p in a.metrics}))
        return
    reqs = plan()
    total = sum(r["chars"] for r in reqs)
    led = ledger_load()
    print(f"PLAN: {len(LINES)} lines x {len(VARIANTS)} delivery variants = {len(reqs)} requests, {total} characters; finishes are free (ffmpeg).")
    for r in reqs:
        print(f"  {r['file']:16s} {r['chars']:3d}  {r['text']}")
    print(f"TOTAL {total} chars; ledger already {led['spent']}; budget {BUDGET}; after: {led['spent'] + total}")
    if led["spent"] + total > BUDGET and not a.report:
        sys.exit("over budget: shorten the lines")
    if a.go:
        render(reqs)
    elif not a.report:
        print("(plan only: nothing spent; pass --go to render)")
        return
    fins = make_finishes(a.best)
    files = [OUT / "clips" / r["file"] for r in reqs if (OUT / "clips" / r["file"]).exists()] + [OUT / "clips" / f["file"] for f in fins]
    met = metrics_for(files) if files else {}
    ref = REF_METRICS
    if REF_WAV.exists():
        ref_m = metrics_for_ref()
        ref = ref_m or REF_METRICS
    (OUT / "metrics.json").write_text(json.dumps({"reference": ref, "clips": met}, indent=1), encoding="utf-8")
    report(reqs, a.best, fins, met, ref)
    print(f"open {OUT / 'index.html'}")
    if met:
        print("style metrics (delivery only) vs reference:", fmt(ref))
        for k in sorted(met):
            if k.endswith(("_A", "_B", "_C")):
                print(f"  {k:12s} {fmt(met[k])}")


def metrics_for_ref() -> dict | None:
    py = SCRATCH_VENV
    if not py.exists():
        return None
    r = subprocess.run([str(py), "-X", "utf8", str(Path(__file__).resolve()), "--metrics", str(REF_WAV)], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return list(json.loads(r.stdout.strip().splitlines()[-1]).values())[0]


if __name__ == "__main__":
    main()
