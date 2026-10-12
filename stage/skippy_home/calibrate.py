"""Measure each output's latency with the microphone and write latency_ms into the private config.

    python stage/skippy_home/calibrate.py [--outputs here,upstairs] [--write] [--repeats 3] [--amp 0.1]
    python stage/skippy_home/calibrate.py --set upstairs=220 --write     # manual numbers when there is no microphone

How: a quiet linear chirp (800..6000 Hz, 0.4 s, amplitude <= 0.15) goes out of one output, preceded by near-silent lead-in so the link is
awake; the microphone records; a matched filter (cross-correlation with the chirp) finds when it arrived. Latency = arrival time - the
time the chirp's first sample was handed to the output's callback, both on PortAudio's clock. That is what the router's `latency_ms`
means (it includes the speaker-to-mic air time, about 3 ms per metre). The microphone must be able to HEAR the speaker: a speaker in
another room gives a low-confidence result (reported, and not written without --force). With no microphone the config's numbers stand
(Bluetooth about 200 ms, HDMI about 60 ms by default) and this says so.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_config as acfg  # noqa: E402

MIN_SNR = 8.0


def make_chirp(rate: int, dur: float = 0.4, f0: float = 800.0, f1: float = 6000.0, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(dur * rate)) / rate
    k = (f1 - f0) / dur
    x = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t)) * np.hanning(len(t))
    return (x * min(amp, 0.15)).astype(np.float32)


def estimate_arrival(mic: np.ndarray, chirp: np.ndarray) -> tuple[int, float]:
    """(sample index in `mic` where the chirp starts, peak-to-noise ratio). Matched filter via FFT."""
    n = len(mic) + len(chirp)
    size = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(mic, size) * np.conj(np.fft.rfft(chirp, size)), size)[: len(mic)]
    env = np.abs(corr)
    i = int(np.argmax(env))
    return i, float(env[i] / (float(np.median(env)) + 1e-12))


def find_mic(hint: str = "Microphone (CMTECK)"):
    from audio_router import find_device
    for api in ("WASAPI", "MME", ""):
        try:
            return find_device(hint, api, want_input=True)
        except OSError:
            continue
    return None


def measure(cfg: dict, name: str, mic_idx: int, amp: float, repeats: int) -> dict:
    import sounddevice as sd
    from audio_router import find_device, NOISE_AMP
    o = cfg["outputs"][name]
    out_idx = find_device(o["device"], o.get("hostapi", "WASAPI"))
    rate = 48000
    chirp = make_chirp(rate, amp=amp)
    mic_info = sd.query_devices(mic_idx)
    mrate = int(mic_info["default_samplerate"])
    mic_chirp = make_chirp(mrate, amp=amp)
    results = []
    tail = int(1.0 * rate)
    for rep in range(repeats):
        rng = np.random.default_rng(rep)
        sig = np.concatenate([rng.uniform(-NOISE_AMP, NOISE_AMP, int(float(o["lead_in_ms"]) / 1000 * rate) + rate // 2).astype(np.float32),
                              chirp, np.zeros(tail, np.float32)])
        start_frame = len(sig) - len(chirp) - tail
        state = {"pos": 0, "t_out": None, "mic": [], "t_mic0": None, "done": threading.Event()}

        def ocb(outdata, frames, tinfo, status):
            a = state["pos"]; b = min(a + frames, len(sig))
            outdata[:, :] = 0
            if b > a:
                outdata[: b - a, 0] = sig[a:b]; outdata[: b - a, 1] = sig[a:b]
            if state["t_out"] is None and a <= start_frame < a + frames:
                state["t_out"] = tinfo.currentTime + (start_frame - a) / rate
            state["pos"] = a + frames
            if a >= len(sig):
                state["done"].set()

        def icb(indata, frames, tinfo, status):
            if state["t_mic0"] is None:
                state["t_mic0"] = tinfo.inputBufferAdcTime
            state["mic"].append(indata[:, 0].copy())

        def wasapi(idx):
            return sd.WasapiSettings(auto_convert=True) if "wasapi" in sd.query_hostapis(sd.query_devices(idx)["hostapi"])["name"].lower() else None

        with sd.InputStream(device=mic_idx, samplerate=mrate, channels=1, dtype="float32", callback=icb, extra_settings=wasapi(mic_idx)):
            time.sleep(0.4)
            with sd.OutputStream(device=out_idx, samplerate=rate, channels=2, dtype="float32", blocksize=1024, latency="high",
                                 callback=ocb, extra_settings=wasapi(out_idx)):
                state["done"].wait(timeout=len(sig) / rate + 3)
                time.sleep(0.5)
        mic = np.concatenate(state["mic"]) if state["mic"] else np.zeros(1, np.float32)
        if state["t_out"] is None or state["t_mic0"] is None:
            results.append({"error": "no timestamps"}); continue
        idx, snr = estimate_arrival(mic, mic_chirp)
        results.append({"latency_ms": (state["t_mic0"] + idx / mrate - state["t_out"]) * 1000.0, "snr": snr,
                        "mic_peak": float(np.max(np.abs(mic)))})
    good = [r for r in results if "latency_ms" in r and r["snr"] >= MIN_SNR]
    res = {"output": name, "runs": results}
    if good:
        lat = [r["latency_ms"] for r in good]
        res.update(latency_ms=statistics.median(lat), spread_ms=(max(lat) - min(lat)) if len(lat) > 1 else 0.0, confident=True)
    else:
        res.update(confident=False, why=f"the chirp was not clearly heard (peak-to-noise below {MIN_SNR:.0f}): is the speaker within earshot of the mic?")
    return res


def write_latency(path: Path, name: str, ms: float):
    d = json.loads(path.read_text(encoding="utf-8-sig"))
    d.setdefault("outputs", {}).setdefault(name, {})["latency_ms"] = round(ms, 1)
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--amp", type=float, default=0.1)
    ap.add_argument("--mic", default="Microphone (CMTECK)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true", help="write even a low-confidence number")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=MS", help="manual latency, no measuring")
    a = ap.parse_args(argv)
    cfg, _ = acfg.load()
    for n in a.set:
        k, v = n.split("=")
        print(f"manual: {k} latency {float(v):.0f} ms" + (" (written)" if a.write else " (not written: add --write)"))
        if a.write:
            write_latency(acfg.CONFIG_PATH, k, float(v))
    if a.set:
        return 0
    mic = find_mic(a.mic)
    if mic is None:
        print(f"NO MICROPHONE matching {a.mic!r} is available, so nothing was measured. The config's latency_ms values stand "
              "(defaults: Bluetooth about 200 ms, HDMI about 60 ms) unless you set them by hand with --set NAME=MS --write.")
        return 2
    names = [n for n in (a.outputs.split(",") if a.outputs else cfg["outputs"]) if n]
    for n in names:
        r = measure(cfg, n, mic, a.amp, a.repeats)
        if r.get("confident"):
            print(f"{n}: latency {r['latency_ms']:.0f} ms (spread {r['spread_ms']:.0f} ms over {len(r['runs'])} runs)")
            if a.write:
                write_latency(acfg.CONFIG_PATH, n, r["latency_ms"]); print("   written to", acfg.CONFIG_PATH)
        else:
            print(f"{n}: NOT MEASURED: {r['why']}  runs={r['runs']}")
            if a.write and a.force and r["runs"] and "latency_ms" in r["runs"][0]:
                write_latency(acfg.CONFIG_PATH, n, r["runs"][0]["latency_ms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
