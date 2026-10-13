"""OPTIONAL, NOT RUN BY ANYONE BUT AARON: create an ORIGINAL designed voice from a text description (no recording, no clone).

ElevenLabs voice design (public docs, checked 2026-10-13): POST https://api.elevenlabs.io/v1/text-to-voice/design returns `previews`
(each with audio_base_64 and generated_voice_id). A preview you like is saved as a voice with POST /v1/text-to-voice (body: voice_name,
voice_description, generated_voice_id; confirm the fields on that page before relying on them).

Usage (Aaron, on his own machine; the key stays in his environment and is never printed or stored by this script):
    export ELEVENLABS_API_KEY=...        # or however he keeps it
    python stage/gmod/holodeck/voice_design_candidate.py              # shows what it would send (no network)
    python stage/gmod/holodeck/voice_design_candidate.py --send       # designs 3 previews, writes mp3s to data/voice_lab/designed/
    python stage/gmod/holodeck/voice_design_candidate.py --save ID NAME   # keeps preview ID as a voice called NAME
Design previews may cost credit; the sample text is 100+ characters because the API requires it.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

API = "https://api.elevenlabs.io/v1"
OUT = Path(__file__).resolve().parent / "data" / "voice_lab" / "designed"
DESCRIPTION = ("A gravelly, smug, mid-low male voice with a slight metallic edge, like a clever machine that enjoys being right. "
               "Snappy, brisk delivery with wide swings in pitch and volume, dry sarcasm, forward and clear. "
               "An original character voice, not an imitation of any real person.")
SAMPLE_TEXT = ("Gravity never takes a day off. Never complains, either. Meanwhile, you found the floor, monkey, which is "
               "a lovely start. Now write that down somewhere safe, and try not to drop it.")      # >= 100 chars (the API minimum)
BODY = {"voice_description": DESCRIPTION, "model_id": "eleven_ttv_v3", "text": SAMPLE_TEXT, "loudness": 0.5,
        "guidance_scale": 5, "seed": 20261013}


def _post(path: str, body: dict, key: str) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "xi-api-key": key})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--save", nargs=2, metavar=("GENERATED_VOICE_ID", "NAME"))
    a = ap.parse_args()
    assert len(SAMPLE_TEXT) >= 100
    if not (a.send or a.save):
        print("Would POST", API + "/text-to-voice/design", "\n", json.dumps(BODY, indent=1))
        return
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        sys.exit("Set ELEVENLABS_API_KEY in your environment first (this script never prints or stores it).")
    if a.send:
        res = _post("/text-to-voice/design", BODY, key)
        OUT.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(res.get("previews", [])):
            f = OUT / f"preview_{i}_{p['generated_voice_id']}.mp3"
            f.write_bytes(base64.b64decode(p["audio_base_64"]))
            print(f, p["generated_voice_id"])
    if a.save:
        gid, name = a.save
        res = _post("/text-to-voice", {"voice_name": name, "voice_description": DESCRIPTION, "generated_voice_id": gid}, key)
        print("saved voice id:", res.get("voice_id"))


if __name__ == "__main__":
    main()
