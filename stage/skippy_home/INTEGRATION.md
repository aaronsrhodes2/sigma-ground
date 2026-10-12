# Wiring the audio router into the SkippyTel house listener

Nothing under SkippyTel was edited. The router runs on the host (it needs the Windows audio endpoints), so the satellite, which also
runs on the host, calls it over loopback. Lines to add:

1. **Start the router** next to the satellite (same supervisor / `run.sh`):
   `python D:/Aaron/development/sigma-ground/stage/skippy_home/audio_router.py serve` (port 8790, token in
   `stage/skippy_home/private/audio_token.txt`; export it to the satellite as `SKIPPY_AUDIO_TOKEN`).

2. **`satellite/audio_tools.py`** (new, thin client, stdlib only):

   ```python
   import json, os, urllib.request
   URL = os.environ.get("SKIPPY_AUDIO_URL", "http://127.0.0.1:8790")
   TOKEN = os.environ.get("SKIPPY_AUDIO_TOKEN", "")

   def _post(path, body):
       req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"X-Skippy-Token": TOKEN, "Content-Type": "application/json"})
       return json.loads(urllib.request.urlopen(req, timeout=8).read())

   def load_tools():                       # same shape as brain.TOOLS; empty if the router is down
       try:
           req = urllib.request.Request(URL + "/tools", headers={"X-Skippy-Token": TOKEN})
           return json.loads(urllib.request.urlopen(req, timeout=3).read())["tools"]
       except OSError:
           return []

   def run(name, args):
       try:
           return _post("/tool", {"name": name, "input": args})["result"]
       except OSError:
           return "The music player is not running."
   ```

   (Or import `tools.TOOLS` directly with `sys.path` pointing at `stage/skippy_home`, if the satellite should not depend on the HTTP hop.)

3. **`satellite/brain.py`**: `from audio_tools import load_tools, run as run_audio`; `AUDIO_TOOLS = load_tools()`;
   `tools=TOOLS + AUDIO_TOOLS` in `messages.create`; in `Brain.run_tool`, before the final `return "I have no such tool."`:
   `if name in {t["name"] for t in AUDIO_TOOLS}: return run_audio(name, args)`. Tool names: `play_music play_audiobook pause_audio
   resume_audio skip_audio stop_audio set_audio_volume move_audio now_playing`; `where` is `here | upstairs | everywhere`.
   Change the SPOKEN prompt's "Use your tools when one fits (timers)" to mention music and audiobooks.

4. **Deterministic fast path (optional, saves a model round-trip)**: in `ear.py` before calling the brain, `POST /say {"text": heard}`; if
   the answer's `intent` is not `unknown`, speak its `say` and skip the brain. Phrases like "pause", "quieter upstairs", "stop
   everything" then answer in milliseconds.

5. **Talking over music**: while Skippy speaks, lower the music and restore it after: `POST /volume {"delta": -0.2}` then `+0.2`
   (or `/pause` then `/resume`). The microphone will hear the music; stop or duck it on the wake word.

6. **Audiobooks**: none found on this PC yet. Put `.m4b` files or one folder per book anywhere under `D:/Aaron/Music`,
   `C:/Users/aaron/Music`, `Documents` or `Downloads`, run `python stage/skippy_home/audio_library.py`, and "play dune" /
   "resume my audiobook" work.
