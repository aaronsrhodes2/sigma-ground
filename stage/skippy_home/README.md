# skippy_home: the host-side multi-room audio player

Aaron is replacing his Echo devices with Skippy. This package is the **player**: it decodes music and audiobooks once and plays them
to `here` (the PC's HDMI stereo), `upstairs` (the Bluetooth speaker "Enclave Audio"), or `everywhere` (both, in sync). It does not hear
or think. The always-on listener (`SkippyTel/services/satellite`: wake word, voice, brain with `TOOLS`) and admin Skippy call it.
The music catalog is **music-gate's** (`SkippyTel/services/music-gate/library.py`), read from its existing tag cache; nothing here scans
the 21k tracks again. See `INTEGRATION.md` for the exact lines the SkippyTel session adds.

## Run it

```bash
python stage/skippy_home/audio_router.py devices                 # Windows output endpoints; marks the ones config uses
python stage/skippy_home/audio_router.py chime upstairs          # a quiet two-note chime on a config output or a device-name substring
python stage/skippy_home/audio_router.py serve                   # service + JSON API on 127.0.0.1:8790
python stage/skippy_home/audio_router.py say "play the wall everywhere"
python stage/skippy_home/audio_router.py play "D:/Books/Dune" --book --to upstairs
python stage/skippy_home/calibrate.py                            # measure each output's latency with the mic (see below)
python stage/skippy_home/audio_library.py                        # private report: audiobook candidates, audio music-gate leaves out
python -m pytest sigma_ground/radiance/tests/test_skippy_home_audio.py -q
```

Needs `numpy av mutagen sounddevice` (official PyPI). Tests need no sound device.

## Config (names, never IP addresses)

`private/audio_config.json` (gitignored; template `audio_config.example.json`). Each **output** is a Windows endpoint name substring
plus link properties; **groups** are named sets (`everywhere`); any extra output you add is a target too.

```json
{"outputs": {"here": {"device": "EPSON PJ (NVIDIA High Definition Audio)", "kind": "hdmi", "latency_ms": 60, "lead_in_ms": 300, "volume": 0.4},
             "upstairs": {"device": "Speakers (Enclave Audio)", "kind": "bluetooth", "latency_ms": 200, "lead_in_ms": 400}},
 "groups": {"everywhere": ["here", "upstairs"]}}
```

`latency_ms` = how long after a sample is handed to the device it is heard (calibrate.py writes it). `delay_ms` (optional) overrides the
computed delay by hand. `volume` 0..1 (gain = volume squared); the router refuses above `max_gain_volume` (0.9). To add a Pi or phone
later: add an output whose backend plays to it (a new class next to `SoundDeviceBackend` with `start/stop/alive`; the `Output` ring,
delay, drift and loss logic is already device-independent), then name it in a group.

## How it keeps rooms in sync

* **Decode once**, push the same blocks into one ring per output; the decoder runs at most `buffer_s` (1 s) ahead of the slowest ring.
* **Delay**: all outputs start so that `prefix + latency` is equal (`start_prefix_ms`): the faster link waits the difference.
* **Lead-in**: each stream starts with `lead_in_ms` of noise at about -80 dBFS (inaudible, not digital zero) so a sleeping Bluetooth/HDMI
  link is awake before the first note; the first samples are never clipped.
* **Synchronous start**: streams open at different moments (Bluetooth took 0.17-0.34 s longer in live tests), so every new stream is
  held silent until all are pulling, then released together.
* **Drift**: positions are compared at a common instant; a PI controller resamples the faster output by at most +-1000 ppm (inaudible);
  a persistent error above 60 ms is repaired by a jump instead (it would take minutes to resample away).
* **Device loss**: an output whose stream dies is dropped from the feed, the others keep playing, it is retried every `reconnect_s` and
  re-joined aligned to where the others are (it shares the reference's queued frames). Events say so in `now()`.
* **Audiobooks**: `.m4b` (chapters read via PyAV, chapter skip/back) or a folder in natural order; position saved every 5 s, on pause,
  skip and stop in `private/audio_state.json`; playing the book again resumes (2 s recap); a finished book starts over.
* Sources: files, folders, `.m3u`, `http(s)` radio streams (no seeking). Corrupt files are skipped with an event.

## Calibrating

`calibrate.py` plays a quiet chirp (<= 0.15) from each output and records it with the microphone `Microphone (CMTECK)`; a matched filter
finds the arrival; latency = arrival minus the time the chirp was handed to the callback. The mic must be able to hear the speaker; a
low-confidence result is reported and not written. **No microphone: nothing is measured**, the config's numbers stand (Bluetooth 200 ms,
HDMI 60 ms defaults) and you can set them by ear with `calibrate.py --set upstairs=230 --write`. (To tune by ear: play a click track
`everywhere` and nudge `latency_ms` until the clicks fuse.)

## Calling it

HTTP on `127.0.0.1:8790`, header `X-Skippy-Token` (env `SKIPPY_AUDIO_TOKEN`, or `private/audio_token.txt`, created on first run);
web pages (Origin/Sec-Fetch-*) and foreign hosts are refused. `GET /health /now /queue /devices /tools`; `POST /play {source,targets,kind,shuffle,append}
/pause /resume /stop {targets} /skip /prev /seek {seconds,relative} /targets {targets} /volume {targets,level|delta|mute} /queue {source}
/say {text} /tool {name,input}`. `tools.py` exports `TOOLS` in the same shape as `satellite/brain.py` (name, description, input_schema)
and `run_tool(name, args, router, library)` returning a sentence; `intents.py` does the deterministic phrase parsing
(`play X everywhere`, `upstairs only`, `quieter upstairs`, `resume my audiobook`, `stop everything`, typo-tolerant) with no LLM.

Finding things is honest: an exact or full-word match plays; two different things with the same name ("Hurt": Johnny Cash or Nine Inch
Nails) is a question, never a pick; a fuzzy match is only offered ("the closest I have is X, shall I play that?") and plays on "yes".

## Verified vs not

| | state |
|---|---|
| Sync/delay maths, lead-in, drift correction (synthetic 500 ppm), staggered start, device loss and rejoin, resume, chapters, seek, m3u, corrupt file, config, intents corpus, resolver honesty, tools shape, HTTP guard, chirp estimator | **tests** (87, offline, fake backend, real decoding) |
| Playing to the Enclave (WASAPI shared, auto-convert 44.1k), to the EPSON PJ HDMI endpoint, and both together; library lookup and intents driving real playback; no underruns; no re-alignment events after the synchronous start | **live, quiet** (amplitude <= 0.15) |
| Whether the two rooms are actually aligned to the ear (needs the real latencies) | **not verified**: no microphone was present |
| Bluetooth drop mid-play with a real speaker, long audiobook drift on real clocks, `.m4b` chapters from a real file, the satellite calling it | not verified live |
| HDMI lip-sync delay of the projector/AVR | not measured |
