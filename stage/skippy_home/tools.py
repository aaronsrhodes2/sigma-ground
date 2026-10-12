"""The audio tools in the SAME shape as SkippyTel/services/satellite/brain.py TOOLS (name, description, input_schema), so the house
listener can adopt them by importing TOOLS and routing run_tool (or by POSTing {"name","input"} to the router's /tool).
run_tool returns the sentence Skippy can say (the brain's run_tool contract: a string)."""
from __future__ import annotations

from intents import Intent, execute

WHERE = {"type": "string", "enum": ["here", "upstairs", "everywhere"],
         "description": "Where: 'here' (the PC's HDMI stereo), 'upstairs' (the Bluetooth speaker) or 'everywhere' (both, in sync). Leave out to keep the current place."}

TOOLS = [
    {"name": "play_music", "description": "Play music from the Captain's library (a song, an album, or an artist) in one room or everywhere. "
     "If the library has no exact match it says what is closest instead of playing something else.",
     "input_schema": {"type": "object", "properties": {"what": {"type": "string", "description": "Song, album or artist, e.g. 'Thriller' or 'Pink Floyd'. Add 'by <artist>' when known."},
                                                       "where": WHERE, "shuffle": {"type": "boolean"}}, "required": ["what"]}},
    {"name": "play_audiobook", "description": "Play an audiobook, resuming where he left off. With no title, resumes the last one.",
     "input_schema": {"type": "object", "properties": {"what": {"type": "string", "description": "Title or author; leave out to resume the last audiobook."}, "where": WHERE}}},
    {"name": "pause_audio", "description": "Pause the music or audiobook.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "resume_audio", "description": "Resume paused audio.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "skip_audio", "description": "Skip to the next track (or chapter), go back to the previous one, or jump forward or back some seconds.",
     "input_schema": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["next", "previous"]},
                                                       "seconds": {"type": "number", "description": "Instead: jump this many seconds (negative = back)."}}}},
    {"name": "stop_audio", "description": "Stop the audio everywhere, or only in one place (the rest keeps playing).",
     "input_schema": {"type": "object", "properties": {"where": WHERE}}},
    {"name": "set_audio_volume", "description": "Change the volume in one place or all playing places: a level in percent, or louder/quieter, or mute.",
     "input_schema": {"type": "object", "properties": {"where": WHERE, "percent": {"type": "integer", "minimum": 0, "maximum": 100},
                                                       "change": {"type": "string", "enum": ["louder", "quieter", "much louder", "much quieter", "a bit louder", "a bit quieter"]},
                                                       "mute": {"type": "boolean"}}}},
    {"name": "move_audio", "description": "Move what is playing to another place (or to everywhere) without restarting it.",
     "input_schema": {"type": "object", "properties": {"where": WHERE}, "required": ["where"]}},
    {"name": "now_playing", "description": "What is playing now and where.", "input_schema": {"type": "object", "properties": {}}},
]

_CHANGE = {"louder": 0.1, "quieter": -0.1, "much louder": 0.2, "much quieter": -0.2, "a bit louder": 0.05, "a bit quieter": -0.05}


def _tg(args) -> list:
    w = args.get("where")
    return [w] if w in ("here", "upstairs", "everywhere") else []


def run_tool(name: str, args: dict, router, library=None) -> str:
    args = args if isinstance(args, dict) else {}
    tg = _tg(args)
    if name == "play_music":
        it = Intent("play", thing=str(args.get("what") or ""), kind="music", targets=tg, shuffle=bool(args.get("shuffle")))
    elif name == "play_audiobook":
        it = Intent("play", thing=str(args.get("what") or ""), kind="book", targets=tg)
    elif name == "pause_audio":
        it = Intent("pause")
    elif name == "resume_audio":
        it = Intent("resume")
    elif name == "skip_audio":
        if args.get("seconds") is not None:
            it = Intent("seek", seconds=float(args["seconds"]))
        else:
            it = Intent("prev" if args.get("direction") == "previous" else "skip")
    elif name == "stop_audio":
        it = Intent("stop", targets=[] if tg == ["everywhere"] else tg)
    elif name == "set_audio_volume":
        if args.get("mute") is not None:
            it = Intent("volume", mute=bool(args["mute"]), targets=tg)
        elif args.get("percent") is not None:
            it = Intent("volume", level=max(0, min(100, int(args["percent"]))) / 100, targets=tg)
        else:
            it = Intent("volume", delta=_CHANGE.get(str(args.get("change")), 0.1), targets=tg)
    elif name == "move_audio":
        it = Intent("move", targets=tg)
    elif name == "now_playing":
        it = Intent("now")
    else:
        return "I have no such audio tool."
    return execute(it, router, library)
