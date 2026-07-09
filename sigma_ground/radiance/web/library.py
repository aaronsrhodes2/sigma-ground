"""Durable conversation + render store for the Mentat web UI — stdlib only.

A *conversation* is one chat thread, keyed by the browser tab's session id. Each
/chat turn (the user's text and Mentat's reply envelope) is appended to
``data/library/conv_<id>.json``; a ``data/library/index.json`` manifest lists
every conversation so the V2 history rail and gallery can show them. Renders
already persist as ``data/<slug>.json`` (front_door._save_bundle) — here we only
LINK their slugs to the conversation that produced them.

Why server-side: the in-memory ``Session`` in serve.py is wiped on restart, so
threads never survived a reload. These files do. Zero dependencies (json + os),
every write is atomic (temp file + os.replace), and the index read-modify-write
is guarded by a process-wide lock so the threading HTTP server can't interleave
two updates. Reads always hit disk — there is no in-memory cache to lose, which
*is* the restart-survival guarantee.

Layering: imports nothing from the sigma_ground tiers — pure stdlib IO — so it
sits cleanly beside serve.py at the web edge.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import threading

# Overridable by tests (monkeypatch this global); paths are derived live so a
# test can repoint the whole store at a tmp dir.
_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "library")
_LOCK = threading.Lock()                       # serialises the index.json read-modify-write
_SAFE = re.compile(r"[^A-Za-z0-9_-]")
_DEFAULT_TITLES = {None, "", "untitled", "New conversation"}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _safe_id(conv_id: str) -> str:
    """Filename-safe, idempotent conversation id (the session id may carry a '.'
    or '/'). Empty → 'conv', so we never write a nameless/hidden file."""
    cid = _SAFE.sub("_", (conv_id or "").strip())[:80]
    return cid or "conv"


def _index_path() -> str:
    return os.path.join(_LIB_DIR, "index.json")


def _conv_path(conv_id: str) -> str:
    return os.path.join(_LIB_DIR, f"conv_{_safe_id(conv_id)}.json")


def _read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path: str, obj) -> None:
    """Atomic write: a concurrent reader (or a crash) never sees a half-file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _title_from(text: str) -> str:
    t = " ".join((text or "").split())
    return (t[:60] + "…") if len(t) > 60 else (t or "untitled")


def _preview(conv: dict) -> str:
    """The newest non-empty line — the rail's secondary text."""
    for m in reversed(conv.get("messages", [])):
        if m.get("text"):
            t = " ".join(m["text"].split())
            return (t[:90] + "…") if len(t) > 90 else t
    return ""


def _summary(conv: dict) -> dict:
    """The per-conversation row the index (history rail / gallery) reads."""
    return {"id": conv["id"], "title": conv.get("title") or "untitled",
            "created": conv.get("created"), "updated": conv.get("updated"),
            "renders": list(conv.get("renders", [])),
            "turns": sum(1 for m in conv.get("messages", []) if m.get("role") == "user"),
            "preview": _preview(conv)}


# ── reads (always hit disk → nothing to lose on a restart) ────────────────
def load_index() -> list:
    """The conversation manifest, newest first. [] when nothing's been saved."""
    idx = _read_json(_index_path(), [])
    return idx if isinstance(idx, list) else []


def load_conversation(conv_id: str):
    """The full thread (messages + linked render slugs), or None if absent."""
    obj = _read_json(_conv_path(conv_id), None)
    return obj if isinstance(obj, dict) else None


# ── writes (lock-guarded; the index is rebuilt from the conversation) ─────
def _reindex(conv: dict) -> dict:
    """Upsert this conversation's summary at the front of index.json. MUST be
    called while holding _LOCK."""
    summary = _summary(conv)
    idx = [s for s in load_index() if s.get("id") != conv["id"]]
    idx.insert(0, summary)                     # most-recently-touched first
    _write_json(_index_path(), idx)
    return summary


def record_turn(conv_id: str, *, user_text: str, mode: str | None,
                envelope: dict) -> str:
    """Append one /chat exchange (the user's message + Mentat's reply) and link
    any render it produced. Returns the sanitised conversation id.

    Stores only the small, displayable fields of the envelope (not the full
    render_handle). Callers should guard the call — persistence must never break
    a /chat reply — but the body stays defensive regardless."""
    cid = _safe_id(conv_id)
    now = _now()
    with _LOCK:
        conv = load_conversation(cid)
        if conv is None:
            conv = {"id": cid, "title": _title_from(user_text), "created": now,
                    "updated": now, "messages": [], "renders": []}
        # Promote a placeholder title to the first real user line.
        if conv.get("title") in _DEFAULT_TITLES and (user_text or "").strip():
            conv["title"] = _title_from(user_text)
        conv["updated"] = now
        conv["messages"].append({"role": "user", "text": user_text or "",
                                 "mode": mode or "auto", "t": now})
        reply = {"role": "mentat", "intent": envelope.get("intent"),
                 "text": envelope.get("text", ""), "t": now}
        if envelope.get("value") is not None:
            reply["value"] = envelope["value"]
        if envelope.get("source"):
            reply["source"] = envelope["source"]
        saved = envelope.get("saved")
        if isinstance(saved, dict) and saved.get("slug"):
            reply["saved"] = {"slug": saved["slug"], "title": saved.get("title"),
                              "url": saved.get("url")}
            if saved["slug"] not in conv["renders"]:
                conv["renders"].append(saved["slug"])   # link once, even on re-render
        conv["messages"].append(reply)
        _write_json(_conv_path(cid), conv)
        _reindex(conv)
    return cid


def upsert_conversation(conv_id: str, *, title: str | None = None) -> dict:
    """Create an empty conversation (the rail's "+ New") or rename one. Returns
    the index summary. Backs POST /conversation."""
    cid = _safe_id(conv_id)
    now = _now()
    with _LOCK:
        conv = load_conversation(cid)
        if conv is None:
            conv = {"id": cid, "title": title or "New conversation", "created": now,
                    "updated": now, "messages": [], "renders": []}
        elif title:
            conv["title"] = title
        conv["updated"] = now
        _write_json(_conv_path(cid), conv)
        return _reindex(conv)


def delete_conversation(conv_id: str) -> bool:
    """Forget a conversation. The render bundles it links (data/<slug>.json) are
    left in place — they may be shared with the curated gallery; Phase 0 only
    unlinks the thread."""
    cid = _safe_id(conv_id)
    with _LOCK:
        try:
            os.remove(_conv_path(cid))
        except OSError:
            pass                                # already gone — deletion is idempotent
        idx = [s for s in load_index() if s.get("id") != cid]
        _write_json(_index_path(), idx)
    return True


__all__ = ["load_index", "load_conversation", "record_turn",
           "upsert_conversation", "delete_conversation"]
