"""The packed-payload cache (PC side): a model that was measured and packed once is a file read forever after.

    python stage/skippyverse/packcache.py            # what is cached, how big, the cap
    python stage/skippyverse/packcache.py --clear    # delete every entry

What an entry is: the finished SVM payload (the exact bytes the game relays), plus the measured facts the bridge reports with it (size,
triangle/vertex/palette counts, the card's source, licence and size basis). One file per entry, `<key>.svmc`:

    b"SVMC1\\n" | u32 header_len | header JSON (utf-8) | payload bytes

The header carries the key, PACK_VERSION, the format, the triangle target, a fingerprint of the source files, the payload length and its
sha256. A read re-hashes the payload; any mismatch deletes the entry and counts as a miss, so a half-written or damaged file can never be served.

The key is sha256 over (card id, format, triangle target, meshpack.PACK_VERSION). The source fingerprint (sha256 of the downloaded files
the packer reads, the NASA size table for NASA cards) is checked on a read WHEN the asset is on disk; when it is not (the download was
pruned, or the entry was prewarmed elsewhere) the entry is trusted: an asset id names an immutable upload. A new PACK_VERSION is a new key,
so the old entries are never read again and age out through eviction.

Size cap (default 500 MB) with least-recently-used eviction: a read touches the file's mtime, and a write that goes over the cap deletes the
oldest-touched entries first (never the one just written). Writes are atomic: a temp file in the same directory, flushed and fsynced, then
os.replace. Temp files left by a crash are swept at the next write.

Concurrency: `get_or_build` takes a lock per key, so two simultaneous first requests for one model build once and the second one reads the
first one's entry. Standard library only; no network.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "cache" / "packed"                 # gitignored (stage/skippyverse/cache/)
DEFAULT_CAP = 500 * 1024 * 1024
MAGIC = b"SVMC1\n"
SUFFIX = ".svmc"
MAX_HEADER = 64 * 1024


def make_key(card_id: str, fmt: str, tris: int, pack_version) -> str:
    """The name of one model's packed payload. 32 hex characters."""
    return hashlib.sha256(f"{card_id}|{fmt}|{int(tris)}|{pack_version}".encode()).hexdigest()[:32]


def fingerprint(asset_dir, extra: bytes = b"") -> str | None:
    """sha256 over the files in a downloaded asset (relative name and content, sorted), plus `extra`; None when there is no such directory.
    The acquire marker is left out (it carries nothing the packer reads)."""
    if asset_dir is None:
        return None
    d = Path(asset_dir)
    if not d.is_dir():
        return None
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file() and x.name != "acquired.json"):
        h.update(p.relative_to(d).as_posix().encode())
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    h.update(extra)
    return h.hexdigest()


class Entry:
    __slots__ = ("key", "payload", "facts", "header")

    def __init__(self, key, payload, facts, header):
        self.key, self.payload, self.facts, self.header = key, payload, facts, header


class PackCache:
    def __init__(self, directory=None, cap_bytes: int = DEFAULT_CAP, clock=time.time):
        self.dir = Path(directory) if directory else DEFAULT_DIR
        self.cap = int(cap_bytes)
        self.clock = clock
        self._locks: dict = {}
        self._master = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "builds": 0, "corrupt": 0, "stale": 0, "evicted": 0}

    # ---------------------------------------------------------------------------------------------- files
    def path(self, key: str) -> Path:
        if not key or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("bad cache key")
        return self.dir / (key + SUFFIX)

    def _drop(self, p: Path) -> None:
        try:
            p.unlink()
        except OSError:
            pass

    def _read(self, p: Path, key: str):
        """The Entry in file `p`, or None when it is damaged in any way (the caller deletes it)."""
        try:
            raw = p.read_bytes()
        except OSError:
            return None
        if raw[:len(MAGIC)] != MAGIC or len(raw) < len(MAGIC) + 4:
            return None
        (hl,) = struct.unpack_from("<I", raw, len(MAGIC))
        start = len(MAGIC) + 4
        if hl > MAX_HEADER or start + hl > len(raw):
            return None
        try:
            header = json.loads(raw[start:start + hl].decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        payload = raw[start + hl:]
        if not isinstance(header, dict) or header.get("key") != key or header.get("payload_len") != len(payload):
            return None
        if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
            return None
        return Entry(key, payload, header.get("facts") or {}, header)

    # ---------------------------------------------------------------------------------------------- the API
    def get(self, key: str, fp: str | None = None, pack_version=None):
        """The cached Entry, or None. A damaged entry, one packed with another PACK_VERSION, or one whose source fingerprint differs from `fp`
        (when both are known) is deleted and counted as a miss."""
        p = self.path(key)
        if not p.is_file():
            self.stats["misses"] += 1
            return None
        e = self._read(p, key)
        if e is None:
            self.stats["corrupt"] += 1
            self.stats["misses"] += 1
            self._drop(p)
            return None
        if (pack_version is not None and str(e.header.get("pack_version")) != str(pack_version)) or \
                (fp is not None and e.header.get("fingerprint") is not None and e.header["fingerprint"] != fp):
            self.stats["stale"] += 1
            self.stats["misses"] += 1
            self._drop(p)
            return None
        try:
            now = self.clock()
            os.utime(p, (now, now))                              # least recently used = oldest mtime
        except OSError:
            pass
        self.stats["hits"] += 1
        return e

    def put(self, key: str, payload: bytes, facts: dict, *, fmt: str, tris: int, pack_version, card_id: str = "", fp: str | None = None) -> bool:
        """Store a payload atomically. False when it does not fit the cap at all."""
        if len(payload) + 2048 > self.cap:
            return False
        header = {"key": key, "pack_version": str(pack_version), "format": fmt, "tris_target": int(tris), "card": card_id, "fingerprint": fp,
                  "payload_len": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest(), "created": self.clock(), "facts": facts}
        hb = json.dumps(header, sort_keys=True).encode("utf-8")
        data = MAGIC + struct.pack("<I", len(hb)) + hb + payload
        self.dir.mkdir(parents=True, exist_ok=True)
        self._sweep_temp()
        final = self.path(key)
        tmp = self.dir / f".tmp-{uuid.uuid4().hex}"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        try:
            now = self.clock()
            os.utime(final, (now, now))
        except OSError:
            pass
        self.evict(keep=final)
        return True

    def _sweep_temp(self, older_than: float = 3600.0) -> None:
        now = time.time()
        for p in self.dir.glob(".tmp-*"):
            try:
                if now - p.stat().st_mtime > older_than:
                    p.unlink()
            except OSError:
                pass

    def entries(self):
        out = []
        for p in self.dir.glob("*" + SUFFIX) if self.dir.is_dir() else []:
            try:
                st = p.stat()
            except OSError:
                continue
            out.append((st.st_mtime, st.st_size, p))
        return out

    def total_bytes(self) -> int:
        return sum(s for _, s, _ in self.entries())

    def evict(self, keep: Path | None = None) -> int:
        """Delete least-recently-used entries until the cache fits its cap. Returns how many went."""
        ents = sorted(self.entries(), key=lambda t: (t[0], t[2].name))
        total = sum(s for _, s, _ in ents)
        n = 0
        for _, size, p in ents:
            if total <= self.cap:
                break
            if keep is not None and p == keep:
                continue
            self._drop(p)
            total -= size
            n += 1
        self.stats["evicted"] += n
        return n

    def clear(self) -> int:
        n = 0
        for _, _, p in self.entries():
            self._drop(p)
            n += 1
        return n

    # ---------------------------------------------------------------------------------------------- one build per key
    def key_lock(self, key: str) -> threading.Lock:
        with self._master:
            return self._locks.setdefault(key, threading.Lock())

    def get_or_build(self, key: str, build, *, fp: str | None = None, pack_version=None, fmt: str = "", tris: int = 0, card_id: str = ""):
        """-> (Entry, hit). `build()` returns (payload bytes, facts dict) and runs at most once at a time per key: a second caller waits and then
        reads the first one's entry."""
        e = self.get(key, fp, pack_version)
        if e is not None:
            return e, True
        with self.key_lock(key):
            e = self.get(key, fp, pack_version)
            if e is not None:
                return e, True
            payload, facts = build()
            self.stats["builds"] += 1
            self.put(key, payload, facts, fmt=fmt, tris=tris, pack_version=pack_version, card_id=card_id, fp=fp)
            return Entry(key, payload, facts, {"key": key}), False


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=None)
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args(argv)
    c = PackCache(a.dir)
    if a.clear:
        print(f"deleted {c.clear()} entries from {c.dir}")
        return 0
    ents = c.entries()
    print(f"{c.dir}: {len(ents)} entries, {sum(s for _, s, _ in ents) / 1e6:.2f} MB of {c.cap / 1e6:.0f} MB")
    for mt, size, p in sorted(ents, reverse=True):
        e = c._read(p, p.stem)
        print(f"  {p.stem}  {size / 1024:7.1f} KB  {(e.header.get('card') if e else '(damaged)')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
