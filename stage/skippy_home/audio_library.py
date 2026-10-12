"""Finding things to play. The MUSIC catalog is music-gate's (SkippyTel/services/music-gate/library.py): we import that module read-only
(it needs only mutagen, not Flask), build its Library from the tag cache it already keeps in SkippyTel/state/music-gate (no scan, nothing
written there), and read the same catalog rows (fields p,t,a,al,n,d,g,s,art,y,m). We add only what music-gate leaves out on purpose:
audiobooks (.m4b, spoken-word folders) and the non-MP3/AAC files, found by `discover` into a private report.

Resolution is honest like the SkippyVerse resolver: exact / strong match plays; a fuzzy match is only OFFERED ("the closest I have is X");
two different things with the same name is a question, never a pick.
"""
from __future__ import annotations

import difflib
import gzip
import importlib.util
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import audio_config as acfg

HERE = Path(__file__).resolve().parent
DEV_ROOT = HERE.parents[2]                                  # .../development
MG_LIB = DEV_ROOT / "SkippyTel" / "services" / "music-gate" / "library.py"
MG_STATE = DEV_ROOT / "SkippyTel" / "state" / "music-gate"
MUSIC_ROOT = Path(os.environ.get("SKIPPY_MUSIC_ROOT", "D:/Aaron/Music/corrected_music"))   # what music-gate mounts as /music
BOOKS_PATH = acfg.PRIVATE / "audiobooks.json"
REPORT_PATH = acfg.PRIVATE / "library_report.json"
FIELDS = ["p", "t", "a", "al", "n", "d", "g", "s", "art", "y", "m"]
STOP = {"the", "a", "an", "some", "any", "my", "of", "by", "to", "please", "song", "songs", "music", "album", "audiobook", "book", "from", "and", "&"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(\s*\d{4}\s*\)", " ", s)                  # '(2008)' album folder year
    s = re.sub(r"[^a-z0-9 ]+", " ", s.replace("&", " and "))
    return re.sub(r"\s+", " ", s).strip()


def toks(s: str) -> list[str]:
    return [t for t in norm(s).split() if t not in STOP]


@dataclass
class Entity:
    kind: str                  # track | album | artist | book
    name: str                  # display name
    artist: str = ""
    key: str = ""              # normalized name
    sources: list = field(default_factory=list)   # file paths (absolute) to play, in order
    book: bool = False

    def say(self) -> str:
        if self.kind == "artist":
            return f"{self.name}"
        return f"{self.name} by {self.artist}" if self.artist and self.artist != "Unknown artist" else self.name


@dataclass
class Resolution:
    status: str                # exact | strong | closest | ambiguous | not_found
    entity: Entity | None = None
    options: list = field(default_factory=list)
    say: str = ""

    @property
    def playable(self) -> bool:
        return self.status in ("exact", "strong") and self.entity is not None


def load_music_gate_rows(root: Path = MUSIC_ROOT, mg_lib: Path = MG_LIB, state: Path = MG_STATE) -> list[list]:
    """The catalog rows exactly as music-gate serves them, or [] with a reason if its code or cache is not there."""
    if not mg_lib.exists():
        raise FileNotFoundError(f"music-gate code not found at {mg_lib}")
    spec = importlib.util.spec_from_file_location("music_gate_library", mg_lib)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.Library(root, state)                          # reads state/tags.json only; scan() is never called
    if not lib.catalog_gz:
        raise FileNotFoundError(f"music-gate has no tag cache in {state}")
    return json.loads(gzip.decompress(lib.catalog_gz))["tracks"]


class Library:
    def __init__(self, rows: list[list], books: list[dict] | None = None, root: Path | str = MUSIC_ROOT):
        self.root = Path(root)
        self.ents: list[Entity] = []
        self.index: dict[str, set[int]] = {}
        self.vocab: set[str] = set()
        albums: dict[tuple, list] = {}
        artists: dict[str, list] = {}
        for r in rows:
            p, t, a, al = r[0], r[1], r[2], r[3]
            path = str(self.root / p)
            self._add(Entity("track", t, a, norm(t), [path]))
            albums.setdefault((a, al), []).append((r[5] or 0, r[4] or 0, path))
            artists.setdefault(a, []).append(path)
        for (a, al), items in albums.items():
            if al and len(items) > 1:
                self._add(Entity("album", re.sub(r"\s*\(\d{4}\)\s*$", "", al), a, norm(al), [p for *_, p in sorted(items)]))
        for a, paths in artists.items():
            self._add(Entity("artist", a, "", norm(a), paths))
        for b in books or []:
            self._add(Entity("book", b["title"], b.get("author", ""), norm(b["title"]), [b["path"]], book=True))
        self.track_count = len(rows)

    def _add(self, e: Entity):
        i = len(self.ents)
        self.ents.append(e)
        words = set(toks(e.name)) | set(toks(e.artist)) if e.kind != "artist" else set(toks(e.name))
        for w in words:
            self.index.setdefault(w, set()).add(i)
            self.vocab.add(w)

    @classmethod
    def load_default(cls) -> "Library":
        books = []
        try:
            books = json.loads(BOOKS_PATH.read_text(encoding="utf-8")).get("books", [])
        except (OSError, ValueError):
            pass
        return cls(load_music_gate_rows(), books)

    # ---- resolution
    def _candidates(self, qtoks: list[str]) -> set[int]:
        sets = [self.index.get(t, set()) for t in qtoks]
        if not sets or any(not s for s in sets):
            return set()
        out = set.intersection(*sets)
        return out

    def resolve(self, query: str, kind: str | None = None) -> Resolution:
        """kind: None | 'book' | 'music'. 'play X by Y' narrows by artist."""
        q = query.strip()
        who = ""
        m = re.match(r"^(.*?)\s+by\s+(.+)$", q, re.I)
        if m:
            q, who = m.group(1), m.group(2)
        qn, wn = norm(q), norm(who)
        qt = toks(q) or qn.split()
        ents = [e for e in self.ents if (kind != "book" or e.book) and (kind != "music" or not e.book)]
        if not qn:
            return Resolution("not_found", say="I did not catch what to play.")
        pool = self._candidates(qt)
        pool_e = [self.ents[i] for i in pool if self.ents[i] in ents] if pool else []
        if wn:
            wt = toks(who)
            pool_e = [e for e in pool_e if all(any(difflib.SequenceMatcher(None, w, x).ratio() > 0.8 for x in toks(e.artist) + toks(e.name)) for w in wt)] \
                if wt else pool_e
        exact = [e for e in pool_e if e.key == qn]
        order = {"book": 0, "album": 1, "artist": 2, "track": 3}
        pick = exact or []
        status = "exact"
        if not pick and pool_e:
            # strong: every query word is in the name (not merely in the artist), shortest name first
            strong = [e for e in pool_e if all(t in toks(e.name) for t in qt)] or []
            if strong:
                strong.sort(key=lambda e: (order[e.kind], len(toks(e.name))))
                best = strong[0]
                near = [e for e in strong if e.kind == best.kind and len(toks(e.name)) == len(toks(best.name))]
                pick, status = (near or [best]), "strong"
        if pick:
            kinds = sorted({e.kind for e in pick}, key=lambda k: order[k])
            pick = [e for e in pick if e.kind == kinds[0]]
            distinct = {(e.key, norm(e.artist)) for e in pick}
            if len(distinct) > 1:
                opts = [e.say() for e in pick[:5]]
                return Resolution("ambiguous", options=pick[:5], say="I have more than one: " + "; or ".join(opts) + ". Which one?")
            return Resolution(status, pick[0], say=f"Playing {pick[0].say()}.")
        return self._closest(qn, ents)

    def _closest(self, qn: str, ents: list[Entity]) -> Resolution:
        keys = {}
        for e in ents:
            keys.setdefault(e.key, e)
        best = difflib.get_close_matches(qn, list(keys), n=3, cutoff=0.78)
        if best:
            e = keys[best[0]]
            return Resolution("closest", e, options=[keys[b] for b in best],
                              say=f"I do not have '{qn}'. The closest I have is {e.say()}. Shall I play that?")
        return Resolution("not_found", say=f"I could not find '{qn}' in your library.")


# ---- discovery (what music-gate leaves out, plus audiobook candidates)
NON_GATE = {".flac", ".ogg", ".oga", ".opus", ".wav", ".wma", ".m4p", ".aax", ".aa", ".m4b"}
AUDIO_ALL = {".mp3", ".m4a", ".aac"} | NON_GATE


def discover(roots: list[Path], seq_min: int = 8) -> dict:
    """Walk `roots` (names only; never opens a file) and report: counts by type, audiobook candidates (.m4b/.aax, and folders of
    >= seq_min sequentially numbered files outside the music library), and non-MP3/AAC audio music-gate excludes."""
    counts: dict[str, int] = {}
    books, folders, extras = [], [], []
    for root in roots:
        if not root.is_dir():
            continue
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if not d.startswith(".")]
            audio = [f for f in fn if Path(f).suffix.lower() in AUDIO_ALL]
            for f in audio:
                ext = Path(f).suffix.lower()
                counts[ext] = counts.get(ext, 0) + 1
                full = str(Path(dp) / f)
                if ext in (".m4b", ".aax", ".aa"):
                    books.append({"title": Path(f).stem, "author": Path(dp).parent.name if Path(dp).parent != root else "", "path": full})
                elif ext in NON_GATE and len(extras) < 5000:
                    extras.append(full)
            numbered = [f for f in audio if re.match(r"^\s*(chapter|part|disc|cd|track)?\s*\d{1,3}\b", f, re.I)]
            if len(numbered) >= seq_min and "corrected_music" not in dp:
                folders.append({"path": dp, "files": len(numbered)})
    return {"counts": counts, "books": books, "sequential_folders": folders, "non_gate_audio_sample": extras[:200],
            "non_gate_audio_total": sum(v for k, v in counts.items() if k in NON_GATE)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Discover audio outside music-gate's catalog; writes the private report. Reads names only.")
    ap.add_argument("roots", nargs="*", default=["D:/Aaron/Music", "C:/Users/aaron/Music", "C:/Users/aaron/Documents", "C:/Users/aaron/Downloads", "D:/Music", "D:/Audiobooks"])
    a = ap.parse_args()
    rep = discover([Path(r) for r in a.roots])
    try:
        rows = load_music_gate_rows()
        rep["music_gate_tracks"] = len(rows)
    except (OSError, FileNotFoundError) as e:
        rep["music_gate_tracks"] = f"unavailable: {e}"
    acfg.PRIVATE.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    BOOKS_PATH.write_text(json.dumps({"books": rep["books"]}, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("counts", "music_gate_tracks", "non_gate_audio_total")}), "books:", len(rep["books"]),
          "sequential folders:", len(rep["sequential_folders"]))


if __name__ == "__main__":
    main()
