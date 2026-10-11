"""SQLite FTS5 index of cards: names, aliases, approximate names, tags, category closure, WordNet-style synonym groups, plus a
size-sanity check. `version.json` is written LAST (the pattern doomloop.py uses): a reader that finds no version.json, or one whose
sha256 does not match the database, refuses the index, so a half-built index is never served.

    python stage/skippyverse/index.py build [--cards build/cards.json] [--out build/index] [--web]
    python stage/skippyverse/index.py search "couch" [--index build/index]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skippyverse import cards as C  # noqa: E402
from skippyverse.taxonomy import band_for, category_closure  # noqa: E402
from skippyverse.text import norm, tokens  # noqa: E402

SCHEMA_V = 1
REPO = Path(__file__).resolve().parents[2]
SHAPE_ALIASES = REPO / "sigma_ground" / "inventory" / "data" / "shape_aliases.json"
HERE = Path(__file__).resolve().parent


def fts5_available() -> bool:
    try:
        c = sqlite3.connect(":memory:")
        c.execute("create virtual table t using fts5(a)")
        return True
    except sqlite3.Error:
        return False


def load_synonym_groups(path: Path | None = SHAPE_ALIASES) -> list[list[str]]:
    """WordNet-derived lemma groups shipped in the repo (vocabulary only). [] if the file is absent."""
    if not path or not Path(path).exists():
        return []
    return [list(g) for g in json.loads(Path(path).read_text(encoding="utf-8")).get("groups", [])]


# ----------------------------------------------------------------------------------------------------------------- size sanity
def size_sanity(cards: list[dict]) -> dict[str, list[str]]:
    """card id -> reasons its size or mass looks implausible. Deliberately conservative: a finding means 'look at this', and the
    resolver then declines to place the card on its size. Checks: (1) longest side outside the category's band (taxonomy.SIZE_BANDS,
    judgement, not physics), (2) an earlier cross-check against another source disagreed by more than 3x, (3) mass density far
    outside any real object's bounding-box density."""
    out: dict[str, list[str]] = {}
    for c in cards:
        s = c.get("size_m")
        if not s:
            continue
        why = []
        longest = max(s["w"], s["d"], s["h"])
        lo, hi = band_for(c.get("category", "misc"))
        if not lo <= longest <= hi:
            why.append(f"longest side {longest:.3g} m is outside the plausible {lo:g}..{hi:g} m for category {c.get('category')!r}")
        if "size_crosscheck_disagrees" in c.get("flags", []):
            why.append("another source's size for the same name disagrees by more than 3x")
        m = c.get("mass")
        if m and m.get("kg"):
            dens = m["kg"] / (s["w"] * s["d"] * s["h"])
            if not 0.01 <= dens <= 30000:
                why.append(f"mass/bounding-box density {dens:.3g} kg/m3 is outside 0.01..30000")
        if why:
            out[c["id"]] = why
    return out


# ------------------------------------------------------------------------------------------------------------------------ build
_DDL = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE cards(id TEXT PRIMARY KEY, kind TEXT, name TEXT, category TEXT, cls TEXT, redistributable INTEGER, size_ok INTEGER,
                   longest REAL, json TEXT);
CREATE TABLE terms(term TEXT, card_id TEXT, role TEXT);
CREATE INDEX terms_term ON terms(term);
CREATE TABLE cat_closure(card_id TEXT, category TEXT);
CREATE INDEX cat_cat ON cat_closure(category);
CREATE TABLE syn(term TEXT, grp INTEGER);
CREATE INDEX syn_term ON syn(term);
CREATE VIRTUAL TABLE fts USING fts5(card_id UNINDEXED, name, aliases, tags, cats, tokenize='unicode61 remove_diacritics 2');
"""


def build_index(cards: list[dict], out_dir: str | Path, *, synonym_groups=None, web: bool = False) -> dict:
    """Write out_dir/index.sqlite, then out_dir/version.json LAST. `web=True` stores the web-safe copies (strip_for_web), which is what
    a published replica must be. Returns the version record."""
    if not fts5_available():
        raise RuntimeError("this Python's SQLite has no FTS5")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "version.json").unlink(missing_ok=True)                     # an interrupted rebuild must not look complete
    tmp, final = out / "index.sqlite.tmp", out / "index.sqlite"
    tmp.unlink(missing_ok=True)
    cards = sorted(cards, key=lambda c: c["id"])
    ids = [c["id"] for c in cards]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate card ids")
    sanity = size_sanity(cards)
    groups = load_synonym_groups() if synonym_groups is None else synonym_groups
    db = sqlite3.connect(tmp)
    db.executescript(_DDL)
    for c in cards:
        c = dict(c)
        if c["id"] in sanity:
            c["flags"] = [*c.get("flags", []), "size_suspect"] if "size_suspect" not in c.get("flags", []) else c["flags"]
            c["size_findings"] = sanity[c["id"]]
        stored = C.strip_for_web(c) if web else c
        s = c.get("size_m")
        db.execute("INSERT INTO cards VALUES(?,?,?,?,?,?,?,?,?)",
                   (c["id"], norm(c["kind"]), c["name"], c["category"], c["visual"]["class"], int(c["redistributable"]),
                    int(c["id"] not in sanity and s is not None), max(s["w"], s["d"], s["h"]) if s else None,
                    json.dumps(stored, sort_keys=True)))
        names = [(norm(c["name"]), "name"), (norm(c["kind"]), "name")]
        names += [(norm(a), "alias") for a in c["aliases"]] + [(norm(a), "approx") for a in c.get("approx_names", [])]
        names += [(t.replace(" ", ""), r) for t, r in names if " " in t]          # 'coffee table' is also findable as 'coffeetable'
        for term, role in dict.fromkeys(names):
            if term:
                db.execute("INSERT INTO terms VALUES(?,?,?)", (term, c["id"], role))
        closure = category_closure(c["category"])
        for cat in closure:
            db.execute("INSERT INTO cat_closure VALUES(?,?)", (c["id"], cat))
        db.execute("INSERT INTO fts(card_id,name,aliases,tags,cats) VALUES(?,?,?,?,?)",
                   (c["id"], " ".join(norm(x) for x in (c["name"], c["kind"])), " ".join(norm(a) for a in c["aliases"]),
                    " ".join(norm(t) for t in c.get("tags", [])), " ".join(norm(x.replace("/", " ")) for x in closure)))
    for gi, g in enumerate(groups):
        for term in {norm(w) for w in g}:
            if term:
                db.execute("INSERT INTO syn VALUES(?,?)", (term, gi))
    db.execute("INSERT INTO meta VALUES('schema_v', ?)", (str(SCHEMA_V),))
    db.commit()
    db.close()
    tmp.replace(final)
    sha = hashlib.sha256(final.read_bytes()).hexdigest()
    version = {"schema_v": SCHEMA_V, "card_v": C.CARD_V, "n_cards": len(cards), "n_size_suspect": len(sanity), "web": bool(web),
               "n_synonym_groups": len(groups), "sha256": sha, "files": ["index.sqlite"]}
    (out / "version.json").write_text(json.dumps(version, indent=1, sort_keys=True), encoding="utf-8")       # LAST
    return version


class IndexError_(RuntimeError):
    pass


class Index:
    """Read-only handle on a built index."""

    def __init__(self, path: Path, version: dict, shared: bool = False):
        """shared=True: the handle may be used from several threads (a server's request threads); the index is read-only."""
        self.path, self.version = Path(path), version
        self.db = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, check_same_thread=not shared)
        self.db.row_factory = sqlite3.Row
        self._cards: dict[str, dict] = {}
        self._vocab: dict[str, set] | None = None

    # -- cards
    def card(self, cid: str) -> dict:
        if cid not in self._cards:
            row = self.db.execute("SELECT json FROM cards WHERE id=?", (cid,)).fetchone()
            if row is None:
                raise KeyError(cid)
            self._cards[cid] = json.loads(row["json"])
        return self._cards[cid]

    def all_ids(self) -> list[str]:
        return [r["id"] for r in self.db.execute("SELECT id FROM cards ORDER BY id")]

    def n_cards(self) -> int:
        return self.db.execute("SELECT count(*) FROM cards").fetchone()[0]

    def size_ok(self, cid: str) -> bool:
        return bool(self.db.execute("SELECT size_ok FROM cards WHERE id=?", (cid,)).fetchone()[0])

    # -- lookups
    def by_term(self, term: str) -> list[tuple[dict, str]]:
        """Cards whose normalised name/alias/approx name equals `term`: [(card, role)], role in name|alias|approx."""
        rows = self.db.execute("SELECT card_id, role FROM terms WHERE term=?", (term,)).fetchall()
        return [(self.card(r["card_id"]), r["role"]) for r in rows]

    def synonyms(self, term: str) -> set[str]:
        grp = [r["grp"] for r in self.db.execute("SELECT grp FROM syn WHERE term=?", (term,))]
        out: set[str] = set()
        for g in grp:
            out.update(r["term"] for r in self.db.execute("SELECT term FROM syn WHERE grp=?", (g,)))
        out.discard(term)
        return out

    def vocabulary(self) -> dict[str, set]:
        """normalised full term -> {card ids}, for typo matching."""
        if self._vocab is None:
            v: dict[str, set] = defaultdict(set)
            for r in self.db.execute("SELECT term, card_id FROM terms"):
                v[r["term"]].add(r["card_id"])
            self._vocab = dict(v)
        return self._vocab

    def category_cards(self, category: str) -> list[dict]:
        return [self.card(r["card_id"]) for r in self.db.execute("SELECT card_id FROM cat_closure WHERE category=?", (category,))]

    def categories(self) -> set[str]:
        return {r["category"] for r in self.db.execute("SELECT DISTINCT category FROM cat_closure")}

    def search(self, text: str, limit: int = 25, any_token: bool = True) -> list[tuple[dict, float]]:
        """FTS5 bm25 search (name weighted highest). Scores are positive, bigger = better."""
        toks = [t for t in dict.fromkeys(norm(text).split()) if t]
        if not toks:
            return []
        q = (" OR " if any_token else " AND ").join(f'"{t}"' for t in toks)
        rows = self.db.execute(
            "SELECT card_id, bm25(fts, 0.0, 10.0, 6.0, 3.0, 1.0) AS s FROM fts WHERE fts MATCH ? ORDER BY s LIMIT ?", (q, limit)).fetchall()
        return [(self.card(r["card_id"]), -r["s"]) for r in rows]

    def name_tokens(self, card: dict) -> set[str]:
        """Words of the name, kind and aliases (not tags): what a partial match must be measured against."""
        t = set()
        for x in (card["name"], card["kind"], *card["aliases"]):
            t.update(norm(x).split())
        return t

    def searchable_tokens(self, card: dict) -> set[str]:
        t = set()
        for x in (card["name"], card["kind"], *card["aliases"], *card.get("tags", [])):
            t.update(norm(x).split())
        return t


def load_index(index_dir: str | Path, shared: bool = False) -> Index:
    d = Path(index_dir)
    vj = d / "version.json"
    if not vj.exists():
        raise IndexError_(f"{d}: no version.json - the index is missing or a build did not finish")
    version = json.loads(vj.read_text(encoding="utf-8"))
    db = d / "index.sqlite"
    if not db.exists():
        raise IndexError_(f"{d}: version.json present but index.sqlite missing")
    if hashlib.sha256(db.read_bytes()).hexdigest() != version.get("sha256"):
        raise IndexError_(f"{d}: index.sqlite does not match version.json (partial or tampered)")
    if version.get("schema_v") != SCHEMA_V:
        raise IndexError_(f"{d}: schema {version.get('schema_v')} is not {SCHEMA_V}")
    return Index(db, version, shared=shared)


def build_local(out_dir: str | Path | None = None, *, web: bool = False) -> Index:
    """Convenience: ingest everything local and build the index (used by tests, metrics and the CLI)."""
    from skippyverse.ingest_local import ingest_all
    cards, _ = ingest_all()
    out = Path(out_dir) if out_dir else HERE / "build" / ("index_web" if web else "index")
    build_index(cards, out, web=web)
    return load_index(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out")
    b.add_argument("--web", action="store_true", help="store web-safe copies (strip_for_web), as a published replica must")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--index")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        idx = build_local(a.out, web=a.web)
        print(json.dumps(idx.version, indent=1))
        return 0
    idx = load_index(a.index or HERE / "build" / "index")
    for card, score in idx.search(a.query, 10):
        print(f"{score:7.2f}  {card['id']:50s} {card['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
