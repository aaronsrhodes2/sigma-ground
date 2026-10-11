"""Rung 2 of the lookup ladder, executed lazily: download ONE asset on demand, measure it, return an updated card.

One of only two modules allowed to import network modules (the other is ingest_real.py). The network is reached through an
injectable `fetch(url, *, max_bytes) -> bytes`, so every test runs offline.

Sources (URL patterns verified live 2026-10-10, no keys, no accounts):
    NASA 3D       https://raw.githubusercontent.com/nasa/NASA-3D-Resources/master/<url-quoted repo path>      one .glb
    Gazebo Fuel   https://fuel.gazebosim.org/1.0/<owner>/models/<url-quoted name>.zip                          a zip (SDF + mesh + textures)
    Poly Haven    https://api.polyhaven.com/files/<id> -> JSON; the 1k glTF and its .bin only (textures are never fetched)

Hard rules: the licence gate must pass (cards.compute_redistributable); a single file over 40 MB is refused unless force=True; one
card per call, never bulk; textures are not kept.

Sizes: a mesh in KNOWN units (glTF metres by spec, Collada <unit>, SDF models, Poly Haven) gives basis `measured`. NASA meshes have
arbitrary units: `guessed`, unless data/nasa_real_sizes.json holds a verified published figure for that model (then `sourced`).
"""
from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from skippyverse import cards as C  # noqa: E402
from skippyverse import measure as M  # noqa: E402

MAX_BYTES = 40 * 1024 * 1024
FORCE_CEILING = 2 * 1024 ** 3
UA = {"User-Agent": "SkippyVerse-acquire"}
DEFAULT_CACHE = HERE / "cache" / "assets"
NASA_SOURCE, FUEL_SOURCE, PH_SOURCE = "NASA 3D Resources", "Gazebo Fuel", "Poly Haven"
SOURCE_LABEL = {NASA_SOURCE: "NASA", FUEL_SOURCE: "Gazebo Fuel", PH_SOURCE: "Poly Haven"}
NASA_RAW = "https://raw.githubusercontent.com/nasa/NASA-3D-Resources/master/"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".dds", ".hdr", ".exr"}


class AcquireRefused(Exception):
    """The card may not (or must not) be downloaded: licence gate, no downloader, too big."""


class AcquireError(Exception):
    """The download or the measurement failed."""


class TooLarge(AcquireRefused):
    pass


def default_fetch(url: str, *, max_bytes: int = MAX_BYTES) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        n = r.headers.get("Content-Length")
        if n and int(n) > max_bytes:
            raise TooLarge(f"{url} is {int(n)/1e6:.1f} MB, over the {max_bytes/1e6:.0f} MB limit (force=True to override)")
        buf = bytearray()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            if len(buf) > max_bytes:
                raise TooLarge(f"{url} exceeds the {max_bytes/1e6:.0f} MB limit (force=True to override)")
        return bytes(buf)


def has_downloader(card: dict) -> bool:
    return (card.get("provenance") or {}).get("source") in SOURCE_LABEL


def _gate(card: dict) -> None:
    ok, why = C.compute_redistributable(card)
    if not ok or not card.get("redistributable"):
        raise AcquireRefused(f"licence gate: {why if not ok else 'card is not redistributable'}")
    if not has_downloader(card):
        raise AcquireRefused(f"no downloader for source {(card.get('provenance') or {}).get('source')!r}")


def _safe_dir(cache_dir, card_id: str) -> Path:
    return Path(cache_dir) / re.sub(r"[^A-Za-z0-9._-]+", "_", card_id)[:120]


def _write(d: Path, rel: str, data: bytes) -> str:
    rel = rel.replace("\\", "/").lstrip("/")
    if ".." in rel.split("/") or re.match(r"^[A-Za-z]:", rel):
        raise AcquireError(f"unsafe path {rel!r}")
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return rel


# ----------------------------------------------------------------------------------------------------------------- per source
def nasa_path(card: dict, tree_file: Path | None = None) -> str:
    prov = card.get("provenance") or {}
    if prov.get("asset"):
        return prov["asset"]
    tf = tree_file or (HERE / "cache" / "nasa_tree.json")
    if tf.is_file():
        tree = json.loads(tf.read_text(encoding="utf-8"))
        want = card["id"].split(".", 2)[-1]
        for e in tree.get("tree", []):
            if e.get("type") == "blob" and e["path"].lower().endswith(".glb") and C.slug(e["path"]) == want:
                return e["path"]
    raise AcquireError("cannot tell which repo file this NASA card is (no provenance.asset and no listing cache)")


def nasa_url(path: str) -> str:
    return NASA_RAW + quote(path)


def fuel_ref(card: dict) -> tuple[str, str]:
    m = re.match(r"https://fuel\.gazebosim\.org/([^/]+)/models/([^/?#]+)$", (card.get("provenance") or {}).get("url") or "")
    if not m:
        raise AcquireError("Fuel card without a model page url")
    return unquote(m.group(1)), unquote(m.group(2))


def fuel_url(owner: str, name: str) -> str:
    return f"https://fuel.gazebosim.org/1.0/{quote(owner)}/models/{quote(name)}.zip"


def polyhaven_id(card: dict) -> str:
    m = re.match(r"https://polyhaven\.com/a/([^/?#]+)$", (card.get("provenance") or {}).get("url") or "")
    if not m:
        raise AcquireError("Poly Haven card without an asset page url")
    return unquote(m.group(1))


def _fetch_nasa(card, fetch, d, limit, state) -> dict:
    path = nasa_path(card)
    data = fetch(nasa_url(path), max_bytes=limit)
    if data[:4] != b"glTF":
        raise AcquireError("NASA file is not a binary glTF" + (" (a git-lfs pointer?)" if b"git-lfs" in data[:80] else ""))
    state["bytes"] += len(data)
    name = Path(path).name
    return {"mesh": _write(d, name, data), "files": [name], "url": nasa_url(path)}


def _fetch_fuel(card, fetch, d, limit, state, force) -> dict:
    owner, name = fuel_ref(card)
    url = fuel_url(owner, name)
    data = fetch(url, max_bytes=limit)
    state["bytes"] += len(data)
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise AcquireError("Fuel download is not a zip") from None
    infos = [i for i in z.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() not in IMAGE_EXT]
    if not force and sum(i.file_size for i in infos) > 4 * limit:
        raise AcquireRefused("zip would expand to more than four times the size limit")
    files = [_write(d, i.filename, z.read(i)) for i in infos]
    sdfs = sorted(f for f in files if f.lower().endswith(".sdf"))
    pick = next((f for f in sdfs if f.lower() == "model.sdf"), sdfs[0] if sdfs else None)
    if pick is None:
        raise AcquireError("Fuel zip has no .sdf")
    return {"sdf": pick, "files": files, "url": url}


def _fetch_polyhaven(card, fetch, d, limit, state, resolution="1k") -> dict:
    pid = polyhaven_id(card)
    raw = fetch(f"https://api.polyhaven.com/files/{quote(pid)}", max_bytes=limit)
    try:
        g = json.loads(raw.decode("utf-8"))["gltf"]
    except (ValueError, KeyError):
        raise AcquireError("unexpected Poly Haven /files response (no gltf)") from None
    res = resolution if resolution in g else sorted(g)[0]
    entry = g[res]["gltf"]
    gl = fetch(entry["url"], max_bytes=limit)
    state["bytes"] += len(gl)
    mesh = _write(d, Path(entry["url"]).name, gl)
    files = [mesh]
    for rel, inc in (entry.get("include") or {}).items():
        if Path(rel).suffix.lower() == ".bin":                     # geometry only; never textures
            b = fetch(inc["url"], max_bytes=limit)
            state["bytes"] += len(b)
            files.append(_write(d, rel, b))
    return {"mesh": mesh, "files": files, "url": entry["url"], "resolution": res}


# ----------------------------------------------------------------------------------------------------------------- acquire
def acquire(card: dict, fetch=None, cache_dir=None, *, force: bool = False) -> dict:
    """Download ONE asset (or find it in the cache). -> {"dir", "source", "mesh"|"sdf", "files", "bytes", "from_cache", "url"}."""
    _gate(card)
    fetch = fetch or default_fetch
    d = _safe_dir(cache_dir or DEFAULT_CACHE, card["id"])
    marker = d / "acquired.json"
    if marker.is_file():
        info = json.loads(marker.read_text(encoding="utf-8"))
        info.update(dir=str(d), from_cache=True, bytes=0)
        return info
    limit = FORCE_CEILING if force else MAX_BYTES
    src = card["provenance"]["source"]
    state = {"bytes": 0}
    d.mkdir(parents=True, exist_ok=True)
    try:
        if src == NASA_SOURCE:
            info = _fetch_nasa(card, fetch, d, limit, state)
        elif src == FUEL_SOURCE:
            info = _fetch_fuel(card, fetch, d, limit, state, force)
        else:
            info = _fetch_polyhaven(card, fetch, d, limit, state)
    except (AcquireRefused, AcquireError):
        raise
    except Exception as e:                                       # network, JSON, OS: one error type for the caller
        raise AcquireError(f"{type(e).__name__}: {e}") from e
    info.update(source=src, bytes=state["bytes"])
    marker.write_text(json.dumps(info), encoding="utf-8")        # written last: an interrupted download is not a cache hit
    info.update(dir=str(d), from_cache=False)
    return info


# ----------------------------------------------------------------------------------------------------------------- measuring a download
def load_nasa_sizes(path: Path | None = None) -> dict:
    p = path or (HERE / "data" / "nasa_real_sizes.json")
    return json.loads(p.read_text(encoding="utf-8")).get("entries", {}) if p.is_file() else {}


def nasa_folder(path: str) -> str:
    parts = path.split("/")
    return parts[1] if len(parts) >= 3 else parts[-1]


def measure_acquired(card: dict, asset: dict, *, nasa_sizes: dict | None = None) -> dict:
    """-> {"w","d","h","basis","source","measurement": dict}. Raises AcquireError when the file cannot be measured."""
    d = Path(asset["dir"])
    src = asset["source"]
    try:
        if src == FUEL_SOURCE:
            m = M.measure_sdf(d / asset["sdf"], d)
            what = f"Gazebo Fuel SDF visuals ({asset['sdf']}), scale and poses applied"
        else:
            m = M.measure_file(d / asset["mesh"])
            what = f"{src} mesh {asset['mesh']}"
    except (M.MeasureError, OSError, KeyError, ValueError) as e:
        raise AcquireError(f"could not measure: {e}") from e
    w, dd, h = m.size_whd()
    out = {"measurement": m.to_dict()}
    if src == NASA_SOURCE:
        folder = nasa_folder(nasa_path(card))
        ent = (nasa_sizes if nasa_sizes is not None else load_nasa_sizes()).get(folder)
        if ent and ent.get("verified") and ent.get("longest_m"):
            k = ent["longest_m"] / max(w, dd, h)
            out.update(w=w * k, d=dd * k, h=h * k, basis="sourced",
                       source=f"longest side set to the published {ent['longest_m']} m ({ent['source_url']}); proportions measured from the mesh")
            out["measurement"]["scale_to_m"] = k
            return out
        out.update(w=w, d=dd, h=h, basis="guessed", source="NASA mesh units are arbitrary: file units taken as metres (a guess)")
        return out
    out.update(w=w, d=dd, h=h, basis="measured" if m.unit_known else "guessed",
               source=f"measured from the {what}" + ("" if m.unit_known else "; units unknown, taken as metres"))
    return out


def updated_card(card: dict, measured: dict, asset: dict) -> dict:
    """The card after acquire+measure: size with its basis (never weaker than the card already had), a guessed mass prior, the local
    asset recorded, `say` regenerated. Front stays unknown."""
    c = json.loads(json.dumps(card))
    new = {k: measured[k] for k in ("w", "d", "h", "basis", "source")}
    old = c.get("size_m")
    flags = [f for f in c.get("flags", []) if f != "size_unknown"]
    if old and C.basis_rank(old["basis"]) > C.basis_rank(new["basis"]):
        flags.append("download_size_weaker_than_listing")         # keep the stronger published size
    else:
        if old and old["basis"] == "sourced" and new["basis"] == "measured":
            if max(abs(new[k] / old[k] - 1) for k in ("w", "d", "h")) > 0.05:
                flags.append("measured_size_disagrees_with_listing")
        c["size_m"] = new
        c["mass"] = C.prior_mass(new)
        if new["basis"] == "guessed" and "scale_arbitrary" not in flags:
            flags.append("scale_arbitrary")
    c["flags"] = list(dict.fromkeys(flags))
    vis = c.setdefault("visual", {})
    vis["local_asset"] = {"dir": asset["dir"], "mesh": asset.get("mesh") or asset.get("sdf")}
    c["say"] = C.honesty_clause(c)
    C.assert_valid(c)
    return c


def acquire_and_measure(card: dict, fetch=None, cache_dir=None, *, force: bool = False, nasa_sizes: dict | None = None) -> dict:
    """The one call the resolver's rung 2 makes. -> {"card", "pose", "asset", "measurement", "bytes", "from_cache"}.
    `pose` is the transform the future client needs, computed and reported, applied to no file: a row-major 4x4 taking the source
    scene's coordinates (node transforms already inside the scene) to metres, +Z up, origin on the floor under the footprint centre."""
    asset = acquire(card, fetch, cache_dir, force=force)
    meas = measure_acquired(card, asset, nasa_sizes=nasa_sizes)
    m = meas["measurement"]
    mat = m["normalise_matrix"]
    k = m.get("scale_to_m", 1.0)
    if k != 1.0:
        mat = M.mat_mul(M.mat_scale((k, k, k)), mat)
    pose = {"matrix": mat, "up": "+Z", "units": "metres" if meas["basis"] != "guessed" else "metres (guessed)",
            "origin": "floor under footprint centre", "source_up": m["up_in"], "format": m["format"]}
    return {"card": updated_card(card, meas, asset), "pose": pose, "asset": asset, "measurement": m,
            "bytes": asset["bytes"], "from_cache": asset["from_cache"]}
