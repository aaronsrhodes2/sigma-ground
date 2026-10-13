"""skippy_faces.py -- the bridge's /face action: a player's picture, made safe and small, for the game to wear as a head.

Hooked into skippy_bridge.py by one branch in Handler._route (POST /face). Everything else lives here so the bridge file stays small.

Sources (the game server picks; the bridge never sees a client's word for who someone is):
  action "steam"   {steamid64}   the player's PUBLIC Steam avatar (the default, no setup). The server passes ply:SteamID64() itself.
  action "url"     {url}         a picture the player chose: https only, from the allow-list below.
  action "path"    {path}        a local file; only when the bridge was started with SKIPPY_FACE_ALLOW_LOCAL=1, and the game only ever asks
                                 for it from the server console / listen-server host. Still decoded and shrunk like any picture.
  action "redeem"  {code}        OPTIONAL portal link code -> the portal's picture URL (STUB until the portal side exists; see
                                 stage/gmod/FACE_PORTAL_HANDOFF.md) -> fetched like "url".

Answer: {"ok": true, "hash", "crc32", "bytes", "b64", "border": [r,g,b], "detector", "source"} or
        {"ok": false, "kind": "no_picture" | "refused" | "bad_image" | "unavailable", "error": "..."} (HTTP 200: it is an answer, not a fault).

Safety, in order: allow-listed host (https, port 443, no user info, no IP literals) -> DNS answer must be public (no private, loopback,
link-local, reserved) -> we connect to THAT address (no second lookup to rebind) -> redirects re-checked, at most 3 -> size cap
(2 MB for pictures, 64 KB for the Steam profile XML) -> content type and magic bytes -> Pillow with a pixel limit (decompression bomb).
The result is a 128x128 JPEG (<= 20 KB) cached by content hash in a private directory.
"""
from __future__ import annotations

import base64
import fnmatch
import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import sys
import threading
import time
import urllib.parse
import urllib.request
import warnings
import zlib
from io import BytesIO
from pathlib import Path

OUT_SIZE = 128
MAX_OUT_BYTES = 20 * 1024
MAX_PICTURE_BYTES = 2 * 1024 * 1024
MAX_XML_BYTES = 64 * 1024
MAX_PIXELS = 24_000_000                  # Pillow warns above this and raises above twice it; both end in a refusal here
MAX_REDIRECTS = 3
TIMEOUT_S = 6.0
STEAM_PROFILE_HOST = "steamcommunity.com"      # the only host besides the picture allow-list, and only for the Steam XML
DEFAULT_ALLOW = ("*.googleusercontent.com", "www.gravatar.com", "secure.gravatar.com", "gravatar.com", "avatars.steamstatic.com",
                 "*.steamstatic.com", "steamcdn-a.akamaihd.net")
IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp", "image/gif")
STEAM_CACHE_S = 600.0

CACHE_DIR = Path(os.environ.get("SKIPPY_FACE_CACHE") or (Path(__file__).resolve().parent / "data" / "faces_cache"))
CACHE_KEEP_DAYS = 30
ALLOW_LOCAL = os.environ.get("SKIPPY_FACE_ALLOW_LOCAL", "") == "1"


class FaceError(Exception):
    """A refusal or a failure with a kind the game understands."""

    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind, self.msg = kind, msg


def allow_list() -> tuple[str, ...]:
    extra = os.environ.get("SKIPPY_FACE_HOSTS", "")
    return DEFAULT_ALLOW + tuple(h.strip().lower() for h in extra.split(",") if h.strip())


def host_allowed(host: str, allow=None) -> bool:
    host = host.lower().rstrip(".")
    return any(fnmatch.fnmatchcase(host, pat) and (not pat.startswith("*.") or host != pat[2:]) for pat in (allow or allow_list()))


def check_url(url, allow=None, extra_hosts=()) -> tuple[str, int, str]:
    """(host, port, path+query) of an acceptable https URL, else FaceError("refused")."""
    if not isinstance(url, str) or not url or len(url) > 600 or any(ord(c) < 33 or ord(c) > 126 for c in url):
        raise FaceError("refused", "not a usable link")
    p = urllib.parse.urlsplit(url)
    if p.scheme != "https":
        raise FaceError("refused", "only https links are used")
    if p.username or p.password or "@" in p.netloc:
        raise FaceError("refused", "links with a user name are refused")
    host = (p.hostname or "").lower().rstrip(".")
    try:
        port = p.port or 443
    except ValueError:
        raise FaceError("refused", "bad port") from None
    if not host or port != 443:
        raise FaceError("refused", "only the standard https port is used")
    try:
        ipaddress.ip_address(host)
        raise FaceError("refused", "addresses are not accepted, only the allowed sites")
    except ValueError:
        pass
    if host not in extra_hosts and not host_allowed(host, allow):
        raise FaceError("refused", f"{host} is not one of the allowed picture sites")
    return host, port, (p.path or "/") + (("?" + p.query) if p.query else "")


def _getaddrinfo(host, port):
    return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)


def resolve_public(host: str, port: int = 443) -> list[str]:
    """Every address the name has, all of which must be public; else FaceError. Returns them (first is connected to)."""
    try:
        infos = _getaddrinfo(host, port)
    except OSError:
        raise FaceError("unavailable", "the site could not be found") from None
    ips = []
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
        except ValueError:
            continue
        if not ip.is_global or ip.is_multicast:
            raise FaceError("refused", "that name points at a private address")
        ips.append(str(ip))
    if not ips:
        raise FaceError("unavailable", "the site has no address")
    return ips


def _transport(host: str, ip: str, port: int, path: str, timeout: float, max_bytes: int):
    """One GET to a vetted address with TLS checked against the NAME. -> (status, headers (lower-case keys), body)."""
    ctx = ssl.create_default_context()

    class Conn(http.client.HTTPSConnection):
        def connect(self):
            sock = socket.create_connection((ip, port), timeout=timeout)
            self.sock = ctx.wrap_socket(sock, server_hostname=host)

    c = Conn(host, port, timeout=timeout, context=ctx)
    try:
        c.request("GET", path, headers={"Host": host, "User-Agent": "SkippyBridge-Faces/1", "Accept": "image/*,text/xml;q=0.5",
                                          "Accept-Encoding": "identity", "Connection": "close"})
        r = c.getresponse()
        headers = {k.lower(): v for k, v in r.getheaders()}
        n = headers.get("content-length")
        if n and n.isdigit() and int(n) > max_bytes:
            raise FaceError("bad_image", "that file is too large")
        body = r.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise FaceError("bad_image", "that file is too large")
        return r.status, headers, body
    except (OSError, http.client.HTTPException):
        raise FaceError("unavailable", "the site did not answer") from None
    finally:
        c.close()


def fetch(url: str, max_bytes: int = MAX_PICTURE_BYTES, allow=None, extra_hosts=(), want_types=IMAGE_TYPES, deadline: float | None = None):
    """GET an allow-listed https URL with the protections above. Returns (body, content_type)."""
    deadline = deadline or time.monotonic() + 15
    for _ in range(MAX_REDIRECTS + 1):
        host, port, path = check_url(url, allow, extra_hosts)
        ips = resolve_public(host, port)
        left = deadline - time.monotonic()
        if left <= 0:
            raise FaceError("unavailable", "took too long")
        status, headers, body = _transport(host, ips[0], port, path, min(TIMEOUT_S, left), max_bytes)
        if status in (301, 302, 303, 307, 308):
            loc = headers.get("location", "")
            url = urllib.parse.urljoin(url, loc)           # the next hop goes through check_url again: off the list = refused
            continue
        if status == 404 or status == 403 or status == 410:
            raise FaceError("no_picture", "no picture there")
        if status != 200:
            raise FaceError("unavailable", f"the site answered {status}")
        if len(body) > max_bytes:
            raise FaceError("bad_image", "that file is too large")
        ctype = headers.get("content-type", "").split(";")[0].strip().lower()
        if want_types and ctype not in want_types:
            raise FaceError("bad_image", "that is not a picture")
        return body, ctype
    raise FaceError("refused", "too many redirects")


# ---------------------------------------------------------------------------------------------------------------- pictures
MAGIC = ((b"\xff\xd8\xff", "JPEG"), (b"\x89PNG\r\n\x1a\n", "PNG"), (b"GIF87a", "GIF"), (b"GIF89a", "GIF"))


def sniff(data: bytes) -> str:
    for m, name in MAGIC:
        if data.startswith(m):
            return name
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    raise FaceError("bad_image", "that is not a picture")


_cascade = None
_cascade_tried = False


def _face_box(im):
    """The largest face (x, y, w, h) found by OpenCV's Haar cascade, or None. Optional: without opencv the crop is centred."""
    global _cascade, _cascade_tried
    if not _cascade_tried:
        _cascade_tried = True
        try:
            import cv2
            c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            _cascade = c if not c.empty() else None
        except Exception:                                    # opencv is an optional extra
            _cascade = None
    if _cascade is None:
        return None
    try:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
        found = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(24, im.width // 12),) * 2)
    except Exception:
        return None
    if len(found) == 0:
        return None
    return max((tuple(int(v) for v in f) for f in found), key=lambda f: f[2] * f[3])


def crop_square(im, use_detector: bool = True):
    """(square image, detector name). Centred on the face when one is found, else the middle."""
    w, h = im.size
    side = min(w, h)
    cx, cy, how = w / 2, h / 2, "centre"
    box = _face_box(im) if use_detector else None
    if box:
        x, y, fw, fh = box
        side = int(min(min(w, h), max(fw, fh) * 2.2))
        cx, cy, how = x + fw / 2, y + fh / 2 + fh * 0.1, "haar"
    left = int(round(min(max(cx - side / 2, 0), w - side)))
    top = int(round(min(max(cy - side / 2, 0), h - side)))
    return im.crop((left, top, left + side, top + side)), how


def border_colour(im) -> list[int]:
    """The mean colour of the picture's outer ring (the head's sides and back take it)."""
    w, h = im.size
    k = max(1, w // 16)
    px = im.convert("RGB")
    tot, n = [0, 0, 0], 0
    for x in range(w):
        for y in list(range(k)) + list(range(h - k, h)):
            for i, v in enumerate(px.getpixel((x, y))):
                tot[i] += v
            n += 1
    for y in range(k, h - k):
        for x in list(range(k)) + list(range(w - k, w)):
            for i, v in enumerate(px.getpixel((x, y))):
                tot[i] += v
            n += 1
    return [int(t / n) for t in tot]


def process(data: bytes, use_detector: bool = True) -> dict:
    """Raw picture bytes -> {"jpeg", "border", "detector"}; FaceError("bad_image") for anything that is not a modest picture."""
    sniff(data)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(BytesIO(data))
            if im.width * im.height > MAX_PIXELS:
                raise FaceError("bad_image", "that picture is far too large")
            im.seek(0)
            im.load()
    except FaceError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise FaceError("bad_image", "that picture is far too large") from None
    except Exception:
        raise FaceError("bad_image", "that picture could not be read") from None
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (128, 128, 128))
        bg.paste(im, mask=im.split()[3])
        im = bg
    im = im.convert("RGB")
    sq, how = crop_square(im, use_detector)
    sq = sq.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    out = b""
    for q in (85, 78, 70, 60, 50, 40):
        buf = BytesIO()
        sq.save(buf, "JPEG", quality=q, optimize=True)
        out = buf.getvalue()
        if len(out) <= MAX_OUT_BYTES:
            break
    if len(out) > MAX_OUT_BYTES:
        raise FaceError("bad_image", "that picture does not shrink small enough")
    return {"jpeg": out, "border": border_colour(sq), "detector": how}


# ---------------------------------------------------------------------------------------------------------------- the cache
_cache_lock = threading.Lock()


def cache_put(res: dict) -> str:
    h = hashlib.sha256(res["jpeg"]).hexdigest()[:24]
    with _cache_lock:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            p = CACHE_DIR / (h + ".jpg")
            if not p.exists():
                tmp = p.with_suffix(".tmp")
                tmp.write_bytes(res["jpeg"])
                tmp.replace(p)
            (CACHE_DIR / (h + ".json")).write_text(json.dumps({"border": res["border"], "detector": res["detector"]}), encoding="utf-8")
        except OSError:
            pass                                                   # the cache is a convenience; the answer does not depend on it
    return h


def cache_get(h: str):
    if not re.fullmatch(r"[0-9a-f]{24}", h or ""):
        return None
    try:
        jpeg = (CACHE_DIR / (h + ".jpg")).read_bytes()
        meta = json.loads((CACHE_DIR / (h + ".json")).read_text(encoding="utf-8"))
        return {"jpeg": jpeg, "border": meta["border"], "detector": meta["detector"]}
    except (OSError, ValueError, KeyError):
        return None


def prune_cache(days: float = CACHE_KEEP_DAYS, now: float | None = None) -> int:
    """Delete cached pictures not used for `days` days (a picture used again is rewritten, which refreshes it). Returns files removed."""
    now = now or time.time()
    n = 0
    try:
        for p in CACHE_DIR.iterdir():
            if p.suffix in (".jpg", ".json", ".tmp") and now - p.stat().st_mtime > days * 86400:
                p.unlink()
                n += 1
    except OSError:
        pass
    return n


def _answer(res: dict, source: str) -> dict:
    h = cache_put(res)
    jpeg = res["jpeg"]
    return {"ok": True, "hash": h, "crc32": zlib.crc32(jpeg) & 0xFFFFFFFF, "bytes": len(jpeg), "b64": base64.b64encode(jpeg).decode(),
            "border": res["border"], "detector": res["detector"], "source": source}


# ---------------------------------------------------------------------------------------------------------------- sources
_STEAMID = re.compile(r"7656119\d{10}\Z")
_XML_FULL = re.compile(r"<avatarFull>\s*(?:<!\[CDATA\[)?\s*(https://[^\s<\]]+)", re.I)
_XML_MEDIUM = re.compile(r"<avatarMedium>\s*(?:<!\[CDATA\[)?\s*(https://[^\s<\]]+)", re.I)
_steam_seen: dict[str, tuple[float, dict | None]] = {}


def steam_avatar_url(steamid64: str, deadline=None) -> str:
    """The avatarFull URL from the public profile XML (https://steamcommunity.com/profiles/<id>/?xml=1), else FaceError("no_picture").
    The XML is read with regexes (no XML parser, so no entity tricks); the URL it names must pass the same allow-list."""
    if not _STEAMID.match(steamid64 or ""):
        raise FaceError("refused", "not a Steam id")
    body, _ = fetch(f"https://{STEAM_PROFILE_HOST}/profiles/{steamid64}/?xml=1", MAX_XML_BYTES, extra_hosts=(STEAM_PROFILE_HOST,),
                    want_types=("text/xml", "application/xml", "text/html"), deadline=deadline)
    text = body.decode("utf-8", "replace")
    if "<profile>" not in text:                                   # an unknown id answers a web page, not the profile
        raise FaceError("no_picture", "no public Steam profile")
    m = _XML_FULL.search(text) or _XML_MEDIUM.search(text)
    if not m:
        raise FaceError("no_picture", "the profile shows no picture")
    return m.group(1)


def from_steam(steamid64: str, deadline=None) -> dict:
    now = time.monotonic()
    hit = _steam_seen.get(steamid64)
    if hit and now - hit[0] < STEAM_CACHE_S:
        if hit[1] is None:
            raise FaceError("no_picture", "no public Steam picture")
        return hit[1]
    try:
        url = steam_avatar_url(steamid64, deadline)
        body, _ = fetch(url, deadline=deadline)
        ans = _answer(process(body), "steam")
    except FaceError as e:
        if e.kind in ("no_picture", "bad_image"):
            _steam_seen[steamid64] = (now, None)
        raise
    if len(_steam_seen) > 500:
        _steam_seen.clear()
    _steam_seen[steamid64] = (now, ans)
    return ans


def size_param(url: str, size: int = 256) -> str:
    """Google picture URLs end in =s96-c; ask for a bigger one so the shrink is ours."""
    return re.sub(r"=s\d+(-c)?$", f"=s{size}", url)


def from_url(url: str, deadline=None) -> dict:
    body, _ = fetch(size_param(url) if "googleusercontent.com" in url else url, deadline=deadline)
    return _answer(process(body), "url")


def from_path(path: str) -> dict:
    if not ALLOW_LOCAL:
        raise FaceError("refused", "local pictures are switched off on this bridge (SKIPPY_FACE_ALLOW_LOCAL=1 enables them)")
    p = Path(str(path)).expanduser()
    try:
        if not p.is_file() or p.stat().st_size > MAX_PICTURE_BYTES:
            raise FaceError("bad_image", "no such picture, or it is too large")
        data = p.read_bytes()
    except OSError:
        raise FaceError("bad_image", "that file could not be read") from None
    return _answer(process(data), "path")


class PortalUnavailable(FaceError):
    def __init__(self, msg="the portal link is not set up yet"):
        super().__init__("unavailable", msg)


def portal_redeem(code: str) -> dict:
    """STUB until the portal side exists (stage/gmod/FACE_PORTAL_HANDOFF.md is the contract).

    With SKIPPY_PORTAL_URL and SKIPPY_PORTAL_SECRET set it POSTs {"code": code} to <url>/api/game-link/redeem with the secret in the
    X-Game-Link-Secret header and expects {"ok": true, "picture_url": "https://...", "display_name": "..."}; otherwise PortalUnavailable.
    Tests replace this function."""
    base, secret = os.environ.get("SKIPPY_PORTAL_URL", ""), os.environ.get("SKIPPY_PORTAL_SECRET", "")
    if not base or not secret:
        raise PortalUnavailable()
    req = urllib.request.Request(base.rstrip("/") + "/api/game-link/redeem", data=json.dumps({"code": code}).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Game-Link-Secret": secret})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            obj = json.loads(r.read(8192))
    except urllib.error.HTTPError as e:
        raise FaceError("no_picture", "that code was not accepted") if e.code in (400, 404, 410, 429) else PortalUnavailable() from None
    except (OSError, ValueError):
        raise PortalUnavailable() from None
    if not (isinstance(obj, dict) and obj.get("ok") is True and isinstance(obj.get("picture_url"), str)):
        raise FaceError("no_picture", "that code was not accepted")
    return obj


def from_code(code: str, deadline=None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9-]{4,16}", code or ""):
        raise FaceError("refused", "that is not a link code")
    got = portal_redeem(code)
    ans = from_url(got["picture_url"], deadline)
    ans["source"] = "link"
    if isinstance(got.get("display_name"), str):
        ans["display_name"] = got["display_name"][:60]
    return ans


# ---------------------------------------------------------------------------------------------------------------- the bridge hook
def ensure_gate(bridge) -> None:
    """Give the bridge's Gate a budget and a slot for /face (the Gate was built before this module was first used)."""
    g = bridge.GATE
    g.per_minute.setdefault("face", 30)
    g.per_caller.setdefault("face", 6)
    if "face" not in g.rate:
        g.rate["face"] = bridge.Window(g.per_minute["face"], 60, g.clock)
    if "face" not in g.res:
        g.res["face"] = bridge.Resource(2, 2, 3.0)


def handle(body: dict, deadline=None) -> dict:
    """The pure part of the route: body -> answer dict (never raises)."""
    act = body.get("action")
    try:
        if act == "steam":
            return from_steam(str(body.get("steamid64", "")), deadline)
        if act == "url":
            return from_url(str(body.get("url", "")), deadline)
        if act == "path":
            return from_path(body.get("path", ""))
        if act == "redeem":
            return from_code(str(body.get("code", "")), deadline)
        return {"ok": False, "kind": "refused", "error": "unknown action"}
    except FaceError as e:
        return {"ok": False, "kind": e.kind, "error": e.msg}


def route(handler, body: dict, who: str, bridge) -> None:
    ensure_gate(bridge)
    bridge.GATE.admit("face", who)
    deadline = time.monotonic() + 15
    with bridge.GATE.slot("face", getattr(bridge._ctx, "cancel", None)):
        out = handle(body, deadline)
    handler._json(200, out)
