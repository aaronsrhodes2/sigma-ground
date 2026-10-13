"""The bridge's /face action (stage/gmod/holodeck/skippy_faces.py), offline: the network is monkeypatched (DNS and the one-GET transport),
the pictures are synthetic (drawn with Pillow), nothing real is fetched and no real person's photo is used."""
from __future__ import annotations

import sys
import time
from io import BytesIO
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

HOLO = Path(__file__).resolve().parents[3] / "stage" / "gmod" / "holodeck"
sys.path.insert(0, str(HOLO))
import skippy_faces as F  # noqa: E402

PUBLIC = "142.250.1.1"
STEAM_ID = "76561197960287930"
AVATAR = "https://avatars.akamai.steamstatic.com/0123456789abcdef0123456789abcdef01234567_full.jpg"
# the shape of the real profile XML (fetched once for Valve's public profile 76561197960287930 on 2026-10-12; hash replaced)
PROFILE_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><profile>
\t<steamID64>{STEAM_ID}</steamID64>
\t<steamID><![CDATA[Someone]]></steamID>
\t<privacyState>friendsonly</privacyState>
\t<avatarIcon><![CDATA[{AVATAR.replace('_full', '')}]]></avatarIcon>
\t<avatarMedium><![CDATA[{AVATAR.replace('_full', '_medium')}]]></avatarMedium>
\t<avatarFull><![CDATA[{AVATAR}]]></avatarFull>
</profile>""".encode()


def synthetic_face(w=300, h=200, fmt="PNG", mode="RGB"):
    im = Image.new("RGB", (w, h), (30, 90, 160))
    d = ImageDraw.Draw(im)
    d.ellipse((w * .3, h * .1, w * .7, h * .9), fill=(230, 190, 160))
    d.ellipse((w * .4, h * .35, w * .46, h * .45), fill=(20, 20, 20))
    d.ellipse((w * .54, h * .35, w * .6, h * .45), fill=(20, 20, 20))
    buf = BytesIO()
    im.convert(mode).save(buf, fmt)
    return buf.getvalue()


class Net:
    """A fake internet: name -> addresses; (host, path) -> (status, headers, body). Records every transport call."""

    def __init__(self, mp):
        self.dns, self.pages, self.calls = {}, {}, []
        mp.setattr(F, "_getaddrinfo", self.getaddrinfo)
        mp.setattr(F, "_transport", self.transport)

    def getaddrinfo(self, host, port):
        if host not in self.dns:
            raise OSError("no such host")
        return [(2, 1, 6, "", (a, port)) for a in self.dns[host]]

    def transport(self, host, ip, port, path, timeout, max_bytes):
        self.calls.append((host, ip, path))
        status, headers, body = self.pages[(host, path)]
        return status, {k.lower(): v for k, v in headers.items()}, body

    def page(self, host, path, body, ctype="image/png", status=200, **headers):
        self.dns.setdefault(host, [PUBLIC])
        self.pages[(host, path)] = (status, {"Content-Type": ctype, **headers}, body)


@pytest.fixture
def net(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "CACHE_DIR", tmp_path / "cache")
    F._steam_seen.clear()
    return Net(monkeypatch)


# ------------------------------------------------------------------------------------------------------------ allow-list
@pytest.mark.parametrize("url", [
    "http://lh3.googleusercontent.com/a", "https://example.com/a.png", "https://evilgooglecontent.com/a",
    "https://lh3.googleusercontent.com.evil.com/a", "https://googleusercontent.com/a", "https://127.0.0.1/a", "https://[::1]/a",
    "https://user:pw@lh3.googleusercontent.com/a", "https://lh3.googleusercontent.com:8443/a", "ftp://lh3.googleusercontent.com/a",
    "file:///etc/passwd", "https://lh3.googleusercontent.com/a b", "", "https://steamcommunity.com/profiles/1/?xml=1",
    "https://lh3.googleusercontent.com@evil.com/a"])
def test_urls_off_the_list_or_not_plain_https_are_refused(url):
    with pytest.raises(F.FaceError) as e:
        F.check_url(url)
    assert e.value.kind == "refused"


@pytest.mark.parametrize("url", ["https://lh3.googleusercontent.com/a/xyz=s96-c", "https://www.gravatar.com/avatar/abc?s=200",
                                 "https://avatars.akamai.steamstatic.com/x_full.jpg", "https://avatars.steamstatic.com/x.jpg"])
def test_the_named_picture_sites_are_accepted(url):
    assert F.check_url(url)[2].startswith("/")


def test_the_allow_list_is_configurable(monkeypatch):
    monkeypatch.setenv("SKIPPY_FACE_HOSTS", "pics.example.org")
    assert F.check_url("https://pics.example.org/a.png")[0] == "pics.example.org"


# ------------------------------------------------------------------------------------------------------------ SSRF
@pytest.mark.parametrize("addr", ["127.0.0.1", "10.0.0.5", "192.168.1.9", "172.16.0.1", "169.254.169.254", "100.64.0.1", "::1", "fe80::1", "0.0.0.0"])
def test_a_name_that_points_at_a_private_address_is_refused_before_any_connection(net, addr):
    net.dns["lh3.googleusercontent.com"] = [addr]
    net.pages[("lh3.googleusercontent.com", "/a")] = (200, {}, b"")
    with pytest.raises(F.FaceError) as e:
        F.fetch("https://lh3.googleusercontent.com/a")
    assert e.value.kind == "refused" and net.calls == []


def test_one_private_answer_among_public_ones_refuses_the_name(net):
    net.dns["lh3.googleusercontent.com"] = [PUBLIC, "10.1.1.1"]
    with pytest.raises(F.FaceError):
        F.fetch("https://lh3.googleusercontent.com/a")
    assert net.calls == []


def test_the_connection_goes_to_the_vetted_address_not_a_second_lookup(net):
    net.page("lh3.googleusercontent.com", "/a", synthetic_face())
    F.fetch("https://lh3.googleusercontent.com/a")
    assert net.calls == [("lh3.googleusercontent.com", PUBLIC, "/a")]


def test_a_redirect_off_the_list_is_refused_and_one_on_the_list_is_followed(net):
    net.page("lh3.googleusercontent.com", "/a", b"", status=302, Location="https://evil.example.com/x")
    with pytest.raises(F.FaceError) as e:
        F.fetch("https://lh3.googleusercontent.com/a")
    assert e.value.kind == "refused" and len(net.calls) == 1
    net.page("lh3.googleusercontent.com", "/b", b"", status=302, Location="/c")
    net.page("lh3.googleusercontent.com", "/c", synthetic_face())
    assert F.fetch("https://lh3.googleusercontent.com/b")[1] == "image/png"


def test_a_redirect_to_a_private_address_via_an_allowed_name_is_refused(net):
    net.page("lh3.googleusercontent.com", "/a", b"", status=302, Location="https://lh4.googleusercontent.com/z")
    net.dns["lh4.googleusercontent.com"] = ["192.168.0.2"]
    with pytest.raises(F.FaceError) as e:
        F.fetch("https://lh3.googleusercontent.com/a")
    assert e.value.kind == "refused"


def test_a_redirect_loop_is_cut_after_three_hops(net):
    net.page("lh3.googleusercontent.com", "/a", b"", status=302, Location="/a")
    with pytest.raises(F.FaceError) as e:
        F.fetch("https://lh3.googleusercontent.com/a")
    assert "redirect" in e.value.msg and len(net.calls) == F.MAX_REDIRECTS + 1


def test_size_type_and_magic_checks(net):
    net.page("lh3.googleusercontent.com", "/big", b"\x89PNG" + b"0" * (F.MAX_PICTURE_BYTES + 10))
    with pytest.raises(F.FaceError):
        F.fetch("https://lh3.googleusercontent.com/big")
    net.page("lh3.googleusercontent.com", "/html", b"<html>", ctype="text/html")
    with pytest.raises(F.FaceError) as e:
        F.fetch("https://lh3.googleusercontent.com/html")
    assert e.value.kind == "bad_image"
    with pytest.raises(F.FaceError):
        F.process(b"<html><body>not a picture at all</body></html>")
    with pytest.raises(F.FaceError):
        F.process(b"\xff\xd8\xff" + b"garbage" * 10)               # right magic, not a picture


def test_a_decompression_bomb_is_refused(net):
    big = Image.new("1", (20000, 20000))                          # 4e8 pixels, a few KB as PNG
    buf = BytesIO()
    big.save(buf, "PNG")
    assert len(buf.getvalue()) < F.MAX_PICTURE_BYTES
    with pytest.raises(F.FaceError) as e:
        F.process(buf.getvalue())
    assert "large" in e.value.msg


# ------------------------------------------------------------------------------------------------------------ pictures
@pytest.mark.parametrize("fmt,size", [("PNG", (300, 200)), ("JPEG", (640, 480)), ("PNG", (50, 90)), ("GIF", (200, 200)), ("WEBP", (256, 300))])
def test_any_picture_comes_out_128_square_jpeg_under_20kb(fmt, size):
    out = F.process(synthetic_face(*size, fmt=fmt), use_detector=False)
    im = Image.open(BytesIO(out["jpeg"]))
    assert im.size == (128, 128) and im.format == "JPEG" and len(out["jpeg"]) <= F.MAX_OUT_BYTES
    assert out["detector"] == "centre" and len(out["border"]) == 3


def test_transparent_pictures_land_on_grey_and_the_border_colour_is_the_rim():
    im = Image.new("RGBA", (100, 100), (255, 0, 0, 0))
    buf = BytesIO()
    im.save(buf, "PNG")
    out = F.process(buf.getvalue(), use_detector=False)
    assert all(abs(c - 128) < 6 for c in out["border"])
    out = F.process(synthetic_face(200, 200), use_detector=False)
    assert out["border"][2] > out["border"][0], "the synthetic background is blue"


def test_the_crop_centres_on_a_found_face_and_falls_back_to_the_middle(monkeypatch):
    im = Image.new("RGB", (400, 200), (0, 0, 0))
    ImageDraw.Draw(im).rectangle((300, 60, 360, 120), fill=(255, 255, 255))
    monkeypatch.setattr(F, "_face_box", lambda i: (300, 60, 60, 60))
    sq, how = F.crop_square(im)
    assert how == "haar" and sq.size[0] == sq.size[1] and sq.getpixel((sq.width // 2, sq.height // 2))[0] > 200
    monkeypatch.setattr(F, "_face_box", lambda i: None)
    sq, how = F.crop_square(im)
    assert how == "centre" and sq.size == (200, 200)


def test_the_cache_is_by_content_hash(net):
    a = F._answer(F.process(synthetic_face(), False), "url")
    b = F._answer(F.process(synthetic_face(), False), "url")
    other = F._answer(F.process(synthetic_face(260, 260), False), "url")
    assert a["hash"] == b["hash"] != other["hash"]
    assert len(list(F.CACHE_DIR.glob("*.jpg"))) == 2
    got = F.cache_get(a["hash"])
    assert got and got["border"] == a["border"] and F.cache_get("../../etc/passwd") is None


def test_old_cache_files_are_pruned(net):
    a = F._answer(F.process(synthetic_face(), False), "url")
    assert F.prune_cache(30) == 0
    assert F.prune_cache(30, now=time.time() + 31 * 86400) >= 2 and F.cache_get(a["hash"]) is None


# ------------------------------------------------------------------------------------------------------------ the sources
def serve_steam(net, xml=PROFILE_XML, picture=None):
    net.page("steamcommunity.com", f"/profiles/{STEAM_ID}/?xml=1", xml, ctype="text/xml")
    if picture is not None:
        net.page("avatars.akamai.steamstatic.com", AVATAR.split("steamstatic.com")[1], picture, ctype="image/jpeg")


def test_the_steam_avatar_is_found_in_the_profile_xml_and_fetched_from_the_cdn(net):
    serve_steam(net, picture=synthetic_face(184, 184, "JPEG"))
    out = F.handle({"action": "steam", "steamid64": STEAM_ID})
    assert out["ok"] and out["source"] == "steam" and out["bytes"] <= F.MAX_OUT_BYTES
    assert [c[0] for c in net.calls] == ["steamcommunity.com", "avatars.akamai.steamstatic.com"]


def test_a_missing_profile_or_a_profile_without_a_picture_is_a_quiet_no_picture(net):
    serve_steam(net, xml=b"<!DOCTYPE html><html><body>The specified profile could not be found.</body></html>")
    out = F.handle({"action": "steam", "steamid64": STEAM_ID})
    assert out == {"ok": False, "kind": "no_picture", "error": "no public Steam profile"}
    F._steam_seen.clear()
    serve_steam(net, xml=b"<profile><steamID64>1</steamID64></profile>")
    assert F.handle({"action": "steam", "steamid64": STEAM_ID})["kind"] == "no_picture"
    F._steam_seen.clear()
    net.pages[("steamcommunity.com", f"/profiles/{STEAM_ID}/?xml=1")] = (500, {}, b"")
    assert F.handle({"action": "steam", "steamid64": STEAM_ID})["kind"] == "unavailable"


def test_a_profile_that_names_an_avatar_off_the_cdn_list_is_refused(net):
    serve_steam(net, xml=PROFILE_XML.replace(b"avatars.akamai.steamstatic.com", b"169.254.169.254"))
    out = F.handle({"action": "steam", "steamid64": STEAM_ID})
    assert not out["ok"] and out["kind"] == "refused" and [c[0] for c in net.calls] == ["steamcommunity.com"]


def test_only_real_steam_ids_are_looked_up_and_failures_are_remembered_briefly(net):
    for bad in ("", "abc", "76561197960287930/../x", "123", "7656119796028793"):
        assert F.handle({"action": "steam", "steamid64": bad})["kind"] == "refused"
    assert net.calls == []
    serve_steam(net, xml=b"<html>no</html>")
    F.handle({"action": "steam", "steamid64": STEAM_ID})
    F.handle({"action": "steam", "steamid64": STEAM_ID})
    assert len(net.calls) == 1, "a private profile is not asked again within the minutes the answer is kept"


def test_steam_profile_host_is_not_a_picture_site(net):
    out = F.handle({"action": "url", "url": f"https://steamcommunity.com/profiles/{STEAM_ID}/?xml=1"})
    assert out["kind"] == "refused" and net.calls == []


def test_a_google_picture_url_gets_a_bigger_size_asked_for(net):
    net.page("lh3.googleusercontent.com", "/a/abc=s256", synthetic_face(256, 256, "JPEG"), ctype="image/jpeg")
    out = F.handle({"action": "url", "url": "https://lh3.googleusercontent.com/a/abc=s96-c"})
    assert out["ok"] and net.calls[0][2] == "/a/abc=s256"


def test_local_files_need_the_bridge_switch(tmp_path, monkeypatch, net):
    p = tmp_path / "me.png"
    p.write_bytes(synthetic_face())
    monkeypatch.setattr(F, "ALLOW_LOCAL", False)
    assert F.handle({"action": "path", "path": str(p)})["kind"] == "refused"
    monkeypatch.setattr(F, "ALLOW_LOCAL", True)
    assert F.handle({"action": "path", "path": str(p)})["ok"]
    (tmp_path / "notes.txt").write_text("secret")
    assert F.handle({"action": "path", "path": str(tmp_path / "notes.txt")})["kind"] == "bad_image"


def test_link_codes_go_through_the_portal_stub(net, monkeypatch):
    monkeypatch.delenv("SKIPPY_PORTAL_URL", raising=False)
    assert F.handle({"action": "redeem", "code": "AB12-CD34"})["kind"] == "unavailable"
    assert F.handle({"action": "redeem", "code": "no way; drop"})["kind"] == "refused"
    net.page("lh3.googleusercontent.com", "/a/me=s256", synthetic_face(256, 256, "JPEG"), ctype="image/jpeg")
    monkeypatch.setattr(F, "portal_redeem", lambda c: {"ok": True, "picture_url": "https://lh3.googleusercontent.com/a/me=s96-c", "display_name": "Pat"})
    out = F.handle({"action": "redeem", "code": "AB12-CD34"})
    assert out["ok"] and out["source"] == "link" and out["display_name"] == "Pat"
    monkeypatch.setattr(F, "portal_redeem", lambda c: {"ok": True, "picture_url": "https://evil.example.com/x.png"})
    assert F.handle({"action": "redeem", "code": "AB12-CD34"})["kind"] == "refused"


# ------------------------------------------------------------------------------------------------------------ the bridge hook
def test_the_bridge_routes_face_through_its_gate_and_rations_it(net, monkeypatch):
    import skippy_bridge as B
    monkeypatch.setattr(B, "GATE", B.Gate())
    sent = []

    class H:
        def _json(self, code, obj):
            sent.append((code, obj))
    serve_steam(net, picture=synthetic_face(184, 184, "JPEG"))
    for _ in range(6):
        F.route(H(), {"action": "steam", "steamid64": STEAM_ID}, "7656", B)
    assert len(sent) == 6 and all(c == 200 and o["ok"] for c, o in sent)
    with pytest.raises(B.Over):
        F.route(H(), {"action": "steam", "steamid64": STEAM_ID}, "7656", B)
    src = (HOLO / "skippy_bridge.py").read_text(encoding="utf-8")
    assert 'self.path == "/face"' in src
