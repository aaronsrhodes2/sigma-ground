"""Deckard is a polite data citizen: it identifies itself, throttles per host,
and caches so it never re-pulls. All offline — no real network is touched.
"""
import time

from sigma_ground.deckard.sources import web


def test_user_agent_identifies_with_a_contact_url():
    assert "Deckard" in web._UA
    assert "http" in web._UA          # a reachable contact, per Wikimedia API policy


def test_rate_limiter_throttles_repeat_hits_to_a_host(monkeypatch):
    monkeypatch.setattr(web, "_MIN_INTERVAL_S", 0.05)
    web._last_hit.clear()
    t0 = time.monotonic()
    web._rate_limit("https://example.org/a")      # first hit: no wait
    web._rate_limit("https://example.org/b")      # same host again: must wait
    assert (time.monotonic() - t0) >= 0.05
    # a different host is independent (not throttled by the first)
    t1 = time.monotonic()
    web._rate_limit("https://other.example.net/x")
    assert (time.monotonic() - t1) < 0.05


def test_cache_prevents_a_second_network_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(web, "_MIN_INTERVAL_S", 0.0)
    web._last_hit.clear()
    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": 1}'

    def fake_urlopen(req, timeout=10.0):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(web.urllib.request, "urlopen", fake_urlopen)
    url = "https://example.org/data.json"
    assert web.get_json(url) == {"ok": 1}         # network hit, then cached
    assert web.get_json(url) == {"ok": 1}         # served from the on-disk cache
    assert calls["n"] == 1                        # only ONE real network hit
