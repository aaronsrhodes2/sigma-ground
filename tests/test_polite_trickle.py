"""scripts/polite_trickle_fetch.py with an injected fetcher and a fake clock: nothing touches the network.

Ground truth: the behaviour the Wolfram trickle promises (one request per item ever, a daily budget, a pause when the server asks, a network error costs nothing) and plain arithmetic for the delays.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import polite_trickle_fetch as pt  # noqa: E402

Q = [{"id": n, "url": f"https://example.invalid/{n}", "expect": n, "file": f"{n}.json"} for n in ("eris", "haumea", "orcus")]


def make(tmp, fetch, **kw):
    sleeps = []
    t = pt.Trickle(Q, Path(tmp), fetch=fetch, sleep=sleeps.append, rand=lambda: 0.5, log=lambda m: None, **kw)
    return t, sleeps


class TestTrickle(unittest.TestCase):
    def test_each_item_is_asked_once_ever_and_a_rerun_asks_nothing(self):                # BUG: the Wolfram promise
        calls = []

        def fetch(url):
            calls.append(url)
            return 200, f'{{"object":"{url.rsplit("/", 1)[1]}"}}'.encode(), {}
        with tempfile.TemporaryDirectory() as d:
            t, _ = make(d, fetch)
            self.assertEqual(t.run(), 0)
            self.assertEqual(len(calls), 3)
            t2, _ = make(d, fetch)                                                       # a new process reading the saved state
            self.assertEqual(t2.run(), 0)
            self.assertEqual(len(calls), 3)

    def test_delays_are_the_interval_with_jitter_and_none_after_the_last(self):          # BUG: arithmetic
        with tempfile.TemporaryDirectory() as d:
            t, sleeps = make(d, lambda u: (200, u.encode(), {}), interval_s=300.0, jitter=0.3)
            t.run()
            self.assertEqual(sleeps, [300.0, 300.0])                                      # rand() = 0.5 -> zero jitter; 3 items -> 2 gaps
            self.assertAlmostEqual(pt.Trickle(Q, Path(d), rand=lambda: 1.0, interval_s=300.0, jitter=0.3, log=lambda m: None).next_delay(), 390.0)
            self.assertAlmostEqual(pt.Trickle(Q, Path(d), rand=lambda: 0.0, interval_s=300.0, jitter=0.3, log=lambda m: None).next_delay(), 210.0)

    def test_the_daily_budget_stops_the_queue_for_the_day(self):                         # BUG
        with tempfile.TemporaryDirectory() as d:
            t, _ = make(d, lambda u: (200, u.encode(), {}), daily_budget=2)
            self.assertEqual([t.step(), t.step(), t.step()], ["done", "done", "budget"])

    def test_retry_after_is_honoured_and_three_failures_stop_the_job(self):              # BUG
        with tempfile.TemporaryDirectory() as d:
            t, sleeps = make(d, lambda u: (429, b"slow down", {"Retry-After": "120"}))
            self.assertEqual(t.run(), 2)
            self.assertEqual(sleeps, [120.0, 120.0])                                      # slept as asked, then stopped on the third failure
            self.assertEqual(t.state["done"], {})

    def test_a_network_error_costs_no_budget_and_leaves_the_item_queued(self):           # BUG
        def boom(url):
            raise OSError("no route")
        with tempfile.TemporaryDirectory() as d:
            t, _ = make(d, boom)
            self.assertEqual(t.step(), "retry")
            self.assertEqual(t.calls_today(), 0)
            self.assertEqual(len(t.pending()), 3)

    def test_a_wrong_body_is_rejected_kept_aside_and_never_used(self):                   # BUG
        with tempfile.TemporaryDirectory() as d:
            t, _ = make(d, lambda u: (200, b'{"object":"Pluto"}', {}))
            self.assertEqual(t.step(), "rejected")
            self.assertTrue((Path(d) / "rejected" / "eris.json").exists())
            self.assertFalse((Path(d) / "eris.json").exists())
            self.assertNotIn("eris", [q["id"] for q in t.pending()])

    def test_a_good_file_is_never_overwritten(self):                                      # BY CONSTRUCTION
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "eris.json").write_bytes(b"ORIGINAL eris")
            t, _ = make(d, lambda u: (200, b"NEW eris", {}))
            t.step()
            self.assertEqual((Path(d) / "eris.json").read_bytes(), b"ORIGINAL eris")


if __name__ == "__main__":
    unittest.main()
