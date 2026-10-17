"""A polite trickle for public APIs: one request at a time, minutes apart, a daily budget, each target asked ONCE ever, pause when the server asks us to slow down.

Modelled on SkippyTel/services/wa-trickle (the Wolfram|Alpha trickle): the answer is frozen when it arrives; a network error costs nothing and the item stays queued; a refusal pauses the queue.

queue file (JSON list): [{"id": "eris", "url": "https://...", "expect": "Eris", "file": "sbdb_eris.json"}, ...]
  expect : text that must appear in the response (case-insensitive) or the response is REJECTED (kept under rejected/, never used)
state  : <out-dir>/trickle_state.json  (done / rejected / attempts; delete nothing by hand)

Politeness rules: sequential only; --interval-s between requests (default 300) with +/- --jitter (default 0.3); --daily-budget requests per UTC day; honours Retry-After on HTTP 429/503 (else pauses
one hour); stops after 3 consecutive failures; a User-Agent that names this job and carries no personal data.

usage: python scripts/polite_trickle_fetch.py --queue Q.json --out-dir DIR [--interval-s 300] [--daily-budget 24] [--once]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = "sigma-ground-trickle/1.0 (research job, one request per few minutes)"


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def http_get(url: str, timeout_s: float = 60.0) -> tuple[int, bytes, dict]:
    """(status, body, headers). HTTP errors are returned, not raised; network errors raise OSError."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers or {})


class Trickle:
    def __init__(self, queue: list[dict], out_dir: Path, *, interval_s: float = 300.0, jitter: float = 0.3, daily_budget: int = 24, fetch=http_get, sleep=time.sleep,
                 rand=random.random, log=print) -> None:
        self.queue, self.out, self.interval, self.jitter, self.budget = queue, Path(out_dir), interval_s, jitter, daily_budget
        self.fetch, self.sleep, self.rand, self.log = fetch, sleep, rand, log
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "rejected").mkdir(exist_ok=True)
        self.state_path = self.out / "trickle_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"done": {}, "rejected": {}, "attempts": {}, "budget": {}}

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def pending(self) -> list[dict]:
        return [q for q in self.queue if q["id"] not in self.state["done"] and q["id"] not in self.state["rejected"]]

    def calls_today(self) -> int:
        return self.state["budget"].get(today(), 0)

    def next_delay(self) -> float:
        return self.interval * (1.0 + self.jitter * (2.0 * self.rand() - 1.0))

    def step(self) -> str:
        """Ask the next pending item once. -> 'done' | 'rejected' | 'retry' | 'paused' | 'budget' | 'empty'"""
        todo = self.pending()
        if not todo:
            return "empty"
        if self.calls_today() >= self.budget:
            return "budget"
        item = todo[0]
        try:
            status, body, headers = self.fetch(item["url"])
        except OSError as e:                                          # a network error costs no budget; the item stays queued
            self.state["attempts"][item["id"]] = self.state["attempts"].get(item["id"], 0) + 1
            self._save()
            self.log(f"{item['id']}: network error ({e}); stays queued")
            return "retry"
        self.state["budget"][today()] = self.calls_today() + 1
        self.state["attempts"][item["id"]] = self.state["attempts"].get(item["id"], 0) + 1
        if status in (429, 503):
            wait = 3600.0
            ra = (headers.get("Retry-After") or headers.get("retry-after") or "").strip()
            if ra.isdigit():
                wait = float(ra)
            self.state["pause_s"] = wait
            self._save()
            self.log(f"{item['id']}: HTTP {status}; pausing {wait:.0f} s as asked")
            return "paused"
        text = body.decode("utf-8", errors="replace")
        if status != 200 or item["expect"].lower() not in text.lower():
            (self.out / "rejected" / item["file"]).write_bytes(body)
            self.state["rejected"][item["id"]] = f"HTTP {status}, expected text '{item['expect']}' " + ("missing" if status == 200 else "n/a")
            self._save()
            self.log(f"{item['id']}: REJECTED ({self.state['rejected'][item['id']]})")
            return "rejected"
        target = self.out / item["file"]
        if not target.exists():                                       # a good file is never overwritten
            target.write_bytes(body)
        self.state["done"][item["id"]] = {"file": item["file"], "bytes": len(body), "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self._save()
        self.log(f"{item['id']}: ok ({len(body)} bytes) [{len(self.state['done'])}/{len(self.queue)}]")
        return "done"

    def run(self) -> int:
        fails = 0
        while self.pending():
            r = self.step()
            if r == "budget":
                self.log("daily budget used; sleeping until the next UTC day (checking hourly)")
                self.sleep(3600.0)
                continue
            if r in ("retry", "paused"):
                fails += 1
                if fails >= 3:
                    self.log("3 consecutive failures: stopping (rerun later; nothing is lost)")
                    return 2
                self.sleep(float(self.state.pop("pause_s", 0.0)) or self.next_delay())
                continue
            fails = 0
            if self.pending():
                self.sleep(self.next_delay())
        self.log(f"queue finished: {len(self.state['done'])} good, {len(self.state['rejected'])} rejected")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--interval-s", type=float, default=300.0)
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--daily-budget", type=int, default=24)
    ap.add_argument("--once", action="store_true", help="ask one item and exit")
    a = ap.parse_args()
    t = Trickle(json.loads(a.queue.read_text()), a.out_dir, interval_s=a.interval_s, jitter=a.jitter, daily_budget=a.daily_budget, log=lambda m: print(m, flush=True))
    if a.once:
        print(t.step())
        return 0
    return t.run()


if __name__ == "__main__":
    sys.exit(main())
