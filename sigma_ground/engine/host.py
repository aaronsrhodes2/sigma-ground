"""WorldHost — the authoritative world stepper (engine phase E2).

Owns a World + ONE Region (E1's StackSimRegion today) and is the SINGLE
WRITER of its hot state: exactly one thread ever touches the region — the
paced loop while start() is active, the calling thread otherwise. No locks
guard sim state because no second toucher exists; everything from other
threads (commands, save, hash) marshals through a queue that is drained
ONLY at step boundaries — the same apply-then-step-at-index-k convention
E1's golden replay proved byte-identical (region.py's binding contract:
Region.apply is SYNCHRONOUS at a step boundary, never queued inside).

Determinism doctrine, honored to the letter:
* fixed dt; the wall clock only decides HOW MANY fixed-dt steps to take —
  it never enters sim state. StackSim.apply_force computes
  t_end = t + duration and apply_tilt captures t0 = t at CALL time; every
  command applies at a recorded step_count where sim.t is the identical
  accumulated +=dt sequence, so replay matches bit-for-bit.
* the command LOG is the state: save() carries {scenario spec, dt,
  step_count, command log, format version} and NO raw float state, NO wall
  clock. load() rebuilds the scenario and replays the log headlessly; the
  embedded state_hash is a self-verification checksum, not state.
* one hashing scheme: bench.determinism._floats_to_hash over a fixed-order,
  length-prefixed float list (never a second sha256 composition).

DISCLOSED couplings and deferrals:
* _full_state_floats reads StackSim privates (_forces/_tilt/_holds) so a
  save mid-force-window or mid-tilt-ramp hashes differently from a dropped
  window — positions alone would lie. A push_off refactor breaks this
  LOUDLY (hash mismatch is the drift signal); push_off is frozen this phase.
* zero physics arithmetic lives in this file — the only float handling is
  hash-list encoding and pace clamping (an engine command, never physics).
* live multi-submitter ordering is FIFO-by-arrival (inherently
  nondeterministic in advance), but the log records whatever order actually
  applied, so replay is still exact.
* observer queues are UNBOUNDED: a stuck observer accumulates memory and
  never blocks physics. Backpressure/eviction is E4's lane.
* snapshot() may be called from any thread (the cockpit's per-client hello
  frame); it is a display read only — the hash never reads frame().
"""
from __future__ import annotations

import collections
import math
import queue
import threading
import time
from dataclasses import dataclass, field

from ..deckard.scene import build_stack
from ..dynamics.mechanisms.applied import AppliedForce
from ..radiance.push_off import make_lookups
from .bench.determinism import _floats_to_hash, _state_stacksim
from .ecs import StaleEntityError, World
from .region import RegionRefusal, StackSimRegion

SIM_DT = 1.0 / 960.0
SEND_DT = 1.0 / 30.0            # sim-time between broadcast snapshots
MAX_CATCHUP = 4000              # bounded catch-up: max steps per iteration
PACE_MIN, PACE_MAX = 0.01, 10.0
SAVE_FORMAT = "sg-world-save"
SAVE_VERSION = 1


class HostBusy(RuntimeError):
    """A headless-only call arrived while the paced loop owns stepping."""


class WorldReplayError(RuntimeError):
    """load() replayed the command log but the hash mismatched the save."""


def _non_finite_path(obj, _path="force", _depth=0) -> "str | None":
    """First path inside a wire dict holding a non-finite/overflowing number,
    or None if every number is a finite double. Guards the command log's
    strict-JSON restorability invariant (json.dumps(save, allow_nan=False)
    must always succeed — a logged NaN/Infinity makes every later save
    unreadable by a browser's JSON.parse). Arbitrary-precision JSON ints
    beyond double range count as overflowing (float() would raise
    OverflowError on the loop thread otherwise). Depth-capped: deeper
    nesting than any legitimate wire dict is itself refused."""
    if _depth > 6:
        return _path + " (nesting too deep for a wire dict)"
    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)):
        try:
            return None if math.isfinite(float(obj)) else _path
        except OverflowError:
            return _path
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad = _non_finite_path(v, f"{_path}.{k}", _depth + 1)
            if bad is not None:
                return bad
        return None
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad = _non_finite_path(v, f"{_path}[{i}]", _depth + 1)
            if bad is not None:
                return bad
        return None
    return None                     # strings/None: not numbers, not ours


def _canonical(af: AppliedForce) -> dict:
    """The log/wire NORMAL FORM of an AppliedForce — exact floats.

    Never to_plug(): the plug disclosure rounds vector/twist/point to 6 dp,
    while region routing reads the raw fields. Round-trip guarantee
    (verified against applied.py): AppliedForce.from_dict(_canonical(af))
    == af (dataclass eq), and json.dumps/loads round-trips every double
    exactly (shortest-round-trip repr).
    """
    return {"kind": af.kind, "target": af.target, "label": af.label,
            "vector": list(af.vector) if af.vector is not None else None,
            "twist": list(af.twist) if af.twist is not None else None,
            "point": list(af.point) if af.point is not None else None,
            "t0": af.t0, "duration": af.duration, "power_w": af.power_w,
            "variables": dict(af.variables)}


def _full_state_floats(sim, step_count) -> list:
    """The FULL hot state as a flat float list — hash input, fixed order.

    Includes the in-flight god-hand state (_forces windows, _tilt ramp,
    _holds) because GATE 2 saves mid-window: positions alone would call a
    dropped force window "equal" at save time. Variable-length sections are
    length-prefixed (and _tilt carries a presence flag) so the encoding is
    unambiguous. DISCLOSED: reads StackSim privates — see module docstring.
    """
    floats = list(_state_stacksim(sim))                # dx,dz,vx,vz per body
    floats += [sim.t, float(step_count), sim.gx, sim.gz]
    floats.append(float(len(sim._forces)))
    for (b, a, t_end) in sim._forces:                  # push_off shape: (body, ax, t_end)
        floats += [float(b), a, t_end]
    if sim._tilt is not None:
        ang, t0, t1 = sim._tilt
        floats += [1.0, ang, t0, t1]                   # presence flag, no NaN games
    else:
        floats += [0.0, 0.0, 0.0, 0.0]
    floats.append(float(len(sim._holds)))
    floats += sorted(float(i) for i in sim._holds)
    floats += [-1.0 if s is None else -2.0 if isinstance(s, str) else float(s)
               for s in sim.support]
    return floats


@dataclass
class _Command:
    """One cross-thread ticket, resolved at a step boundary by the owner."""
    op: str                     # "apply" | "save" | "hash"
    target: "str | None" = None
    force: "dict | None" = None
    id: object = None           # opaque client correlation token
    done: threading.Event = field(default_factory=threading.Event)
    result: "dict | None" = None

    def wait(self, timeout: float = 5.0) -> dict:
        if not self.done.wait(timeout):
            raise TimeoutError(
                f"command {self.op!r} was never drained within {timeout}s — "
                f"is the host loop running?")
        return self.result


class WorldHost:
    """One authoritative world: a Region, a command queue, observers, a log.

    See the module docstring for the single-writer/threading contract. The
    ``region`` parameter is singular on purpose — E5 multi-region enters via
    a SAVE_VERSION bump, and nothing multi-region is built here.
    """

    def __init__(self, region, *, scenario: dict, dt: float = SIM_DT):
        self.region = region
        self.scenario = dict(scenario)
        self.dt = float(dt)
        self.step_count = 0
        self.command_log = []           # [(step_index, target|None, canonical dict)]
        sim = region.sim
        self._labels = {sim.names[i]: region.entity_ids[i]
                        for i in range(sim.n)}
        self._queue = collections.deque()
        self._qlock = threading.Lock()          # guards the queue ONLY, never sim
        self._observers = []
        self._olock = threading.Lock()
        self._ctl = threading.Lock()            # guards _pace/_paused/_rebase ONLY
        self._pace = 1.0
        self._paused = False
        self._rebase = False
        self._thread = None
        self._stop = threading.Event()

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_items(cls, items, *, pushed_index: int = 1,
                   dt: float = SIM_DT) -> "WorldHost":
        built = build_stack(list(items), allow_llm=False)
        return cls.from_built(built, items=list(items),
                              pushed_index=pushed_index, dt=dt)

    @classmethod
    def from_built(cls, built, *, items, pushed_index: int = 1,
                   dt: float = SIM_DT) -> "WorldHost":
        """Test/perf seam: skip the rebuild when a built list is at hand.

        The scenario still records ``items`` honestly — determinism.py
        separately proves double-builds are bit-equal, so a later
        from_items() of the same items reproduces this world.
        """
        mu, e = make_lookups(pushed_index)
        region = StackSimRegion(World(), built, mu_lookup=mu, e_lookup=e)
        scenario = {"builder": "stacksim", "items": list(items),
                    "pushed_index": int(pushed_index)}
        return cls(region, scenario=scenario, dt=dt)

    # ── the ONE apply path (owning thread, at a step boundary only) ───────
    def _apply_at_boundary(self, target, force_dict) -> dict:
        """Validate + route one raw force dict; refuse EVERYTHING invalid.

        This runs on the loop thread when the paced loop is live — an
        exception escaping here KILLS the world (the loop thread dies, the
        world stays a corpse in any registry holding it, every observer's
        frames stop). Adversarially proven, not hypothetical: a raw
        force="impulse" string (AttributeError inside from_dict) and a
        400-digit JSON-integer duration (OverflowError from float()) both
        did exactly that before the guards below existed. Hence: (a) the
        dict/finite pre-checks, (b) the widened except tuple, (c) the
        last-resort Exception arm — hostile DATA must never become an
        escaped exception on the authoritative thread.
        """
        try:
            if not isinstance(force_dict, dict):
                return {"ok": False, "refusal":
                        f"force must be a JSON object (an AppliedForce wire "
                        f"dict), got {type(force_dict).__name__}"}
            bad = _non_finite_path(force_dict)
            if bad is not None:
                return {"ok": False, "refusal":
                        f"non-finite or overflowing number at {bad} — the "
                        f"command log must stay strict-JSON restorable "
                        f"(NaN/Infinity would poison every later save), so "
                        f"non-finite floats are refused at the boundary"}
            af = AppliedForce.from_dict(force_dict)     # ValueError = refusal
            if target is None:
                eid = None                              # scene-scoped (tilt)
            elif target in self._labels:
                eid = self._labels[target]
            else:
                raise KeyError(f"unknown target {target!r} — bodies here: "
                               f"{sorted(self._labels)}")
            self.region.apply(eid, af)
        except (ValueError, TypeError, KeyError, AttributeError,
                ArithmeticError, RegionRefusal, StaleEntityError) as e:
            return {"ok": False, "refusal": str(e)}     # honest; NEVER logged
        except Exception as e:                          # pragma: no cover
            # Last resort: an UNANTICIPATED shape of hostile data. Refusing
            # is strictly better than killing the world's loop thread; the
            # exception type is named so a real engine bug still surfaces
            # loudly in the refusal text instead of being classified away.
            return {"ok": False, "refusal":
                    f"internal: unhandled {type(e).__name__} validating "
                    f"this force ({e}) — refused rather than killing the "
                    f"world loop; report this shape"}
        canon = _canonical(af)
        self.command_log.append((self.step_count, target, canon))
        return {"ok": True, "step_index": self.step_count,
                "target": target, "force": canon}

    def _step_once(self):
        self.region.step(self.dt)
        self.step_count += 1

    def _drain(self):
        """Resolve every queued ticket — called ONLY at step boundaries."""
        with self._qlock:
            cmds = list(self._queue)
            self._queue.clear()
        for cmd in cmds:                                # FIFO by arrival
            if cmd.op == "apply":
                cmd.result = self._apply_at_boundary(cmd.target, cmd.force)
            elif cmd.op == "save":
                cmd.result = {"ok": True, "save": self._save_dict()}
            elif cmd.op == "hash":
                cmd.result = {"ok": True, "hash": self._hash_now()}
            else:                                       # unreachable by API
                cmd.result = {"ok": False,
                              "refusal": f"unknown control op {cmd.op!r}"}
            cmd.done.set()

    # ── headless API (the caller IS the owning thread) ────────────────────
    def _loop_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def _require_headless(self, what: str):
        if self._loop_running():
            raise HostBusy(
                f"{what} is a headless call — the paced loop owns stepping "
                f"while start() is active (paused or not); use submit()/"
                f"request_save()/request_hash(), or stop() first")

    def apply(self, target, force_dict) -> dict:
        """Apply one wire-shape force dict at the current step boundary."""
        self._require_headless("apply()")
        return self._apply_at_boundary(target, force_dict)

    def advance(self, n_steps: int):
        """Advance exactly n fixed-dt steps, draining due commands at every
        step boundary (the headless twin of the paced loop's iteration)."""
        self._require_headless("advance()")
        for _ in range(int(n_steps)):
            self._drain()
            self._step_once()

    def emit_snapshot(self):
        """Headless broadcast: hand the current snapshot to every observer."""
        self._require_headless("emit_snapshot()")
        self._emit()

    # ── threaded API (any thread) ─────────────────────────────────────────
    def _enqueue(self, cmd: _Command) -> _Command:
        with self._qlock:
            self._queue.append(cmd)
        if not self._loop_running():
            self._drain()       # no loop: "now" IS a step boundary
        return cmd

    def submit(self, target, force_dict, *, id=None) -> _Command:
        return self._enqueue(_Command(op="apply", target=target,
                                      force=force_dict, id=id))

    def request_save(self) -> _Command:
        return self._enqueue(_Command(op="save"))

    def request_hash(self) -> _Command:
        return self._enqueue(_Command(op="hash"))

    def set_pace(self, factor) -> float:
        f = max(PACE_MIN, min(PACE_MAX, float(factor)))
        with self._ctl:
            self._pace = f
            self._rebase = True
        return f

    def pause(self):
        """Stop consuming wall time. sim.t is untouched; commands still
        drain at the frozen boundary (refusals surface promptly)."""
        with self._ctl:
            self._paused = True

    def resume(self):
        with self._ctl:
            self._paused = False
            self._rebase = True         # rebase NEVER touches sim state

    @property
    def paused(self) -> bool:
        with self._ctl:
            return self._paused

    @property
    def pace(self) -> float:
        with self._ctl:
            return self._pace

    def start(self, pace: float = 1.0):
        if self._loop_running():
            raise HostBusy("the paced loop is already running")
        self.set_pace(pace)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="WorldHost-loop")
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        t = self._thread
        if t is None:
            return
        self._stop.set()
        t.join(timeout)
        if t.is_alive():                                # pragma: no cover
            raise TimeoutError("paced loop did not stop within timeout")
        self._thread = None

    # ── the paced loop (accumulator lifted from serve._live_session) ──────
    def _run(self):
        sim = self.region.sim
        wall0 = sim0 = None
        next_send = 0.0
        while not self._stop.is_set():
            self._drain()               # boundary: BEFORE any step this
            #                             iteration — apply-then-step-at-k,
            #                             enforced structurally
            with self._ctl:
                paused, pace, reb = self._paused, self._pace, self._rebase
                self._rebase = False
            if not paused:
                if reb or wall0 is None:
                    wall0, sim0 = time.monotonic(), sim.t
                target_t = sim0 + (time.monotonic() - wall0) * pace
                steps = 0
                while sim.t < target_t and steps < MAX_CATCHUP:
                    self._step_once()
                    steps += 1
                if sim.t >= next_send:
                    self._emit()
                    next_send = sim.t + SEND_DT
            else:
                wall0 = None            # pause = stop consuming wall time
            time.sleep(0.002)
        self._drain()                   # pending tickets always resolve

    # ── observers ─────────────────────────────────────────────────────────
    def subscribe(self) -> queue.Queue:
        q = queue.Queue()               # unbounded — see module docstring
        with self._olock:
            self._observers.append(q)
        return q

    def unsubscribe(self, q):
        with self._olock:
            try:
                self._observers.remove(q)
            except ValueError:
                pass

    def snapshot(self) -> dict:
        """One display frame. sim.frame()'s t_sim/pos are rounded display
        values — the hash never reads frame(). No world name here: the host
        is registry-agnostic (the cockpit's ready message names the world)."""
        sim = self.region.sim
        with self._ctl:
            pace, paused = self._pace, self._paused
        fr = sim.frame()
        fr["type"] = "frame"
        fr["step"] = self.step_count
        fr["t"] = sim.t                                 # EXACT float
        fr["support"] = [s if s is None or isinstance(s, str)
                         else sim.names[s] for s in sim.support]
        fr["settled"] = sim.settled()
        fr["pace"] = pace
        fr["paused"] = paused
        return fr

    def _emit(self):
        """ONE snapshot dict handed to EVERY observer queue — identical
        sequences by construction (same object, same order, no filtering;
        read-only by documented convention)."""
        snap = self.snapshot()
        with self._olock:
            for q in self._observers:
                q.put(snap)

    # ── persistence: the log IS the state ─────────────────────────────────
    def _save_dict(self) -> dict:
        import json
        return {
            "format": SAVE_FORMAT,
            "version": SAVE_VERSION,
            "scenario": {"builder": self.scenario.get("builder", "stacksim"),
                         "items": list(self.scenario["items"]),
                         "pushed_index": int(self.scenario.get("pushed_index", 1))},
            "dt": self.dt,
            "step_count": self.step_count,
            # deep JSON round-trip: the save owns its lists (no aliasing into
            # the live log) and is guaranteed strict-JSON clean
            "command_log": json.loads(json.dumps(
                [[k, target, force] for (k, target, force)
                 in self.command_log])),
            # a checksum for self-verification, NOT state — the log is the
            # state. Queued-but-undrained commands are intents, not state:
            # deliberately excluded.
            "state_hash": self._hash_now(),
        }

    def save(self) -> dict:
        if self._loop_running():
            return self.request_save().wait()["save"]
        return self._save_dict()

    def _hash_now(self) -> str:
        return _floats_to_hash(
            _full_state_floats(self.region.sim, self.step_count))

    def state_hash(self) -> str:
        if self._loop_running():
            return self.request_hash().wait()["hash"]
        return self._hash_now()

    @classmethod
    def load(cls, saved: dict, *, prebuilt=None,
             verify: bool = True) -> "WorldHost":
        """Rebuild the scenario and REPLAY the command log headlessly.

        The loop structure handles the trailing-boundary case: an entry at
        k == saved step_count is applied after the final top-up positions
        at k — exactly as it was applied-but-unstepped at save time.
        """
        if not isinstance(saved, dict) or saved.get("format") != SAVE_FORMAT:
            raise ValueError(
                f"not a {SAVE_FORMAT!r} save "
                f"(format={None if not isinstance(saved, dict) else saved.get('format')!r})")
        if saved.get("version") != SAVE_VERSION:
            raise ValueError(
                f"unsupported save version {saved.get('version')!r} — this "
                f"host reads version {SAVE_VERSION}")
        scen = saved["scenario"]
        if scen.get("builder") != "stacksim":
            raise ValueError(
                f"unknown scenario builder {scen.get('builder')!r} — this "
                f"host rebuilds 'stacksim' scenarios only")
        items = list(scen["items"])
        pushed_index = int(scen.get("pushed_index", 1))
        dt = float(saved["dt"])
        if prebuilt is not None:
            host = cls.from_built(prebuilt, items=items,
                                  pushed_index=pushed_index, dt=dt)
        else:
            host = cls.from_items(items, pushed_index=pushed_index, dt=dt)
        for entry in saved["command_log"]:
            k, target, force = entry[0], entry[1], entry[2]
            while host.step_count < k:
                host._step_once()
            res = host._apply_at_boundary(target, force)
            if not res["ok"]:
                raise ValueError(
                    f"corrupt save: logged command refused on replay: "
                    f"{res['refusal']}")
        while host.step_count < int(saved["step_count"]):
            host._step_once()
        if verify and saved.get("state_hash"):
            got = host._hash_now()
            if got != saved["state_hash"]:
                raise WorldReplayError(
                    f"replay hash mismatch: save says "
                    f"{saved['state_hash'][:16]}…, replay produced "
                    f"{got[:16]}… — physics or the save format drifted")
        return host


__all__ = ["SIM_DT", "SEND_DT", "MAX_CATCHUP", "PACE_MIN", "PACE_MAX",
           "SAVE_FORMAT", "SAVE_VERSION", "HostBusy", "WorldReplayError",
           "WorldHost"]
