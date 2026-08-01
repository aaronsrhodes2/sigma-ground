"""God-hand live layer: StackSim's taxonomy entry points + websocket frames.

The live loop steps the SAME StackSim the recorder bakes through — these
tests pin the four live kinds (impulse, force, tilt, hold) doing real
physics, and the stdlib RFC 6455 endpoint round-tripping frames.
"""
import json
import math
import socket
import struct
import unittest

from sigma_ground.deckard.scene import build_stack
from sigma_ground.radiance.push_off import StackSim, make_lookups

_BUILT = None


def _fresh_sim():
    global _BUILT
    if _BUILT is None:
        _BUILT = build_stack(["1700s table", "plate", "Doughnut"],
                             allow_llm=False)
    mu, e = make_lookups(1)
    return StackSim(_BUILT, mu_lookup=mu, e_lookup=e)


def _run(sim, seconds, dt=1.0 / 960.0):
    for _ in range(int(seconds / dt)):
        sim.step(dt)


class TestGodHandKinds(unittest.TestCase):
    def test_untouched_stack_stays_put(self):
        sim = _fresh_sim()
        _run(sim, 1.0)
        self.assertEqual(sim.dx, [0.0] * sim.n)
        self.assertEqual(sim.support, [None, 0, 1])

    def test_impulse_mid_run_is_the_push(self):
        sim = _fresh_sim()
        _run(sim, 0.5)                       # idle first — then the god-hand
        sim.apply_impulse(1, 2.2)
        _run(sim, 4.0)
        # the tablecloth outcome, same physics as the recorder at 2.2 m/s
        self.assertEqual(sim.support[1], "floor")
        self.assertEqual(sim.support[2], 0)

    def test_force_below_static_grip_does_nothing(self):
        sim = _fresh_sim()
        m_plate = sim.mass[1] + sim.mass[2]  # plate carries the doughnut
        weak = 0.3 * 0.35 * 9.80665 * m_plate
        sim.apply_force(1, weak, 1.0)
        _run(sim, 2.0)
        self.assertEqual(sim.support, [None, 0, 1])
        self.assertLess(abs(sim.dx[1]), 1e-6)

    def test_strong_sustained_force_slides_the_plate(self):
        sim = _fresh_sim()
        m_plate = sim.mass[1] + sim.mass[2]
        strong = 3.0 * 0.35 * 9.80665 * m_plate
        sim.apply_force(1, strong, 1.5)
        _run(sim, 4.0)
        self.assertNotEqual(sim.support[1], 0)   # driven off the table

    def test_tilt_beyond_friction_angle_slides_everything(self):
        """tan(25 deg) = 0.466 > mu 0.35: once the ramp completes, the
        plate must slip — gravity re-aimed IS the operator's tray."""
        sim = _fresh_sim()
        sim.apply_tilt(25.0, 1.0)
        _run(sim, 4.0)
        self.assertGreater(abs(sim.dx[1]), 0.01)

    def test_tilt_below_friction_angle_holds(self):
        sim = _fresh_sim()
        sim.apply_tilt(10.0, 1.0)            # tan = 0.176 < 0.35
        _run(sim, 3.0)
        self.assertEqual(sim.support, [None, 0, 1])
        self.assertLess(abs(sim.dx[1]), 1e-3)

    def test_hold_is_an_artificial_bracket(self):
        """Pin the doughnut, push the plate away — the doughnut hangs at
        its height (the disclosed artificial bracket), then release drops
        it onto the table."""
        sim = _fresh_sim()
        sim.hold(2, True)
        sim.apply_impulse(1, 3.0)
        _run(sim, 3.0)
        self.assertEqual(sim.support[1], "floor")     # plate long gone
        self.assertEqual(sim.dz[2], 0.0)              # doughnut pinned midair
        sim.hold(2, False)
        _run(sim, 2.0)
        self.assertEqual(sim.support[2], 0)           # released → lands on table
        self.assertLess(sim.dz[2], -1e-4)


class TestWebSocketFraming(unittest.TestCase):
    def test_round_trip_over_a_socketpair(self):
        from sigma_ground.radiance.web.serve import _WebSocket
        a, b = socket.socketpair()
        try:
            ws = _WebSocket(a)
            # server → client: unmasked text frame
            ws.send_json({"type": "frame", "t": 1.5})
            raw = b.recv(65536)
            self.assertEqual(raw[0], 0x81)
            n = raw[1]
            self.assertEqual(json.loads(raw[2:2 + n].decode()),
                             {"type": "frame", "t": 1.5})
            # client → server: masked text frame (per RFC, clients mask)
            payload = json.dumps({"type": "apply", "x": 2}).encode()
            mask = b"\x01\x02\x03\x04"
            masked = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
            b.sendall(struct.pack("!BB", 0x81, 0x80 | len(payload))
                      + mask + masked)
            msg = ws.recv_json(timeout=1.0)
            self.assertEqual(msg, {"type": "apply", "x": 2})
            # ping (unmasked, empty) → parser replies pong and keeps reading
            b.sendall(struct.pack("!BB", 0x89, 0))
            self.assertIsNone(ws.recv_json(timeout=0.05))   # nothing else queued
            pong = b.recv(16)
            self.assertEqual(pong[0], 0x8A)
        finally:
            a.close()
            b.close()

    def test_close_frame_returns_none(self):
        from sigma_ground.radiance.web.serve import _WebSocket
        a, b = socket.socketpair()
        try:
            ws = _WebSocket(a)
            b.sendall(struct.pack("!BB", 0x88, 0x80) + b"\x00\x00\x00\x00")
            self.assertIsNone(ws.recv_json(timeout=1.0))
        finally:
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
