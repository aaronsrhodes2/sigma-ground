"""
SkippyCraft Phase 1: a LIVE theater. sigma-ground's ``VoxelWorld`` ticks; Minecraft shows each tick (misc/SKIPPYCRAFT_PLAN.md).

The service owns the world. Every tick it steps ``VoxelWorld`` (heat, phase, settling), compares what each cell now holds with
what Minecraft was last told, and sends only the changed blocks over RCON (``setblock``). A boss bar shows the readout, straight
from the world's own ledger: simulated time, mass, energy, what radiated or vented, what changed. Minecraft simulates nothing:
every block it draws is a carrier repainted by the shared resource pack (``export_mc``), and carriers never move by themselves.

    python stage/skippycraft/theater.py sink --at 4 -60 20      # gold sinks through mercury, iron floats
    python stage/skippycraft/theater.py melt --at 20 -60 4      # lead melts on a 1500 K tungsten plate, in vacuum
    python stage/skippycraft/theater.py looks                    # (re)write the resource pack for every scene; F3+T in game

RCON is 127.0.0.1:42575 (dev-server/docker-compose.yml); its password is read from dev-server/data/server.properties.
Phase 1 drives the server with commands; no plugin is needed yet.
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_mc  # noqa: E402

TANK = ('box(0,0,0,7,0,7,"granite"); box(1,1,1,6,5,6,"glass"); box(2,1,2,5,5,5,"air"); box(2,1,2,5,2,5,"mercury");'
        'block(3,3,3,"iron"); block(4,4,4,"gold"); box(9,0,0,10,2,1,"copper")')

# name → (MineBench code, block size m, atmosphere, temperature_k(field) or None, ticks, dt s)
SCENES = {
    "sink": (TANK.replace('block(4,4,4,"gold")', 'block(4,5,4,"gold")'), 0.05, "STP", None, 12, 0.05),
    "melt": ('box(0,0,0,9,0,9,"tungsten"); box(4,1,4,5,3,5,"lead")', 0.01, "ISM", "plate_1500", 120, 0.05),
}
BOSSBAR = "skippycraft:theater"


# ── RCON (the Source RCON protocol Minecraft speaks) ──────────────────────────────────────────────────────────────────────
class Rcon:
    def __init__(self, host="127.0.0.1", port=42575, password=None, timeout=10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self._id = 0
        if self._request(3, password or server_password()) == -1:
            raise PermissionError("RCON refused the password")

    def _packet(self, kind, body):
        self._id += 1
        data = struct.pack("<ii", self._id, kind) + body.encode("utf-8") + b"\x00\x00"
        self.sock.sendall(struct.pack("<i", len(data)) + data)
        return self._id

    def _read(self):
        def exact(n):
            buf = b""
            while len(buf) < n:
                chunk = self.sock.recv(n - len(buf))
                if not chunk:
                    raise ConnectionError("RCON closed")
                buf += chunk
            return buf
        size = struct.unpack("<i", exact(4))[0]
        rid, kind = struct.unpack("<ii", exact(8))
        body = exact(size - 8)[:-2].decode("utf-8", "replace")
        return rid, kind, body

    def _request(self, kind, body):
        self._packet(kind, body)
        rid, _, _ = self._read()
        return rid

    def command(self, cmd):
        self._packet(2, cmd)
        return self._read()[2]

    def close(self):
        self.sock.close()


def server_password(dev_server=HERE / "dev-server"):
    for line in (dev_server / "data" / "server.properties").read_text(encoding="utf-8").splitlines():
        if line.startswith("rcon.password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no rcon.password in server.properties")


# ── scenes and their shared looks ──────────────────────────────────────────────────────────────────────────────────────────
def build(name):
    """``(field, world)`` for a scene."""
    import numpy as np
    from sigma_ground.deckard.voxel_exec import field_from_voxel_exec
    from sigma_ground.materia.voxel_world import VoxelWorld

    code, block_m, env, temp, _, _ = SCENES[name]
    field, _ = field_from_voxel_exec(code, block_m=block_m)
    T = None
    if temp == "plate_1500":
        T = np.where(field.label_grid == field.materials.index("tungsten"), 1500.0, 293.15)
    return field, VoxelWorld(field, temperature_k=T, env=env)


def shared_looks():
    """One carrier per (material, block size) across every scene, so all theaters share one resource pack."""
    fields = [build(n)[0] for n in SCENES]
    looks, appearances = {}, {}
    for f in fields:
        lk, ap = export_mc.derive_looks(f)
        for m in lk:
            appearances.setdefault((m, f.voxel_size), ap[m])
    opaque, clear = iter(export_mc.CARRIERS_OPAQUE), iter(export_mc.CARRIERS_CLEAR)
    for key, a in appearances.items():
        looks[key] = next(clear if a["opacity"] < export_mc.CLEAR_BELOW else opaque)
    return looks, appearances


# ── a live theater ─────────────────────────────────────────────────────────────────────────────────────────────────────────
class Theater:
    """A ``VoxelWorld`` shown in Minecraft with its low corner at ``origin``. ``send(commands)`` delivers commands."""

    def __init__(self, world, looks, origin, send):
        import numpy as np
        self.world, self.send = world, send
        self.origin = tuple(int(v) for v in origin)
        occ = np.argwhere(world.label > 0)
        self.i0, self.k0, self.jhi = int(occ[:, 0].min()), int(occ[:, 2].min()), int(occ[:, 1].max())
        self.block = ["air"] + [looks[(m, world.pitch)] for m in world.materials[1:]]
        # unknown (-1) from the theater's ground up, so the first draw also clears what an earlier run left there; below
        # the ground is Minecraft's own ground and is never touched
        self.shown = np.full_like(world.label, -1)
        self.shown[:, :, :self.k0] = world.label[:, :, :self.k0]
        self.changed_total = 0

    def mc(self, i, j, k):
        x0, y0, z0 = self.origin
        return x0 + int(i) - self.i0, y0 + int(k) - self.k0, z0 + self.jhi - int(j)

    def draw(self):
        """Send every cell whose block differs from what Minecraft shows. Returns how many."""
        import numpy as np
        diff = np.argwhere(self.world.label != self.shown)
        cmds = []
        for i, j, k in diff:
            x, y, z = self.mc(i, j, k)
            cmds.append(f"setblock {x} {y} {z} {self.block[self.world.label[i, j, k]]}")
        self.send(cmds)
        self.shown = self.world.label.copy()
        self.changed_total += len(cmds)
        return len(cmds)

    def readout(self, changed):
        w = self.world
        text = (f"t {w.t:.2f} s | {w.mass_kg():.4g} kg | E {w.energy_J():.6g} J | radiated "
                f"{w.ledger['radiated_J']:.4g} J | vented {w.ledger['vented_kg']:.3g} kg | changed {changed}")
        self.send([f"bossbar set {BOSSBAR} name {json.dumps(text)}"])
        return text

    def open(self):
        self.send([f"bossbar add {BOSSBAR} {json.dumps('SkippyCraft')}", f"bossbar set {BOSSBAR} players @a",
                   f"bossbar set {BOSSBAR} color green"])
        self.readout(self.draw())

    def run(self, ticks, dt, rate_hz=4.0, on_tick=None):
        period = 1.0 / rate_hz
        for _ in range(ticks):
            t0 = time.monotonic()
            self.world.step(dt)
            text = self.readout(self.draw())
            if on_tick:
                on_tick(text)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def clear(self):
        import numpy as np
        self.world.label[:] = 0
        self.draw()
        self.shown = np.zeros_like(self.world.label)


def verify(theater, rcon):
    """Ask Minecraft, cell by cell over the grid from the theater's ground up, whether it shows what the world holds.
    (Cells below the ground are Minecraft's own ground.) Returns the mismatches."""
    import numpy as np
    bad = []
    for i, j, k in np.ndindex(theater.world.label.shape):
        if k < theater.k0:
            continue
        x, y, z = theater.mc(i, j, k)
        want = theater.block[theater.world.label[i, j, k]]
        if "passed" not in rcon.command(f"execute if block {x} {y} {z} minecraft:{want}"):
            bad.append(((x, y, z), want))
    return bad


def main(argv=None):
    import argparse
    import os
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scene", choices=[*SCENES, "looks"])
    ap.add_argument("--at", nargs=3, type=int, default=[4, -60, 20], metavar=("X", "Y", "Z"))
    ap.add_argument("--rate", type=float, default=4.0, help="ticks shown per real second")
    ap.add_argument("--ticks", type=int)
    ap.add_argument("--verify", action="store_true", help="check every cell in Minecraft afterwards")
    a = ap.parse_args(argv)

    looks, appearances = shared_looks()
    packs = os.path.join(os.environ["APPDATA"], ".minecraft", "resourcepacks")
    export_mc.write_resourcepack(looks, appearances, packs)
    if a.scene == "looks":
        print(json.dumps({f"{m}@{s:g}m": c for (m, s), c in looks.items()}, indent=2))
        return
    _, world = build(a.scene)
    rcon = Rcon()
    th = Theater(world, looks, a.at, lambda cmds: [rcon.command(c) for c in cmds])
    th.open()
    _, _, _, _, ticks, dt = SCENES[a.scene]
    th.run(a.ticks or ticks, dt, a.rate, on_tick=print)
    if a.verify:
        bad = verify(th, rcon)
        print(f"verify: {len(bad)} mismatched cells" + (f", first {bad[:5]}" if bad else ""))
    rcon.close()


if __name__ == "__main__":
    main()
