"""
SkippyCraft Phase 1: a LIVE theater. sigma-ground's ``VoxelWorld`` ticks; Minecraft shows each tick (misc/SKIPPYCRAFT_PLAN.md).

The service owns the world. Every tick it steps ``VoxelWorld`` (heat, phase, settling), compares what each cell now holds with
what Minecraft was last told, and sends only the changed blocks over RCON (``setblock``). A boss bar shows the readout, straight
from the world's own ledger: simulated time, mass, energy, what radiated or vented, what changed. Minecraft simulates nothing:
every block it draws is a carrier repainted by the shared resource pack (``export_mc``), and carriers never move by themselves.

    python stage/skippycraft/theater.py sink --at 4 -60 20      # gold sinks through mercury, iron floats
    python stage/skippycraft/theater.py melt --at 20 -60 4      # lead melts on a 1500 K tungsten plate, in vacuum
    python stage/skippycraft/theater.py looks                    # (re)write the resource pack for every scene; F3+T in game
    python stage/skippycraft/theater.py serve                    # the service: buttons in the world, the pack on join

``serve`` is how the Captain uses it: it builds a small panel in front of each theater (a button to run it, a button to stop),
serves the resource pack over HTTP so the server hands it to him when he joins (no F3+T), and runs whichever scene he clicks.
A button sits on a command block that records the request in the scoreboard (``#request`` in objective ``skippycraft``);
the service polls it over RCON.

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

# name → (MineBench code, its written block size m, atmosphere, temperature_k(field) or None, ticks, dt s). The theater is
# built at RESOLUTION times finer than written (the Captain: higher resolution), fixed for its life: every written block is
# RESOLUTION³ blocks, each 1/RESOLUTION as wide.
RESOLUTION = 4
SCENES = {
    "sink": (TANK.replace('block(4,4,4,"gold")', 'block(4,5,4,"gold")'), 0.05, "STP", None, 40, 0.05),
    "melt": ('box(0,0,0,9,0,9,"tungsten"); box(4,1,4,5,3,5,"lead")', 0.01, "ISM", "plate_1500", 160, 0.05),
}
BOSSBAR = "skippycraft:theater"

# where each scene stands, its control desk, its request number, and its pace. A desk (the Captain: buttons must be easy to
# find at this scale) floats at the theater's viewing spot: a glowing platform to stand on, and in front of it at eye level a
# board of three buttons with big labels: Run, Stop, and Go to the other theater's desk. ``desk`` is the board's middle block;
# he stands 2.5 blocks behind it, facing +z, looking down at ``pitch`` degrees.
STAGES = {
    "sink": {"origin": (0, -60, 24), "desk": (16, -46, 14), "pitch": 25, "other": "melt", "request": 1, "rate": 4.0,
             "tell": "Sink: a gold block drops into the mercury and sinks; the iron floats (gold is denser than mercury, iron is not)."},
    "melt": {"origin": (56, -60, 24), "desk": (75, -46, 14), "pitch": 35, "other": "sink", "request": 2, "rate": 6.0,
             "tell": "Melt: a lead column on a 1500 K tungsten plate, in vacuum. It slumps when it melts; the boss bar keeps the books."},
}
STOP = 9
DESK_LOOK = {"fill": "minecraft:sea_lantern", "board": "minecraft:command_block"}
PACK_DIR = HERE / "dev-server" / "pack"
PACK_HOST, PACK_PORT = "100.122.71.14", 42566           # skippy-pc's tailnet address: the server's RESOURCE_PACK points here


# ── RCON (the Source RCON protocol Minecraft speaks) ──────────────────────────────────────────────────────────────────────
class Rcon:
    def __init__(self, host="127.0.0.1", port=42575, password=None, timeout=10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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

    def commands(self, cmds):
        """Many commands, one at a time: Minecraft's RCON reads one packet per read and closes the connection on
        pipelined packets (seen 2026-10-31), so each waits for its reply. Rows are merged into fills to keep this short."""
        return [self.command(c) for c in cmds]

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
    field, _ = field_from_voxel_exec(code, block_m=block_m, resolution=RESOLUTION)
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
        label = self.world.label
        changed = label != self.shown
        cmds = []
        for j, k in zip(*np.nonzero(changed.any(axis=0))):          # rows along x that have a change
            row, todo = label[:, j, k], changed[:, j, k]
            i = 0
            while i < len(row):
                if not todo[i]:
                    i += 1
                    continue
                end = i
                while end + 1 < len(row) and todo[end + 1] and row[end + 1] == row[i]:
                    end += 1
                (x0, y, z), x1, block = self.mc(i, j, k), self.mc(end, j, k)[0], self.block[row[i]]
                cmds.append(f"setblock {x0} {y} {z} {block}" if end == i else f"fill {x0} {y} {z} {x1} {y} {z} {block}")
                i = end + 1
        self.send(cmds)
        n = int(changed.sum())
        self.shown = label.copy()
        self.changed_total += n
        return n

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


# ── the service: panels, the served pack, requests ────────────────────────────────────────────────────────────────────────
def build_pack():
    """Write the shared resource pack and its zip for the server to hand out. Returns the looks."""
    import shutil
    looks, appearances = shared_looks()
    export_mc.write_resourcepack(looks, appearances, PACK_DIR)
    shutil.make_archive(str(PACK_DIR / "skippycraft"), "zip", PACK_DIR / "skippycraft")
    return looks


def serve_pack():
    """Serve ``PACK_DIR`` over HTTP on the tailnet address, in a background thread."""
    import functools
    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    httpd = ThreadingHTTPServer((PACK_HOST, PACK_PORT), functools.partial(Quiet, directory=str(PACK_DIR)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def stand(name):
    """Where he stands at a desk, and how he looks: ``(x, y, z, yaw, pitch)``."""
    x, y, z = STAGES[name]["desk"]
    return x + 0.5, y, z - 2.0, 0, STAGES[name]["pitch"]


def label(tag, x, y, z, text):
    """A big bold green label on a dark backing, always facing him."""
    comp = f'{{text:{json.dumps(text)},color:"green",bold:true}}'
    return [f"kill @e[type=text_display,tag={tag}]",
            f'summon minecraft:text_display {x} {y} {z} {{text:{comp},billboard:"center",background:-1442840576,'
            f"transformation:{{left_rotation:[0f,0f,0f,1f],right_rotation:[0f,0f,0f,1f],translation:[0f,0f,0f],"
            f'scale:[1.4f,1.4f,1.4f]}},Tags:["skippycraft_label","{tag}"]}}']


def desk_commands(name):
    """The desk: a glowing platform, a board of command blocks with buttons on its near face, a big label over each."""
    st = STAGES[name]
    x, y, z = st["desk"]
    sx, sy, sz, yaw, pitch = stand(st["other"])
    cmds = [f"fill {x - 2} {y - 1} {z - 4} {x + 2} {y - 1} {z - 1} {DESK_LOOK['fill']}"]
    buttons = ((x - 2, f"Run {name}", f"scoreboard players set #request skippycraft {st['request']}"),
               (x, "Stop", f"scoreboard players set #request skippycraft {STOP}"),
               (x + 2, f"Go to {st['other']}", f"tp @p {sx} {sy} {sz} {yaw} {pitch}"))
    for n, (bx, text, command) in enumerate(buttons):
        cmds += [f"setblock {bx} {y} {z} {DESK_LOOK['board']}{{Command:{json.dumps(command)}}}",
                 f"setblock {bx} {y} {z - 1} minecraft:stone_button[face=wall,facing=north]"]
        cmds += label(f"skippycraft_{name}_{n}", bx + 0.5, y + 1.3, z - 0.6, text)
    return cmds


def desk_checks(name):
    """``execute if block`` tests that pass only while the desk's buttons and command blocks are all there."""
    x, y, z = STAGES[name]["desk"]
    return [f"execute if block {bx} {y} {z} {DESK_LOOK['board']}" for bx in (x - 2, x, x + 2)] +            [f"execute if block {bx} {y} {z - 1} minecraft:stone_button" for bx in (x - 2, x, x + 2)] +            [f"execute if block {x} {y - 1} {z - 2} {DESK_LOOK['fill']}"]


def repair_desks(rcon):
    """Rebuild any desk with a missing piece (in creative one left-click breaks a block). Returns the desks rebuilt."""
    fixed = []
    for name in STAGES:
        if not all("passed" in r for r in rcon.commands(desk_checks(name))):
            rcon.commands(desk_commands(name))
            fixed.append(name)
    return fixed


def read_request(rcon):
    out = rcon.command("scoreboard players get #request skippycraft")
    if " has " not in out:
        return 0
    value = int(out.split(" has ", 1)[1].split()[0])
    if value:
        rcon.command("scoreboard players set #request skippycraft 0")
    return value


def serve(player="aaronsrhodes"):
    """Run forever: answer the panel buttons with live theaters."""
    looks = build_pack()
    serve_pack()
    rcon = Rcon()
    send = rcon.commands
    send(["forceload add -16 0 111 79", "kill @e[type=text_display,tag=skippycraft_label]", "scoreboard objectives add skippycraft dummy",
          "scoreboard players set #request skippycraft 0"])
    # forceload loads chunks asynchronously: build nothing until every panel and theater spot is loaded
    spots = [st[k] for st in STAGES.values() for k in ("origin", "desk")]
    for _ in range(100):
        if all("passed" in rcon.command(f"execute if loaded {x} {y} {z}") for x, y, z in spots):
            break
        time.sleep(0.2)
    else:
        raise RuntimeError("theater chunks never loaded")
    for name in STAGES:
        send(desk_commands(name))
    sx, sy, sz, _, _ = stand("sink")
    send([f"setworldspawn {int(sx)} {sy} {int(sz)}", "gamerule respawn_radius 0"])
    by_number = {st["request"]: name for name, st in STAGES.items()}
    tell = lambda text: rcon.command("tellraw @a " + json.dumps({"text": text, "color": "green"}))
    print("serving: panels built, pack at", f"http://{PACK_HOST}:{PACK_PORT}/skippycraft.zip", flush=True)
    pending, last_check = 0, 0.0
    while True:
        req = pending or read_request(rcon)
        pending = 0
        name = by_number.get(req)
        if name is None:
            if time.monotonic() - last_check > 3.0:              # idle: put back anything he broke
                last_check = time.monotonic()
                for fixed in repair_desks(rcon):
                    print("repaired the", fixed, "desk", flush=True)
                    tell(f"Put the {fixed} desk back together.")
            time.sleep(0.3)
            continue
        st = STAGES[name]
        _, world = build(name)
        th = Theater(world, looks, st["origin"], send)
        rcon.command(f'title @a actionbar {json.dumps({"text": "Running " + name, "color": "green"})}')
        tell(f"Building {name} at {RESOLUTION}x resolution (a few seconds)...")
        th.open()
        tell(st["tell"])
        _, _, _, _, ticks, dt = SCENES[name]
        print("run", name, flush=True)
        for _ in range(ticks):
            t0 = time.monotonic()
            nxt = read_request(rcon)
            if nxt:
                pending = 0 if nxt == STOP else nxt
                tell(f"Stopped {name}.")
                break
            world.step(dt)
            th.readout(th.draw())
            time.sleep(max(0.0, 1.0 / st["rate"] - (time.monotonic() - t0)))
        else:
            tell(f"{name} done: the world ran {world.t:.2f} simulated seconds.")


def main(argv=None):
    import argparse
    import os
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scene", choices=[*SCENES, "looks", "serve"])
    ap.add_argument("--at", nargs=3, type=int, default=[4, -60, 20], metavar=("X", "Y", "Z"))
    ap.add_argument("--rate", type=float, default=4.0, help="ticks shown per real second")
    ap.add_argument("--ticks", type=int)
    ap.add_argument("--verify", action="store_true", help="check every cell in Minecraft afterwards")
    a = ap.parse_args(argv)

    if a.scene == "serve":
        serve()
        return
    looks, appearances = shared_looks()
    packs = os.path.join(os.environ["APPDATA"], ".minecraft", "resourcepacks")
    export_mc.write_resourcepack(looks, appearances, packs)
    if a.scene == "looks":
        print(json.dumps({f"{m}@{s:g}m": c for (m, s), c in looks.items()}, indent=2))
        return
    _, world = build(a.scene)
    rcon = Rcon()
    th = Theater(world, looks, a.at, rcon.commands)
    th.open()
    _, _, _, _, ticks, dt = SCENES[a.scene]
    th.run(a.ticks or ticks, dt, a.rate, on_tick=print)
    if a.verify:
        bad = verify(th, rcon)
        print(f"verify: {len(bad)} mismatched cells" + (f", first {bad[:5]}" if bad else ""))
    rcon.close()


if __name__ == "__main__":
    main()
