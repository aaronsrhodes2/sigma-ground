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
Blocks go through the SkippyCraft plugin's block writer (plugin/, 127.0.0.1:42576) when it answers: one binary frame per
tick, tens of thousands of blocks; without it, through RCON fills (about 600 commands a second).
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
    # the wave machine: a glass tank, an aluminium paddle near one end, water 15 cm deep; the paddle is driven at the
    # tank's own first sloshing period (fluid_surface.sloshing_angular_frequency) unless he changes it at the desk
    "slosh": ('box(0,0,0,17,6,7,"glass"); box(1,1,1,16,6,6,"air"); box(2,1,1,2,6,6,"aluminum"); '
              'box(3,1,1,16,3,6,"water"); block(8,5,3,"gold"); block(12,5,4,"wood_oak")', 0.05, "STP", "flip", 750, 0.04),
    # the double slit, one photon at a time, on a screen of 1 mm blocks (materia.voxel_slits): no build code, its own world
    "slit": (None, 0.001, None, "slits", 240, 0.1),
    # the shooting range: Weapon vs. Armor's physics carved into an armour target of 2 mm blocks (shooting_range.py)
    "range": (None, 0.002, "STP", "range", 150, 0.02),
}
PADDLE = {"material": "aluminum", "axis": 0, "amplitude_m": 0.015, "period_s": None}
BODIES = ("gold", "wood_oak")          # free bodies in the slosh tank: they fall in when it runs (gold sinks, oak floats)
BOSSBAR = "skippycraft:theater"

# where each scene stands, its control desk, its request number, and its pace. A desk (the Captain: buttons must be easy to
# find at this scale) floats at the theater's viewing spot: a glowing platform to stand on, and in front of it at eye level a
# board of three buttons with big labels: Run, Stop, and Go to the other theater's desk. ``desk`` is the board's middle block;
# he stands 2.5 blocks behind it, facing +z, looking down at ``pitch`` degrees.
STAGES = {
    "sink": {"origin": (0, -60, 24), "desk": (16, -46, 14), "pitch": 25, "other": "melt", "request": 1, "rate": 4.0,
             "tell": "Sink: a gold block drops into the mercury and sinks; the iron floats (gold is denser than mercury, iron is not)."},
    "melt": {"origin": (56, -60, 24), "desk": (75, -46, 14), "pitch": 35, "other": "slosh", "request": 2, "rate": 6.0,
             "tell": "Melt: a lead column on a 1500 K tungsten plate, in vacuum. It slumps when it melts; the boss bar keeps the books."},
    "slosh": {"origin": (0, -60, 100), "desk": (36, -46, 88), "pitch": 25, "other": "slit", "request": 3, "rate": 25.0,
              "extra": (("Slower", 4), ("Faster", 5)),
              "tell": "Slosh: the paddle pushes the water at the tank's own sloshing period, so each push adds to the last. "
                      "A gold block and an oak block drop in: the water's own pressure sinks one and floats the other."},
    # the screen stands at z = 200 (128 x 32 blocks, 1 mm each); the desk is 50 blocks back so the whole screen is in view
    "slit": {"origin": (0, -60, 200), "desk": (64, -45, 150), "pitch": 0, "other": "range", "request": 8, "rate": 8.0,
             "extra": (("Observe", 6), ("One slit", 7), ("Colour", 10)),
             "tell": "Double slit: a laser through two slits 0.25 mm apart, onto a screen 2 m away, drawn at true scale (1 block "
                     "= 1 mm). Photons arrive one at a time; the fringes appear only as they pile up. Observe = learn which slit "
                     "each photon took (the fringes vanish). One slit = close one. Colour = change the laser."},
    # the target's face is 64 x 64 blocks (12.8 cm square, 2 mm blocks) at z = 60; the desk floats 40 blocks back at its centre
    "range": {"origin": (120, -60, 60), "desk": (152, -28, 20), "pitch": 0, "other": "sink", "request": 11, "rate": 8.0,
              "extra": (("Weapon", 12), ("Armor", 13)),
              "tell": "Range: Weapon vs. Armor's own physics, carved into a real target of 2 mm blocks. Run fires; the hole "
                      "goes through every layer the shot defeats, the crater glows at the temperature the game computes, then "
                      "cools by conduction. Weapon / Armor change the setup."},
}
STOP = 9
SLOWER, FASTER = 4, 5
SCREEN_BUTTONS = {6: "observe", 7: "one_slit", 10: "colour"}
RANGE_BUTTONS = {12: "weapon", 13: "armor"}
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


class Blocks:
    """The SkippyCraft plugin's block writer: one binary frame of runs along +x per call (plugin/SkippyCraftPlugin.java)."""

    def __init__(self, host="127.0.0.1", port=42576, timeout=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(self, palette, runs):
        """``palette``: block names; ``runs``: (x, y, z, length, palette index). Returns (blocks set, server ms)."""
        out = [struct.pack(">ii", 0x534B4346, len(palette))]
        for name in palette:
            b = name.encode("utf-8")
            out.append(struct.pack(">H", len(b)) + b)
        out.append(struct.pack(">i", len(runs)))
        out.append(b"".join(struct.pack(">iiihh", *r) for r in runs))
        self.sock.sendall(b"".join(out))
        buf = b""
        while len(buf) < 8:
            chunk = self.sock.recv(8 - len(buf))
            if not chunk:
                raise ConnectionError("block writer closed")
            buf += chunk
        return struct.unpack(">ii", buf)

    def close(self):
        self.sock.close()


def block_writer():
    """The plugin's block writer, or None when it isn't there (then blocks go over RCON)."""
    try:
        return Blocks()
    except OSError:
        return None


def server_password(dev_server=HERE / "dev-server"):
    for line in (dev_server / "data" / "server.properties").read_text(encoding="utf-8").splitlines():
        if line.startswith("rcon.password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no rcon.password in server.properties")


# ── scenes and their shared looks ──────────────────────────────────────────────────────────────────────────────────────────
def scene_field(name):
    """The scene's VoxelField (None for the slit screen, which has no build)."""
    from sigma_ground.deckard.voxel_exec import field_from_voxel_exec
    code, block_m = SCENES[name][:2]
    if code is None:
        return None
    return field_from_voxel_exec(code, block_m=block_m, resolution=RESOLUTION)[0]


def build(name):
    """``(field, world)`` for a scene."""
    import numpy as np
    from sigma_ground.materia.voxel_fluid import FluidVoxelWorld
    from sigma_ground.materia.voxel_world import VoxelWorld

    _, _, env, temp, _, _ = SCENES[name]
    if temp == "range":
        import shooting_range
        return None, shooting_range.RangeWorld(armed=False)
    if temp == "slits":
        from sigma_ground.materia.voxel_slits import SlitScreen
        return None, SlitScreen()
    field = scene_field(name)
    if temp == "flip":
        return field, FluidVoxelWorld(field, paddle=dict(PADDLE), bodies=BODIES)
    T = None
    if temp == "plate_1500":
        T = np.where(field.label_grid == field.materials.index("tungsten"), 1500.0, 293.15)
    return field, VoxelWorld(field, temperature_k=T, env=env)


def shared_looks():
    """One carrier per (material, block size) across every scene, so all theaters share one resource pack."""
    fields = [f for f in (scene_field(n) for n in SCENES) if f is not None]
    looks, appearances = {}, {}
    for f in fields:
        lk, ap = export_mc.derive_looks(f)
        for m in lk:
            appearances.setdefault((m, f.voxel_size), ap[m])
    for key, a in range_appearances().items():
        appearances.setdefault(key, a)
    opaque, clear = iter(export_mc.CARRIERS_OPAQUE), iter(export_mc.CARRIERS_CLEAR)
    for key, a in appearances.items():
        looks[key] = next(clear if a["opacity"] < export_mc.CLEAR_BELOW else opaque)
    return looks, appearances


GLOW_HUE_OF = "blackbody"   # the band's hue: each material's spectral emissivity shifts it by under 2% (checked: tungsten, lead)


def shared_glows():
    """One light-emitting carrier per glow band, shared by every material (the range can reach any band): its hue and
    brightness from ``export_mc.glow_display`` at the band's lower edge; dimmer bands on dimmer blocks. Returns
    ``{band: (carrier index, rgba, brightness)}``."""
    lights = sorted(range(len(export_mc.GLOW_CARRIERS)), key=lambda i: export_mc.GLOW_CARRIERS[i][1])
    n = len(export_mc.GLOW_BANDS_K)
    pick = [lights[round(i * (len(lights) - 1) / (n - 1))] for i in range(n)]
    out = {}
    for band, (edge, ci) in enumerate(zip(export_mc.GLOW_BANDS_K, pick)):
        rgba, v, _ = export_mc.glow_display(GLOW_HUE_OF, edge)
        out[band] = (ci, rgba, v)
    return out


def range_appearances():
    """``{(material, 0.002): appearance}`` for every range material. A material sigma-ground has no colour for takes the
    game's hand-picked cross-section colour (weapon_vs_armor.materials.MATERIAL_COLORS), named as a gap."""
    import shooting_range
    from sigma_ground.field.interface.block import appearance
    from weapon_vs_armor.materials import MATERIAL_COLORS
    out = {}
    for m in shooting_range.materials():
        a = dict(appearance(m, shooting_range.PITCH))
        if a["color_rgb"] is None and m in MATERIAL_COLORS:
            a["color_rgb"] = tuple(c / 255 for c in MATERIAL_COLORS[m])
            a["sources"] = dict(a["sources"], color_rgb="weapon_vs_armor MATERIAL_COLORS (hand-picked)")
            a["gaps"] = [g for g in a["gaps"] if "colour" not in g] + ["colour hand-picked by the game"]
        out[(m, shooting_range.PITCH)] = a
    return out


def slit_looks():
    """The slit screen's looks: the bare screen is plain black terracotta; each (wavelength, brightness) is a wool block
    repainted in the laser's colour. Returns ``(looks, {carrier: rgba})``."""
    from sigma_ground.materia.voxel_slits import LEVELS, WAVELENGTHS_NM, SlitScreen
    pitch = SCENES["slit"][1]
    looks, paint = {("screen", pitch): "black_terracotta"}, {}
    wool = iter(f"{d}_wool" for d in export_mc._DYES)
    for nm in WAVELENGTHS_NM:
        for lv in range(1, LEVELS + 1):
            carrier = next(wool)
            looks[(f"light {nm} {lv}", pitch)] = carrier
            rgb = SlitScreen.light_rgb(nm, lv)
            paint[carrier] = tuple(round(255 * c) for c in rgb) + (255,)
    return looks, paint


# ── a live theater ─────────────────────────────────────────────────────────────────────────────────────────────────────────
class Theater:
    """A ``VoxelWorld`` shown in Minecraft with its low corner at ``origin``. ``send(commands)`` delivers commands."""

    def __init__(self, world, looks, origin, send, blocks=None, glows=None):
        import numpy as np
        self.world, self.send, self.blocks = world, send, blocks
        self.origin = tuple(int(v) for v in origin)
        occ = np.argwhere(world.label > 0)
        self.i0, self.k0, self.jhi = int(occ[:, 0].min()), int(occ[:, 2].min()), int(occ[:, 1].max())
        self.block = ["air"] + [looks[(m, world.pitch)] for m in world.materials[1:]]
        # glowing cells: (material index, band) → an index into self.block past the materials
        nb = len(export_mc.GLOW_BANDS_K)
        self.glow_lut = np.tile(np.arange(len(self.block))[:, None], (1, nb))
        for band, (ci, _, _) in (glows or {}).items():
            self.glow_lut[1:, band] = len(self.block)
            self.block.append(export_mc.GLOW_CARRIERS[ci][0])
        self.glowing = bool(glows) and hasattr(world, "T")
        # unknown (-1) from the theater's ground up, so the first draw also clears what an earlier run left there; below
        # the ground is Minecraft's own ground and is never touched
        self.shown = np.full_like(world.label, -1)
        self.shown[:, :, :self.k0] = world.label[:, :, :self.k0]
        self.changed_total = 0

    def mc(self, i, j, k):
        x0, y0, z0 = self.origin
        return x0 + int(i) - self.i0, y0 + int(k) - self.k0, z0 + self.jhi - int(j)

    def display(self):
        """What each cell is drawn as: its material, or its material's glow when it is past the Draper point."""
        import numpy as np
        label = self.world.label
        if not self.glowing:
            return label
        band = np.searchsorted(export_mc.GLOW_BANDS_K, self.world.T, side="right") - 1
        hot = (band >= 0) & (label > 0)
        out = label.copy()
        out[hot] = self.glow_lut[label[hot], band[hot]]
        return out

    def draw(self):
        """Send every cell whose block differs from what Minecraft shows. Returns how many."""
        import numpy as np
        label = self.display()
        changed = label != self.shown
        if self.blocks is not None:
            self._draw_frame(label, changed)
            n = int(changed.sum())
            self.shown = label.copy()
            self.changed_total += n
            return n
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

    def _draw_frame(self, label, changed):
        """Every changed cell, as runs along x, in one frame to the plugin."""
        import numpy as np
        runs = []
        for j, k in zip(*np.nonzero(changed.any(axis=0))):
            row, todo = label[:, j, k], changed[:, j, k]
            i = 0
            while i < len(row):
                if not todo[i]:
                    i += 1
                    continue
                end = i
                while end + 1 < len(row) and todo[end + 1] and row[end + 1] == row[i] and end - i < 30000:
                    end += 1
                x, y, z = self.mc(i, j, k)
                runs.append((x, y, z, end - i + 1, int(row[i])))
                i = end + 1
        if runs:
            self.blocks.send(["minecraft:" + b for b in self.block], runs)

    def readout(self, changed):
        w = self.world
        if hasattr(w, "readout"):
            text = f"{w.readout()} | changed {changed}"
            self.send([f"bossbar set {BOSSBAR} name {json.dumps(text)}"])
            return text
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
    shown = theater.display()
    for i, j, k in np.ndindex(theater.world.label.shape):
        if k < theater.k0:
            continue
        x, y, z = theater.mc(i, j, k)
        want = theater.block[shown[i, j, k]]
        if "passed" not in rcon.command(f"execute if block {x} {y} {z} minecraft:{want}"):
            bad.append(((x, y, z), want))
    return bad


# ── the service: panels, the served pack, requests ────────────────────────────────────────────────────────────────────────
def build_pack():
    """Write the shared resource pack and its zip for the server to hand out. Returns the looks."""
    import shutil
    looks, appearances = shared_looks()
    export_mc.write_resourcepack(looks, appearances, PACK_DIR)
    export_mc.write_glow_textures(shared_glows(), PACK_DIR)
    screen, paint = slit_looks()
    export_mc.write_flat_textures(paint, PACK_DIR)
    shutil.make_archive(str(PACK_DIR / "skippycraft"), "zip", PACK_DIR / "skippycraft")
    return {**looks, **screen}


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
    buttons = [(x - 2, f"Run {name}", f"scoreboard players set #request skippycraft {st['request']}"),
               (x, "Stop", f"scoreboard players set #request skippycraft {STOP}"),
               (x + 2, f"Go to {st['other']}", f"tp @p {sx} {sy} {sz} {yaw} {pitch}")]
    for n, (text, request) in enumerate(st.get("extra", ())):          # more buttons to the right
        buttons.append((x + 4 + 2 * n, text, f"scoreboard players set #request skippycraft {request}"))
    cmds = [f"fill {x - 2} {y - 1} {z - 4} {buttons[-1][0]} {y - 1} {z - 1} {DESK_LOOK['fill']}"]
    for n, (bx, text, command) in enumerate(buttons):
        cmds += [f"setblock {bx} {y} {z} {DESK_LOOK['board']}{{Command:{json.dumps(command)}}}",
                 f"setblock {bx} {y} {z - 1} minecraft:stone_button[face=wall,facing=north]"]
        cmds += label(f"skippycraft_{name}_{n}", bx + 0.5, y + 1.3, z - 0.6, text)
    return cmds


def desk_checks(name):
    """``execute if block`` tests that pass only while the desk's buttons and command blocks are all there."""
    x, y, z = STAGES[name]["desk"]
    xs = [x - 2, x, x + 2] + [x + 4 + 2 * n for n in range(len(STAGES[name].get("extra", ())))]
    return [f"execute if block {bx} {y} {z} {DESK_LOOK['board']}" for bx in xs] +            [f"execute if block {bx} {y} {z - 1} minecraft:stone_button" for bx in xs] +            [f"execute if block {x} {y - 1} {z - 2} {DESK_LOOK['fill']}"]


def repair_desks(rcon):
    """Rebuild any desk with a missing piece (in creative one left-click breaks a block). Returns the desks rebuilt."""
    fixed = []
    for name in STAGES:
        if not all("passed" in r for r in rcon.commands(desk_checks(name))):
            rcon.commands(desk_commands(name))
            fixed.append(name)
    return fixed


def range_press(aim, what):
    """Cycle the range's weapon or armour. Returns what it is now, in words."""
    import shooting_range
    weapons, armors = shooting_range.presets()
    names = shooting_range.WEAPONS if what == "weapon" else shooting_range.ARMORS
    aim[what] = names[(names.index(aim[what]) + 1) % len(names)]
    table = weapons if what == "weapon" else armors
    return f"{what.capitalize()}: {table[aim[what]]['name']}. Press Run to fire."


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
    glows = shared_glows()
    serve_pack()
    rcon = Rcon()
    send = rcon.commands
    blocks = block_writer()
    print("block writer:", "plugin" if blocks else "none (RCON fills)", flush=True)
    send(["forceload add -16 0 191 207", "kill @e[type=text_display,tag=skippycraft_label]", "scoreboard objectives add skippycraft dummy",
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
    def reconnect():
        while True:
            try:
                r = Rcon()
                return r, r.commands, block_writer()
            except OSError:
                time.sleep(2.0)

    by_number = {st["request"]: name for name, st in STAGES.items()}
    kept = {}
    import shooting_range
    aim, preview = {"weapon": shooting_range.WEAPONS[0], "armor": shooting_range.ARMORS[0]}, False
    tell = lambda text: rcon.command("tellraw @a " + json.dumps({"text": text, "color": "green"}))
    print("serving: panels built, pack at", f"http://{PACK_HOST}:{PACK_PORT}/skippycraft.zip", flush=True)
    pending, last_check = 0, 0.0
    while True:
        try:
            req = pending or read_request(rcon)
            pending = 0
            if req in SCREEN_BUTTONS:                                     # pressed between runs: change the setup, then run
                tell(kept.setdefault("slit", build("slit")[1]).press(SCREEN_BUTTONS[req]))
                req = STAGES["slit"]["request"]
            if req in RANGE_BUTTONS:                                      # a new weapon or armour: show the fresh target
                tell(range_press(aim, RANGE_BUTTONS[req]))
                req, preview = STAGES["range"]["request"], True
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
            if name == "slit":                                            # the screen keeps its setup between runs
                world = kept.setdefault("slit", build("slit")[1])
                world.reset()
            elif name == "range":
                import shooting_range
                world = shooting_range.RangeWorld(aim["weapon"], aim["armor"], armed=not preview)
            else:
                _, world = build(name)
            th = Theater(world, looks, st["origin"], send, blocks, glows)
            rcon.command(f'title @a actionbar {json.dumps({"text": "Running " + name, "color": "green"})}')
            tell(f"Building {name} at {RESOLUTION}x resolution (a few seconds)...")
            th.open()
            tell(st["tell"])
            _, _, _, _, ticks, dt = SCENES[name]
            if preview:                                                   # just show the fresh target
                ticks, preview = 0, False
            print("run", name, flush=True)
            for tick in range(ticks):
                t0 = time.monotonic()
                nxt = read_request(rcon)
                if nxt in RANGE_BUTTONS:                                  # change the setup mid-run: start over with it
                    tell(range_press(aim, RANGE_BUTTONS[nxt]))
                    pending, preview = STAGES["range"]["request"], True
                    break
                if nxt in SCREEN_BUTTONS and hasattr(world, "press"):
                    tell(world.press(SCREEN_BUTTONS[nxt]))
                    nxt = 0
                if nxt in (SLOWER, FASTER) and getattr(world, "paddle", None):
                    period = world.paddle["period_s"] * (1.1 if nxt == SLOWER else 1 / 1.1)
                    world.set_period(period)
                    tune = period / world.paddle["resonant_s"]
                    tell(f"Paddle period {period:.2f} s ({tune:.2f} x the tank's sloshing period).")
                    nxt = 0
                if nxt:
                    pending = 0 if nxt == STOP else nxt
                    tell(f"Stopped {name}.")
                    break
                world.step(dt)
                text = th.readout(th.draw())
                if tick % 25 == 0:                                        # the service's log: what it last drew, and where
                    where = " ".join(f"{b['material']}@{th.mc(*b['cell'])}" for b in getattr(world, "bodies", ()))
                    print(f"{name} tick {tick}: {text} {where}", flush=True)
                time.sleep(max(0.0, 1.0 / st["rate"] - (time.monotonic() - t0)))
            else:
                tell(f"{name} done: the world ran {world.t:.2f} simulated seconds.")

        except (TimeoutError, ConnectionError, OSError) as e:
            # the server can stall for many seconds (a world autosave to a slow disk): reconnect and carry on
            print("lost the server:", repr(e), "- reconnecting", flush=True)
            rcon, send, blocks = reconnect()
            pending = 0

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
    screen, paint = slit_looks()
    export_mc.write_flat_textures(paint, packs)
    looks = {**looks, **screen}
    if a.scene == "looks":
        print(json.dumps({f"{m}@{s:g}m": c for (m, s), c in looks.items()}, indent=2))
        return
    _, world = build(a.scene)
    rcon = Rcon()
    glows = shared_glows()
    export_mc.write_glow_textures(glows, packs)
    th = Theater(world, looks, a.at, rcon.commands, block_writer(), glows)
    th.open()
    _, _, _, _, ticks, dt = SCENES[a.scene]
    th.run(a.ticks or ticks, dt, a.rate, on_tick=print)
    if a.verify:
        bad = verify(th, rcon)
        print(f"verify: {len(bad)} mismatched cells" + (f", first {bad[:5]}" if bad else ""))
    rcon.close()


if __name__ == "__main__":
    main()
