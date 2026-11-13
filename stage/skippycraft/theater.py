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
    # Schrodinger's cat in a sealed box, computed (field.interface.schrodinger_cat via materia.voxel_cat): no build code, its own world
    "cat": (None, 0.25, None, "cat", 300, 0.1),
    # the shooting range: Weapon vs. Armor's physics carved into an armour target of 2 mm blocks (shooting_range.py)
    "range": (None, 0.002, "STP", "range", 150, 0.02),
    # the playground (SKIPPYCRAFT_PLAN 10b): heat, melting, liquid momentum and bodies at once (materia.playground). A 2200 K
    # tungsten plate on a cold granite floor melts the lead slab on it; the melt runs off onto the granite and freezes where
    # it stops; the gold cube on the slab sinks through the melt to the plate. Every value from each cell's live state.
    "playground": ('box(0,0,0,19,0,19,"granite"); box(6,1,6,13,1,13,"tungsten"); box(7,2,7,12,4,12,"lead"); '
                   'box(9,5,9,10,6,10,"gold")', 0.01, "ISM", "playground", 400, 0.05),
    # strength (SKIPPYCRAFT_PLAN 10b, P3): bodies that rotate, touch, slide, topple and break (dynamics.voxel_rigid), in the
    # same world as heat. A stack of five granite blocks, each 2 cm further out than the one below, topples; a 3000 K
    # tungsten block rests on a steel beam between two granite piers, its heat runs into the beam, and where the steel is
    # hot enough to yield (Eurocode 3's measured fire table) the beam forms a plastic hinge, sags, tears and drops it.
    "strength": ('box(0,0,0,69,0,29,"granite"); box(4,1,12,8,18,17,"granite"); box(61,1,12,65,18,17,"granite"); '
                 'box(5,19,14,64,19,15,"steel"); box(32,20,14,37,31,15,"tungsten")', 0.01, "ISM", "strength", 300, 0.05),
    # impact (SKIPPYCRAFT_PLAN 10b, P4): the range's shot as an event in the playground. A wall of four 10 cm steel blocks
    # stands on a granite floor with a granite backstop behind it; 0.2 s in, the range's weapon fires at the top-left
    # block. The game's per-layer solve says how deep it goes and what momentum and energy it loses in each layer
    # (shooting_range.layer_solver); the world knocks out the bore, pushes the block with that momentum, and heats the
    # bore's walls with the energy the block does not take as motion (materia.playground.PlaygroundWorld.impact).
    "impact": ('box(0,0,0,47,0,39,"granite"); box(0,1,26,47,24,29,"granite")', 0.01, "STP", "impact", 200, 0.05),
}
IMPACT_BLOCKS = tuple((14 + 10 * i, 1 + 10 * j, 10, 23 + 10 * i, 10 + 10 * j, 19) for j in range(2) for i in range(2))
IMPACT_FIRE_S = 0.2
IMPACT_ROOM = (0, 0, 40)                 # empty cells above the build: what is knocked flying stays in the world longer
IMPACT_WEAPON = 17                       # the impact desk's Weapon button: the next round that can push, then fire
# the strength scene's stack: separate granite bodies that touch (written apart, so each is its own body)
STRENGTH_STACK = tuple((20 + 2 * k, 1 + 4 * k, 22, 23 + 2 * k, 4 + 4 * k, 25) for k in range(5))
# scenes whose blocks of one material are separate bodies: (boxes, material written, canonical name)
SEPARATE = {"strength": (STRENGTH_STACK, "granite", "granite"), "impact": (IMPACT_BLOCKS, "steel", "steel_mild")}
STRENGTH_LOAD_K = 3000.0
PLAYGROUND_BODIES = ("gold", "lead")
# a scene may be built at its own resolution (fixed for its life): the playground runs heat, a liquid and bodies at once,
# and at 4x (2.5 mm blocks) a tick took ~3.7 s; at 2x (5 mm) it is watchable
SCENE_RESOLUTION = {"playground": 2, "strength": 1, "impact": 2}
PADDLE = {"material": "aluminum", "axis": 0, "amplitude_m": 0.015, "period_s": None}
BODIES = ("gold", "wood_oak")          # free bodies in the slosh tank: they fall in when it runs (gold sinks, oak floats)
BOSSBAR = "skippycraft:theater"
MAX_MATERIALS = 64                       # palette slots per theater: a scene's materials plus any placed from inside

# where each scene stands, its control desk, its request number, and its pace. A desk (the Captain: buttons must be easy to
# find at this scale) floats at the theater's viewing spot: a glowing platform to stand on, and in front of it at eye level a
# board of three buttons with big labels: Run, Stop, and Go to the other theater's desk. ``desk`` is the board's middle block;
# he stands 2.5 blocks behind it, facing +z, looking down at ``pitch`` degrees.
STAGES = {
    "sink": {"origin": (0, -60, 24), "desk": (16, -46, 14), "pitch": 25, "other": "melt", "request": 1, "rate": 4.0, "matter": True,
             "tell": "Sink: a gold block drops into the mercury and sinks; the iron floats (gold is denser than mercury, iron is not)."},
    "melt": {"origin": (56, -60, 24), "desk": (75, -46, 14), "pitch": 35, "other": "slosh", "request": 2, "rate": 6.0, "matter": True,
             "tell": "Melt: a lead column on a 1500 K tungsten plate, in vacuum. It slumps when it melts; the boss bar keeps the books."},
    "slosh": {"origin": (0, -60, 100), "desk": (36, -46, 88), "pitch": 25, "other": "slit", "request": 3, "rate": 25.0, "matter": True,
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
    "range": {"origin": (120, -60, 60), "desk": (152, -28, 20), "pitch": 0, "other": "playground", "request": 11, "rate": 8.0,
              "extra": (("Weapon", 12), ("Armor", 13)),
              "tell": "Range: Weapon vs. Armor's own physics, carved into a real target of 2 mm blocks. Run fires; the hole "
                      "goes through every layer the shot defeats, the crater glows at the temperature the game computes, then "
                      "cools by conduction. Weapon / Armor change the setup."},
    # the playground's grid is ~90 blocks square at x 100-190, z 105-195; the desk floats in front of it
    "playground": {"origin": (100, -60, 105), "desk": (145, -38, 92), "pitch": 35, "other": "strength", "request": 14, "rate": 6.0, "matter": True,
                   "tell": "Playground: everything at once. A 2200 K tungsten plate melts the lead slab on it; the melt flows with "
                           "momentum, runs off onto the cold granite and freezes where it stops; the gold cube sinks through the "
                           "melt. Every number comes from each block's own temperature; the boss bar keeps the books."},
    # the strength scene is 74 x 34 blocks (1 cm each) at x 0-74, z 260-294, 36 high; the desk floats in front of it
    "strength": {"origin": (0, -60, 260), "desk": (37, -40, 244), "pitch": 30, "other": "impact", "request": 15, "rate": 6.0, "matter": True,
                 "tell": "Strength: bodies that rotate, slide and break. The leaning granite stack topples. The 3000 K tungsten "
                         "block heats the steel beam under it; where the steel is hot enough to yield, the beam bends, sags, "
                         "tears and drops it. Every strength, friction and bounce comes from the blocks' own state."},
    # the impact scene is 48 x 40 written blocks at 5 mm (96 x 80 blocks) at x 80-176, z 300-380; the desk in front of it
    "impact": {"origin": (80, -60, 312), "desk": (128, -42, 300), "pitch": 30, "other": "cat", "request": 16, "rate": 6.0, "matter": True,
               "extra": (("Weapon", IMPACT_WEAPON),),
               "tell": "Impact: the range's weapon fires at the steel wall. The game's own solve says how far the round "
                       "goes and what it loses in each layer; that momentum pushes the block into the backstop, and the "
                       "energy the block does not take as motion heats the bore, which glows. Weapon = next round."},
    # the cat stage is 86 x 14 blocks (25 cm each) at x 0-85, z 430-443: the box on the left, the ledger of its quantum state on the right
    "cat": {"origin": (0, -60, 430), "desk": (42, -40, 392), "pitch": 0, "other": "sink", "request": 18, "rate": 6.0,
            "extra": (("Pinhole", 19), ("Air", 20), ("Open box", 21), ("Step inside", 22), ("Step outside", 23), ("Subject", 24), ("Lump T", 25)),
            "tell": "Schrodinger's cat, computed: a sealed box, a cobalt-60 sample sized for even odds in an hour, a counter, a vial, a "
                    "cat, and you. The box's own air records the cat in 1.7e-27 s, long before anyone could look; sealed, the whole "
                    "content is still one quantum state in principle (cyan meter) and no experiment can see it in practice (gold "
                    "meter). Step inside to be the observer: you see one outcome. Open box: the outside looks, and sees the same. "
                    "Pinhole: one photon tells the outside. Air: pump it out and only the cat's own light records it. Subject: swap the "
                    "cat for a 95 g lump of osmium in a vacuum box; Lump T steps it 0 K, 1 mK, 1 K, 300 K. At 0 K nothing records it, "
                    "until you step inside: a 305 K body."},
}
STOP = 9
SLOWER, FASTER = 4, 5
SCREEN_BUTTONS = {6: "observe", 7: "one_slit", 10: "colour"}
RANGE_BUTTONS = {12: "weapon", 13: "armor"}
CAT_BUTTONS = {19: "leak", 20: "air", 21: "open", 24: "subject", 25: "temp"}
CAT_INSIDE, CAT_OUTSIDE = 22, 23
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

    def send_matter(self, records, runs):
        """Matter blocks (frame 'SKCN'): ``records``: matter-record JSON texts ('' is air); ``runs``: (x, y, z, length, index
        into records). Returns (blocks set, server ms)."""
        out = [struct.pack(">ii", 0x534B434E, len(records))]
        for text in records:
            b = text.encode("utf-8")
            out.append(struct.pack(">i", len(b)) + b)
        out.append(struct.pack(">i", len(runs)))
        out.append(b"".join(struct.pack(">iiihh", *r) for r in runs))
        self.sock.sendall(b"".join(out))
        return struct.unpack(">ii", self._exact(8))

    def _exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("block writer closed")
            buf += chunk
        return buf

    def poll(self):
        """What he said to Skippy since the last poll (P6): ``[{"player", "text", "eye", "look", "hit", "block"}]``."""
        self.sock.sendall(struct.pack(">i", 0x534B4350))
        out = []
        for _ in range(struct.unpack(">i", self._exact(4))[0]):
            player = self._exact(struct.unpack(">H", self._exact(2))[0]).decode("utf-8")
            text = self._exact(struct.unpack(">H", self._exact(2))[0]).decode("utf-8")
            x, y, z, dx, dy, dz = struct.unpack(">6d", self._exact(48))
            hit = bool(self._exact(1)[0])
            bx, by, bz = struct.unpack(">iii", self._exact(12))
            out.append({"player": player, "text": text, "eye": (x, y, z), "look": (dx, dy, dz), "hit": hit,
                        "block": (bx, by, bz)})
        return out

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
    if name in SEPARATE:
        return separate_blocks(name, *SEPARATE[name])[0]
    return field_from_voxel_exec(code, block_m=block_m, resolution=SCENE_RESOLUTION.get(name, RESOLUTION))[0]


def separate_blocks(name, boxes, material, canonical):
    """A scene's field with ``boxes`` of ``material`` added, and each box's cells as its own body ``(canonical, mask)``:
    blocks of one material that touch would otherwise be one body. Each mask comes from the same build with that box
    marked, so the grids match."""
    import numpy as np
    from sigma_ground.deckard.voxel_exec import field_from_voxel_exec
    code, block_m = SCENES[name][:2]
    res = SCENE_RESOLUTION.get(name, RESOLUTION)
    full = code + "; " + "; ".join('box(%d,%d,%d,%d,%d,%d,"%s")' % (*b, material) for b in boxes)
    field = field_from_voxel_exec(full, block_m=block_m, resolution=res)[0]
    masks = []
    for b in boxes:
        probe = field_from_voxel_exec(full + '; box(%d,%d,%d,%d,%d,%d,"gold")' % b, block_m=block_m, resolution=res)[0]
        assert np.shape(probe.label_grid) == np.shape(field.label_grid)
        masks.append((canonical, np.asarray(probe.label_grid) == probe.materials.index("gold")))
    return field, masks


IMPACT_ROUNDS = ("m2_browning", "ak47", "pistol_9mm", "crossbow_bolt", "armor_piercing_30mm", "sabot_apfsds",
                 "railgun_hypersonic", "hel_100kw")          # the range's weapons that carry momentum in the game


def impact_world(weapon="m2_browning"):
    """The impact scene with ``weapon`` set to fire at IMPACT_FIRE_S, at the middle of the top-left block's face."""
    import numpy as np
    import shooting_range
    from sigma_ground.materia.playground import PlaygroundWorld
    field, masks = separate_blocks("impact", *SEPARATE["impact"])
    w = PlaygroundWorld(field, liquid="steel_mild", bodies=masks, env=SCENES["impact"][2], room=IMPACT_ROOM)
    target = min((b for b in w.bodies), key=lambda b: (-round(b.x[2], 3), b.x[0]))     # top course, lowest x
    start = np.array([target.x[0], w.label.shape[1] * w.pitch - 1e-4, target.x[2]])   # from the desk's side
    solve = shooting_range.layer_solver(weapon)
    w.at(IMPACT_FIRE_S, lambda W: W.impact(start, (0.0, -1.0, 0.0), solve, shooting_range.hole_radius_m(weapon)))
    w.weapon = weapon
    return field, w


def _body_name(b):
    return b["material"] if isinstance(b, dict) else b.material


def _body_cell(b):
    """A body's lowest corner cell: the slosh tank's dict bodies, or a rigid voxel body (``dynamics.voxel_rigid``)."""
    return b["cell"] if isinstance(b, dict) else b.grid.min(0)


def build(name):
    """``(field, world)`` for a scene."""
    import numpy as np
    from sigma_ground.materia.voxel_fluid import FluidVoxelWorld
    from sigma_ground.materia.voxel_world import VoxelWorld

    _, _, env, temp, _, _ = SCENES[name]
    if temp == "range":
        import shooting_range
        return None, shooting_range.RangeWorld(armed=False)
    if temp == "cat":
        from sigma_ground.materia.voxel_cat import CatBox
        return None, CatBox()
    if temp == "slits":
        from sigma_ground.materia.voxel_slits import SlitScreen
        return None, SlitScreen()
    field = scene_field(name)
    if temp == "playground":
        from sigma_ground.materia.playground import PlaygroundWorld
        T = np.where(field.label_grid == field.materials.index("tungsten"), 2200.0, 293.15)
        return field, PlaygroundWorld(field, liquid="lead", temperature_k=T, bodies=PLAYGROUND_BODIES, env=env)
    if temp == "strength":
        from sigma_ground.materia.playground import PlaygroundWorld
        field, masks = separate_blocks(name, *SEPARATE[name])
        lab = np.asarray(field.label_grid)
        T = np.where(lab == field.materials.index("tungsten"), STRENGTH_LOAD_K, 293.15)
        return field, PlaygroundWorld(field, liquid="steel_mild", temperature_k=T, env=env,
                                      bodies=["steel_mild", "tungsten"] + masks)
    if temp == "impact":
        return impact_world()
    if temp == "flip":
        return field, FluidVoxelWorld(field, paddle=dict(PADDLE), bodies=BODIES)
    T = None
    if temp == "plate_1500":
        T = np.where(field.label_grid == field.materials.index("tungsten"), 1500.0, 293.15)
    return field, VoxelWorld(field, temperature_k=T, env=env)


def shared_looks():
    """One carrier per (material, block size) across every scene that still draws in carrier blocks (the screens and the range: not matter
    mode), so those theaters share one resource pack. Matter-mode scenes need none: their cells carry sigma-ground's records."""
    fields = [f for f in (scene_field(n) for n in SCENES if not STAGES.get(n, {}).get("matter")) if f is not None]
    looks, appearances = {}, {}
    for f in fields:
        lk, ap = export_mc.derive_looks(f)
        for m in lk:
            appearances.setdefault((m, f.voxel_size), ap[m])
    for key, a in range_appearances().items():
        appearances.setdefault(key, a)
    opaque, clear = iter(export_mc.CARRIERS_OPAQUE), iter(export_mc.CARRIERS_CLEAR)
    for key, a in appearances.items():
        try:
            looks[key] = next(clear if a["opacity"] < export_mc.CLEAR_BELOW else opaque)
        except StopIteration:
            raise ValueError(f"the pack is out of carrier blocks at {key}: fewer materials or scenes") from None
    return looks, appearances


SLOT_BASE = 1000      # matter-mode palette slots start here: below are the materials' own indices and the free bodies' slots

def range_appearances():
    """``{(material, 0.002): appearance}`` for every range material. A material sigma-ground has no colour for (Kevlar,
    ABS) is drawn in the pack's flat grey and named as a gap; the game's hand-picked cross-section colours are not used
    (the Captain's rule: a sourced colour or a flagged gap, never a borrowed number)."""
    import shooting_range
    from sigma_ground.field.interface.block import appearance
    return {(m, shooting_range.PITCH): appearance(m, shooting_range.PITCH) for m in shooting_range.materials()}


def cat_looks():
    """The cat stage's looks: each material is a plain block that the pack leaves alone (nothing is repainted)."""
    from sigma_ground.materia.voxel_cat import LOOKS, PITCH
    return {(m, PITCH): block for m, block in LOOKS.items()}


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

    def __init__(self, world, looks, origin, send, blocks=None, matter=False):
        import numpy as np
        self.world, self.send, self.blocks = world, send, blocks
        # matter mode: each cell is a skippycraft:matter block carrying sigma-ground's record for its material (and a live one for a
        # moving body), not a carrier block repainted by the resource pack. Needs the mod's block writer.
        self.matter = bool(matter and blocks is not None)
        self.records, self.body_json, self.shown_body = {}, {}, {}
        self.slot_of = {}                 # (material index, temperature step, melt step) -> palette slot, stable so a cell only changes when its state does
        self.ambient_k = float(getattr(world, "ambient_T", getattr(world, "temperature_k", 293.15)))
        self.origin = tuple(int(v) for v in origin)
        occ = np.argwhere(world.label > 0)
        self.i0, self.k0, self.jhi = int(occ[:, 0].min()), int(occ[:, 2].min()), int(occ[:, 1].max())
        # material slots 0..MAX_MATERIALS-1 (a material placed from inside the game takes the next free one: sync);
        # glow carriers past them
        self.looks = looks
        self.block = ["air"] + ([] if self.matter else [looks[(m, world.pitch)] for m in world.materials[1:]])
        self.block += ["air"] * (MAX_MATERIALS - len(self.block))
        self.n_materials = len(world.materials)
        # unknown (-1) from the theater's ground up, so the first draw also clears what an earlier run left there; below
        # the ground is Minecraft's own ground and is never touched
        self.shown = np.full(world.label.shape, -1, np.int32 if self.matter else world.label.dtype)
        self.shown[:, :, :self.k0] = world.label[:, :, :self.k0]
        self.changed_total = 0

    def mc(self, i, j, k):
        x0, y0, z0 = self.origin
        return x0 + int(i) - self.i0, y0 + int(k) - self.k0, z0 + self.jhi - int(j)

    def cell(self, x, y, z):
        """The grid cell shown at Minecraft block (x, y, z), or None when it is outside the grid."""
        x0, y0, z0 = self.origin
        i, k, j = int(x) - x0 + self.i0, int(y) - y0 + self.k0, self.jhi - (int(z) - z0)
        if all(0 <= c < n for c, n in zip((i, j, k), self.world.label.shape)):
            return (i, j, k)
        return None

    def cell_float(self, x, y, z):
        """A point in Minecraft's continuous coordinates as grid cell units (cell i spans [i, i+1))."""
        x0, y0, z0 = self.origin
        return (x - x0 + self.i0, self.jhi + 1 - (z - z0), y - y0 + self.k0)

    def sync(self):
        """Give every material the world gained since the last draw its carrier block."""
        mats = self.world.materials
        if self.matter:
            self.n_materials = len(mats)
            return
        for idx in range(self.n_materials, len(mats)):
            key = (mats[idx], self.world.pitch)
            if key not in self.looks:
                raise ValueError(f"no carrier block for {mats[idx]} at {self.world.pitch:g} m in the pack")
            if idx >= MAX_MATERIALS:
                raise ValueError("too many materials for the theater's palette")
            self.block[idx] = self.looks[key]
        self.n_materials = len(mats)

    def display(self):
        """What each cell is drawn as: its material (carrier mode; a matter scene draws each cell from its record instead)."""
        return getattr(self.world, "drawn", self.world.label)

    def draw(self):
        """Send every cell whose block differs from what Minecraft shows. Returns how many."""
        import numpy as np
        self.sync()
        label = getattr(self.world, "drawn", self.world.label) if self.matter else self.display()
        if self.matter:
            label, changed = self._matter_label(label)
        else:
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

    def _body_record(self, b):
        """A free body's record now: its velocity (in Minecraft's axes) and the time it is as of, only when it is moving."""
        import numpy as np
        import matter
        v = np.asarray(b["vel"], float)
        mc_v = (float(v[0]), float(v[2]), -float(v[1]))                 # world (i, j, k up) -> Minecraft (x, y up, z)
        moving = any(abs(c) > 1e-6 for c in mc_v)
        return matter.to_wire(matter.record(b["material"], None, self.world.pitch, self.ambient_k,
                                            t=float(self.world.t) if moving else None, v=mc_v if moving else (0.0, 0.0, 0.0)))

    TEMPERATURE_STEP_K, MELT_STEPS = 10.0, 10     # the resolution a cell's state is told apart at (data resolution, not physics)

    def _slots(self, label):
        """Each non-empty cell's palette slot: its material at its (stepped) live temperature and melt fraction, so a record is made once
        per distinct state and a cell changes block only when its state does."""
        import numpy as np
        import matter
        w = self.world
        T = getattr(w, "T", None)
        tq = np.rint((np.full(label.shape, self.ambient_k) if T is None else T) / self.TEMPERATURE_STEP_K).astype(np.int64)
        mq = np.rint(getattr(w, "melt", np.zeros(label.shape)) * self.MELT_STEPS).astype(np.int64)
        out = np.zeros(label.shape, np.int32)
        live = label > 0
        keys = (label.astype(np.int64) * 100000 + np.clip(tq, 0, 99999)) * 16 + np.clip(mq, 0, 15)
        for key in np.unique(keys[live]):
            slot = self.slot_of.get(int(key))
            if slot is None:
                slot = SLOT_BASE + len(self.slot_of)
                self.slot_of[int(key)] = slot
                m, rest = divmod(int(key) // 16, 100000)
                self.records[slot] = matter.to_wire(matter.record(
                    w.materials[m], None, w.pitch, rest * self.TEMPERATURE_STEP_K, melt=(int(key) % 16) / self.MELT_STEPS))
            out[live & (keys == key)] = slot
        return out

    def _matter_label(self, label):
        """The label as palette slots, with each free body's cells under its own live record (a slot past the materials), and which cells
        changed: a cell whose block differs, or any cell of a body whose record changed (its velocity, its clock)."""
        label = self._slots(label)
        live = []
        for bi, b in enumerate(getattr(self.world, "bodies", ())):
            if not isinstance(b, dict) or getattr(self, "_clearing", False):
                continue
            idx = tuple(self.world._body_cells(b).T)
            label[idx] = MAX_MATERIALS + bi
            live.append((bi, idx, self._body_record(b)))
        changed = label != self.shown
        for bi, idx, rec in live:
            self.body_json[bi] = rec
            if self.shown_body.get(bi) != rec:
                changed[idx] = True
                self.shown_body[bi] = rec
        return label, changed

    def _draw_matter(self, runs):
        """Send the runs as one matter frame, its palette only the records the runs use."""
        used = sorted({r[4] for r in runs})
        slot = {old: new for new, old in enumerate(used)}
        records = [self.records.get(old, "") if old >= SLOT_BASE else self.body_json.get(old - MAX_MATERIALS, "") if old >= MAX_MATERIALS else ""
                   for old in used]
        self.blocks.send_matter(records, [(*r[:4], slot[r[4]]) for r in runs])

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
        if runs and self.matter:
            self._draw_matter(runs)
        elif runs:
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
        self._clearing = True
        try:
            self.draw()
        finally:
            self._clearing = False
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
        want = ("skippycraft:matter" if shown[i, j, k] > 0 else "minecraft:air") if theater.matter else "minecraft:" + theater.block[shown[i, j, k]]
        if "passed" not in rcon.command(f"execute if block {x} {y} {z} {want}"):
            bad.append(((x, y, z), want))
    return bad


# ── the service: panels, the served pack, requests ────────────────────────────────────────────────────────────────────────
def build_pack():
    """Write the shared resource pack and its zip for the server to hand out. Returns the looks."""
    import shutil
    looks, appearances = shared_looks()
    export_mc.write_resourcepack(looks, appearances, PACK_DIR)
    screen, paint = slit_looks()
    export_mc.write_flat_textures(paint, PACK_DIR)
    shutil.make_archive(str(PACK_DIR / "skippycraft"), "zip", PACK_DIR / "skippycraft")
    return {**looks, **screen, **cat_looks()}


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


def cat_captions(th, world):
    """Text over the cat stage's panels (``world.captions()``: grid cells and words), in front of the board."""
    cmds = []
    for n, (i, j, k, text) in enumerate(world.captions()):
        x, y, z = th.mc(i, j, int(k))
        cmds += label(f"skippycraft_cat_{n}", x + 0.5, y + 0.5 + (k - int(k)), z - 0.7, text)
    return cmds


def step_through(rcon, world, th, inside, tell):
    """Move him into the sealed box (to stand beside the cat, facing it) or back to the desk."""
    if inside:
        if world is None or th is None:
            tell("Run the cat first, then step inside.")
            return
        x, y, z = th.mc(*world.OBSERVER_CELL)
        rcon.command(f"tp @p {x + 0.5} {y} {z + 0.5} 90 10")
        tell(world.set_observer(True))
        th.draw()
    else:
        sx, sy, sz, yaw, pitch = stand("cat")
        rcon.command(f"tp @p {sx} {sy} {sz} {yaw} {pitch}")
        tell(world.set_observer(False) if world is not None else "You are outside the box, at the desk.")
        if th is not None:
            th.draw()


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
    serve_pack()
    rcon = Rcon()
    send = rcon.commands
    blocks = block_writer()
    print("block writer:", "plugin" if blocks else "none (RCON fills)", flush=True)
    send(["forceload add -16 0 191 207", "forceload add -16 208 191 399", "forceload add -16 400 191 463", "kill @e[type=text_display,tag=skippycraft_label]", "scoreboard objectives add skippycraft dummy",
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
    impact_aim = {"weapon": IMPACT_ROUNDS[0]}

    def impact_press():
        impact_aim["weapon"] = IMPACT_ROUNDS[(IMPACT_ROUNDS.index(impact_aim["weapon"]) + 1) % len(IMPACT_ROUNDS)]
        return f"Impact: next round {shooting_range.presets()[0][impact_aim['weapon']]['name']}."
    tell = lambda text: rcon.command("tellraw @a " + json.dumps({"text": text, "color": "green"}))
    import controls
    session = controls.Session()

    def hear(world=None, th=None):
        """What he said since the last tick, acted on and answered (P6). Returns a request the loop must act on."""
        if blocks is None:
            return None
        for m in blocks.poll():
            reply = session.apply(m, world, th)
            print(f"{m['player']} said: {m['text']!r} -> {reply}", flush=True)
            tell(reply)
        req, session.request = session.request, None
        return req
    print("serving: panels built, pack at", f"http://{PACK_HOST}:{PACK_PORT}/skippycraft.zip", flush=True)
    pending, last_check = 0, 0.0
    while True:
        try:
            req = pending or read_request(rcon)
            pending = 0
            if req in SCREEN_BUTTONS:                                     # pressed between runs: change the setup, then run
                tell(kept.setdefault("slit", build("slit")[1]).press(SCREEN_BUTTONS[req]))
                req = STAGES["slit"]["request"]
            if req in CAT_BUTTONS:                                        # the cat box: change the setup, then run
                tell(kept.setdefault("cat", build("cat")[1]).press(CAT_BUTTONS[req]))
                if CAT_BUTTONS[req] == "open":                            # the door opens where it stands: no new run
                    if kept.get("cat_th"):
                        kept["cat_th"].draw()
                    continue
                req = STAGES["cat"]["request"]
            if req in (CAT_INSIDE, CAT_OUTSIDE):                          # between runs: just move him
                step_through(rcon, kept.get("cat"), kept.get("cat_th"), req == CAT_INSIDE, tell)
                continue
            if req in RANGE_BUTTONS:                                      # a new weapon or armour: show the fresh target
                tell(range_press(aim, RANGE_BUTTONS[req]))
                req, preview = STAGES["range"]["request"], True
            if req == IMPACT_WEAPON:                                      # the next round, and fire it
                tell(impact_press())
                req = STAGES["impact"]["request"]
            heard = hear()
            if heard in STAGES:
                req = STAGES[heard]["request"]
            elif heard and heard not in ("stop", "reset"):
                tell(f"No theater called {heard}: " + ", ".join(STAGES) + ".")
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
            elif name == "cat":                                           # the box keeps its setup between runs; a new run draws the Born rule again
                world = kept.setdefault("cat", build("cat")[1])
                world.reset()
            elif name == "range":
                import shooting_range
                world = shooting_range.RangeWorld(aim["weapon"], aim["armor"], armed=not preview)
            elif name == "impact":
                _, world = impact_world(impact_aim["weapon"])
            else:
                _, world = build(name)
            th = Theater(world, looks, st["origin"], send, blocks, matter=st.get("matter", False))
            rcon.command(f'title @a actionbar {json.dumps({"text": "Running " + name, "color": "green"})}')
            tell(f"Building {name} at {SCENE_RESOLUTION.get(name, RESOLUTION)}x resolution (a few seconds)...")
            th.open()
            if name == "cat":
                kept["cat_th"] = th
                rcon.commands(cat_captions(th, world))
            tell(st["tell"])
            _, _, _, _, ticks, dt = SCENES[name]
            if preview:                                                   # just show the fresh target
                ticks, preview = 0, False
            print("run", name, flush=True)
            tick = 0
            while tick < ticks:
                t0 = time.monotonic()
                heard = hear(world, th)
                if heard == "stop":
                    tell(f"Stopped {name}.")
                    break
                if heard == "reset":
                    pending = st["request"]
                    break
                if heard in STAGES:
                    pending = STAGES[heard]["request"]
                    break
                if not session.tick_allowed():                             # paused: show what he changed, wait
                    th.readout(th.draw())
                    time.sleep(0.1)
                    continue
                tick += 1
                nxt = read_request(rcon)
                if nxt in RANGE_BUTTONS:                                  # change the setup mid-run: start over with it
                    tell(range_press(aim, RANGE_BUTTONS[nxt]))
                    pending, preview = STAGES["range"]["request"], True
                    break
                if nxt == IMPACT_WEAPON:
                    tell(impact_press())
                    pending = STAGES["impact"]["request"]
                    break
                if nxt in CAT_BUTTONS and name == "cat":
                    tell(world.press(CAT_BUTTONS[nxt]))
                    if CAT_BUTTONS[nxt] != "open":                         # the switches start a fresh run
                        pending = st["request"]
                        break
                    nxt = 0
                if nxt in (CAT_INSIDE, CAT_OUTSIDE) and name == "cat":
                    step_through(rcon, world, th, nxt == CAT_INSIDE, tell)
                    nxt = 0
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
                    where = " ".join(f"{_body_name(b)}@{th.mc(*_body_cell(b))}" for b in getattr(world, "bodies", ()))
                    print(f"{name} tick {tick}: {text} {where}", flush=True)
                time.sleep(max(0.0, 1.0 / (st["rate"] * session.speed) - (time.monotonic() - t0)))
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
    looks = {**looks, **screen, **cat_looks()}
    if a.scene == "looks":
        print(json.dumps({f"{m}@{s:g}m": c for (m, s), c in looks.items()}, indent=2))
        return
    _, world = build(a.scene)
    rcon = Rcon()
    th = Theater(world, looks, a.at, rcon.commands, block_writer())
    th.open()
    _, _, _, _, ticks, dt = SCENES[a.scene]
    th.run(a.ticks or ticks, dt, a.rate, on_tick=print)
    if a.verify:
        bad = verify(th, rcon)
        print(f"verify: {len(bad)} mismatched cells" + (f", first {bad[:5]}" if bad else ""))
    rcon.close()


if __name__ == "__main__":
    main()
