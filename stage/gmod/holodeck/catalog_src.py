"""The curated half of the holodeck's shape library: WHICH real models stand for each everyday thing, what people call
it, and the few facts about its use that a model file cannot tell (does it sit against a wall, can things stand on it).

Everything geometric (size, mass, material) is measured in the engine by stage/gmod/addon/lua/mentat/tools/measure_models.lua
and merged by build_catalog.py. Which side is the FRONT was estimated from the collision mesh and then CHECKED BY EYE on
photographs of every model (stage/gmod/holodeck/contact_sheet.py); the corrections live in data/front_overrides.json.
Models the photographs showed to be wrong for their name were removed rather than patched (an open fridge shell without its
door, a lamp shade that hangs from a ceiling, two mattresses with no bunk frame, a shop counter called a water cooler...).

Fields of a kind
    kind        canonical id
    names       singular names people use (multi-word allowed); plurals are made by the parser
    approx      names that are only the NEAREST thing the library has ("fireplace" -> wood stove); the holodeck says so
    category    seating | table | bed | storage | lighting | appliance | electronics | decor | bath | container
    seat        True if a person sits on it (drives conversation-group layouts)
    surface     True if its top is a flat surface things can stand on
    wall        "prefer" (backs onto a wall when one is near) | "free" (stands anywhere)
    front       "auto_back" (front = opposite the tall side, from the physics mesh, then corrected by eye) | "none" (looks the
                same from every side) | an explicit yaw in degrees of the front direction in the model's own frame
    variants    model paths (an entry may also be a dict with its own `model`, `front`)
"""
from __future__ import annotations

KINDS = [
    # ------------------------------------------------------------------------------------------ seating
    dict(kind="couch", names=["couch", "sofa", "settee", "davenport", "sectional"], approx=["chaise", "futon", "daybed", "chesterfield"],
         category="seating", seat=True, wall="prefer", front="auto_back",
         variants=["models/props_c17/furniturecouch001a.mdl", "models/props_interiors/furniture_couch01a.mdl",
                   "models/props/cs_office/sofa.mdl", "models/props/cs_militia/couch.mdl"]),
    dict(kind="loveseat", names=["loveseat", "love seat", "two seater", "2 seater", "two seat sofa"], category="seating", seat=True,
         wall="prefer", front="auto_back", variants=["models/props_c17/furniturecouch002a.mdl"]),
    dict(kind="armchair", names=["armchair", "arm chair", "easy chair", "recliner", "lounge chair", "club chair"],
         approx=["wingback", "wing chair", "lounger"], category="seating", seat=True, wall="free", front="auto_back",
         variants=["models/props_c17/furniturearmchair001a.mdl", "models/props/cs_office/sofa_chair.mdl",
                   "models/props_interiors/furniture_couch02a.mdl"]),
    dict(kind="chair", names=["chair", "dining chair", "kitchen chair", "wooden chair", "seat"], category="seating", seat=True,
         wall="free", front="auto_back",
         variants=["models/props_c17/furniturechair001a.mdl", "models/props_c17/chair02a.mdl",
                   "models/props_interiors/furniture_chair01a.mdl", "models/props_interiors/furniture_chair03a.mdl",
                   "models/props/de_inferno/chairantique.mdl", "models/props/de_tides/patio_chair.mdl",
                   "models/props/de_tides/patio_chair2.mdl"]),
    dict(kind="office chair", names=["office chair", "swivel chair", "desk chair", "computer chair", "task chair"],
         category="seating", seat=True, wall="free", front="auto_back",
         variants=["models/props/cs_office/chair_office.mdl", "models/props_c17/chair_office01a.mdl",
                   "models/props_combine/breenchair.mdl"]),
    dict(kind="stool", names=["stool", "bar stool", "barstool"], approx=["ottoman", "footstool", "pouf"], category="seating", seat=True,
         wall="free", front="none", variants=["models/props_c17/chair_stool01a.mdl", "models/props/cs_militia/barstool01.mdl"]),
    dict(kind="bench", names=["bench", "park bench", "pew"], category="seating", seat=True, wall="prefer", front="auto_back",
         variants=["models/props_c17/bench01a.mdl", "models/props_trainstation/bench_indoor001a.mdl",
                   "models/props/de_inferno/bench_wood.mdl", "models/props/cs_militia/wood_bench.mdl",
                   "models/props_wasteland/cafeteria_bench001a.mdl"]),
    # ------------------------------------------------------------------------------------------ tables and desks
    dict(kind="table", names=["table", "dining table", "kitchen table", "wooden table"], approx=["counter", "workbench", "worktable"],
         category="table", surface=True, wall="free", front="none",
         variants=["models/props_c17/furnituretable001a.mdl", "models/props_c17/furnituretable002a.mdl",
                   "models/props/cs_militia/table_kitchen.mdl",
                   "models/props/de_inferno/tableantique.mdl", "models/props/de_tides/restaurant_table.mdl",
                   "models/props_wasteland/cafeteria_table001a.mdl"]),
    dict(kind="coffee table", names=["coffee table", "low table", "cocktail table"], category="table", surface=True, wall="free",
         front="none", variants=["models/props/cs_office/table_coffee.mdl", "models/props/de_inferno/tablecoffee.mdl",
                                 "models/props_c17/furnituretable003a.mdl"]),         # 0.42 m high: a low table, not a dining table
    dict(kind="side table", names=["side table", "end table", "nightstand", "night stand", "bedside table", "bedside cabinet"],
         approx=["lamp table", "occasional table"], category="table", surface=True, wall="prefer", front=0,
         variants=["models/props_c17/furnituredrawer002a.mdl"]),
    dict(kind="desk", names=["desk", "writing desk", "office desk", "workstation"], approx=["bureau desk", "escritoire"],
         category="table", surface=True, wall="prefer", front="auto_back",
         variants=[{"model": "models/props_interiors/furniture_desk01a.mdl", "front": 0},
                   {"model": "models/props_combine/breendesk.mdl", "front": 180},
                   {"model": "models/props_wasteland/controlroom_desk001a.mdl", "front": 0},
                   {"model": "models/props_wasteland/controlroom_desk001b.mdl", "front": 0}]),
    dict(kind="meeting table", names=["meeting table", "conference table", "boardroom table"], category="table", surface=True,
         wall="free", front="none", variants=["models/props/cs_office/table_meeting.mdl"]),
    dict(kind="patio table", names=["patio table", "garden table", "picnic table", "outdoor table", "umbrella table"], category="table",
         surface=False, wall="free", front="none",           # the physics mesh's highest flat is the umbrella canopy, not the table
         variants=["models/props/de_tides/patio_table.mdl", "models/props/de_tides/patio_table2.mdl"]),
    # ------------------------------------------------------------------------------------------ beds
    dict(kind="bed", names=["bed", "double bed", "single bed", "bed frame"], category="bed", wall="prefer", front="auto_back",
         variants=["models/props_c17/furniturebed001a.mdl", "models/props/de_inferno/bed.mdl",
                   "models/props_wasteland/prison_bedframe001a.mdl"]),
    dict(kind="mattress", names=["mattress"], category="bed", wall="free", front="none",
         variants=["models/props_c17/furnituremattress001a.mdl"]),
    dict(kind="bunk bed", names=["bunk bed", "bunk", "bunkbed"], category="bed", wall="prefer", front="auto_back",
         variants=["models/props/cs_militia/bunkbed.mdl"]),
    dict(kind="hospital bed", names=["hospital bed", "cot", "stretcher bed"], category="bed", wall="free", front="auto_back",
         variants=["models/props_c17/hospital_bed01.mdl"]),
    # ------------------------------------------------------------------------------------------ storage
    dict(kind="bookshelf", names=["bookshelf", "bookcase", "book shelf"], category="storage", wall="prefer",
         front="auto_back",
         variants=["models/props/cs_office/bookshelf1.mdl", "models/props/cs_office/bookshelf2.mdl",
                   "models/props/cs_office/bookshelf3.mdl", "models/props/cs_havana/bookcase_small.mdl"]),
    dict(kind="shelf", names=["shelf", "shelving unit", "rack"], category="storage", wall="prefer", front="auto_back",
         variants=["models/props_c17/shelfunit01a.mdl", "models/props_c17/furnitureshelf001a.mdl",
                   "models/props_interiors/furniture_shelf01a.mdl", "models/props_wasteland/kitchen_shelf001a.mdl"]),
    dict(kind="dresser", names=["dresser", "chest of drawers", "bureau", "drawers", "sideboard"],
         approx=["credenza", "cupboard", "buffet", "hutch", "cabinet unit"], category="storage", surface=True, wall="prefer",
         front="auto_back",
         variants=["models/props_c17/furnituredrawer001a.mdl", "models/props_c17/furnituredrawer003a.mdl",
                   "models/props_interiors/furniture_cabinetdrawer01a.mdl", "models/props_interiors/furniture_cabinetdrawer02a.mdl"]),
    dict(kind="wardrobe", names=["wardrobe", "armoire", "closet"], approx=["clothes cupboard"], category="storage", wall="prefer",
         front=0, variants=["models/props_c17/furnituredresser001a.mdl"]),
    dict(kind="filing cabinet", names=["filing cabinet", "file cabinet", "cabinet"], category="storage", wall="prefer",
         front="auto_back",
         variants=["models/props/cs_office/file_cabinet1.mdl", "models/props/cs_office/file_cabinet2.mdl",
                   "models/props/cs_office/file_cabinet3.mdl", "models/props_wasteland/controlroom_filecabinet001a.mdl",
                   "models/props_wasteland/controlroom_filecabinet002a.mdl", "models/props_lab/filecabinet02.mdl"]),
    dict(kind="locker", names=["locker", "gym locker"], category="storage", wall="prefer", front="auto_back",
         variants=["models/props_c17/lockers001a.mdl", "models/props/de_train/lockers001a.mdl", "models/props_lab/lockers.mdl"]),
    dict(kind="crate", names=["crate", "wooden crate", "shipping crate"], category="container", wall="free", front="none",
         variants=["models/props_junk/wood_crate001a.mdl", "models/props_junk/wood_crate002a.mdl",
                   "models/props_junk/plasticcrate01a.mdl", "models/props/de_dust/du_crate_64x64.mdl",
                   "models/props/de_nuke/crate_small.mdl", "models/props/de_nuke/crate_large.mdl"]),
    dict(kind="box", names=["box", "cardboard box", "carton"], category="container", wall="free", front="none",
         variants=["models/props/cs_office/cardboard_box01.mdl", "models/props/cs_office/cardboard_box02.mdl",
                   "models/props/cs_office/cardboard_box03.mdl", "models/props/cs_office/file_box.mdl"]),
    dict(kind="barrel", names=["barrel", "drum", "oil drum", "keg"], category="container", wall="free", front="none",
         variants=["models/props_c17/woodbarrel001.mdl", "models/props_borealis/bluebarrel001.mdl",
                   "models/props_c17/oildrum001.mdl", "models/props/de_train/barrel.mdl"]),
    dict(kind="chest", names=["chest", "footlocker", "trunk", "treasure chest"], category="container", wall="free", front=0,
         variants=["models/props/cs_militia/footlocker01_closed.mdl"]),
    # ------------------------------------------------------------------------------------------ lighting
    dict(kind="floor lamp", names=["floor lamp", "standing lamp", "standard lamp", "lamp"], category="lighting", wall="free",
         front="none", variants=["models/props_interiors/furniture_lamp01a.mdl"]),
    dict(kind="desk lamp", names=["desk lamp", "table lamp", "reading lamp", "bedside lamp"], category="lighting", wall="free",
         front="none", variants=["models/props_lab/desklamp01.mdl", "models/props_c17/lamp001a.mdl"]),
    # ------------------------------------------------------------------------------------------ kitchen and bath
    dict(kind="fridge", names=["fridge", "refrigerator", "icebox", "freezer"], category="appliance", wall="prefer", front="auto_back",
         variants=["models/props_c17/furniturefridge001a.mdl", "models/props/cs_militia/refrigerator01.mdl",
                   "models/props_wasteland/kitchen_fridge001a.mdl"]),
    dict(kind="stove", names=["stove", "oven", "cooker", "range", "cooktop"], category="appliance", wall="prefer", front="auto_back",
         variants=["models/props_c17/furniturestove001a.mdl", "models/props/cs_militia/stove01.mdl",
                   "models/props_wasteland/kitchen_stove001a.mdl"]),
    dict(kind="microwave", names=["microwave", "microwave oven"], category="appliance", wall="prefer", front="auto_back",
         variants=["models/props/cs_office/microwave.mdl", "models/props/cs_militia/microwave01.mdl"]),
    dict(kind="sink", names=["sink", "kitchen sink", "basin", "washbasin"], category="appliance", wall="prefer", front="auto_back",
         variants=["models/props_c17/furnituresink001a.mdl", "models/props_interiors/sinkkitchen01a.mdl",
                   "models/props_wasteland/prison_sink001a.mdl"]),
    dict(kind="toilet", names=["toilet", "loo", "lavatory", "wc"], category="bath", wall="prefer", front="auto_back",
         variants=["models/props_c17/furnituretoilet001a.mdl", "models/props/cs_militia/toilet.mdl",
                   "models/props_wasteland/prison_toilet01.mdl"]),
    dict(kind="bathtub", names=["bathtub", "bath tub", "bath", "tub"], category="bath", wall="prefer", front="auto_back",
         variants=["models/props_c17/furniturebathtub001a.mdl", "models/props_interiors/bathtub01a.mdl"]),
    dict(kind="radiator", names=["radiator"], approx=["heater", "baseboard heater"], category="appliance", wall="prefer",
         front="auto_back",
         variants=["models/props_c17/furnitureradiator001a.mdl", "models/props_interiors/radiator01a.mdl"]),
    dict(kind="wood stove", names=["wood stove", "woodstove", "potbelly stove", "pot belly stove", "log burner"],
         approx=["fireplace", "hearth", "fire", "furnace"], category="appliance", wall="prefer", front="auto_back",
         variants=["models/props_c17/furniturefireplace001a.mdl"]),
    dict(kind="vending machine", names=["vending machine", "soda machine", "snack machine", "drinks machine"], category="appliance",
         wall="prefer", front="auto_back",
         variants=[{"model": "models/props_interiors/vendingmachinesoda01a.mdl", "front": 270},
                   "models/props/cs_office/vending_machine.mdl"]),
    dict(kind="display case", names=["display case", "display cabinet", "shop counter", "glass counter", "deli counter",
                                     "display cooler"], approx=["water cooler", "cooler"], category="storage", surface=True, wall="prefer", front="auto_back",
         variants=["models/props_c17/display_cooler01a.mdl"]),
    # ------------------------------------------------------------------------------------------ electronics
    dict(kind="tv", names=["tv", "television", "telly", "flat screen", "tv set"], category="electronics", wall="prefer",
         front="auto_back",
         variants=[{"model": "models/props/cs_office/tv_plasma.mdl", "front": 0},
                   {"model": "models/props_c17/tv_monitor01.mdl", "front": 0},
                   {"model": "models/props/cs_militia/television_console01.mdl", "front": 0}]),
    dict(kind="computer", names=["computer", "pc", "desktop computer", "monitor"], category="electronics", wall="free",
         front="auto_back", variants=[{"model": "models/props/cs_office/computer.mdl", "front": 0}]),
    dict(kind="phone", names=["phone", "telephone"], category="electronics", wall="free", front="none",
         variants=["models/props/cs_office/phone.mdl"]),
    # ------------------------------------------------------------------------------------------ decor
    dict(kind="plant", names=["plant", "potted plant", "houseplant", "pot plant", "fern", "flower pot"], approx=["tree", "shrub", "flowers"],
         category="decor", wall="free", front="none",
         variants=["models/props/cs_office/plant01.mdl", "models/props/de_inferno/potted_plant1.mdl",
                   "models/props/de_inferno/potted_plant2.mdl", "models/props/de_inferno/potted_plant3.mdl"]),
    dict(kind="trash can", names=["trash can", "trashcan", "bin", "rubbish bin", "garbage can", "wastebasket", "waste bin"],
         category="container", wall="free", front="none",
         variants=["models/props/cs_office/trash_can.mdl", "models/props_trainstation/trashcan_indoor001a.mdl",
                   "models/props_junk/trashbin01a.mdl"]),
    dict(kind="dumpster", names=["dumpster", "skip"], category="container", wall="free", front="auto_back",
         variants=["models/props_junk/trashdumpster01a.mdl", "models/props_junk/trashdumpster02.mdl"]),
    dict(kind="mug", names=["mug", "coffee mug", "cup"], category="decor", wall="free", front="none",
         variants=["models/props/cs_office/coffee_mug.mdl", "models/props/cs_office/coffee_mug2.mdl",
                   "models/props/cs_office/coffee_mug3.mdl"]),
]

# Scene recipes: a request such as "a living room" expands to these items, each arranged by the layout rules.
# Each line is (count, kind, role); roles are interpreted by the layout engine, never by this file.
RECIPES = {
    "living room": [(1, "couch", "main_seat"), (2, "armchair", "side_seat"), (1, "coffee table", "center"), (1, "tv", "focus"),
                    (1, "floor lamp", "corner"), (1, "plant", "corner"), (1, "bookshelf", "wall")],
    "dining room": [(1, "table", "center"), (4, "chair", "around"), (1, "dresser", "wall"), (1, "plant", "corner")],
    "kitchen": [(1, "fridge", "wall"), (1, "stove", "wall"), (1, "sink", "wall"), (1, "table", "center"), (4, "chair", "around"),
                (1, "microwave", "on_wall_unit")],
    "bedroom": [(1, "bed", "wall_head"), (1, "side table", "beside_bed"), (1, "wardrobe", "wall"), (1, "floor lamp", "corner")],
    "office": [(1, "desk", "wall"), (1, "office chair", "at_desk"), (1, "computer", "on_desk"), (1, "filing cabinet", "wall"),
               (1, "bookshelf", "wall"), (1, "plant", "corner")],
    "waiting room": [(4, "chair", "row_wall"), (1, "coffee table", "center"), (1, "plant", "corner"), (1, "trash can", "corner")],
    "bathroom": [(1, "bathtub", "wall"), (1, "toilet", "wall"), (1, "sink", "wall")],
}
