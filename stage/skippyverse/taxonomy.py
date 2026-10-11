"""Small hand-written taxonomy: category parents (the closure table), keyword -> coarse category for placeholder names, and the
size-sanity bands per coarse category.

The bands are JUDGEMENT (a plausible longest-dimension range in metres for a thing of that sort), not physics and not sourced;
their only job is to catch absurd entries such as a 0.27 m "jet" (design section 9: "a per-class size sanity gate"). They are
deliberately wide, so a flag means "look at this", never "this is wrong"."""
from __future__ import annotations

import re

# flat catalog categories -> parent (the catalog's own categories are single words)
CATEGORY_PARENTS = {
    "seating": "furniture", "table": "furniture", "bed": "furniture", "storage": "furniture", "lighting": "furniture",
    "furniture": None, "appliance": "household", "bath": "household", "electronics": "household", "decor": "household",
    "container": "household", "household": None,
    "vehicle/aircraft": "vehicle", "vehicle/road": "vehicle", "vehicle/cycle": "vehicle", "vehicle": None,
    "plant": "nature", "nature": None, "tool": None, "tableware": "household", "building": None, "misc": None,
}


def category_closure(category: str) -> list[str]:
    """['vehicle/aircraft'] -> ['vehicle/aircraft', 'vehicle']; ['seating'] -> ['seating', 'furniture']."""
    out, seen, cur = [], set(), category
    while cur and cur not in seen:
        out.append(cur)
        seen.add(cur)
        nxt = CATEGORY_PARENTS.get(cur)
        if nxt is None and "/" in cur:
            nxt = cur.rsplit("/", 1)[0]
        cur = nxt
    return out


# first match wins; matched against whole words of a lowercased name
KEYWORD_CATEGORY = [
    (r"airplane|aeroplane|aircraft|jet|jumbo|plane|helicopter|glider|spacecraft|space shuttle|rocket ship", "vehicle/aircraft"),
    (r"bicycle|bike|motorcycle|scooter|motorbike|moped", "vehicle/cycle"),
    (r"car|motorcar|automobile|truck|van|bus|suv|taxi|tractor|vehicle|jeep|limo|limousine", "vehicle/road"),
    (r"sofa|couch|chair|stool|bench|armchair|settee|recliner|ottoman", "seating"),
    (r"table|desk|counter", "table"),
    (r"bed|mattress|crib|cot", "bed"),
    (r"shelf|shelves|bookshelf|bookcase|cabinet|dresser|wardrobe|locker|drawer|cupboard|armoire|bureau|sideboard", "storage"),
    (r"lamp|chandelier|lantern|sconce", "lighting"),
    (r"fridge|refrigerator|stove|oven|microwave|dishwasher|washer|dryer|toaster|kettle|radiator|heater", "appliance"),
    (r"toilet|bathtub|sink|shower|basin", "bath"),
    (r"tv|television|monitor|computer|laptop|phone|keyboard|speaker|camera|printer|radio", "electronics"),
    (r"mug|cup|glass|bottle|bowl|plate|vase|jar|pot|pan|knife|fork|spoon", "tableware"),
    (r"tree|plant|flower|bush|shrub|palm", "plant"),
    (r"house|building|tower|church|barn|shed|garage", "building"),
    (r"hammer|screwdriver|wrench|saw|drill|pliers", "tool"),
]
_KW = [(re.compile(r"\b(?:" + pat + r")s?\b"), cat) for pat, cat in KEYWORD_CATEGORY]


def guess_category(name: str) -> str:
    n = name.lower()
    for rx, cat in _KW:
        if rx.search(n):
            return cat
    return "misc"


# plausible LONGEST dimension per coarse category, metres: (low, high). Judgement, see module docstring.
SIZE_BANDS = {
    "vehicle/aircraft": (1.5, 80.0), "vehicle/road": (1.2, 20.0), "vehicle/cycle": (0.6, 3.0),
    "seating": (0.3, 3.5), "table": (0.3, 6.0), "bed": (0.8, 3.0), "storage": (0.2, 3.5), "lighting": (0.1, 3.0),
    "appliance": (0.2, 3.0), "bath": (0.3, 2.5), "electronics": (0.03, 2.5), "tableware": (0.02, 0.8),
    "plant": (0.05, 40.0), "building": (2.0, 300.0), "tool": (0.04, 1.8),
    "furniture": (0.2, 6.0), "household": (0.03, 3.5), "container": (0.1, 3.5), "decor": (0.03, 3.0),
}
DEFAULT_BAND = (0.003, 120.0)


def band_for(category: str) -> tuple[float, float]:
    for c in category_closure(category):
        if c in SIZE_BANDS:
            return SIZE_BANDS[c]
    return DEFAULT_BAND
