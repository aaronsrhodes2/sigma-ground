# Deckard data sources — licenses, terms & politeness

Deckard grounds the *facts* it cites (material densities, object dimensions,
shape outlines, part decompositions) in real sources. We are a **polite** data
citizen: every source's usage terms are recorded here **before** we pull from it,
and the shared fetch layer (`deckard/sources/web.py`) enforces the etiquette —

- **Identifies itself**: User-Agent `Deckard/0.2 (sigma-ground physics shape
  researcher; https://github.com/aaronsrhodes2/sigma-ground)` — a real client
  name plus a contact URL, as Wikimedia's API policy requires.
- **Throttles**: at most ~1 request/second per host, issued **sequentially**
  (`_rate_limit`), never in parallel.
- **Caches**: every fetched URL is cached on disk, so we essentially never hit a
  source twice for the same datum.
- **Degrades silently**: any failure returns `None`; research still runs.

A grounded value's provenance (`Fact.source` + `Fact.license`) carries the
attribution below, so a Construct-Spec's `## Sources` section credits each source.

| Source | Used for | License | Access / terms | Attribution |
|---|---|---|---|---|
| **Wikipedia** (REST `…/page/summary/<title>`) | a free-text extract to *ground the prompt* (typical proportions + composition) | CC BY-SA 4.0 | public REST API; UA + sequential; we read a short summary, never bulk-scrape | "Wikipedia, CC BY-SA 4.0" |
| **Wikidata** (Action API: `wbsearchentities`, `wbgetclaims`) | material **density** (P2054) and object **dimensions** (P2048 height / P2049 width / P2386 diameter / P2043 length) | CC0 | public API; UA + sequential; only whitelisted units trusted, unknown units refused | "Wikidata <QID>, CC0" |
| **Quick, Draw!** (`storage.googleapis.com/quickdraw_dataset/full/simplified/<category>.ndjson`) | a canonical **2D outline** per noun (medoid stroke) → the `Outline` primitive | **CC BY 4.0** | public GCS bucket; per-category NDJSON fetched **once**, distilled offline into `inventory/data/outlines/<noun>.json`; the dataset itself is not redistributed | **"Quick, Draw! by Google, Inc., CC BY 4.0"** |
| **Standard dimension tables** (ISO 216 paper, ISO 4014/4017 + ISO 261 fasteners, nominal lumber, sports balls, brick) | exact **dimensions** of standardized objects | the numeric values are facts (uncopyrightable); cited by standard number | encoded locally in `inventory/data/dimensions.json` — no network | "ISO 216", "ISO 4014", … |
| **PartNet** (part hierarchies) | a **part-decomposition** prior for composites (deferred) | research license, free account + approval, **non-commercial** | NOT yet pulled; would be distilled OFFLINE into a tiny `compositions.json` table, never the meshes; terms to be re-confirmed before any download | "PartNet (Mo et al. 2019)" |
| **ShapeNetSem** (real-world dims/volume/weight) | object **scale** grounding (deferred) | research license, account-gated, **non-commercial** | NOT yet pulled; same offline-distill-only policy as PartNet | "ShapeNet (Chang et al. 2015)" |

## Notes
- **PartNet / ShapeNetSem are not pulled yet.** They are account-gated and
  carry non-commercial research terms; before any download we re-read their
  current license and only ship a small *distilled* fact-table (part graphs /
  bounding-box sizes), never the datasets or meshes.
- **Quick Draw** is fetched per-category (a few MB each, e.g. `feather.ndjson`),
  cached, and reduced to one canonical outline per noun. We ship the distilled
  outline, attributed to Google under CC BY 4.0.
- To slow down further (or speed up offline tests), set
  `DECKARD_FETCH_MIN_INTERVAL_S` (seconds between same-host requests).
