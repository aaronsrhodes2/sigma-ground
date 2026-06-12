# Construct Spec — hammer

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 
- ShapeNetSem (Savva et al. 2015) — aspect ratio corrected to median proportions (n=28) — ShapeNet Terms of Use (non-commercial research)
- typical size (everyday object) — overall size (construct scaled to it) — 
- curated common-knowledge decomposition (replace with PartNet) — 

## Parts (primitives)

- **handle** — cylinder (oak) @ (0.0, 0.0, -0.15675)
    - dims: radius_m=0.003879 (scaled to typical size (everyday object); conf 0.45), height_m=0.2145 (scaled to typical size (everyday object); conf 0.45)
    - density: 700.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **head** — box (cast iron) @ (0.0, 0.0, 0.057749999999999996)
    - dims: x_m=0.12932 (scaled to typical size (everyday object); conf 0.45), y_m=0.12932 (scaled to typical size (everyday object); conf 0.45), z_m=0.0165 (scaled to typical size (everyday object); conf 0.45)
    - density: 7874.0 (field.interface.surface.MATERIALS; conf 0.60) kg/m³

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "hammer",
  "kind": "composite",
  "identified": true,
  "geometry": {},
  "layers": [],
  "parts": [
    {
      "name": "handle",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.003879,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.2145,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "oak",
      "density_kg_m3": {
        "value": 700.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.0,
        0.0,
        -0.15675
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "head",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.12932,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.12932,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.0165,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "cast iron",
      "density_kg_m3": {
        "value": 7874.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.6
      },
      "center_m": [
        0.0,
        0.0,
        0.057749999999999996
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    }
  ],
  "sources": [
    {
      "name": "qwen2.5:7b — researched proportions (estimates)",
      "license": ""
    },
    {
      "name": "field.interface.surface.MATERIALS",
      "license": ""
    },
    {
      "name": "ShapeNetSem (Savva et al. 2015) — aspect ratio corrected to median proportions (n=28)",
      "license": "ShapeNet Terms of Use (non-commercial research)"
    },
    {
      "name": "typical size (everyday object) — overall size (construct scaled to it)",
      "license": ""
    },
    {
      "name": "curated common-knowledge decomposition (replace with PartNet)",
      "license": ""
    }
  ],
  "notes": "Researched by qwen2.5:7b."
}
```
