# Construct Spec — a dumbbell

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 
- typical size (everyday object) — overall size (construct scaled to it) — 
- curated common-knowledge decomposition (replace with PartNet) — 

## Parts (primitives)

- **handle** — cylinder (steel) @ (0.0, 0.0, 0.0)
    - dims: radius_m=0.003077 (scaled to typical size (everyday object); conf 0.45), height_m=0.246154 (scaled to typical size (everyday object); conf 0.45)
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **weight1** — cylinder (cast iron) @ (0.18461538461538463, 0.0, 0.0)
    - dims: radius_m=0.015385 (scaled to typical size (everyday object); conf 0.45), height_m=0.061538 (scaled to typical size (everyday object); conf 0.45)
    - density: 7874.0 (field.interface.surface.MATERIALS; conf 0.60) kg/m³
- **weight2** — cylinder (cast iron) @ (-0.18461538461538463, 0.0, 0.0)
    - dims: radius_m=0.015385 (scaled to typical size (everyday object); conf 0.45), height_m=0.061538 (scaled to typical size (everyday object); conf 0.45)
    - density: 7874.0 (field.interface.surface.MATERIALS; conf 0.60) kg/m³

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "a dumbbell",
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
          "value": 0.003077,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.246154,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "steel",
      "density_kg_m3": {
        "value": 7850.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.0,
        0.0,
        0.0
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "weight1",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.015385,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.061538,
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
        0.18461538461538463,
        0.0,
        0.0
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "weight2",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.015385,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.061538,
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
        -0.18461538461538463,
        0.0,
        0.0
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
