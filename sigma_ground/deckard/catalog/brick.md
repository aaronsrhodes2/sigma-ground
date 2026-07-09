# Construct Spec — brick

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- US modular brick (8 x 3 5/8 x 2 1/4 in) — 

## Parts (primitives)

- **brick_body** — box (clay_brick) @ (0.0, 0.0, 0.0)
    - dims: x_m=0.203 (US modular brick (8 x 3 5/8 x 2 1/4 in); conf 0.90), y_m=0.102 (US modular brick (8 x 3 5/8 x 2 1/4 in); conf 0.90), z_m=0.057 (US modular brick (8 x 3 5/8 x 2 1/4 in); conf 0.90)
    - density: 1000.0  [estimated] kg/m³

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "brick",
  "kind": "composite",
  "identified": true,
  "geometry": {},
  "layers": [],
  "parts": [
    {
      "name": "brick_body",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.203,
          "source": "US modular brick (8 x 3 5/8 x 2 1/4 in)",
          "license": "",
          "confidence": 0.9
        },
        "y_m": {
          "value": 0.102,
          "source": "US modular brick (8 x 3 5/8 x 2 1/4 in)",
          "license": "",
          "confidence": 0.9
        },
        "z_m": {
          "value": 0.057,
          "source": "US modular brick (8 x 3 5/8 x 2 1/4 in)",
          "license": "",
          "confidence": 0.9
        }
      },
      "material": "clay_brick",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
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
    }
  ],
  "sources": [
    {
      "name": "qwen2.5:7b — researched proportions (estimates)",
      "license": ""
    },
    {
      "name": "US modular brick (8 x 3 5/8 x 2 1/4 in)",
      "license": ""
    }
  ],
  "notes": "Researched by qwen2.5:7b."
}
```
