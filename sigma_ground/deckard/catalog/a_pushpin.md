# Construct Spec — a pushpin

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 

## Parts (primitives)

- **shaft** — cylinder (steel) @ (0.0, 0.0, 0.0)
    - dims: radius_m=0.0005  [estimated], height_m=0.02  [estimated]
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **head** — sphere (plastic) @ (0.0, 0.0, 0.02)
    - dims: radius_m=0.003  [estimated]
    - density: 1000.0  [estimated] kg/m³

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "a pushpin",
  "kind": "composite",
  "identified": true,
  "geometry": {},
  "layers": [],
  "parts": [
    {
      "name": "shaft",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.0005,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "height_m": {
          "value": 0.02,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
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
      "name": "head",
      "shape": "sphere",
      "dims": {
        "radius_m": {
          "value": 0.003,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        }
      },
      "material": "plastic",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        0.0,
        0.02
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
    }
  ],
  "notes": "Researched by qwen2.5:7b."
}
```
