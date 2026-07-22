# Construct Spec — plate

- **kind**: composite
- **identified**: True

## Sources

- qwen3:8b — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 
- ShapeNetSem (Savva et al. 2015) — aspect ratio corrected to median proportions (n=28) — ShapeNet Terms of Use (non-commercial research)
- ShapeNetSem (Savva et al. 2015), median of 28 models — overall size (construct scaled to it) — ShapeNet Terms of Use (non-commercial research)

## Parts (primitives)

- **plate** — cylinder (ceramic) @ (0.0, 0.0, 0.0)
    - dims: radius_m=0.20625 (scaled to ShapeNetSem (Savva et al. 2015), median of 28 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), height_m=0.0132 (scaled to ShapeNetSem (Savva et al. 2015), median of 28 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 3950.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³

## Notes

Researched by qwen3:8b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "plate",
  "kind": "composite",
  "identified": true,
  "geometry": {},
  "layers": [],
  "parts": [
    {
      "name": "plate",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.20625,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 28 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.0132,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 28 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        }
      },
      "material": "ceramic",
      "density_kg_m3": {
        "value": 3950.0,
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
    }
  ],
  "sources": [
    {
      "name": "qwen3:8b — researched proportions (estimates)",
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
      "name": "ShapeNetSem (Savva et al. 2015), median of 28 models — overall size (construct scaled to it)",
      "license": "ShapeNet Terms of Use (non-commercial research)"
    }
  ],
  "notes": "Researched by qwen3:8b."
}
```
