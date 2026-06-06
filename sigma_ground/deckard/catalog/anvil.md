# Construct Spec — anvil

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 

## Parts (primitives)

- **base** — box (cast iron) @ (0.0, -0.15, 0.0)
    - dims: x_m=0.5  [estimated], y_m=0.3  [estimated], z_m=0.1  [estimated]
    - density: 7874.0 (field.interface.surface.MATERIALS; conf 0.60) kg/m³
- **top** — box (cast iron) @ (0.0, -0.06, 0.0)
    - dims: x_m=0.48  [estimated], y_m=0.28  [estimated], z_m=0.1  [estimated]
    - density: 7874.0 (field.interface.surface.MATERIALS; conf 0.60) kg/m³
- **handle** — cylinder (steel) @ (0.24, 0.0, 0.0)
    - dims: radius_m=0.05  [estimated], height_m=1.0  [estimated]
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "anvil",
  "kind": "composite",
  "identified": true,
  "geometry": {},
  "layers": [],
  "parts": [
    {
      "name": "base",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.5,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.3,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "z_m": {
          "value": 0.1,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
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
        -0.15,
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
      "name": "top",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.48,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.28,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "z_m": {
          "value": 0.1,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
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
        -0.06,
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
      "name": "handle",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.05,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "height_m": {
          "value": 1.0,
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
        0.24,
        0.0,
        0.0
      ],
      "euler_deg": [
        90.0,
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
