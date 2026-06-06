# Construct Spec — electric kettle

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- inventory/data/materials.json — 
- typical size (everyday object) — overall size (construct scaled to it) — 
- modeled hollow - bulk density from a thin-wall shell estimate (appliances/containers are mostly air) — 

## Parts (primitives)

- **body** — cylinder (stainless steel) @ (0.0, 0.0, 0.0)
    - dims: radius_m=0.072816 (scaled to typical size (everyday object); conf 0.45), height_m=0.145631 (scaled to typical size (everyday object); conf 0.45)
    - density: 32.599  [estimated] kg/m³
- **lid** — box (stainless steel) @ (0.0, 0.0, 0.14563106796116504)
    - dims: x_m=0.087379 (scaled to typical size (everyday object); conf 0.45), y_m=0.116505 (scaled to typical size (everyday object); conf 0.45), z_m=0.024272 (scaled to typical size (everyday object); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **handle** — cylinder (stainless steel) @ (0.06310679611650485, 0.0, -0.048543689320388356)
    - dims: radius_m=0.012136 (scaled to typical size (everyday object); conf 0.45), height_m=0.087379 (scaled to typical size (everyday object); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **spout** — cylinder (stainless steel) @ (-0.06310679611650485, 0.0, -0.048543689320388356)
    - dims: radius_m=0.012136 (scaled to typical size (everyday object); conf 0.45), height_m=0.087379 (scaled to typical size (everyday object); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **fill** — fill (water) @ (0.0, 0.0, 0.0)
    - density: 997.0 (inventory/data/materials.json; conf 0.90) kg/m³
    - fills: body to 0.85 (gas on top: air)

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "electric kettle",
  "kind": "composite",
  "identified": true,
  "geometry": {},
  "layers": [],
  "parts": [
    {
      "name": "body",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.072816,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.145631,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "stainless steel",
      "density_kg_m3": {
        "value": 32.599,
        "source": "estimated",
        "license": "",
        "confidence": 0.4
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
      "name": "lid",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.087379,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.116505,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.024272,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "stainless steel",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        0.0,
        0.14563106796116504
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
          "value": 0.012136,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.087379,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "stainless steel",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.06310679611650485,
        0.0,
        -0.048543689320388356
      ],
      "euler_deg": [
        90.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "spout",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.012136,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.087379,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "stainless steel",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        -0.06310679611650485,
        0.0,
        -0.048543689320388356
      ],
      "euler_deg": [
        90.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "fill",
      "shape": "fill",
      "dims": {},
      "material": "water",
      "density_kg_m3": {
        "value": 997.0,
        "source": "inventory/data/materials.json",
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
      "op": "add",
      "fill": {
        "of": "body",
        "fraction": 0.85,
        "gas": "air"
      }
    }
  ],
  "sources": [
    {
      "name": "qwen2.5:7b — researched proportions (estimates)",
      "license": ""
    },
    {
      "name": "inventory/data/materials.json",
      "license": ""
    },
    {
      "name": "typical size (everyday object) — overall size (construct scaled to it)",
      "license": ""
    },
    {
      "name": "modeled hollow - bulk density from a thin-wall shell estimate (appliances/containers are mostly air)",
      "license": ""
    }
  ],
  "notes": "Researched by qwen2.5:7b."
}
```
