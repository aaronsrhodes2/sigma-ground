# Construct Spec — cast iron skillet

- **kind**: layered_vessel
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 
- inventory/data/materials.json — 

## Geometry

- **outer_radius_m**: 0.15  [estimated]
- **height_m**: 0.12  [estimated]
- **wall_m**: 0.003  [estimated]
- **glaze_m**: 0.001  [estimated]
- **base_m**: 0.002  [estimated]
- **fill_fraction**: 0.0  [estimated]

## Layers (outer→inner)

- **glaze** — enamel
    - density: 1000.0  [estimated] kg/m³
    - thickness: 0.001  [estimated] m
    - interfaces: air, ceramic
- **ceramic** — stoneware
    - density: 3950.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
    - thickness: 0.003  [estimated] m
    - interfaces: glaze, air, water
- **water** — liquid water
    - density: 997.0 (inventory/data/materials.json; conf 0.90) kg/m³
    - thickness: 0.0  [estimated] m
    - interfaces: ceramic, air

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "cast iron skillet",
  "kind": "layered_vessel",
  "identified": true,
  "geometry": {
    "outer_radius_m": {
      "value": 0.15,
      "source": "estimated",
      "license": "",
      "confidence": 0.5
    },
    "height_m": {
      "value": 0.12,
      "source": "estimated",
      "license": "",
      "confidence": 0.5
    },
    "wall_m": {
      "value": 0.003,
      "source": "estimated",
      "license": "",
      "confidence": 0.5
    },
    "glaze_m": {
      "value": 0.001,
      "source": "estimated",
      "license": "",
      "confidence": 0.5
    },
    "base_m": {
      "value": 0.002,
      "source": "estimated",
      "license": "",
      "confidence": 0.5
    },
    "fill_fraction": {
      "value": 0.0,
      "source": "estimated",
      "license": "",
      "confidence": 0.5
    }
  },
  "layers": [
    {
      "name": "glaze",
      "material": "enamel",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "thickness_m": {
        "value": 0.001,
        "source": "estimated",
        "license": "",
        "confidence": 0.4
      },
      "interfaces": [
        "air",
        "ceramic"
      ]
    },
    {
      "name": "ceramic",
      "material": "stoneware",
      "density_kg_m3": {
        "value": 3950.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "thickness_m": {
        "value": 0.003,
        "source": "estimated",
        "license": "",
        "confidence": 0.4
      },
      "interfaces": [
        "glaze",
        "air",
        "water"
      ]
    },
    {
      "name": "water",
      "material": "liquid water",
      "density_kg_m3": {
        "value": 997.0,
        "source": "inventory/data/materials.json",
        "license": "",
        "confidence": 0.9
      },
      "thickness_m": {
        "value": 0.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.4
      },
      "interfaces": [
        "ceramic",
        "air"
      ]
    }
  ],
  "parts": [],
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
      "name": "inventory/data/materials.json",
      "license": ""
    }
  ],
  "notes": "Researched by qwen2.5:7b."
}
```
