# Construct Spec — drinking glass

- **kind**: layered_vessel
- **identified**: True

## Sources

- gemini-2.5-flash — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 
- inventory/data/materials.json — 

## Geometry

- **outer_radius_m**: 0.035  [estimated]
- **height_m**: 0.12  [estimated]
- **wall_m**: 0.002  [estimated]
- **glaze_m**: 0.0  [estimated]
- **base_m**: 0.005  [estimated]
- **fill_fraction**: 0.8  [estimated]

## Layers (outer→inner)

- **glaze** — glass
    - density: 2500.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
    - thickness: 0.0  [estimated] m
    - interfaces: air, ceramic
- **ceramic** — glass
    - density: 2500.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
    - thickness: 0.002  [estimated] m
    - interfaces: glaze, air, water
- **water** — liquid water
    - density: 997.0 (inventory/data/materials.json; conf 0.90) kg/m³
    - thickness: 0.092  [estimated] m
    - interfaces: ceramic, air

## Notes

Common cylindrical drinking vessel made of soda-lime glass, partially filled with water.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "drinking glass",
  "kind": "layered_vessel",
  "identified": true,
  "geometry": {
    "outer_radius_m": {
      "value": 0.035,
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
      "value": 0.002,
      "source": "estimated",
      "license": "",
      "confidence": 0.5
    },
    "glaze_m": {
      "value": 0.0,
      "source": "estimated",
      "license": "",
      "confidence": 0.5
    },
    "base_m": {
      "value": 0.005,
      "source": "estimated",
      "license": "",
      "confidence": 0.5
    },
    "fill_fraction": {
      "value": 0.8,
      "source": "estimated",
      "license": "",
      "confidence": 0.5
    }
  },
  "layers": [
    {
      "name": "glaze",
      "material": "glass",
      "density_kg_m3": {
        "value": 2500.0,
        "source": "field.interface.surface.MATERIALS",
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
        "air",
        "ceramic"
      ]
    },
    {
      "name": "ceramic",
      "material": "glass",
      "density_kg_m3": {
        "value": 2500.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "thickness_m": {
        "value": 0.002,
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
        "value": 0.092,
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
  "sources": [
    {
      "name": "gemini-2.5-flash — researched proportions (estimates)",
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
  "notes": "Common cylindrical drinking vessel made of soda-lime glass, partially filled with water."
}
```
