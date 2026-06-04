# Construct Spec — coffee cup

- **kind**: layered_vessel
- **identified**: True

## Sources

- typical stoneware coffee mug (approx.) — 
- liquid water at 20C = 998 kg/m^3 — public-domain

## Geometry

- **outer_radius_m**: 0.04 (typical stoneware mug (approx.); conf 0.60)
- **height_m**: 0.095 (typical stoneware mug (approx.); conf 0.60)
- **wall_m**: 0.005 (typical stoneware mug (approx.); conf 0.60)
- **glaze_m**: 0.0003 (typical stoneware mug (approx.); conf 0.50)
- **base_m**: 0.007 (typical stoneware mug (approx.); conf 0.60)
- **fill_fraction**: 0.8 (filled to ~80% of interior height; conf 0.50)

## Layers (outer→inner)

- **glaze** — glaze (glassy)
    - density: 2400.0 (ceramic glaze ~ glass; conf 0.60) kg/m³
    - thickness: 0.0003 (typical stoneware mug (approx.); conf 0.50) m
    - interfaces: air, ceramic
- **ceramic** — stoneware
    - density: 2300.0 (stoneware/earthenware body; conf 0.70) kg/m³
    - thickness: 0.005 (typical stoneware mug (approx.); conf 0.60) m
    - interfaces: glaze, air, water
- **water** — liquid water
    - density: 998.0 (liquid water at 20C, public-domain; conf 0.95) kg/m³
    - thickness: 0.03 (fill; conf 0.50) m
    - interfaces: ceramic, air

## Notes

Seed spec - the hand-built reference cup (~599 g, CoM ~40.7 mm).

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "coffee cup",
  "kind": "layered_vessel",
  "identified": true,
  "geometry": {
    "outer_radius_m": {
      "value": 0.04,
      "source": "typical stoneware mug (approx.)",
      "license": "",
      "confidence": 0.6
    },
    "height_m": {
      "value": 0.095,
      "source": "typical stoneware mug (approx.)",
      "license": "",
      "confidence": 0.6
    },
    "wall_m": {
      "value": 0.005,
      "source": "typical stoneware mug (approx.)",
      "license": "",
      "confidence": 0.6
    },
    "glaze_m": {
      "value": 0.0003,
      "source": "typical stoneware mug (approx.)",
      "license": "",
      "confidence": 0.5
    },
    "base_m": {
      "value": 0.007,
      "source": "typical stoneware mug (approx.)",
      "license": "",
      "confidence": 0.6
    },
    "fill_fraction": {
      "value": 0.8,
      "source": "filled to ~80% of interior height",
      "license": "",
      "confidence": 0.5
    }
  },
  "layers": [
    {
      "name": "glaze",
      "material": "glaze (glassy)",
      "density_kg_m3": {
        "value": 2400.0,
        "source": "ceramic glaze ~ glass",
        "license": "",
        "confidence": 0.6
      },
      "thickness_m": {
        "value": 0.0003,
        "source": "typical stoneware mug (approx.)",
        "license": "",
        "confidence": 0.5
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
        "value": 2300.0,
        "source": "stoneware/earthenware body",
        "license": "",
        "confidence": 0.7
      },
      "thickness_m": {
        "value": 0.005,
        "source": "typical stoneware mug (approx.)",
        "license": "",
        "confidence": 0.6
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
        "value": 998.0,
        "source": "liquid water at 20C",
        "license": "public-domain",
        "confidence": 0.95
      },
      "thickness_m": {
        "value": 0.03,
        "source": "fill",
        "license": "",
        "confidence": 0.5
      },
      "interfaces": [
        "ceramic",
        "air"
      ]
    }
  ],
  "sources": [
    {
      "name": "typical stoneware coffee mug (approx.)",
      "license": ""
    },
    {
      "name": "liquid water at 20C = 998 kg/m^3",
      "license": "public-domain"
    }
  ],
  "notes": "Seed spec - the hand-built reference cup (~599 g, CoM ~40.7 mm)."
}
```
