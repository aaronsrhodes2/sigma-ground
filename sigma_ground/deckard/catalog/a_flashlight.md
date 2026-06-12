# Construct Spec — a flashlight

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- ShapeNetSem (Savva et al. 2015), median of 17 models — overall size (construct scaled to it) — ShapeNet Terms of Use (non-commercial research)

## Parts (primitives)

- **body** — cylinder (plastic) @ (0.0, 0.0, 0.0)
    - dims: radius_m=0.009582 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), height_m=0.479106 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **cap** — cylinder (metal) @ (0.0, 0.0, 0.47910638297872343)
    - dims: radius_m=0.009582 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), height_m=0.063881 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **switch** — box (metal) @ (0.0, -0.00319404255319149, 0.2235829787234043)
    - dims: x_m=0.006388 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), y_m=0.006388 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), z_m=0.03194 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **light_emitter** — box (plastic) @ (0.0, 0.0, -0.2235829787234043)
    - dims: x_m=0.006388 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), y_m=0.006388 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), z_m=0.03194 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 1000.0  [estimated] kg/m³

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "a flashlight",
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
          "value": 0.009582,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.479106,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
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
      "name": "cap",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.009582,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.063881,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        }
      },
      "material": "metal",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        0.0,
        0.47910638297872343
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "switch",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.006388,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.006388,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.03194,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        }
      },
      "material": "metal",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        -0.00319404255319149,
        0.2235829787234043
      ],
      "euler_deg": [
        0.0,
        90.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "light_emitter",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.006388,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.006388,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.03194,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
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
        -0.2235829787234043
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
      "name": "ShapeNetSem (Savva et al. 2015), median of 17 models — overall size (construct scaled to it)",
      "license": "ShapeNet Terms of Use (non-commercial research)"
    }
  ],
  "notes": "Researched by qwen2.5:7b."
}
```
