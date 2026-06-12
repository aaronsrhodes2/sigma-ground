# Construct Spec — a disassembled flashlight

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 
- ShapeNetSem (Savva et al. 2015), median of 17 models — overall size (construct scaled to it) — ShapeNet Terms of Use (non-commercial research)

## Parts (primitives)

- **body** — cylinder (aluminium) @ (0.0, 0.0, 0.0)
    - dims: radius_m=0.011259 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), height_m=0.56295 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 2700.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **cap** — cylinder (plastic) @ (0.0, 0.0, 0.3753)
    - dims: radius_m=0.011259 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), height_m=0.18765 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **switch** — box (metal) @ (0.0, -0.168885, 0.0)
    - dims: x_m=0.007506 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), y_m=0.007506 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), z_m=0.03753 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **light_emitter** — box (plastic) @ (0.0, -0.168885, 0.03753)
    - dims: x_m=0.003753 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), y_m=0.003753 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), z_m=0.07506 (scaled to ShapeNetSem (Savva et al. 2015), median of 17 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 1000.0  [estimated] kg/m³

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "a disassembled flashlight",
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
          "value": 0.011259,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.56295,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        }
      },
      "material": "aluminium",
      "density_kg_m3": {
        "value": 2700.0,
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
      "name": "cap",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.011259,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.18765,
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
        0.3753
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
          "value": 0.007506,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.007506,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.03753,
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
        -0.168885,
        0.0
      ],
      "euler_deg": [
        90.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "light_emitter",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.003753,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.003753,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 17 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.07506,
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
        -0.168885,
        0.03753
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
    },
    {
      "name": "ShapeNetSem (Savva et al. 2015), median of 17 models — overall size (construct scaled to it)",
      "license": "ShapeNet Terms of Use (non-commercial research)"
    }
  ],
  "notes": "Researched by qwen2.5:7b."
}
```
