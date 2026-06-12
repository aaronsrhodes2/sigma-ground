# Construct Spec — a car

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 
- ShapeNetSem (Savva et al. 2015), median of 20 models — overall size (construct scaled to it) — ShapeNet Terms of Use (non-commercial research)

## Parts (primitives)

- **body** — box (steel) @ (0.0, 0.0, -0.020466666666666668)
    - dims: x_m=0.04912 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), y_m=0.1228 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), z_m=0.040933 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **hood** — box (steel) @ (-0.024560000000000002, 0.0, -0.01773777777777778)
    - dims: x_m=0.032747 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), y_m=0.1228 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), z_m=0.009551 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **trunk** — box (steel) @ (0.024560000000000002, 0.0, -0.01773777777777778)
    - dims: x_m=0.032747 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), y_m=0.1228 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), z_m=0.009551 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **tires** — cylinder (rubber) @ (-0.024560000000000002, 0.0, -0.03411111111111111)
    - dims: radius_m=0.008187 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), height_m=0.009551 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 920.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **tires** — cylinder (rubber) @ (0.024560000000000002, 0.0, -0.03411111111111111)
    - dims: radius_m=0.008187 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), height_m=0.009551 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 920.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **tires** — cylinder (rubber) @ (-0.024560000000000002, 0.0, -0.03411111111111111)
    - dims: radius_m=0.008187 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), height_m=0.009551 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 920.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **tires** — cylinder (rubber) @ (0.024560000000000002, 0.0, -0.03411111111111111)
    - dims: radius_m=0.008187 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), height_m=0.009551 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 920.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **windshield** — box (glass) @ (0.0, 0.05457777777777778, -0.020466666666666668)
    - dims: x_m=0.043662 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), y_m=0.035476 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), z_m=0.010916 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 2500.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **roof** — box (steel) @ (0.0, 0.0614, -0.01773777777777778)
    - dims: x_m=0.04912 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), y_m=0.1228 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45), z_m=0.009551 (scaled to ShapeNetSem (Savva et al. 2015), median of 20 models, ShapeNet Terms of Use (non-commercial research); conf 0.45)
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "a car",
  "kind": "composite",
  "identified": true,
  "geometry": {},
  "layers": [],
  "parts": [
    {
      "name": "body",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.04912,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.1228,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.040933,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
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
        -0.020466666666666668
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "hood",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.032747,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.1228,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.009551,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
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
        -0.024560000000000002,
        0.0,
        -0.01773777777777778
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "trunk",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.032747,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.1228,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.009551,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
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
        0.024560000000000002,
        0.0,
        -0.01773777777777778
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "tires",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.008187,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.009551,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        }
      },
      "material": "rubber",
      "density_kg_m3": {
        "value": 920.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        -0.024560000000000002,
        0.0,
        -0.03411111111111111
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add",
      "attach": {
        "to": "body",
        "my": "bottom",
        "their": "bottom"
      }
    },
    {
      "name": "tires",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.008187,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.009551,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        }
      },
      "material": "rubber",
      "density_kg_m3": {
        "value": 920.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.024560000000000002,
        0.0,
        -0.03411111111111111
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add",
      "attach": {
        "to": "body",
        "my": "bottom",
        "their": "bottom"
      }
    },
    {
      "name": "tires",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.008187,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.009551,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        }
      },
      "material": "rubber",
      "density_kg_m3": {
        "value": 920.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        -0.024560000000000002,
        0.0,
        -0.03411111111111111
      ],
      "euler_deg": [
        0.0,
        0.0,
        180.0
      ],
      "op": "add",
      "attach": {
        "to": "body",
        "my": "bottom",
        "their": "bottom"
      }
    },
    {
      "name": "tires",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.008187,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.009551,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        }
      },
      "material": "rubber",
      "density_kg_m3": {
        "value": 920.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.024560000000000002,
        0.0,
        -0.03411111111111111
      ],
      "euler_deg": [
        0.0,
        0.0,
        180.0
      ],
      "op": "add",
      "attach": {
        "to": "body",
        "my": "bottom",
        "their": "bottom"
      }
    },
    {
      "name": "windshield",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.043662,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.035476,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.010916,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        }
      },
      "material": "glass",
      "density_kg_m3": {
        "value": 2500.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.0,
        0.05457777777777778,
        -0.020466666666666668
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "roof",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.04912,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.1228,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.009551,
          "source": "scaled to ShapeNetSem (Savva et al. 2015), median of 20 models",
          "license": "ShapeNet Terms of Use (non-commercial research)",
          "confidence": 0.45
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
        0.0614,
        -0.01773777777777778
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
    },
    {
      "name": "ShapeNetSem (Savva et al. 2015), median of 20 models — overall size (construct scaled to it)",
      "license": "ShapeNet Terms of Use (non-commercial research)"
    }
  ],
  "notes": "Researched by qwen2.5:7b."
}
```
