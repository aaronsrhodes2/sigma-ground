# Construct Spec — a wooden chair

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 
- typical size (everyday object) — overall size (construct scaled to it) — 
- PartNet (Mo et al. 2019) — aggregate medians over 8176 models (geometry n=150) — ShapeNet/PartNet Terms of Use (non-commercial research); derived aggregate facts only, no raw rows

## Parts (primitives)

- **seat** — box (oak) @ (0.0, -0.2663706992230855, 0.0)
    - dims: x_m=0.532741 (scaled to typical size (everyday object); conf 0.45), y_m=0.499445 (scaled to typical size (everyday object); conf 0.45), z_m=0.019978 (scaled to typical size (everyday object); conf 0.45)
    - density: 700.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **seat_surface** — box (fabric) @ (0.0, -0.2730299667036626, 0.0)
    - dims: x_m=0.54606 (scaled to typical size (everyday object); conf 0.45), y_m=0.516093 (scaled to typical size (everyday object); conf 0.45), z_m=0.001998 (scaled to typical size (everyday object); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **back_surface** — box (leather) @ (0.0, -0.09322974472807992, 0.3995560488346282)
    - dims: x_m=0.332963 (scaled to typical size (everyday object); conf 0.45), y_m=0.166482 (scaled to typical size (everyday object); conf 0.45), z_m=0.001998 (scaled to typical size (everyday object); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **base** — box (oak) @ (0.0, -0.2663706992230855, -0.13318534961154274)
    - dims: x_m=0.566038 (scaled to typical size (everyday object); conf 0.45), y_m=0.466149 (scaled to typical size (everyday object); conf 0.45), z_m=0.019978 (scaled to typical size (everyday object); conf 0.45)
    - density: 700.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **seat_single_surface** — box (fabric) @ (0.0, -0.27968923418423974, 0.0)
    - dims: x_m=0.54273 (scaled to typical size (everyday object); conf 0.45), y_m=0.496115 (scaled to typical size (everyday object); conf 0.45), z_m=0.001998 (scaled to typical size (everyday object); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **back_single_surface** — box (leather) @ (0.0, -0.0978912319644839, 0.3995560488346282)
    - dims: x_m=0.329634 (scaled to typical size (everyday object); conf 0.45), y_m=0.163152 (scaled to typical size (everyday object); conf 0.45), z_m=0.001998 (scaled to typical size (everyday object); conf 0.45)
    - density: 1000.0  [estimated] kg/m³
- **leg_1** — cylinder (oak) @ (-0.2830188679245283, -0.2663706992230855, -0.13318534961154274)
    - dims: radius_m=0.016648 (scaled to typical size (everyday object); conf 0.45), height_m=0.199778 (scaled to typical size (everyday object); conf 0.45)
    - density: 700.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **leg_2** — cylinder (oak) @ (0.2830188679245283, -0.2663706992230855, -0.13318534961154274)
    - dims: radius_m=0.016648 (scaled to typical size (everyday object); conf 0.45), height_m=0.199778 (scaled to typical size (everyday object); conf 0.45)
    - density: 700.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **leg_3** — cylinder (oak) @ (-0.2830188679245283, -0.2663706992230855, -0.3995560488346282)
    - dims: radius_m=0.016648 (scaled to typical size (everyday object); conf 0.45), height_m=0.199778 (scaled to typical size (everyday object); conf 0.45)
    - density: 700.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **leg_4** — cylinder (oak) @ (0.2830188679245283, -0.2663706992230855, -0.3995560488346282)
    - dims: radius_m=0.016648 (scaled to typical size (everyday object); conf 0.45), height_m=0.199778 (scaled to typical size (everyday object); conf 0.45)
    - density: 700.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **arm_1** — box (oak) @ (-0.2663706992230855, -0.3995560488346282, 0.0)
    - dims: x_m=0.199778 (scaled to typical size (everyday object); conf 0.45), y_m=0.099889 (scaled to typical size (everyday object); conf 0.45), z_m=0.013319 (scaled to typical size (everyday object); conf 0.45)
    - density: 700.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **arm_2** — box (oak) @ (0.2663706992230855, -0.3995560488346282, 0.0)
    - dims: x_m=0.199778 (scaled to typical size (everyday object); conf 0.45), y_m=0.099889 (scaled to typical size (everyday object); conf 0.45), z_m=0.013319 (scaled to typical size (everyday object); conf 0.45)
    - density: 700.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "a wooden chair",
  "kind": "composite",
  "identified": true,
  "geometry": {},
  "layers": [],
  "parts": [
    {
      "name": "seat",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.532741,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.499445,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.019978,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "oak",
      "density_kg_m3": {
        "value": 700.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.0,
        -0.2663706992230855,
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
      "name": "seat_surface",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.54606,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.516093,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.001998,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "fabric",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        -0.2730299667036626,
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
      "name": "back_surface",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.332963,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.166482,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.001998,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "leather",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        -0.09322974472807992,
        0.3995560488346282
      ],
      "euler_deg": [
        0.0,
        0.0,
        90.0
      ],
      "op": "add"
    },
    {
      "name": "base",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.566038,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.466149,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.019978,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "oak",
      "density_kg_m3": {
        "value": 700.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.0,
        -0.2663706992230855,
        -0.13318534961154274
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "seat_single_surface",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.54273,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.496115,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.001998,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "fabric",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        -0.27968923418423974,
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
      "name": "back_single_surface",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.329634,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.163152,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.001998,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "leather",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        -0.0978912319644839,
        0.3995560488346282
      ],
      "euler_deg": [
        0.0,
        0.0,
        90.0
      ],
      "op": "add"
    },
    {
      "name": "leg_1",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.016648,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.199778,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "oak",
      "density_kg_m3": {
        "value": 700.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        -0.2830188679245283,
        -0.2663706992230855,
        -0.13318534961154274
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "leg_2",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.016648,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.199778,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "oak",
      "density_kg_m3": {
        "value": 700.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.2830188679245283,
        -0.2663706992230855,
        -0.13318534961154274
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "leg_3",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.016648,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.199778,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "oak",
      "density_kg_m3": {
        "value": 700.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        -0.2830188679245283,
        -0.2663706992230855,
        -0.3995560488346282
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "leg_4",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.016648,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "height_m": {
          "value": 0.199778,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "oak",
      "density_kg_m3": {
        "value": 700.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.2830188679245283,
        -0.2663706992230855,
        -0.3995560488346282
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "arm_1",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.199778,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.099889,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.013319,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "oak",
      "density_kg_m3": {
        "value": 700.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        -0.2663706992230855,
        -0.3995560488346282,
        0.0
      ],
      "euler_deg": [
        0.0,
        90.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "arm_2",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.199778,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "y_m": {
          "value": 0.099889,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        },
        "z_m": {
          "value": 0.013319,
          "source": "scaled to typical size (everyday object)",
          "license": "",
          "confidence": 0.45
        }
      },
      "material": "oak",
      "density_kg_m3": {
        "value": 700.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.2663706992230855,
        -0.3995560488346282,
        0.0
      ],
      "euler_deg": [
        0.0,
        90.0,
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
      "name": "typical size (everyday object) — overall size (construct scaled to it)",
      "license": ""
    },
    {
      "name": "PartNet (Mo et al. 2019) — aggregate medians over 8176 models (geometry n=150)",
      "license": "ShapeNet/PartNet Terms of Use (non-commercial research); derived aggregate facts only, no raw rows"
    }
  ],
  "notes": "Researched by qwen2.5:7b."
}
```
