# Construct Spec — steel anvil

- **kind**: composite
- **identified**: True

## Sources

- gemini-2.5-flash — researched proportions (estimates) — 
- field.interface.surface.MATERIALS — 

## Parts (primitives)

- **base** — box (steel) @ (0.0, 0.0, 0.04)
    - dims: x_m=0.4  [estimated], y_m=0.2  [estimated], z_m=0.08  [estimated]
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **body** — box (steel) @ (0.0, 0.0, 0.0)
    - dims: x_m=0.35  [estimated], y_m=0.18  [estimated], z_m=0.12  [estimated]
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **face** — box (hardened steel) @ (0.0, 0.0, 0.0)
    - dims: x_m=0.4  [estimated], y_m=0.15  [estimated], z_m=0.05  [estimated]
    - density: 1000.0  [estimated] kg/m³
- **horn_cylinder** — cylinder (steel) @ (0.26, 0.0, 0.185)
    - dims: radius_m=0.04  [estimated], height_m=0.12  [estimated]
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **horn_cone** — cone (steel) @ (0.0, 0.0, 0.0)
    - dims: radius_m=0.04  [estimated], height_m=0.08  [estimated]
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **hardy_hole** — box (void) @ (-0.1, 0.03, 0.185)
    - dims: x_m=0.025  [estimated], y_m=0.025  [estimated], z_m=0.05  [estimated]
    - density: 1000.0  [estimated] kg/m³
- **pritchel_hole** — cylinder (void) @ (-0.06, -0.03, 0.185)
    - dims: radius_m=0.0075  [estimated], height_m=0.05  [estimated]
    - density: 1000.0  [estimated] kg/m³

## Notes

Researched by gemini-2.5-flash.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "steel anvil",
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
          "value": 0.4,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.2,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "z_m": {
          "value": 0.08,
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
        0.0,
        0.0,
        0.04
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "body",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.35,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.18,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "z_m": {
          "value": 0.12,
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
      "attach": {
        "to": "base",
        "my": "bottom",
        "their": "top"
      }
    },
    {
      "name": "face",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.4,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.15,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "z_m": {
          "value": 0.05,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        }
      },
      "material": "hardened steel",
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
      "op": "add",
      "attach": {
        "to": "body",
        "my": "bottom",
        "their": "top"
      }
    },
    {
      "name": "horn_cylinder",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.04,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "height_m": {
          "value": 0.12,
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
        0.26,
        0.0,
        0.185
      ],
      "euler_deg": [
        0.0,
        90.0,
        0.0
      ],
      "op": "add"
    },
    {
      "name": "horn_cone",
      "shape": "cone",
      "dims": {
        "radius_m": {
          "value": 0.04,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "height_m": {
          "value": 0.08,
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
        0.0,
        0.0,
        0.0
      ],
      "euler_deg": [
        0.0,
        90.0,
        0.0
      ],
      "op": "add",
      "attach": {
        "to": "horn_cylinder",
        "my": "-x",
        "their": "+x"
      }
    },
    {
      "name": "hardy_hole",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.025,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.025,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "z_m": {
          "value": 0.05,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        }
      },
      "material": "void",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        -0.1,
        0.03,
        0.185
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "subtract"
    },
    {
      "name": "pritchel_hole",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.0075,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "height_m": {
          "value": 0.05,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        }
      },
      "material": "void",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        -0.06,
        -0.03,
        0.185
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "subtract"
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
    }
  ],
  "notes": "Researched by gemini-2.5-flash."
}
```
