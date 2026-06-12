# Construct Spec — an acoustic guitar

- **kind**: composite
- **identified**: True

## Sources

- qwen2.5:7b — researched proportions (estimates) — 
- Quick, Draw! by Google, Inc. — CC BY 4.0
- field.interface.surface.MATERIALS — 
- inventory/data/materials.json — 
- modeled hollow - bulk density from a thin-wall shell estimate (appliances/containers are mostly air) — 

## Parts (primitives)

- **body** — box (mahogany) @ (0.0, 0.0, -0.05)
    - dims: x_m=0.45  [estimated], y_m=0.35  [estimated], z_m=0.1  [estimated]
    - density: 29.931  [estimated] kg/m³
- **top** — outline (spruce) @ (0.0, 0.0, 0.05)
    - density: 1000.0  [estimated] kg/m³
- **back** — outline (mahogany) @ (0.0, 0.0, -0.05)
    - density: 1000.0  [estimated] kg/m³
- **neck** — cylinder (maple) @ (0.0, -0.25, 0.0)
    - dims: radius_m=0.007  [estimated], height_m=0.6  [estimated]
    - density: 1000.0  [estimated] kg/m³
- **headstock** — box (ebony) @ (0.0, -0.65, 0.0)
    - dims: x_m=0.1  [estimated], y_m=0.03  [estimated], z_m=0.08  [estimated]
    - density: 1000.0  [estimated] kg/m³
- **nut** — box (bone) @ (0.0, -0.58, 0.0)
    - dims: x_m=0.01  [estimated], y_m=0.04  [estimated], z_m=0.02  [estimated]
    - density: 1900.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **frets** — box (steel) @ (0.0, -0.58, 0.0)
    - dims: x_m=0.01  [estimated], y_m=0.02  [estimated], z_m=0.003  [estimated]
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
- **bridge** — box (ebony) @ (0.0, -0.3, 0.0)
    - dims: x_m=0.1  [estimated], y_m=0.02  [estimated], z_m=0.05  [estimated]
    - density: 1000.0  [estimated] kg/m³
- **strings** — fill (steel) @ (0.0, 0.0, 0.0)
    - density: 7850.0 (field.interface.surface.MATERIALS; conf 0.90) kg/m³
    - fills: bridge to 1.0 (gas on top: air)

## Notes

Researched by qwen2.5:7b.

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "an acoustic guitar",
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
          "value": 0.45,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.35,
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
      "material": "mahogany",
      "density_kg_m3": {
        "value": 29.931,
        "source": "estimated",
        "license": "",
        "confidence": 0.4
      },
      "center_m": [
        0.0,
        0.0,
        -0.05
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
      "shape": "outline",
      "dims": {},
      "material": "spruce",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        0.0,
        0.05
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add",
      "outline": {
        "profile": [
          [
            -0.07923,
            -0.07707
          ],
          [
            -0.12006,
            -0.07524
          ],
          [
            -0.1554,
            -0.054509999999999996
          ],
          [
            -0.16473000000000002,
            -0.016319999999999998
          ],
          [
            -0.16212,
            0.025500000000000002
          ],
          [
            -0.15063,
            0.0648
          ],
          [
            -0.1281,
            0.09924
          ],
          [
            -0.09042,
            0.09002999999999999
          ],
          [
            -0.058769999999999996,
            0.06255
          ],
          [
            -0.02004,
            0.05204999999999999
          ],
          [
            0.01812,
            0.06813
          ],
          [
            0.05886,
            0.06294
          ],
          [
            0.08979,
            0.03621
          ],
          [
            0.11931,
            0.01656
          ],
          [
            0.16122,
            0.01698
          ],
          [
            0.20313,
            0.017429999999999998
          ],
          [
            0.24500999999999998,
            0.017849999999999998
          ],
          [
            0.28692,
            0.0183
          ],
          [
            0.28896,
            -0.005399999999999999
          ],
          [
            0.24731999999999998,
            -0.01017
          ],
          [
            0.20546999999999999,
            -0.009359999999999999
          ],
          [
            0.16358999999999999,
            -0.008159999999999999
          ],
          [
            0.12171,
            -0.0069299999999999995
          ],
          [
            0.10526999999999999,
            -0.03684
          ],
          [
            0.08085,
            -0.06863999999999999
          ],
          [
            0.041159999999999995,
            -0.07818
          ],
          [
            -0.00045,
            -0.08256
          ],
          [
            -0.03975,
            -0.07089
          ]
        ],
        "mode": "extrude",
        "thickness": 0.002
      }
    },
    {
      "name": "back",
      "shape": "outline",
      "dims": {},
      "material": "mahogany",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        0.0,
        -0.05
      ],
      "euler_deg": [
        0.0,
        0.0,
        0.0
      ],
      "op": "add",
      "outline": {
        "profile": [
          [
            -0.07923,
            -0.07707
          ],
          [
            -0.12006,
            -0.07524
          ],
          [
            -0.1554,
            -0.054509999999999996
          ],
          [
            -0.16473000000000002,
            -0.016319999999999998
          ],
          [
            -0.16212,
            0.025500000000000002
          ],
          [
            -0.15063,
            0.0648
          ],
          [
            -0.1281,
            0.09924
          ],
          [
            -0.09042,
            0.09002999999999999
          ],
          [
            -0.058769999999999996,
            0.06255
          ],
          [
            -0.02004,
            0.05204999999999999
          ],
          [
            0.01812,
            0.06813
          ],
          [
            0.05886,
            0.06294
          ],
          [
            0.08979,
            0.03621
          ],
          [
            0.11931,
            0.01656
          ],
          [
            0.16122,
            0.01698
          ],
          [
            0.20313,
            0.017429999999999998
          ],
          [
            0.24500999999999998,
            0.017849999999999998
          ],
          [
            0.28692,
            0.0183
          ],
          [
            0.28896,
            -0.005399999999999999
          ],
          [
            0.24731999999999998,
            -0.01017
          ],
          [
            0.20546999999999999,
            -0.009359999999999999
          ],
          [
            0.16358999999999999,
            -0.008159999999999999
          ],
          [
            0.12171,
            -0.0069299999999999995
          ],
          [
            0.10526999999999999,
            -0.03684
          ],
          [
            0.08085,
            -0.06863999999999999
          ],
          [
            0.041159999999999995,
            -0.07818
          ],
          [
            -0.00045,
            -0.08256
          ],
          [
            -0.03975,
            -0.07089
          ]
        ],
        "mode": "extrude",
        "thickness": 0.002
      }
    },
    {
      "name": "neck",
      "shape": "cylinder",
      "dims": {
        "radius_m": {
          "value": 0.007,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "height_m": {
          "value": 0.6,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        }
      },
      "material": "maple",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        -0.25,
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
      "name": "headstock",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.1,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.03,
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
      "material": "ebony",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        -0.65,
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
      "name": "nut",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.01,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.04,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "z_m": {
          "value": 0.02,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        }
      },
      "material": "bone",
      "density_kg_m3": {
        "value": 1900.0,
        "source": "field.interface.surface.MATERIALS",
        "license": "",
        "confidence": 0.9
      },
      "center_m": [
        0.0,
        -0.58,
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
      "name": "frets",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.01,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.02,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "z_m": {
          "value": 0.003,
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
        -0.58,
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
      "name": "bridge",
      "shape": "box",
      "dims": {
        "x_m": {
          "value": 0.1,
          "source": "estimated",
          "license": "",
          "confidence": 0.5
        },
        "y_m": {
          "value": 0.02,
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
      "material": "ebony",
      "density_kg_m3": {
        "value": 1000.0,
        "source": "estimated",
        "license": "",
        "confidence": 0.2
      },
      "center_m": [
        0.0,
        -0.3,
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
      "name": "strings",
      "shape": "fill",
      "dims": {},
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
      "fill": {
        "of": "bridge",
        "fraction": 1.0,
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
      "name": "Quick, Draw! by Google, Inc.",
      "license": "CC BY 4.0"
    },
    {
      "name": "field.interface.surface.MATERIALS",
      "license": ""
    },
    {
      "name": "inventory/data/materials.json",
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
