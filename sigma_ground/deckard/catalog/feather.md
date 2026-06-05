# Construct Spec — feather

- **kind**: composite
- **identified**: True

## Sources

- feather anatomy (rachis + flattened vane); keratin density estimated (beta-keratin ~1300 kg/m3, not in local data) — 

## Parts (primitives)

- **rachis** — cone (keratin) @ (0.0, 0.0, 0.0)
    - dims: radius_m=0.0008 (feather anatomy (typical flight feather); conf 0.40), height_m=0.12 (feather anatomy (typical flight feather); conf 0.40)
    - density: 1300.0  [estimated] kg/m³
- **vane** — ellipsoid (keratin) @ (0.0, 0.0, 0.0)
    - dims: rx_m=0.0003 (feather anatomy (typical flight feather); conf 0.40), ry_m=0.012 (feather anatomy (typical flight feather); conf 0.40), rz_m=0.05 (feather anatomy (typical flight feather); conf 0.40)
    - density: 1300.0  [estimated] kg/m³

## Notes

Primitive-kit approximation of a flight feather: tapered keratin shaft (cone, 120mm) plus a flattened webbed vane (thin ellipsoid, 100x24x0.6mm). The LLM researcher returns kind:unknown for a feather (not primitive-friendly), so this is a hand-specified, deliberately approximate shape; dimensions are typical-flight-feather estimates and the keratin density is flagged [estimated].

## Canonical payload

<!-- compile() reads this block; keep the prose above in sync. -->
```json
{
  "name": "feather",
  "kind": "composite",
  "identified": true,
  "geometry": {},
  "layers": [],
  "parts": [
    {
      "name": "rachis",
      "shape": "cone",
      "dims": {
        "radius_m": {
          "value": 0.0008,
          "source": "feather anatomy (typical flight feather)",
          "license": "",
          "confidence": 0.4
        },
        "height_m": {
          "value": 0.12,
          "source": "feather anatomy (typical flight feather)",
          "license": "",
          "confidence": 0.4
        }
      },
      "material": "keratin",
      "density_kg_m3": {
        "value": 1300.0,
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
      "name": "vane",
      "shape": "ellipsoid",
      "dims": {
        "rx_m": {
          "value": 0.0003,
          "source": "feather anatomy (typical flight feather)",
          "license": "",
          "confidence": 0.4
        },
        "ry_m": {
          "value": 0.012,
          "source": "feather anatomy (typical flight feather)",
          "license": "",
          "confidence": 0.4
        },
        "rz_m": {
          "value": 0.05,
          "source": "feather anatomy (typical flight feather)",
          "license": "",
          "confidence": 0.4
        }
      },
      "material": "keratin",
      "density_kg_m3": {
        "value": 1300.0,
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
    }
  ],
  "sources": [
    {
      "name": "feather anatomy (rachis + flattened vane); keratin density estimated (beta-keratin ~1300 kg/m3, not in local data)",
      "license": ""
    }
  ],
  "notes": "Primitive-kit approximation of a flight feather: tapered keratin shaft (cone, 120mm) plus a flattened webbed vane (thin ellipsoid, 100x24x0.6mm). The LLM researcher returns kind:unknown for a feather (not primitive-friendly), so this is a hand-specified, deliberately approximate shape; dimensions are typical-flight-feather estimates and the keratin density is flagged [estimated]."
}
```
