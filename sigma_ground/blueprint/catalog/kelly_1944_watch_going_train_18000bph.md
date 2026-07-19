# Blueprint Spec — kelly_1944_watch_going_train_18000bph

- **identified**: True

## Sources

- Kelly, Harold C. — A Practical Course in Horology (1944) — public scan, freely accessible (archive.org + survivorlibrary.com) — https://archive.org/stream/practicalcoursei00kellrich/practicalcoursei00kellrich_djvu.txt (pp. 16-18 (going train), Part I Ch. 3 opening (escapement type))

## Gears

- **barrel** — wheel (teeth): 72 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.16, 'Calculating the number of turns of a pinion'; conf 0.95)
- **center_pinion** — pinion (leaves): 12 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.16, 'Calculating the number of turns of a pinion'; conf 0.95)
- **center_wheel** — wheel (teeth): 80 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.18, going-train formula 'CTF/tfe'; conf 0.95)
- **third_pinion** — pinion (leaves): 10 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.18, 'the fourth wheel must make 60 turns to one of the center wheel'; conf 0.95)
- **third_wheel** — wheel (teeth): 75 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.18, going-train formula 'CTF/tfe'; conf 0.95)
- **fourth_pinion** — pinion (leaves): 10 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.18, 'the fourth wheel must make 60 turns to one of the center wheel'; conf 0.95)
- **fourth_wheel** — wheel (teeth): 80 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.18, going-train formula 'CTF/tfe'; conf 0.95)
- **escape_pinion** — pinion (leaves): 8 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.18, 'Calculating the number of beats'; conf 0.85)
- **escape_wheel** — wheel (teeth): 15 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.18, 'Calculating the number of beats'; conf 0.95)

## Meshes

- **barrel** → **center_pinion**
- **center_wheel** → **third_pinion**
- **third_wheel** → **fourth_pinion**
- **fourth_wheel** → **escape_pinion**

## Escapement

- kind: lever
- escape wheel teeth: 15 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.18, 'Calculating the number of beats'; conf 0.95)
- beats/hour: 18000 (Kelly, Harold C. — A Practical Course in Horology (1944), public scan, freely accessible (archive.org + survivorlibrary.com) — p.18, 'Calculating the number of beats'; conf 0.95)

## Notes

The standard 18,000-beats/hour lever-escapement watch train Kelly presents as "a modern train" (p.17). No module/pitch/center-distance is cited for this specific example — a real gap (see module docstring), not filled here. Escapement kind 'lever' is a documented cross-reference, not an assumption: the going-train passage describes impulses delivered "first to the receiving pallet and later to the discharging pallet" (p.18), and Ch. 3 (opening quoted above) identifies that two-pallet terminology as the lever escapement, "the superiority of the lever escapement over all other types for portable timepieces."

## Canonical payload

<!-- consumers read this block; keep the prose above in sync. -->
```json
{
  "name": "kelly_1944_watch_going_train_18000bph",
  "identified": true,
  "gears": [
    {
      "name": "barrel",
      "teeth": {
        "value": 72,
        "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
        "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
        "confidence": 0.95,
        "quote": "Suppose, for example, a wheel of 72 teeth gears into a pinion of 12 leaves.",
        "locator": "p.16, 'Calculating the number of turns of a pinion'"
      },
      "is_pinion": false,
      "module_mm": null,
      "pressure_angle_deg": null,
      "tooth_form": null,
      "addendum_coeff": null,
      "dedendum_coeff": null,
      "face_width_mm": null,
      "material": ""
    },
    {
      "name": "center_pinion",
      "teeth": {
        "value": 12,
        "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
        "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
        "confidence": 0.95,
        "quote": "Suppose, for example, a wheel of 72 teeth gears into a pinion of 12 leaves.",
        "locator": "p.16, 'Calculating the number of turns of a pinion'"
      },
      "is_pinion": true,
      "module_mm": null,
      "pressure_angle_deg": null,
      "tooth_form": null,
      "addendum_coeff": null,
      "dedendum_coeff": null,
      "face_width_mm": null,
      "material": ""
    },
    {
      "name": "center_wheel",
      "teeth": {
        "value": 80,
        "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
        "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
        "confidence": 0.95,
        "quote": "CTF / tfe: 80 X 75 X 80 / 10 X 10 X 8 = 600 turns of the escape wheel.",
        "locator": "p.18, going-train formula 'CTF/tfe'"
      },
      "is_pinion": false,
      "module_mm": null,
      "pressure_angle_deg": null,
      "tooth_form": null,
      "addendum_coeff": null,
      "dedendum_coeff": null,
      "face_width_mm": null,
      "material": ""
    },
    {
      "name": "third_pinion",
      "teeth": {
        "value": 10,
        "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
        "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
        "confidence": 0.95,
        "quote": "CT / tf: 80 X 75 / 10 X 10 = 60 turns of the fourth wheel.",
        "locator": "p.18, 'the fourth wheel must make 60 turns to one of the center wheel'"
      },
      "is_pinion": true,
      "module_mm": null,
      "pressure_angle_deg": null,
      "tooth_form": null,
      "addendum_coeff": null,
      "dedendum_coeff": null,
      "face_width_mm": null,
      "material": ""
    },
    {
      "name": "third_wheel",
      "teeth": {
        "value": 75,
        "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
        "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
        "confidence": 0.95,
        "quote": "CTF / tfe: 80 X 75 X 80 / 10 X 10 X 8 = 600 turns of the escape wheel.",
        "locator": "p.18, going-train formula 'CTF/tfe'"
      },
      "is_pinion": false,
      "module_mm": null,
      "pressure_angle_deg": null,
      "tooth_form": null,
      "addendum_coeff": null,
      "dedendum_coeff": null,
      "face_width_mm": null,
      "material": ""
    },
    {
      "name": "fourth_pinion",
      "teeth": {
        "value": 10,
        "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
        "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
        "confidence": 0.95,
        "quote": "CT / tf: 80 X 75 / 10 X 10 = 60 turns of the fourth wheel.",
        "locator": "p.18, 'the fourth wheel must make 60 turns to one of the center wheel'"
      },
      "is_pinion": true,
      "module_mm": null,
      "pressure_angle_deg": null,
      "tooth_form": null,
      "addendum_coeff": null,
      "dedendum_coeff": null,
      "face_width_mm": null,
      "material": ""
    },
    {
      "name": "fourth_wheel",
      "teeth": {
        "value": 80,
        "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
        "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
        "confidence": 0.95,
        "quote": "CTF / tfe: 80 X 75 X 80 / 10 X 10 X 8 = 600 turns of the escape wheel.",
        "locator": "p.18, going-train formula 'CTF/tfe'"
      },
      "is_pinion": false,
      "module_mm": null,
      "pressure_angle_deg": null,
      "tooth_form": null,
      "addendum_coeff": null,
      "dedendum_coeff": null,
      "face_width_mm": null,
      "material": ""
    },
    {
      "name": "escape_pinion",
      "teeth": {
        "value": 8,
        "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
        "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
        "confidence": 0.85,
        "quote": "The escape wheel in most watches contains 15 teeth and delivers twice as many impulses to the balance, since each tooth delivers two impulses, first to the receiving pallet and later to the discharging pallet. ... CTF2E/tfe = number of beats per hour. Substituting the numerical values we have: 80X75X80X2X15 / 10X10X8 = 18,000 beats per hour.",
        "locator": "p.18, 'Calculating the number of beats'"
      },
      "is_pinion": true,
      "module_mm": null,
      "pressure_angle_deg": null,
      "tooth_form": null,
      "addendum_coeff": null,
      "dedendum_coeff": null,
      "face_width_mm": null,
      "material": ""
    },
    {
      "name": "escape_wheel",
      "teeth": {
        "value": 15,
        "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
        "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
        "confidence": 0.95,
        "quote": "The escape wheel in most watches contains 15 teeth and delivers twice as many impulses to the balance, since each tooth delivers two impulses, first to the receiving pallet and later to the discharging pallet. ... CTF2E/tfe = number of beats per hour. Substituting the numerical values we have: 80X75X80X2X15 / 10X10X8 = 18,000 beats per hour.",
        "locator": "p.18, 'Calculating the number of beats'"
      },
      "is_pinion": false,
      "module_mm": null,
      "pressure_angle_deg": null,
      "tooth_form": null,
      "addendum_coeff": null,
      "dedendum_coeff": null,
      "face_width_mm": null,
      "material": ""
    }
  ],
  "meshes": [
    {
      "a": "barrel",
      "b": "center_pinion",
      "center_distance_mm": null
    },
    {
      "a": "center_wheel",
      "b": "third_pinion",
      "center_distance_mm": null
    },
    {
      "a": "third_wheel",
      "b": "fourth_pinion",
      "center_distance_mm": null
    },
    {
      "a": "fourth_wheel",
      "b": "escape_pinion",
      "center_distance_mm": null
    }
  ],
  "spring": null,
  "escapement": {
    "kind": "lever",
    "escape_wheel_teeth": {
      "value": 15,
      "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
      "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
      "confidence": 0.95,
      "quote": "The escape wheel in most watches contains 15 teeth and delivers twice as many impulses to the balance, since each tooth delivers two impulses, first to the receiving pallet and later to the discharging pallet. ... CTF2E/tfe = number of beats per hour. Substituting the numerical values we have: 80X75X80X2X15 / 10X10X8 = 18,000 beats per hour.",
      "locator": "p.18, 'Calculating the number of beats'"
    },
    "beats_per_hour": {
      "value": 18000,
      "source": "Kelly, Harold C. — A Practical Course in Horology (1944)",
      "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
      "confidence": 0.95,
      "quote": "The escape wheel in most watches contains 15 teeth and delivers twice as many impulses to the balance, since each tooth delivers two impulses, first to the receiving pallet and later to the discharging pallet. ... CTF2E/tfe = number of beats per hour. Substituting the numerical values we have: 80X75X80X2X15 / 10X10X8 = 18,000 beats per hour.",
      "locator": "p.18, 'Calculating the number of beats'"
    },
    "lift_angle_deg": null,
    "drop_angle_deg": null
  },
  "sources": [
    {
      "name": "Kelly, Harold C. — A Practical Course in Horology (1944)",
      "license": "public scan, freely accessible (archive.org + survivorlibrary.com)",
      "url": "https://archive.org/stream/practicalcoursei00kellrich/practicalcoursei00kellrich_djvu.txt",
      "locator": "pp. 16-18 (going train), Part I Ch. 3 opening (escapement type)"
    }
  ],
  "notes": "The standard 18,000-beats/hour lever-escapement watch train Kelly presents as \"a modern train\" (p.17). No module/pitch/center-distance is cited for this specific example — a real gap (see module docstring), not filled here. Escapement kind 'lever' is a documented cross-reference, not an assumption: the going-train passage describes impulses delivered \"first to the receiving pallet and later to the discharging pallet\" (p.18), and Ch. 3 (opening quoted above) identifies that two-pallet terminology as the lever escapement, \"the superiority of the lever escapement over all other types for portable timepieces.\""
}
```
