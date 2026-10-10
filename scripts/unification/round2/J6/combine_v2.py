"""J6 v2: merge the two mutation-control runs into jacobson_results_v2.json under 'mutation_controls'.
Controls only: the pre-registered verdict is read from the unmutated run."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "jacobson_results_v2.json")
base = json.load(open(P, encoding="utf-8"))
muts = {m: json.load(open(os.path.join(HERE, f"jacobson_results_v2_mut_{m}.json"), encoding="utf-8"))
        for m in ("unruh_x2", "chi_sign")}
DESC = {"unruh_x2": "A5 broken: T = hbar kappa/pi (Unruh temperature doubled) in S3, S3b, S3c, S8, S9, S10",
        "chi_sign": "A6 broken: chi^a = +kappa lambda k^a (boost field sign flipped) in S2, S3, S3b, S3c, S8, S9"}
steps = {}
for s in base["steps"]:
    row = {"baseline": s["status"], "tag": s["tag"]}
    for m, d in muts.items():
        ms = next(x for x in d["steps"] if x["id"] == s["id"])
        row[m] = ms["status"]; row[m + "_residual"] = ms["residual"][:300]
    steps[s["id"]] = row
numbers = {}
for k in ("N1", "N2", "N3", "N4", "N5"):
    key = "rel" if k == "N1" else "max_rel"
    numbers[k] = {"baseline": base["mentat_numbers"][k][key],
                  **{m: d["mentat_numbers"][k][key] for m, d in muts.items()},
                  "identical": all(d["mentat_numbers"][k][key] == base["mentat_numbers"][k][key] for d in muts.values())}
uses_A5_A6 = [i for i, r in steps.items() if r["unruh_x2"] == "FAILS" or r["chi_sign"] == "FAILS"]
mc = {"description": DESC, "steps": steps, "numbers": numbers,
      "steps_failing_under_at_least_one_mutation": uses_A5_A6,
      "steps_closing_under_both": [i for i in steps if i not in uses_A5_A6],
      "n_close": {"baseline": base["n_steps_close"], **{m: d["n_steps_close"] for m, d in muts.items()}},
      "reading": ("Steps that use A5 or A6 fail when either input is broken (S3 and its descendants S3c, S7, S8, S9, "
                  "S10; S2 under the chi mutation). S1, S1b, S4, S5, S6 do not use A5/A6 and close under both; their "
                  "own live checks are the closed-form ODE, the S4 planar negative control and the S6 alpha solve. "
                  "S3b closes under both because theta(p) = 0 is forced for any Unruh factor and either chi sign. "
                  "N1-N5 are identical to the last digit under both mutations: they test Mentat's code paths, not "
                  "the derivation.")}
base["mutation_controls"] = mc
json.dump(base, open(P, "w", encoding="utf-8"), indent=1, default=str)
for i, r in steps.items():
    print(f"{i:5s} {r['baseline']:7s} {r['unruh_x2']:7s} {r['chi_sign']:7s} {r['tag']}")
for k, v in numbers.items():
    print(k, v)
print(mc["n_close"])
