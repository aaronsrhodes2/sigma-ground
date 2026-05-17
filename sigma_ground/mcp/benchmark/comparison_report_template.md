# sigma-ground physics MCP — 3-way benchmark report

**Date**: TODO  
**Corpus**: 150 questions, 14 physics-major domains  
**Systems tested**: sigma_ground+qwen2.5:7b, Wolfram Alpha free tier, Gemini 2.5 Pro

---

## Methodology

Each of the 150 corpus questions was posed verbatim (natural English, no formula-style prompting) to all three systems. Numerical answers were extracted with light regex parsing and compared against textbook ground truths within stated tolerance bands. Tolerance is per-question (typically 1-5%) and reflects realistic measurement precision rather than arbitrary cutoffs.

System configurations:
- **sigma_ground+Qwen**: `qwen2.5:7b` via Ollama on RTX 4070 Ti 16GB, talking to the sigma-ground MCP server over stdio. System prompt enforces tool-use-or-flag discipline. 94 PRIMARY tools available.
- **Wolfram Alpha**: Free-tier `wolframalpha` Python client, Full Results API, primary "Result" pod extracted.
- **Gemini 2.5 Pro**: `google-generativeai` SDK, default sampling parameters, no tool calls (bare LLM).

---

## Headline numbers

| System | Overall accuracy | Latency (median) | Total cost |
|---|---:|---:|---:|
| sigma_ground + Qwen 7b | TODO% | TODO s | $0 (local) |
| Wolfram Alpha (free) | TODO% | TODO s | $0 (free tier) |
| Gemini 2.5 Pro | TODO% | TODO s | $TODO |

## Per-domain accuracy

| Domain | sigma_ground | Wolfram | Gemini |
|---|---:|---:|---:|
| classical_mechanics_intro    | TODO | TODO | TODO |
| electromagnetism_intro       | TODO | TODO | TODO |
| waves_optics                 | TODO | TODO | TODO |
| thermodynamics_statmech      | TODO | TODO | TODO |
| modern_physics               | TODO | TODO | TODO |
| quantum_mechanics            | TODO | TODO | TODO |
| classical_mechanics_advanced | TODO | TODO | TODO |
| electrodynamics_advanced     | TODO | TODO | TODO |
| general_relativity           | TODO | TODO | TODO |
| cosmology                    | TODO | TODO | TODO |
| astrophysics                 | TODO | TODO | TODO |
| atomic_molecular             | TODO | TODO | TODO |
| nuclear_physics              | TODO | TODO | TODO |
| mathematical_methods         | TODO | TODO | TODO |

---

## Failure analysis

### sigma_ground failures
Questions where the MCP+Qwen pipeline got it wrong, grouped by cause:

- **Tool-call discipline failures**: Qwen answered from memory instead of calling a tool. Count: TODO. Fix: tighter system prompt.
- **Wrong tool selected**: Qwen called a tool but the wrong one. Count: TODO. Fix: improve tool descriptions in manifest.
- **Library gap (fitted)**: Qwen correctly flagged "Fitted due to incompetence". Count: TODO. Fix: add the missing library wrapper.
- **Arithmetic on tool output**: Qwen got the tool result but mis-extracted the number. Count: TODO. Fix: prompt to use `convert_units` instead of mental math.

### Wolfram failures
- Questions Wolfram couldn't parse / returned empty: TODO
- Questions where Wolfram's interpretation was different from intended: TODO

### Gemini failures
- Hallucinated numeric values: TODO
- Used rounded constants: TODO
- Got the right answer but failed to format with "ANSWER:" prefix: TODO

---

## "Fitted due to incompetence" inventory (sigma_ground)

The questions where sigma_ground correctly flagged that the library lacks the required physics. These become the **library development queue**:

| Question | Missing physics | Priority |
|---|---|---|
| TODO | TODO | high/med/low |
| ... | ... | ... |

Top missing modules (ranked by question-count impact):
1. TODO
2. TODO
3. TODO

---

## System prompt improvements (based on observations)

Specific edits to `SYSTEM_INSTRUCTIONS` in `sigma_ground/mcp/server.py`:

1. TODO
2. TODO
3. TODO

---

## Library gap roadmap

Beyond "fitted due to incompetence" tags, additional improvements identified:

| Module | Estimated effort | Question count this would unlock |
|---|---|---:|
| TODO | TODO | TODO |

---

## Verdict by user type

| User type | Best tool | Why |
|---|---|---|
| Student doing homework | TODO | TODO |
| Researcher | TODO | TODO |
| Engineer | TODO | TODO |
| General curiosity | TODO | TODO |
| SSBM-curious | sigma_ground (only) | only one with the framework |

---

## Methodological caveats

- The 150-question corpus is **finite and curated**, not random sampling of all physics questions.
- Wolfram free-tier rate limits forced multi-day runs; one query failure can be retried but state lost.
- Gemini's default sampling has temperature > 0; rerunning gives slightly different answers.
- Numerical extraction via regex fails on some valid free-form answers; manually-corrected scores tracked separately.

---

## Raw data

Results files (gitignored, large):
- `results/sigma_ground_run.json`
- `results/wolfram_run.json`
- `results/gemini_run.json`
- `results/sigma_ground_scored.json`
- `results/wolfram_scored.json`
- `results/gemini_scored.json`
