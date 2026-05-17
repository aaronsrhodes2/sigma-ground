# sigma-ground physics MCP — 3-way benchmark

A reproducible head-to-head test of three physics-Q&A systems:

| System | What it is | Strengths | Weaknesses |
|---|---|---|---|
| **sigma_ground + Qwen** | Local Qwen 2.5:7b orchestrating sigma-ground MCP tools | Transparent provenance, free, local, "fitted due to incompetence" honesty | Limited library breadth |
| **Wolfram Alpha** | Mathematica-backed computational engine via free-tier API | Massive topical breadth, symbolic-numeric integration, plots | No transparent provenance, free tier rate-limited |
| **Gemini 2.5 Pro** | Google's flagship LLM via paid API | Excellent reasoning, ~$1-5 for all 150 Qs, multimodal | No tools by default; hallucinations possible |

## Corpus

150 questions spanning a 4-year physics-major curriculum:

| Domain | Year | Count |
|---|---|---:|
| classical_mechanics_intro    | 1   | 15 |
| electromagnetism_intro       | 1-2 | 15 |
| waves_optics                 | 2   | 12 |
| thermodynamics_statmech      | 2-3 | 12 |
| modern_physics               | 2-3 | 12 |
| quantum_mechanics            | 3   | 12 |
| classical_mechanics_advanced | 3   | 10 |
| electrodynamics_advanced     | 3-4 | 10 |
| general_relativity           | 4   | 10 |
| cosmology                    | 4   |  8 |
| astrophysics                 | 4   | 12 |
| atomic_molecular             | 3-4 |  8 |
| nuclear_physics              | 3-4 |  7 |
| mathematical_methods         | 1-4 |  7 |

All questions phrased in natural English; numerical answers with tolerance bands; ground truth cited from textbooks (Halliday/Griffiths/Schutz/Hecht/Carroll-Ostlie) and standard databases (NIST, CODATA, IUPAC, NASA, Gaia DR3).

## Run all three systems

### 1. Install benchmark deps

```bash
cd D:\Aaron\development\sigma-ground
pip install -e ".[mcp,benchmark]"
```

### 2. Regenerate corpus (if needed)

```bash
python -m sigma_ground.mcp.benchmark.corpus
# Writes questions.json + ground_truth.json
```

### 3. Run each backend (parallel-safe)

**sigma_ground + Qwen** (~75 min runtime on 4070 Ti 16 GB):
```bash
ollama serve &           # if not running
ollama pull qwen2.5:7b   # if not already
python -m sigma_ground.mcp.benchmark.run_sigma_ground \
    --model qwen2.5:7b \
    --output sigma_ground/mcp/benchmark/results/sigma_ground_run.json
```

**Wolfram Alpha** (~2 days, rate-limited free tier):
```bash
export WOLFRAM_ALPHA_APP_ID=<your free APP_ID from developer.wolframalpha.com>
python -m sigma_ground.mcp.benchmark.run_wolfram \
    --output sigma_ground/mcp/benchmark/results/wolfram_run.json \
    --pace-per-day 90 --pause-s 2
# Resumable -- run again the next day to continue
```

**Gemini 2.5 Pro** (~8 min, $1-5 total):
```bash
export GEMINI_API_KEY=<your key from aistudio.google.com>
python -m sigma_ground.mcp.benchmark.run_gemini \
    --model gemini-2.5-pro \
    --output sigma_ground/mcp/benchmark/results/gemini_run.json
```

All runners are **resumable** — they skip questions already in their output file. Safe to ctrl-C and restart.

### 4. Score all three

```bash
python -m sigma_ground.mcp.benchmark.score
# Reads results/*_run.json, writes results/*_scored.json
# Prints per-system + per-domain summary
```

### 5. Generate the comparison report

Comparison-report scaffolding is in `comparison_report_template.md`. Fill in numbers from the scored.json files. Output goes in `misc/mcp_benchmark_results_<DATE>.md`.

## Result file schemas

### `<system>_run.json`

```json
[
  {
    "id": "mech_intro_001",
    "system": "sigma_ground",
    "model": "qwen2.5:7b",
    "answer_text": "<full LLM/Wolfram output>",
    "extracted_value": 1.43,
    "extracted_units": "s",
    "tool_calls": [{"name": "free_fall_time", "args": {...}, "result_text": "..."}],
    "elapsed_s": 12.3,
    "tokens_in": 250, "tokens_out": 80, "cost_usd": 0.0011
  },
  ...
]
```

### `<system>_scored.json`

```json
{
  "summary": {
    "overall_accuracy_pct": 87.3,
    "overall_correct": 131,
    "overall_total": 150,
    "by_domain": {
      "classical_mechanics_intro": {"correct": 14, "total": 15, "pct": 93.3},
      ...
    }
  },
  "rows": [...]
}
```

## What the benchmark measures (and what it doesn't)

**Measures:**
- Numerical accuracy (within stated tolerance)
- Per-domain coverage gaps
- Latency per question
- Cost per question (paid APIs only)
- For sigma_ground: tool-call discipline, "fitted due to incompetence" rate

**Does NOT measure:**
- Conversational quality (subjective; do separate user studies)
- Visualizations / plots (Wolfram does these; the others don't)
- Multimodal (image-based questions)
- Robustness to adversarial phrasing
- Long-form derivations / proofs

The "fitted due to incompetence" rate from sigma_ground is itself a useful output — every flagged answer becomes a library development ticket.
