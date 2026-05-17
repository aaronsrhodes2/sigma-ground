"""Physics benchmark corpus + runners for the 3-way comparison.

Three runners exercise the same 150-question corpus:
  - run_sigma_ground.py: sigma-ground MCP + qwen2.5:7b via Ollama
  - run_wolfram.py:      Wolfram Alpha free-tier API
  - run_gemini.py:       Gemini 2.5 Pro via google-generativeai

Each runner saves results to results/<system>_run.json.
score.py compares against ground_truth.json and emits a report.
"""
