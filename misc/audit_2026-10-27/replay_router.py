"""Replay Mentat's router stage (SkippyTel/mentat/router_stage.py) against the skippy-core shim through the tunnel, with the exception visible.

Same code as deployed: app.py + router_stage.py from SkippyTel, sigma-ground at this checkout (= ef2d4c0 = the container's ref).
"""
import asyncio
import json
import os
import sys
import traceback

os.environ.update({
    "MENTAT_USE_LLM": "1",
    "OLLAMA_URL": "http://127.0.0.1:17434",
    "MENTAT_ROUTER_MODEL": "mentat-router-v31-4b",
    "MENTAT_INSTANCE": "home",
    "SIGMA_DATASETS": "D:/datasets",
})
sys.path.insert(0, "D:/Aaron/development/sigma-ground")
sys.path.insert(0, "D:/Aaron/development/SkippyTel/mentat")

import app as A  # noqa: E402
import router_stage as RS  # noqa: E402

questions = sys.argv[1:] or ["differentiate x^2 sin x"]
for q in questions:
    print("Q:", q)
    try:
        rec = asyncio.run(RS._run(q, A.server, A.TOOLS, A.OLLAMA_URL, A._router_guard))
        keep = {k: rec.get(k) for k in ("router_plan", "router_error", "router_decline", "router_clarify", "extracted_value", "tool_calls")}
        print(json.dumps(keep, indent=1, default=str)[:2500])
    except Exception:
        traceback.print_exc()
    print("answer():", RS.answer(q, server=A.server, tools=A.TOOLS, ollama_url=A.OLLAMA_URL, guard=A._router_guard) is not None)
    print("=" * 60)
