"""QLoRA fine-tune of Qwen3-8B into Mentat's router, and its export to Ollama.

The router reads a question and replies with ONE JSON plan (a tool call, a
request_clarification call, or a decline); Mentat runs the plan. The data is
router_dataset.py's chat JSONL (system = ROUTER_SYSTEM_PROMPT, user = question,
assistant = the plan). See sigma_ground/mcp/benchmark/README.md, "Router mode"
and "Router fine-tune dataset".

Runs in its own CUDA environment (torch, transformers, peft, bitsandbytes,
accelerate; Unsloth when installed), not the repo's. On Windows that is WSL2:

    python tools/train_router.py train --data local-cache/router_dataset \
        --base local-cache/router/Qwen3-8B --out local-cache/router/run1 --max-steps 50   # pilot
    python tools/train_router.py train --data local-cache/router_dataset \
        --base local-cache/router/Qwen3-8B --out local-cache/router/run1                  # full
    python tools/train_router.py merge --base local-cache/router/Qwen3-8B \
        --adapter local-cache/router/run1/adapter --out local-cache/router/merged      # CPU
    python tools/train_router.py modelfile --gguf ./mentat-router-Q4_K_M.gguf --out Modelfile

tools/run_router_study.sh runs the whole study end to end (dataset, pilot,
training, merge, llama.cpp GGUF + Q4_K_M, ollama create, evaluation).

How a record becomes a training example
  prompt     = the chat template over (system, user) with the generation
               prompt and enable_thinking=False, i.e. ending in
               "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
  completion = the plan JSON + "<|im_end|>"
  The two are tokenized separately and concatenated, which is what the model
  sees at inference (Ollama tokenizes the rendered prompt, then generates).
  Loss is on completion tokens only. Before training, the script checks that
  the full template rendering of the record equals prompt + completion + "\\n",
  so the non-thinking template is the one being trained. `meta` is ignored.

Fixed recipe (the study's spec): LoRA r=16, alpha=32, dropout 0, on
q,k,v,o,gate,up,down; 4-bit NF4 base; lr 2e-4, cosine, 3% warmup, 2 epochs,
effective batch 16, seed 0; max length from the measured data, capped at 1024.
train.jsonl trains, val.jsonl evaluates (the split is group-aware already).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Fewer, larger free blocks: less fragmentation after an evaluation pass.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_LEN_CAP = 1024
IM_END = "<|im_end|>"

# What the Ollama build must render, byte for byte, for a (system, user) chat:
# the Qwen3 non-thinking prompt, the same text the HF template gives with
# add_generation_prompt=True, enable_thinking=False.
OLLAMA_TEMPLATE = (
    '{{- range $i, $_ := .Messages }}'
    '{{- $last := eq (len (slice $.Messages $i)) 1 -}}'
    '{{- if eq .Role "system" }}<|im_start|>system\n{{ .Content }}<|im_end|>\n'
    '{{ else if eq .Role "user" }}<|im_start|>user\n{{ .Content }}<|im_end|>\n'
    '{{ else if eq .Role "assistant" }}<|im_start|>assistant\n{{ .Content }}'
    '{{ if not $last }}<|im_end|>\n{{ end }}'
    '{{ end }}'
    '{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant\n<think>\n\n</think>\n\n{{ end }}'
    '{{- end }}'
)


# ── data ─────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render(tokenizer, messages: list[dict], **kw) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         enable_thinking=False, **kw)


def encode(tokenizer, rec: dict) -> dict:
    """prompt ids + completion ids, labels masked to the completion."""
    msgs = rec["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"], msgs
    prompt = render(tokenizer, msgs[:2], add_generation_prompt=True)
    completion = msgs[2]["content"] + IM_END
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    return {"input_ids": p_ids + c_ids,
            "labels": [-100] * len(p_ids) + c_ids,
            "prompt_text": prompt, "completion_text": completion}


def check_template(tokenizer, records: list[dict], n: int = 20) -> None:
    """The trained text must be what the full chat template renders."""
    for rec in records[:n]:
        full = render(tokenizer, rec["messages"])
        enc = encode(tokenizer, rec)
        want = enc["prompt_text"] + enc["completion_text"] + "\n"
        if full != want:
            raise SystemExit("chat template mismatch:\n--- template\n"
                             f"{full!r}\n--- trained\n{want!r}")
        if not enc["prompt_text"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
            raise SystemExit(f"prompt is not the non-thinking form: {enc['prompt_text'][-80:]!r}")


def build_split(tokenizer, records: list[dict], max_len: int) -> tuple[list[dict], int]:
    out, over = [], 0
    for rec in records:
        enc = encode(tokenizer, rec)
        if len(enc["input_ids"]) > max_len:
            over += 1
            continue
        out.append({"input_ids": enc["input_ids"], "labels": enc["labels"]})
    return out, over


def pct(xs: list[int], q: float) -> int:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


# ── model ────────────────────────────────────────────────────────────────

def load_model(base: str, max_len: int, use_unsloth: bool, seed: int, cpu: bool = False):
    if cpu:     # --cpu-smoke: the same LoRA on an unquantized model, to test the loop
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        return model, AutoTokenizer.from_pretrained(base), "cpu-smoke"
    if use_unsloth:
        from unsloth import FastLanguageModel        # must precede transformers/peft
        model, tok = FastLanguageModel.from_pretrained(
            model_name=base, max_seq_length=max_len, load_in_4bit=True, dtype=None)
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=LORA_TARGETS, use_gradient_checkpointing="unsloth",
            random_state=seed)
        return model, tok, "unsloth"
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, device_map={"": 0}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    return model, tok, "transformers+peft"


# ── train ────────────────────────────────────────────────────────────────

def cmd_train(a) -> int:
    use_unsloth = not (a.no_unsloth or a.cpu_smoke)
    if use_unsloth:
        try:
            import unsloth  # noqa: F401  (import first so its patches apply)
        except Exception as e:
            print(f"unsloth unavailable ({e}); using transformers+peft")
            use_unsloth = False
    import torch
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_recs = read_jsonl(Path(a.data) / "train.jsonl")
    val_recs = read_jsonl(Path(a.data) / "val.jsonl")

    # Everything is tokenized with the base model's own tokenizer, the one
    # check_template validates (a training library may patch its copy).
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    check_template(tok, train_recs)
    lens = [len(encode(tok, r)["input_ids"]) for r in train_recs + val_recs]
    max_len = min(MAX_LEN_CAP, int(math.ceil(max(lens) / 8) * 8))
    print(f"records: train {len(train_recs)}, val {len(val_recs)}; tokens per example "
          f"p50 {pct(lens, .5)}, p95 {pct(lens, .95)}, max {max(lens)} -> max_len {max_len}")

    model, _, backend = load_model(a.base, max_len, use_unsloth, a.seed, cpu=a.cpu_smoke)
    gpu = not a.cpu_smoke
    train_ds, over_tr = build_split(tok, train_recs, max_len)
    val_ds, over_va = build_split(tok, val_recs, max_len)
    if over_tr or over_va:
        print(f"dropped over max_len: train {over_tr}, val {over_va}")

    steps_per_epoch = math.ceil(math.ceil(len(train_ds) / a.batch) / a.grad_accum)
    total = a.max_steps if a.max_steps > 0 else steps_per_epoch * a.epochs
    full_total = steps_per_epoch * a.epochs
    warmup = max(1, round(0.03 * full_total))
    eval_steps = a.eval_steps or max(25, full_total // 10)
    print(f"backend {backend}; effective batch {a.batch * a.grad_accum}; "
          f"{steps_per_epoch} steps/epoch; training {total} of {full_total} steps; "
          f"warmup {warmup}; eval every {eval_steps}")

    class Clock(TrainerCallback):
        def __init__(self):
            self.t0 = None
            self.step_times = []

        def on_step_begin(self, args, state, control, **kw):
            if self.t0 is None:
                self.t0 = time.time()

        def on_step_end(self, args, state, control, **kw):
            self.step_times.append(time.time())

    class FreeCache(TrainerCallback):
        """Unsloth's fused cross-entropy asks the driver how much memory is free.
        After an evaluation pass PyTorch still caches those blocks, the driver
        reports almost nothing free, and the first training step afterwards
        raises 'No or negligible GPU memory available' (seen in the pilot, at
        step 29, right after the step-25 evaluation). Hand the cache back."""
        def on_evaluate(self, args, state, control, **kw):
            gc.collect()
            if gpu:
                torch.cuda.empty_cache()

    clock = Clock()
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=max(1, a.batch // 2),
        prediction_loss_only=True,       # eval needs the loss, not 152k-wide logits per token
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=warmup,
        num_train_epochs=a.epochs, max_steps=a.max_steps if a.max_steps > 0 else -1,
        weight_decay=0.0, optim="adamw_8bit" if gpu else "adamw_torch", bf16=gpu,
        use_cpu=not gpu,
        logging_steps=a.logging_steps, eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=2,
        seed=a.seed, data_seed=a.seed, report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tok.pad_token_id), callbacks=[clock, FreeCache()])
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    final_eval = trainer.evaluate()

    # steady-state speed: skip the first 5 steps (compile/warm-up)
    st = clock.step_times
    steady = (len(st) - 6) / (st[-1] - st[5]) if len(st) > 10 else len(st) / wall
    report = {
        "backend": backend, "base": a.base, "data": str(a.data),
        "train_examples": len(train_ds), "val_examples": len(val_ds),
        "dropped_over_max_len": {"train": over_tr, "val": over_va},
        "tokens_per_example": {"p50": pct(lens, .5), "p95": pct(lens, .95), "max": max(lens)},
        "max_len": max_len, "effective_batch": a.batch * a.grad_accum,
        "per_device_batch": a.batch, "grad_accum": a.grad_accum,
        "steps_trained": len(st), "steps_full_run": full_total, "epochs": a.epochs,
        "lr": a.lr, "warmup_steps": warmup, "seed": a.seed,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "targets": LORA_TARGETS},
        "wall_s": round(wall, 1), "steady_steps_per_s": round(steady, 4),
        "projected_full_run_s": round(full_total / steady, 0),
        "vram_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if gpu else None,
        "vram_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2) if gpu else None,
        "final_eval": final_eval,
        "versions": {m: __import__(m).__version__ for m in
                     ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
    }
    if backend == "unsloth":
        import unsloth
        report["versions"]["unsloth"] = unsloth.__version__
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (out / "train_report.json").write_text(json.dumps(report, indent=2, default=str))
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    print(json.dumps(report, indent=2, default=str))
    return 0


# ── export ───────────────────────────────────────────────────────────────

def write_modelfile(gguf: str, path: Path) -> None:
    path.write_text(
        f"FROM {gguf}\n"
        f'TEMPLATE """{OLLAMA_TEMPLATE}"""\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_ctx 4096\n"
        f'PARAMETER stop "{IM_END}"\n'
        'PARAMETER stop "<|endoftext|>"\n', encoding="utf-8")


def cmd_merge(a) -> int:
    """Fold the LoRA into the 16-bit base, one safetensors shard at a time.

    W' = W + (alpha / r) * B @ A, computed in float32 and stored in the base
    dtype. Shard by shard so an 8B merge fits in ~8 GB of RAM (WSL's default
    is half the host's); the output is a plain HF checkpoint for llama.cpp's
    convert_hf_to_gguf.py. CPU only."""
    import shutil
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    base, adapter, out = Path(a.base), Path(a.adapter), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    if cfg.get("use_dora") or cfg.get("modules_to_save"):
        raise SystemExit("merge handles plain LoRA only (no DoRA, no modules_to_save)")
    lora = load_file(str(adapter / "adapter_model.safetensors"))
    pairs = {}
    for k in lora:
        if ".lora_A." in k:
            mod = k.split(".lora_A.")[0].removeprefix("base_model.model.")
            pairs[mod + ".weight"] = (lora[k], lora[k.replace(".lora_A.", ".lora_B.")])
    n_pairs = len(pairs)
    index_path = base / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
              if index_path.exists() else ["model.safetensors"])
    t0 = time.time()
    for shard in shards:
        tensors = {}
        with safe_open(str(base / shard), framework="pt") as f:
            for name in f.keys():
                w = f.get_tensor(name)
                if name in pairs:
                    A, B = pairs.pop(name)
                    w = (w.float() + scale * (B.float() @ A.float())).to(w.dtype)
                tensors[name] = w.contiguous()
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print(f"  {shard}: {len(tensors)} tensors ({time.time() - t0:.0f}s)", flush=True)
    if pairs:
        raise SystemExit(f"{len(pairs)} LoRA targets not found in the base: {sorted(pairs)[:5]}")
    for p in base.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors"):
            shutil.copy2(p, out / p.name)
    pin_non_thinking(out)
    print(f"merged {n_pairs} LoRA pairs at scale {scale} into {out} in {time.time() - t0:.0f}s")
    return 0


PIN = "{%- set enable_thinking = false %}"


def pin_non_thinking(model_dir: Path) -> None:
    """Make the checkpoint's own chat template always render the non-thinking
    prompt the router was trained on. convert_hf_to_gguf embeds this template,
    and Ollama (0.35) renders chats with the GGUF's template, not the
    Modelfile's, so without the pin a client that leaves out think=false
    would get the thinking-mode prompt."""
    hits = 0
    cfg_path = model_dir / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(cfg.get("chat_template"), str) and not cfg["chat_template"].startswith(PIN):
            cfg["chat_template"] = PIN + cfg["chat_template"]
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            hits += 1
    jinja = model_dir / "chat_template.jinja"
    if jinja.exists() and not jinja.read_text(encoding="utf-8").startswith(PIN):
        jinja.write_text(PIN + jinja.read_text(encoding="utf-8"), encoding="utf-8")
        hits += 1
    if not hits:
        raise SystemExit(f"no chat template found to pin in {model_dir}")
    print(f"chat template pinned to non-thinking ({hits} file(s))")


def cmd_modelfile(a) -> int:
    write_modelfile(a.gguf, Path(a.out))   # verbatim: Ollama needs the ./ prefix
    print(Path(a.out).read_text())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="dir with train.jsonl and val.jsonl")
    t.add_argument("--base", default="Qwen/Qwen3-8B", help="HF id or local dir")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=float, default=2)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--batch", type=int, default=8, help="per-device batch")
    t.add_argument("--grad-accum", type=int, default=2, help="batch x accum = 16")
    t.add_argument("--max-steps", type=int, default=0, help=">0: pilot of this many steps")
    t.add_argument("--eval-steps", type=int, default=0, help="0: a tenth of the run")
    t.add_argument("--logging-steps", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--no-unsloth", action="store_true")
    t.add_argument("--cpu-smoke", action="store_true",
                   help="test the training loop on CPU (unquantized; use a tiny base)")
    g = sub.add_parser("merge", help="fold the adapter into the 16-bit base (CPU)")
    g.add_argument("--base", required=True, help="local HF dir of the base model")
    g.add_argument("--adapter", required=True, help="the adapter dir train wrote")
    g.add_argument("--out", required=True)
    m = sub.add_parser("modelfile", help="write a Modelfile for a GGUF")
    m.add_argument("--gguf", required=True, help="path as the Modelfile should name it")
    m.add_argument("--out", required=True)
    a = p.parse_args()
    return {"train": cmd_train, "merge": cmd_merge, "modelfile": cmd_modelfile}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
