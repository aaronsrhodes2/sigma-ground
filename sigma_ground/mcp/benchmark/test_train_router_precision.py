"""Tests for the precision switch in tools/train_router.py: nf4 stays the study's recipe; bf16 never needs bitsandbytes (AMD/ROCm, the Bosgame M5)."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
_spec = importlib.util.spec_from_file_location("train_router", REPO / "tools" / "train_router.py")
T = importlib.util.module_from_spec(_spec)
sys.modules["train_router"] = T
_spec.loader.exec_module(T)


def test_nf4_on_a_gpu_is_the_unchanged_study_recipe():
    c = T.training_config("nf4", gpu=True)
    assert c == {"load_in_4bit": True, "optim": "adamw_8bit", "needs_bitsandbytes": True}


def test_bf16_needs_no_quantization_library():
    c = T.training_config("bf16", gpu=True)
    assert c == {"load_in_4bit": False, "optim": "adamw_torch", "needs_bitsandbytes": False}


@pytest.mark.parametrize("precision", T.PRECISIONS)
def test_the_cpu_smoke_never_uses_bitsandbytes(precision):
    c = T.training_config(precision, gpu=False)
    assert c["optim"] == "adamw_torch" and not c["load_in_4bit"] and not c["needs_bitsandbytes"]


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError):
        T.training_config("fp4", gpu=True)


def test_the_command_line_default_is_still_nf4_and_bf16_is_accepted():
    p = T.build_parser()
    base = ["train", "--data", "d", "--out", "o"]
    assert p.parse_args(base).precision == "nf4"
    assert p.parse_args(base + ["--precision", "bf16"]).precision == "bf16"
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--precision", "int8"])


def test_bitsandbytes_is_only_imported_on_the_quantized_branch():
    src = inspect.getsource(T.load_model)
    gate = src.index('if cfg["load_in_4bit"]:')
    assert src.index("BitsAndBytesConfig") > gate                       # the only place the quantization config is built
    assert src.index("prepare_model_for_kbit_training") > gate          # the 4-bit preparation step too
    assert "import bitsandbytes" not in inspect.getsource(T)             # and nothing imports it at module level


def test_a_missing_module_is_reported_not_raised():
    assert T._version("no_such_module_anywhere") == "not installed"
    assert T._version("pytest") != "not installed"
