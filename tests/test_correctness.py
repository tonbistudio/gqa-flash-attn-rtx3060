"""Correctness tests against the naive PyTorch reference.

Run: pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from gqa_kernels import gqa_flash_attn
from tests.reference import gqa_reference


DEVICE = 'cuda'

# Llama 3.1 8B shape.
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

# (max_abs_err, mean_abs_err) tolerances, per dtype.
TOLERANCES = {
    torch.bfloat16: (1e-2, 1e-3),
    torch.float16: (1e-3, 1e-4),
}


def _make(seq_q, seq_kv, dtype):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    shape_q = (1, NUM_HEADS, seq_q, HEAD_DIM)
    shape_kv = (1, NUM_KV_HEADS, seq_kv, HEAD_DIM)
    q = torch.randn(*shape_q, dtype=dtype, device=DEVICE, generator=g) * 0.1
    k = torch.randn(*shape_kv, dtype=dtype, device=DEVICE, generator=g) * 0.1
    v = torch.randn(*shape_kv, dtype=dtype, device=DEVICE, generator=g) * 0.1
    return q, k, v


def _assert_close(out, ref, dtype):
    diff = (out.float() - ref.float()).abs()
    mx, mn = diff.max().item(), diff.mean().item()
    max_tol, mean_tol = TOLERANCES[dtype]
    assert mx < max_tol, f"max abs err {mx:.2e} >= {max_tol:.2e}"
    assert mn < mean_tol, f"mean abs err {mn:.2e} >= {mean_tol:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seq_len", [512, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prefill(seq_len, dtype):
    """Prefill: seq_q == seq_kv, causal."""
    q, k, v = _make(seq_len, seq_len, dtype)
    out = gqa_flash_attn(q, k, v, causal=True)
    ref = gqa_reference(q, k, v, causal=True)
    _assert_close(out, ref, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seq_kv", [512, 2048, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode(seq_kv, dtype):
    """Decode: seq_q=1, attends to KV cache of length seq_kv."""
    q, k, v = _make(1, seq_kv, dtype)
    out = gqa_flash_attn(q, k, v, causal=True)
    ref = gqa_reference(q, k, v, causal=True)
    _assert_close(out, ref, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seq_q,seq_kv", [(16, 1024), (64, 2048), (256, 4096)])
def test_chunked_prefill(seq_q, seq_kv):
    """Chunked prefill: seq_q < seq_kv, causal mask aligned bottom-right."""
    q, k, v = _make(seq_q, seq_kv, torch.bfloat16)
    out = gqa_flash_attn(q, k, v, causal=True)
    ref = gqa_reference(q, k, v, causal=True)
    _assert_close(out, ref, torch.bfloat16)
