"""Public entrypoint: choose single-pass vs split-KV path per workload."""
import torch
import triton

from .gqa_fwd import gqa_fwd_kernel
from .gqa_split import split_kv_forward

__all__ = ["gqa_flash_attn"]

SM_COUNT = 28  # RTX 3060
LOG2E = 1.4426950408889634


def gqa_flash_attn(q, k, v, causal=True):
    """GQA flash attention.

    Args:
        q: (batch, num_heads,    seq_q,  head_dim)  — fp16 or bf16
        k: (batch, num_kv_heads, seq_kv, head_dim)
        v: (batch, num_kv_heads, seq_kv, head_dim)
        causal: if True, q position i (in K coordinates: i + seq_kv - seq_q)
                attends to K positions [0, q_pos]. Works for prefill
                (seq_q == seq_kv) and decode / chunked-prefill (seq_q < seq_kv).

    Returns:
        (batch, num_heads, seq_q, head_dim), same dtype as q.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, num_kv_heads, seq_kv, _ = k.shape
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    group_size = num_heads // num_kv_heads
    assert head_dim in (64, 128), f"unsupported head_dim {head_dim}"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    # Split-KV when the single-pass grid would underfill the SMs
    # (short seq_q, long seq_kv).
    if seq_q <= 16 and seq_kv >= 1024 and num_kv_heads * batch < SM_COUNT:
        target = 4 * SM_COUNT
        splits = 1
        while splits * num_kv_heads * batch < target and splits < 32:
            splits *= 2
        while splits > 1 and seq_kv // splits < 64:
            splits //= 2
        if splits > 1:
            return split_kv_forward(q, k, v, causal, splits)

    o = torch.empty_like(q)
    scale_log2 = (head_dim ** -0.5) * LOG2E
    grid = lambda META: (num_kv_heads, triton.cdiv(seq_q, META['BLOCK_M']), batch)
    gqa_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_q, seq_kv,
        scale_log2,
        IS_CAUSAL=causal,
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
    )
    return o
