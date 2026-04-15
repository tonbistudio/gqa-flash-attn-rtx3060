"""Naive PyTorch GQA attention. Used as the correctness oracle."""
import torch
import torch.nn.functional as F


def gqa_reference(q, k, v, causal=True):
    """
    q: (batch, num_heads,    seq_q,  head_dim)
    k: (batch, num_kv_heads, seq_kv, head_dim)
    v: (batch, num_kv_heads, seq_kv, head_dim)

    When causal and seq_q < seq_kv, the mask is aligned bottom-right: q at
    row i attends to K positions [0, i + (seq_kv - seq_q)]. Handles both
    prefill (seq_q == seq_kv) and decode-with-cache (seq_q < seq_kv).
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, num_kv_heads, seq_kv, _ = k.shape
    assert num_heads % num_kv_heads == 0
    group_size = num_heads // num_kv_heads
    scale = head_dim ** -0.5

    k_exp = k.repeat_interleave(group_size, dim=1)
    v_exp = v.repeat_interleave(group_size, dim=1)

    attn = torch.matmul(q, k_exp.transpose(-2, -1)) * scale

    if causal:
        q_pos = torch.arange(seq_q, device=q.device) + (seq_kv - seq_q)
        k_pos = torch.arange(seq_kv, device=q.device)
        mask = q_pos[:, None] < k_pos[None, :]
        attn = attn.masked_fill(mask, float('-inf'))

    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v_exp)
