"""Split-KV GQA flash attention for decode and short-Q workloads.

At seq_q=1 the single-pass grid launches only num_kv_heads programs, leaving
most of the RTX 3060's 28 SMs idle. Split-KV partitions the seq_kv axis into
chunks, runs a first pass that writes per-chunk (m, l, acc_unnormalized)
partials, then a second pass combines them via log-sum-exp.

Result on decode@8K: 0.38x FA2 -> 2.95x FA2 (~8x speedup).
"""
import torch
import triton
import triton.language as tl


def _split_kv_configs():
    configs = []
    for BN in (32, 64, 128):
        for nw in (2, 4):
            for ns in (1, 2, 3):
                configs.append(
                    triton.Config({'BLOCK_N': BN}, num_warps=nw, num_stages=ns)
                )
    return configs


@triton.autotune(configs=_split_kv_configs(),
                 key=['seq_kv', 'HEAD_DIM', 'GROUP_SIZE', 'SPLITS'])
@triton.jit
def gqa_split_first_kernel(
    Q, K, V, Mp, Lp, Ap,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_pb, stride_ph, stride_pm, stride_ps,
    stride_ab, stride_ah, stride_am, stride_as, stride_ad,
    seq_q, seq_kv,
    scale,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """First pass: per-chunk partial attention. Groups packed into M axis."""
    pid_kv = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bs = tl.program_id(2)
    pid_s = pid_bs % SPLITS
    pid_b = pid_bs // SPLITS

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    chunk_size = tl.cdiv(seq_kv, SPLITS)
    chunk_start = pid_s * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, seq_kv)
    if IS_CAUSAL:
        max_q_pos = (pid_m + 1) * BLOCK_M + (seq_kv - seq_q) - 1
        chunk_end = tl.minimum(chunk_end, max_q_pos + 1)

    K_base = K + pid_b * stride_kb + pid_kv * stride_kh
    V_base = V + pid_b * stride_vb + pid_kv * stride_vh

    BLOCK_MG: tl.constexpr = BLOCK_M * GROUP_SIZE
    offs_mg = tl.arange(0, BLOCK_MG)
    g_idx = offs_mg // BLOCK_M
    m_idx = offs_mg % BLOCK_M
    row_global = pid_m * BLOCK_M + m_idx
    q_heads = pid_kv * GROUP_SIZE + g_idx
    row_valid = row_global < seq_q
    q_pos = row_global + (seq_kv - seq_q)

    q_ptrs = (Q + pid_b * stride_qb + q_heads[:, None] * stride_qh
              + row_global[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0)

    m_i = tl.full([BLOCK_MG], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_MG], dtype=tl.float32)
    acc = tl.zeros([BLOCK_MG, HEAD_DIM], dtype=tl.float32)

    for n_start in range(chunk_start, chunk_end, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask = n_offs < seq_kv
        k_ptrs = K_base + n_offs[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_base + n_offs[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(n_mask[None, :], qk, float('-inf'))
        if IS_CAUSAL:
            qk = tl.where(q_pos[:, None] >= n_offs[None, :], qk, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Scatter partials (unnormalized acc) per group member.
    mp_ptrs = (Mp + pid_b * stride_pb + q_heads * stride_ph
               + row_global * stride_pm + pid_s * stride_ps)
    lp_ptrs = (Lp + pid_b * stride_pb + q_heads * stride_ph
               + row_global * stride_pm + pid_s * stride_ps)
    ap_ptrs = (Ap + pid_b * stride_ab + q_heads[:, None] * stride_ah
               + row_global[:, None] * stride_am + pid_s * stride_as
               + offs_d[None, :] * stride_ad)
    tl.store(mp_ptrs, m_i, mask=row_valid)
    tl.store(lp_ptrs, l_i, mask=row_valid)
    tl.store(ap_ptrs, acc, mask=row_valid[:, None])


@triton.jit
def gqa_split_reduce_kernel(
    Mp, Lp, Ap, O,
    stride_pb, stride_ph, stride_pm, stride_ps,
    stride_ab, stride_ah, stride_am, stride_as, stride_ad,
    stride_ob, stride_oh, stride_om, stride_od,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Second pass: log-sum-exp combine the per-split partials."""
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_s = tl.arange(0, SPLITS)
    offs_d = tl.arange(0, HEAD_DIM)

    mp_ptrs = Mp + pid_b * stride_pb + pid_h * stride_ph + pid_m * stride_pm + offs_s * stride_ps
    lp_ptrs = Lp + pid_b * stride_pb + pid_h * stride_ph + pid_m * stride_pm + offs_s * stride_ps
    m_vals = tl.load(mp_ptrs)
    l_vals = tl.load(lp_ptrs)

    m_final = tl.max(m_vals)
    alpha = tl.exp2(m_vals - m_final)
    l_final = tl.sum(alpha * l_vals)

    ap_ptrs = (Ap + pid_b * stride_ab + pid_h * stride_ah + pid_m * stride_am
               + offs_s[:, None] * stride_as + offs_d[None, :] * stride_ad)
    acc_vals = tl.load(ap_ptrs)
    acc = tl.sum(alpha[:, None] * acc_vals, axis=0) / l_final

    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + pid_m * stride_om + offs_d * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty))


LOG2E = 1.4426950408889634


def split_kv_forward(q, k, v, causal, splits):
    """Two-pass split-KV forward. Call from dispatch when seq_q is small."""
    batch, num_heads, seq_q, head_dim = q.shape
    _, num_kv_heads, seq_kv, _ = k.shape
    group_size = num_heads // num_kv_heads
    scale_log2 = (head_dim ** -0.5) * LOG2E

    m_p = torch.empty((batch, num_heads, seq_q, splits), dtype=torch.float32, device=q.device)
    l_p = torch.empty_like(m_p)
    acc_p = torch.empty((batch, num_heads, seq_q, splits, head_dim),
                        dtype=torch.float32, device=q.device)
    o = torch.empty_like(q)

    BLOCK_M = 16 if seq_q <= 16 else 32

    grid1 = lambda META: (num_kv_heads, triton.cdiv(seq_q, BLOCK_M), batch * splits)
    gqa_split_first_kernel[grid1](
        q, k, v, m_p, l_p, acc_p,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        m_p.stride(0), m_p.stride(1), m_p.stride(2), m_p.stride(3),
        acc_p.stride(0), acc_p.stride(1), acc_p.stride(2), acc_p.stride(3), acc_p.stride(4),
        seq_q, seq_kv,
        scale_log2,
        IS_CAUSAL=causal,
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        SPLITS=splits,
    )

    grid2 = (num_heads, seq_q, batch)
    gqa_split_reduce_kernel[grid2](
        m_p, l_p, acc_p, o,
        m_p.stride(0), m_p.stride(1), m_p.stride(2), m_p.stride(3),
        acc_p.stride(0), acc_p.stride(1), acc_p.stride(2), acc_p.stride(3), acc_p.stride(4),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        HEAD_DIM=head_dim,
        SPLITS=splits,
    )
    return o
