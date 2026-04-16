"""Single-pass GQA flash attention forward kernel.

Used for prefill and chunked-prefill (seq_q comparable to seq_kv). For
decode (seq_q=1) the dispatcher routes to the split-KV kernels instead.

Design:
- Grid: (num_kv_heads, ceil(seq_q / BLOCK_M), batch). Each program owns one
  (batch, kv_head) and packs GROUP_SIZE Q heads into the M axis, so one
  matmul + one K/V load per chunk handles the whole group.
- Online softmax uses exp2 (fast on Ampere SFU); log2(e) folded into scale.
- Causal offset: q position in K coordinates is `row_global + (seq_kv - seq_q)`,
  so prefill (seq_q == seq_kv) and chunked-prefill both work without branches.
"""
import triton
import triton.language as tl


def _autotune_configs():
    # Effective tile = BLOCK_M * GROUP_SIZE. At head_dim=128 fp32 acc, capping
    # BLOCK_M at 32 keeps the packed tile at 128 rows and register pressure sane.
    configs = []
    for BM in (8, 16, 32):
        for BN in (32, 64, 128):
            for nw in (2, 4, 8):
                for ns in (1, 2, 3):
                    configs.append(
                        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN},
                                      num_warps=nw, num_stages=ns)
                    )
    return configs


@triton.autotune(configs=_autotune_configs(),
                 key=['seq_q', 'seq_kv', 'HEAD_DIM', 'IS_CAUSAL', 'GROUP_SIZE'])
@triton.jit
def gqa_fwd_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_q, seq_kv,
    scale,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # Pack GROUP_SIZE Q tiles into the M axis.
    BLOCK_MG: tl.constexpr = BLOCK_M * GROUP_SIZE
    offs_mg = tl.arange(0, BLOCK_MG)
    g_idx = offs_mg // BLOCK_M
    m_idx = offs_mg % BLOCK_M
    row_global = pid_m * BLOCK_M + m_idx
    q_heads = pid_kv * GROUP_SIZE + g_idx
    row_valid = row_global < seq_q
    q_pos = row_global + (seq_kv - seq_q)

    K_base = K + pid_b * stride_kb + pid_kv * stride_kh
    V_base = V + pid_b * stride_vb + pid_kv * stride_vh

    if IS_CAUSAL:
        kv_end = tl.minimum((pid_m + 1) * BLOCK_M + (seq_kv - seq_q), seq_kv)
    else:
        kv_end = seq_kv

    q_ptrs = (Q + pid_b * stride_qb + q_heads[:, None] * stride_qh
              + row_global[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0)
    q = (q * scale).to(q.dtype)

    m_i = tl.full([BLOCK_MG], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_MG], dtype=tl.float32)
    acc = tl.zeros([BLOCK_MG, HEAD_DIM], dtype=tl.float32)

    for n_start in range(0, kv_end, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask = n_offs < seq_kv

        k_ptrs = K_base + n_offs[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_base + n_offs[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(n_mask[None, :], qk, float('-inf'))
        if IS_CAUSAL:
            qk = tl.where(q_pos[:, None] >= n_offs[None, :], qk, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = (O + pid_b * stride_ob + q_heads[:, None] * stride_oh
              + row_global[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=row_valid[:, None])
