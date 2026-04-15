# Experiment 3: Split-KV decode kernels

**Status:** kept — landmark change
**Impact:** decode@8K 0.41× → 0.86× FA2 (2.1× speedup on the most stubborn workload)

## The problem

The RTX 3060 has 28 SMs. At decode (`seq_q=1, seq_kv=L`), the single-pass grid is:

```
(num_kv_heads, ceil(seq_q / BLOCK_M), batch) = (8, 1, 1) = 8 programs
```

With 8 programs on 28 SMs, **20 SMs sit idle**. The decode kernel is also bottlenecked by streaming K/V through a single program per head — memory-bound work serialized behind launch overhead.

FA2's `flash_attn_with_kvcache` solves this via split-KV: partition `seq_kv` into chunks, launch programs per chunk, then reduce. Our kernel had to do the same.

## The algorithm

**First pass** — one program per `(batch, kv_head, chunk)`:
- Compute partial attention over its K/V chunk
- Write `(m_partial, l_partial, acc_unnormalized)` to a scratch tensor

**Second pass** — one program per `(batch, num_heads, seq_q)`:
- Load all partials for this row
- Combine via log-sum-exp:
  ```
  m_final = max(m_partial_i)
  α_i     = exp2(m_partial_i − m_final)
  l_final = Σ α_i · l_partial_i
  acc     = (Σ α_i · acc_partial_i) / l_final
  ```

Dispatch: `if seq_q ≤ 16 and seq_kv ≥ 1024 and num_kv_heads · batch < SM_COUNT`, target ≥4× SMs worth of programs (so on 3060: `splits=16`, giving `8·16 = 128` programs).

## Implementation notes

- **Two Triton kernels, one Python wrapper.** `gqa_split_first_kernel` + `gqa_split_reduce_kernel`.
- **Partials in fp32.** `m_p`, `l_p`, `acc_p` are fp32 to preserve numerical stability across the reduction. The memory cost is small: at seq_q=1, partial tensor is `batch · num_heads · splits · (head_dim + 2) · 4 bytes` ≈ 260KB at splits=16. Trivial.
- **Causal still works.** The first-pass kernel uses the same `q_pos = row_global + (seq_kv − seq_q)` offset as the main kernel, so decode (q_pos = seq_kv − 1) naturally attends to all of `[0, seq_kv)`.
- **Chunk size.** `chunk_size = ceil(seq_kv / splits)`. For decode@8K with splits=16: 512 elements per chunk, ~8 iterations of BLOCK_N=64.

## Result

| Workload | Before | After |
|---|---|---|
| Decode @2K  | 0.95× FA2 | **1.46×** |
| Decode @4K  | 0.42× FA2 | 0.94× |
| Decode @8K  | **0.41× FA2** | **0.86×** |
| Decode @16K | 0.39× FA2 | 0.87× |

First workload to **beat FA2 outright** (decode@2K at 1.46×).

Prefill was unaffected because the dispatch only routes to split-KV when `seq_q ≤ 16` — prefill always goes to the main kernel.

## Tuning notes (what we tried and reverted)

- **Scaling splits with seq_kv** (e.g. `splits=32` at 8K, `splits=64` at 16K) *regressed* at long context. At higher splits, each chunk is smaller and the per-program first-pass overhead (launch + state init) eats into per-chunk compute. `splits=16` is the sweet spot for this hardware.
- **Expanding `num_stages=4`** in the split-KV autotune made variance worse, not better. The split-KV first kernel has a small loop (few iterations), so deep pipelining offers little benefit and adds register pressure.

## Takeaway

Split-KV is a structural optimization, not a parameter tweak. It directly targets SM underutilization at decode. If your kernel ever launches fewer programs than SMs × 2, you probably need it.

The two-pass reduction is algorithmically simple but easy to get wrong in three places:
1. **Numerical stability**: the partials store raw `m` and `l`, the reduction re-normalizes via log-sum-exp. Don't normalize in the first pass.
2. **Causal chunk exclusion**: for prefill with short `seq_q`, some chunks are entirely beyond the max `q_pos` and should be skipped (`chunk_end = min(chunk_end, max_q_pos + 1)`).
3. **Constexpr SPLITS**: the reduction kernel needs SPLITS as a compile-time constant so `tl.arange(0, SPLITS)` works.
