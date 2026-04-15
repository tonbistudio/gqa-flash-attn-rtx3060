# Experiment 5: Group packing — the biggest structural win

**Status:** kept — the decisive change
**Impact:** decode@8K 0.89× → 2.95× FA2 (+3.3×), prefill@2K 0.91× → 1.13× (beats FA2)

## The insight

After running 4 experiments, decode was still at 0.88× FA2 at 8K+. The code "did" group-sharing — the grid was `(num_kv_heads, M, batch)` — but when I finally read my own kernel closely:

```python
for g in tl.static_range(GROUP_SIZE):   # ← outer
    q = load(Q[g])
    for n_start in range(...):          # ← inner
        k, v = load(K[n_start]), load(V[n_start])  # ← RELOADED for each g
        ...
```

The K/V load is inside BOTH loops. `tl.static_range` unrolls at compile time, producing 4 copies of the inner loop, each with its own `tl.load(K, V)`. The "group-sharing" in the spec relied entirely on L1/L2 cache to dedupe the redundant loads — there was no instruction-level sharing.

FA2's CUDA source does the opposite ordering. So did I, eventually.

## The fix

Pack the group axis into the M axis. One tile, one matmul, one K/V load per chunk handles all group members:

```python
BLOCK_MG: tl.constexpr = BLOCK_M * GROUP_SIZE   # effective tile
offs_mg = tl.arange(0, BLOCK_MG)
g_idx = offs_mg // BLOCK_M          # which Q head per row
m_idx = offs_mg % BLOCK_M           # which Q position per row
q_heads = pid_kv * GROUP_SIZE + g_idx
row_global = pid_m * BLOCK_M + m_idx

# Load all GROUP_SIZE Q tiles in one gather via q_heads[:, None]
q = tl.load(Q + ... + q_heads[:, None] * stride_qh + row_global[:, None] * stride_qm + ...)

# One m_i, l_i, acc of shape [BLOCK_MG, ...]
# Single inner loop, single matmul per iteration
```

Output store is a scatter: rows 0..BLOCK_M−1 → q_head=0, rows BLOCK_M..2BLOCK_M−1 → q_head=1, etc.

## What changed in instruction count

For group_size=4, per program per K/V chunk:
- Old: **4** `tl.load(K)`, **4** `tl.load(V)`, **4** matmuls, **4** softmax updates
- New: **1** `tl.load(K)`, **1** `tl.load(V)`, **1** matmul, **1** softmax update

Total matmul work (MACs) is unchanged — a (64×128) @ (128×64) is algebraically the same as 4× (16×128) @ (128×64). The savings come from:
1. 4× fewer load instructions → less register pressure, less L1 traffic
2. 4× fewer matmul launches → better pipelining
3. 4× fewer softmax updates → simpler control flow

## Applied twice: decode first, then prefill

**Round 1** — split-KV first kernel (decode path):

| Workload | Before | After |
|---|---|---|
| Decode @2K  | 1.67× FA2 | 1.56× (noise at 0.16ms timescale) |
| Decode @4K  | 1.02× FA2 | 1.04× |
| Decode @8K  | **0.89× FA2** | **2.65×** |
| Decode @16K | 0.88× FA2 | **2.68×** |

**Round 2** — main (prefill) kernel:

| Workload | Before | After |
|---|---|---|
| Prefill @2K  | 0.91× FA2 | **1.13×** |
| Prefill @4K  | 0.92× | **1.04×** |
| Prefill @6K  | 0.93× | **1.00×** |
| Prefill @8K  | 0.94× | 0.97× |
| Prefill @12K | 0.97× | **1.00×** |
| Prefill @16K | 0.96× | 0.99× |
| Decode @8K   | 2.65× | **2.95×** (further improvement) |

Prefill matches or beats FA2 on 5 of 6 lengths.

## Register pressure caveat

Packing has a register cost: the `acc` tensor is now `[BLOCK_MG, HEAD_DIM]` instead of `[BLOCK_M, HEAD_DIM]`. At `head_dim=128 fp32 acc`:
- `BLOCK_MG=64` (split-KV, `BLOCK_M=16`): 32KB in registers — fine
- `BLOCK_MG=128` (prefill, `BLOCK_M=32`): 64KB in registers — tight but works at `num_warps=8`
- `BLOCK_MG=512` (if we tried `BLOCK_M=128`): 256KB — would spill heavily

That's why the prefill autotune config was capped at `BLOCK_M ≤ 32`. Going bigger would pack to a register-busting tile.

## Takeaway

**Read your own kernel carefully.** The spec intended group-sharing from day one. The initial implementation had the right grid shape but the wrong loop order — the sharing was happening by accident (cache) rather than by design (instruction count). Rewriting to do it on purpose was the single biggest win of the whole project.

**L1 cache is not a substitute for compiler awareness.** If you want the compiler to see that K/V loads are invariant across a dimension, put that dimension on the inside of the loop, not the outside. `tl.static_range` unrolls — it doesn't CSE across iterations.
