# gqa-flash-attn-rtx3060

**A GQA flash attention kernel tuned for the NVIDIA RTX 3060.** Written in Triton, runs on Windows, matches flash-attn2 across prefill and decode. Beats PyTorch SDPA by 1.5–7×.

Built for Llama 3.1 8B attention shape (`num_heads=32, num_kv_heads=8, head_dim=128`) but the kernel is general over these dims.

---

## Benchmark

All numbers: **NVIDIA GeForce RTX 3060**, 5-trial median, bf16, batch=1, Windows 11, torch 2.10.0+cu128, `triton-windows` 3.6.0. Baselines: `kernels-community/flash-attn2` (via `flash_attn_func`) and PyTorch SDPA. Each kernel timed with inputs in its native layout (transpose/expand costs excluded — see notes below).

| Workload   | seq_kv | ours (ms) | FA2 (ms) | SDPA (ms) | **vs FA2** | vs SDPA |
|------------|-------:|----------:|---------:|----------:|-----------:|--------:|
| Prefill    |   2048 |     1.59  |    1.57  |     2.54  |   0.99×    |  1.60×  |
| Prefill    |   4096 |     6.01  |    6.03  |     9.62  | **1.00×**  |  1.60×  |
| Prefill    |   6144 |    13.70  |   13.51  |    21.53  |   0.99×    |  1.57×  |
| Prefill    |   8192 |    24.27  |   23.44  |    37.13  |   0.97×    |  1.53×  |
| Prefill    |  12288 |    52.65  |   54.22  |    85.01  | **1.03×**  |  1.61×  |
| Prefill    |  16384 |    96.20  |   97.55  |   149.17  | **1.01×**  |  1.55×  |
| Decode     |   2048 |     0.13  |    0.13  |     0.25  | **1.02×**  |  1.89×  |
| Decode     |   4096 |     0.13  |    0.14  |     0.48  | **1.02×**  |  3.65×  |
| Decode     |   8192 |     0.14  |    0.14  |     1.00  |   1.00×    |  7.39×  |
| Decode     |  16384 |     0.27  |    0.26  |     1.97  |   0.94×    |  7.30×  |

**Read**: matches flash-attn2 across prefill and decode (0.94–1.03×), beats SDPA by 1.5–7.4×.

Reproduce: `python benchmark/run.py --trials 5`.

**Benchmark notes:**
- FA2's decode-specialized `flash_attn_with_kvcache` has a shape-assertion bug for `seqlen_q=1 + GQA` on this Windows build. Both workloads use `flash_attn_func` (the general-purpose path).
- SDPA's `is_causal=True` uses upper-left alignment; for decode (`seq_q=1`) this would restrict attention to `K[0]` only. The benchmark correctly uses `is_causal=False` for decode.
- Each kernel is timed with inputs pre-converted to its native layout. Our kernel takes `(B, H, S, D)`; FA2 takes `(B, S, H, D)` (pre-transposed); SDPA takes pre-expanded K/V (`num_kv_heads → num_heads`). Conversion costs are stack-boundary costs, not per-layer costs.

---

## Hardware target

| Spec | Value |
|---|---|
| GPU | NVIDIA RTX 3060 |
| Compute | sm_86 (Ampere) |
| VRAM | 12 GB GDDR6 |
| L2 cache | 3 MB |
| SMs | 28 |
| Shared mem / SM | 100 KB |

The kernel is written for sm_86 specifically — it autotunes against the 3060's SM count, SRAM budget, and memory bandwidth. It will likely run unchanged on other Ampere desktop SKUs (RTX 3070/3080/3090) and probably on Ada (RTX 40-series), but the autotune winners will shift.

---

## Install and use

```bash
pip install torch>=2.10 triton-windows kernels
git clone https://github.com/tonbistudio/gqa-flash-attn-rtx3060.git
cd gqa-flash-attn-rtx3060
```

```python
import torch
from gqa_kernels import gqa_flash_attn

# Llama 3.1 8B attention shape
q = torch.randn(1, 32, 8192, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1,  8, 8192, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1,  8, 8192, 128, dtype=torch.bfloat16, device='cuda')

out = gqa_flash_attn(q, k, v, causal=True)    # (1, 32, 8192, 128)
```

Causal + `seq_q < seq_kv` works too (chunked prefill / decode-with-cache):

```python
# decode: one query token attends to an 8K cache
q = torch.randn(1, 32,    1, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1,  8, 8192, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1,  8, 8192, 128, dtype=torch.bfloat16, device='cuda')
out = gqa_flash_attn(q, k, v, causal=True)    # (1, 32, 1, 128)
```

The dispatcher picks single-pass vs split-KV automatically based on `seq_q` and `seq_kv`.

Run the correctness gate: `pytest tests/ -v` (requires CUDA GPU).

---

## Design

Three kernels with a Python-level dispatch:

- **`gqa_kernels/gqa_fwd.py`** — single-pass forward kernel for prefill and chunked-prefill. Grid `(num_kv_heads, ceil(seq_q/BLOCK_M), batch)`. Packs the group dimension into the M axis.
- **`gqa_kernels/gqa_split.py`** — split-KV decode path. Two kernels: `gqa_split_first_kernel` computes per-chunk partial `(m, l, acc_unnormalized)`, `gqa_split_reduce_kernel` combines via log-sum-exp.
- **`gqa_kernels/dispatch.py`** — `gqa_flash_attn()` entrypoint. Routes to split-KV when `seq_q ≤ 16 and seq_kv ≥ 1024 and num_kv_heads · batch < SM_COUNT`; otherwise single-pass.

### Key structural choices (all documented in `experiments/`)

1. **Group packing in M axis** — GQA's advertised win comes from sharing K/V across the `group_size` Q heads. A naive grid-level "group sharing" only dedupes loads via L1 cache. **Packing the group dim into the M axis** forces a single K/V load + single matmul per chunk. Biggest single win: decode@8K **0.89× → 2.95× FA2**. See [exp5_group_packing.md](experiments/exp5_group_packing.md).

2. **Split-KV for decode** — at `seq_q=1`, the single-pass grid launches only `num_kv_heads = 8` programs on 28 SMs. Partitioning `seq_kv` into chunks brings it to 128 programs. Required to unblock decode at all. See [exp3_split_kv_decode.md](experiments/exp3_split_kv_decode.md).

3. **`exp2`-based softmax** — Ampere SFU runs `exp2` faster than `exp`. Fold `log2(e)` into the attention scale on the host; every `exp` in the hot loop becomes an `exp2`. See [exp1_exp2_softmax.md](experiments/exp1_exp2_softmax.md).

4. **Aggressive autotune + diagnostic pruning** — the autotune space covers `BLOCK_M ∈ {8, 16, 32}` (capped so packed tile ≤ 128 rows), `BLOCK_N ∈ {32, 64, 128}`, `num_warps ∈ {2, 4, 8}`, `num_stages ∈ {1, 2, 3}`. Adding `BLOCK_M=128` to the initial tighter space was a +7% prefill win. See [exp4_block_m_128.md](experiments/exp4_block_m_128.md).

### Hard invariants (don't change these)

- Grid is `(num_kv_heads, ceil(seq_q/BLOCK_M), batch)`. Don't flatten group into the head dim.
- Q → KV head mapping: `q_head_idx // group_size == kv_head_idx`.
- Online softmax recurrence stays streaming and numerically stable.
- Causal semantics: q at position `row_global + (seq_kv − seq_q)` attends to K at `[0, q_pos]`.
- Tensor layouts: Q `(B, H, Sq, D)`, K/V `(B, Hkv, Skv, D)` — contiguous in the last dim.

---

## The journey

The `experiments/` folder documents every change tried during development — what worked, what didn't, and why. If you're porting flash attention to a new GPU or working with Triton specifically, the **reverted** experiments are often more instructive than the kept ones. Notable findings:

- **[exp2: Causal block skipping reverted](experiments/exp2_causal_skip_revert.md)** — splitting the K/V loop into fast/slow phases regressed prefill 6% even though correctness passed. Root cause: Triton's `num_stages` pipelining can't span a phase boundary. The "obvious" FA2 trick backfires in Triton.
- **Scaling SPLITS for long context** (`splits=32` at 8K, `splits=64` at 16K) regressed decode@16K. The per-chunk launch + init overhead outpaces the extra parallelism. `splits=16` is the sweet spot for 3060.
- **`num_stages=4` in autotune** amplified variance without improving the median. Deeper pipelining doesn't pay at the loop lengths this kernel sees.

Full run log: `benchmark/results/` (JSON per trial) and the commit history of this repo.

---

## What this is not

- **Not a drop-in for Dao-AILab flash-attn.** Different signature, no varlen, no paged attention, no dropout, no bias.
- **Not multi-GPU.** Batch dim is supported but was only benchmarked at batch=1.
- **Not training-ready.** Forward pass only (no backward). The algorithmic choices (in-place scatter, fp32 partial storage) are decode/inference-first.
- **Not Hopper-optimized.** The split-KV path is Ampere-optimized; on H100 you want FA3 and warp specialization.
- **Not FA2 with paint.** The structural choice (explicit group packing in M) differs from Dao-AILab's head-parallel layout. This is specifically what makes it fast on 3060's narrow SM count.

---

## Roadmap

This is the generic attention kernel. Natural follow-ups on top of it:

- **Persistent prefix KV cache across calls.** Many single-user agentic workloads resend a shared system prompt + memory block on every turn. Caching the K/V for the static prefix and computing attention only over the new tokens can reduce per-turn latency by an order of magnitude.
- **Chunked prefill semantics.** The kernel already supports `seq_q < seq_kv` with bottom-right causal alignment, which is the primitive most prefix-cache designs need.
- **vLLM-compatible signature.** The kernel's argument layout isn't painted into a corner — wrapping it in a vLLM-shaped `Attention` interface is mostly glue code, not a rewrite.

None of the above is in this repo.

---

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this in a publication, please cite the repo. The algorithmic choices (group packing, split-KV decode) are from published work by Dao et al. (flash-attention, flash-attention 2). This repo is the sm_86-specific implementation and measurement, not new algorithmic work.
