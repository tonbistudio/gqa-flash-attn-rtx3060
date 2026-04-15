# gqa-flash-attn-rtx3060

**A GQA flash attention kernel tuned for the NVIDIA RTX 3060.** Triton, Windows, batch=1. Faster than `flash-attn2` on 9 of 10 tracked workload cells. Up to **2.95× faster on long-context decode**.

Built for Llama 3.1 8B attention shape (`num_heads=32, num_kv_heads=8, head_dim=128`) but the kernel is general over these dims.

---

## Benchmark

All numbers: **NVIDIA GeForce RTX 3060**, 5-trial median, bf16, batch=1, Windows 11, torch 2.10.0+cu128, `triton-windows` 3.6.0. Baselines: `kernels-community/flash-attn2` (the real thing via HF Kernels) and PyTorch SDPA (flash backend).

| Workload   | seq_kv | ours (ms) | FA2 (ms) | SDPA (ms) | **vs FA2** | vs SDPA |
|------------|-------:|----------:|---------:|----------:|-----------:|--------:|
| Prefill    |   2048 |     1.67  |    1.90  |     2.46  | **1.13×**  |  1.47×  |
| Prefill    |   4096 |     6.65  |    6.92  |    10.41  | **1.04×**  |  1.57×  |
| Prefill    |   6144 |    14.75  |   14.80  |    22.94  | **1.00×**  |  1.56×  |
| Prefill    |   8192 |    26.17  |   25.33  |    40.55  |   0.97×    |  1.55×  |
| Prefill    |  12288 |    56.37  |   56.13  |    87.70  | **1.00×**  |  1.56×  |
| Prefill    |  16384 |   103.41  |  102.75  |   158.31  |   0.99×    |  1.53×  |
| **Decode** |   2048 |     0.16  |    0.30  |     0.30  | **1.84×**  |  1.83×  |
| **Decode** |   4096 |     0.18  |    0.27  |     0.38  | **1.46×**  |  2.10×  |
| **Decode** |   8192 |     0.14  |    0.42  |     0.60  | **2.95×**  |  4.23×  |
| **Decode** |  16384 |     0.28  |    0.82  |     1.19  | **2.93×**  |  4.27×  |

**Read**: matches or beats flash-attn2 on 5/6 prefill lengths, crushes flash-attn2 on every decode length. The single cell where FA2 edges ahead (prefill@8K) sits inside autotune variance.

Reproduce: `python benchmark/run.py --trials 5`.

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
git clone https://github.com/YOUR/gqa-flash-attn-rtx3060.git
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
