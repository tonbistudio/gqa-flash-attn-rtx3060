# Experiment 4: `BLOCK_M=128` in autotune

**Status:** kept
**Impact:** prefill@16K 0.84× → 0.99× FA2 (near parity from a one-line change)

## Hypothesis

The initial autotune space had `BLOCK_M ∈ {32, 64}` based on an a priori "sm_86 has less SRAM than A100" heuristic. A100 uses `BLOCK_M=128` in FA2; the 3060 has only 100KB shared memory per SM (vs A100's 192KB), so maybe 128 is too big.

Let's check the math. At `head_dim=128, BLOCK_M=128, BLOCK_N=64` in bf16:
- Q tile in SRAM: 128 × 128 × 2 = 32KB
- K tile: 64 × 128 × 2 = 16KB
- V tile: 64 × 128 × 2 = 16KB
- **Total: 64KB** — well within the 100KB/SM budget.

(Accumulators `acc[BLOCK_M, HEAD_DIM]` live in **registers**, not SRAM, so they don't count toward this budget.)

So the a priori cap was wrong. Adding `BLOCK_M=128` is zero-risk: if it doesn't win, autotune ignores it; if it does, autotune picks it.

## Change

```python
for BM in (32, 64):        # before
for BM in (32, 64, 128):   # after
```

This adds 27 configs to the autotune space (3 BLOCK_N × 3 num_warps × 3 num_stages). Autotune time goes up ~50%.

## Result

| Workload | Before | After | Δ |
|---|---|---|---|
| Prefill @4K  | 0.87× | 0.93× | +7% |
| Prefill @8K  | 0.87× | **0.93×** | +7% |
| Prefill @12K | 0.86× | **0.96×** | +12% |
| Prefill @16K | 0.84× | **0.99×** | +18% |
| Decode @2K   | 1.46× | **1.69×** | +16% |
| Decode @8K   | 0.86× | 0.94× | +9% |

Prefill@16K went from **123ms → 100ms** — one-line change.

Diagnostic run (`TRITON_PRINT_AUTOTUNING=1`) confirmed that autotune picks `BLOCK_M=128` at every prefill length ≥ 2048. The only question was "why wasn't it in the search space to begin with?"

## Takeaway

**Don't trust a priori cap heuristics.** The "smaller SRAM → smaller tiles" intuition is only half the story. What matters is:

1. Does the tile fit in SRAM at the winning `num_warps`?
2. Does register pressure allow enough resident warps for occupancy?
3. Does the larger tile reduce K/V reload traffic enough to matter?

For this hardware/workload/dtype combination, `BLOCK_M=128` passes all three checks. On A100 it does too (with more headroom). The heuristic was wrong because it conflated SRAM budget with tile size without doing the arithmetic.

**General rule:** when adding configs to autotune, the only real cost is autotune time. If a config wouldn't compile (e.g. tile too large for SRAM), Triton skips it gracefully. Be generous with the search space on first pass, then prune based on diagnostics.

## What else we tried from this diagnostic

Running `TRITON_PRINT_AUTOTUNING=1` and inspecting `autotuner.cache` revealed that:
- Prefill picks `BLOCK_M=128` at every length
- `BLOCK_N` alternates 32 ↔ 64 depending on length
- `num_stages` maxes at 3 — adding `num_stages=4` didn't help (tried, reverted — variance grew faster than mean improved)

Diagnostic runs are cheap and often worth more than a blind experiment.
