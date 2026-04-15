# Experiment 2: Causal block skipping — **reverted**

**Status:** discarded
**Impact:** −6% prefill@8K, −24% decode@2K (against the change's intent)

## Hypothesis

For a causal Q tile covering rows `[pid_m·BLOCK_M, (pid_m+1)·BLOCK_M)`, most K blocks are either fully below the diagonal (no mask needed, everything attended) or fully above (entirely masked, nothing attended). Only the diagonal block needs the causal `tl.where`. Reference flash-attn (CUDA) implements this block-level fast path explicitly and saves real cycles at long context.

## Change

Split the inner K loop into two phases:

```python
# Phase 1: unmasked blocks — no bounds check, no causal where
for n_start in range(0, end_unmasked, BLOCK_N):
    ...  # pure dot + online softmax, minimal ops

# Phase 2: diagonal / last block — bounds + causal masks
for n_start in range(end_unmasked, end_full, BLOCK_N):
    ...  # original masking logic
```

Correctness passed. Max abs err unchanged.

## Result

| Workload | Before | After | Δ |
|---|---|---|---|
| Prefill @8K  | 0.85× FA2 | 0.80× | **−6%** |
| Decode @2K   | 0.95× FA2 | 0.72× | **−24%** |
| Decode @8K   | 0.41× FA2 | 0.40× | flat |

Perf regressed everywhere, worst at decode@2K where the optimization should have been most relevant.

## Decision

**Discard.** The "free-looking" optimization broke something more important.

## Takeaway

Splitting a tight Triton loop into two phases **defeats `num_stages` software pipelining**. Triton's autotune-selected pipelining overlaps async K/V loads with compute from earlier iterations of the SAME loop. When the loop is split, each phase is pipelined independently, and the hand-off drains the pipeline. On sm_86 the `tl.where` is already essentially free (select instructions overlap with ALU), so the "savings" are imaginary — the real cost is pipeline fill/drain at the boundary.

This pattern **does** pay off in hand-written CUDA, where explicit register control lets you overlap phase boundaries. Triton doesn't expose that, so this is a case where porting a FA2 trick directly backfires.

**General lesson:** if a Triton restructuring looks like it should help but regresses, suspect pipelining first. Keep the loop as one unit whenever possible and let `num_stages` do its job.
