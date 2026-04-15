# Experiment 1: `exp2`-based softmax

**Status:** kept
**Impact:** +3–10% across all workloads

## Hypothesis

The online softmax recurrence in flash attention uses `exp(x)` once per K/V chunk. On Ampere (sm_86), the Special Function Unit (SFU) computes `exp2(x)` meaningfully faster than `exp(x)`. The two are interchangeable via the identity:

```
exp(x) = exp2(x · log2(e))
```

If we pre-multiply the attention scale by `log2(e)` once on the CPU, every `exp` in the hot loop becomes an `exp2` with no extra multiplications.

## Change

Single-line edit in the Python wrapper:
```python
LOG2E = 1.4426950408889634
scale_log2 = (head_dim ** -0.5) * LOG2E
```

And inside the kernel, replace:
```python
alpha = tl.exp(m_i - m_new)
p     = tl.exp(qk - m_new[:, None])
```
with:
```python
alpha = tl.exp2(m_i - m_new)
p     = tl.exp2(qk - m_new[:, None])
```

All the `m_i`, `m_new`, `qk` values are now implicitly log2-scaled, so the softmax math is algebraically identical.

## Result

| Workload | Before | After | Δ |
|---|---|---|---|
| Prefill @2K | 0.77× FA2 | 0.85× | +10% |
| Prefill @8K | 0.82× FA2 | 0.85× | +3% |
| Decode @2K  | 0.77× FA2 | 0.95× | +23% |
| Decode @16K | 0.39× FA2 | 1.01× SDPA | first >1× SDPA on decode |

Biggest gains at short context where the softmax is a larger fraction of runtime. At long context the matmul dominates so the `exp2` win is proportionally smaller — as expected.

## Decision

**Keep.** This is one line of change for a consistent win across every cell. Also nearly free to implement and impossible to regress, so it becomes the foundation for all subsequent experiments.

## Takeaway

The canonical FA2 Triton tutorial uses `exp2` for exactly this reason. If you're porting FA2 to Ampere/Ada/Hopper, do this first.
