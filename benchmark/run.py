"""Benchmark the GQA kernel vs flash-attn2 and PyTorch SDPA.

Usage:
    python benchmark/run.py                # full sweep, single run
    python benchmark/run.py --trials 5     # 5-trial median (reduces autotune noise)
    python benchmark/run.py --quick        # correctness + 8K only
    python benchmark/run.py --correctness  # correctness gate only

Writes per-trial JSON to benchmark/results/ and prints a markdown-ready table.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Load FA2 from HF `kernels` package (pip install kernels) before adding our
# repo to sys.path — otherwise our gqa_kernels could shadow it.
_FA2 = None
try:
    from kernels import get_kernel  # HF kernels
    _FA2 = get_kernel('kernels-community/flash-attn2')
except Exception as e:
    print(f'[warn] flash-attn2 unavailable: {e}', file=sys.stderr)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from gqa_kernels import gqa_flash_attn  # noqa: E402
from tests.reference import gqa_reference  # noqa: E402


BATCH = 1
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16
DEVICE = 'cuda'

PREFILL_LENS = (2048, 4096, 6144, 8192, 12288, 16384)
DECODE_LENS = (2048, 4096, 8192, 16384)
CORRECTNESS_LENS = (512, 2048, 4096)

WARMUP = 25
ITERS = 100


def _make(seq_q, seq_kv):
    q = torch.randn(BATCH, NUM_HEADS,    seq_q,  HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.1
    k = torch.randn(BATCH, NUM_KV_HEADS, seq_kv, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.1
    v = torch.randn(BATCH, NUM_KV_HEADS, seq_kv, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.1
    return q, k, v


def _sdpa(q, k, v, causal):
    group = NUM_HEADS // NUM_KV_HEADS
    ke, ve = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    return F.scaled_dot_product_attention(q, ke, ve, is_causal=causal)


def _fa2(q, k, v, causal):
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    out = _FA2.flash_attn_func(q_t, k_t, v_t, causal=causal)
    return out.transpose(1, 2)


def correctness_check():
    max_tol, mean_tol = 1e-2, 1e-3
    print(f'[correctness] gate: max_abs < {max_tol:g}, mean_abs < {mean_tol:g}')
    cases = ([(L, L, 'prefill') for L in CORRECTNESS_LENS]
             + [(1, L, 'decode') for L in CORRECTNESS_LENS])
    for sq, sk, label in cases:
        q, k, v = _make(sq, sk)
        out = gqa_flash_attn(q, k, v, causal=True)
        ref = gqa_reference(q, k, v, causal=True)
        diff = (out.float() - ref.float()).abs()
        mx, mn = diff.max().item(), diff.mean().item()
        ok = mx < max_tol and mn < mean_tol
        print(f'[correctness] {label:<7} L={sk:>5d}  max={mx:.2e}  mean={mn:.2e}  {"ok" if ok else "FAIL"}')
        if not ok:
            return False
    return True


def _time(fn):
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS, torch.cuda.max_memory_allocated() / (1024 ** 3)


def run_sweep(prefill_lens, decode_lens):
    rows = []
    for workload, lens in (('prefill', prefill_lens), ('decode', decode_lens)):
        for L in lens:
            sq, sk, causal = (L, L, True) if workload == 'prefill' else (1, L, True)
            q, k, v = _make(sq, sk)
            gqa_flash_attn(q, k, v, causal=causal)
            _sdpa(q, k, v, causal)
            if _FA2 is not None:
                _fa2(q, k, v, causal)
            torch.cuda.synchronize()
            ours_ms, ours_mem = _time(lambda: gqa_flash_attn(q, k, v, causal=causal))
            sdpa_ms, _ = _time(lambda: _sdpa(q, k, v, causal))
            fa2_ms = _time(lambda: _fa2(q, k, v, causal))[0] if _FA2 is not None else float('nan')
            row = {
                'workload': workload, 'seq_kv': L,
                'ours_ms': ours_ms, 'fa2_ms': fa2_ms, 'sdpa_ms': sdpa_ms,
                'tok_s': sq * 1000.0 / ours_ms,
                'speedup_vs_fa2': (fa2_ms / ours_ms) if _FA2 else float('nan'),
                'speedup_vs_sdpa': sdpa_ms / ours_ms,
                'peak_mem_gb': ours_mem,
            }
            rows.append(row)
            fa2s = f'{fa2_ms:7.3f}' if _FA2 else '  n/a  '
            print(f'[{workload}] L={L:>5d}  ours={ours_ms:7.3f}ms  fa2={fa2s}ms  sdpa={sdpa_ms:7.3f}ms  '
                  f'vs_fa2={row["speedup_vs_fa2"]:.2f}x  vs_sdpa={row["speedup_vs_sdpa"]:.2f}x')
    return rows


def aggregate_median(trials):
    keys = sorted(set((r['workload'], r['seq_kv']) for trial in trials for r in trial))
    agg = []
    for k in keys:
        cells = [r for trial in trials for r in trial if (r['workload'], r['seq_kv']) == k]
        ours = statistics.median(c['ours_ms'] for c in cells)
        fa2 = statistics.median(c['fa2_ms'] for c in cells)
        sdpa = statistics.median(c['sdpa_ms'] for c in cells)
        mem = statistics.median(c['peak_mem_gb'] for c in cells)
        sq = k[1] if k[0] == 'prefill' else 1
        agg.append({
            'workload': k[0], 'seq_kv': k[1],
            'ours_ms': ours, 'fa2_ms': fa2, 'sdpa_ms': sdpa,
            'tok_s': sq * 1000.0 / ours,
            'speedup_vs_fa2': fa2 / ours,
            'speedup_vs_sdpa': sdpa / ours,
            'peak_mem_gb': mem,
        })
    return agg


def print_markdown_table(rows):
    print()
    print('| Workload | seq_kv | ours (ms) | FA2 (ms) | SDPA (ms) | vs FA2 | vs SDPA | mem (GB) |')
    print('|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f'| {r["workload"]} | {r["seq_kv"]} | {r["ours_ms"]:.3f} | {r["fa2_ms"]:.3f} | '
              f'{r["sdpa_ms"]:.3f} | **{r["speedup_vs_fa2"]:.2f}×** | '
              f'{r["speedup_vs_sdpa"]:.2f}× | {r["peak_mem_gb"]:.2f} |')


def _commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short=7', 'HEAD'],
                                       cwd=HERE.parent).decode().strip()
    except Exception:
        return 'nogit00'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--correctness', action='store_true')
    args = p.parse_args()

    print(f'device: {torch.cuda.get_device_name(0)}')
    print(f'dtype:  {DTYPE}')
    print(f'config: H={NUM_HEADS} Hkv={NUM_KV_HEADS} D={HEAD_DIM} batch={BATCH}')
    if _FA2 is None:
        print('[warn] flash-attn2 not loaded — vs_fa2 column will be NaN')
    print()

    if not correctness_check():
        sys.exit(1)
    if args.correctness:
        return

    prefill = (8192,) if args.quick else PREFILL_LENS
    decode = (8192,) if args.quick else DECODE_LENS

    trials = []
    for i in range(args.trials):
        if args.trials > 1:
            print(f'\n=== trial {i+1}/{args.trials} ===')
        trials.append(run_sweep(prefill, decode))

    rows = aggregate_median(trials) if args.trials > 1 else trials[0]

    out_dir = HERE / 'results'
    out_dir.mkdir(exist_ok=True)
    commit = _commit()
    suffix = f'_median{args.trials}' if args.trials > 1 else ''
    with open(out_dir / f'{commit}{suffix}.json', 'w') as f:
        json.dump({
            'commit': commit,
            'device': torch.cuda.get_device_name(0),
            'dtype': str(DTYPE),
            'trials': args.trials,
            'config': {'batch': BATCH, 'num_heads': NUM_HEADS,
                       'num_kv_heads': NUM_KV_HEADS, 'head_dim': HEAD_DIM},
            'rows': rows,
        }, f, indent=2)

    label = f'{args.trials}-trial median' if args.trials > 1 else 'single run'
    print(f'\n=== Results ({label}, commit {commit}) ===')
    print_markdown_table(rows)


if __name__ == '__main__':
    main()
