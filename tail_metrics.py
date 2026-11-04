#!/usr/bin/env python3
"""Per-type mean / p95 / max TTFT (and Batch total_time) from results_step2_*.json.
Usage: python3 tail_metrics.py results_step2_*.json"""
import json, sys, glob, statistics as st

def pct(v, p):
    v = sorted(v)
    if not v: return None
    k = (len(v) - 1) * p / 100
    f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

def f(x): return "  -  " if x is None else f"{x:5.2f}"

print(f"{'file':44s} {'type':8s} {'n':>3s} {'ok':>3s} {'mean':>6s} {'p95':>6s} {'max':>6s}")
for path in sorted(sum([glob.glob(a) for a in sys.argv[1:]], [])):
    d = json.load(open(path))
    name = path.split('/')[-1].replace('results_step2_', '').replace('_ts15.json', '')
    for t in ('chat', 'moderate', 'batch'):
        rows = [r for r in d['raw'] if r['client_type'] == t]
        key = 'total_time' if t == 'batch' else 'ttft'
        vals = [r[key] for r in rows if r.get(key) is not None and not r.get('error')]
        print(f"{name:44s} {t:8s} {len(rows):3d} {len(vals):3d} "
              f"{f(st.mean(vals) if vals else None):>6s} {f(pct(vals,95)):>6s} {f(max(vals) if vals else None):>6s}")