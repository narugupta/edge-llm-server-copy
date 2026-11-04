#!/usr/bin/env python3
"""Summarize the matched offered-load sweep (results/sweep/L*/results_sweep_*.json).

For each (slots, level, scenario) and type: n, completed, mean, p95, max, and the
share of requests over the SLO threshold. Chat/Moderate use TTFT; Batch uses total
time. SLO thresholds: chat 0.5s, moderate 2.0s (TTFT), batch 60s (total).
Also writes sweep_summary.csv. Percentiles are linear-interpolated; with n < 20
treat p95 as 'close to the max', and report max alongside it.

Usage: python3 summarize_sweep.py results/sweep [--csv sweep_summary.csv]
"""
import csv, glob, json, os, re, statistics as st, sys

SLO = {"chat": ("ttft", 0.5), "moderate": ("ttft", 2.0), "batch": ("total_time", 60.0)}
NAME = re.compile(r"results_sweep_L([\d.]+)_scenario(\d)_slots(\d+)_ts(\d+)\.json$")

def pct(v, p):
    if not v: return None
    v = sorted(v); k = (len(v) - 1) * p / 100; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

def fmt(x, w=6): return " " * (w - 1) + "-" if x is None else f"{x:{w}.2f}"

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else "results/sweep"
    csv_path = "sweep_summary.csv"
    if "--csv" in sys.argv: csv_path = sys.argv[sys.argv.index("--csv") + 1]
    rows = []
    for path in glob.glob(os.path.join(root, "**", "results_sweep_*.json"), recursive=True):
        m = NAME.search(os.path.basename(path))
        if not m: continue
        level, scen, slots, ts = float(m[1]), int(m[2]), int(m[3]), int(m[4])
        raw = json.load(open(path))["raw"]
        for t, (metric, thr) in SLO.items():
            rs = [r for r in raw if r["client_type"] == t]
            ok = [r[metric] for r in rs if r.get(metric) is not None and not r.get("error")]
            over = sum(1 for r in rs if r.get("error") or r.get(metric) is None or r[metric] > thr)
            rows.append(dict(slots=slots, level=level, scenario=scen, type=t, n=len(rs),
                completed=len(ok), mean=st.mean(ok) if ok else None, p95=pct(ok, 95),
                max=max(ok) if ok else None, viol_pct=100 * over / len(rs) if rs else None,
                slo_mean_pass=(st.mean(ok) < thr) if ok else False))
    if not rows: sys.exit(f"no results_sweep_*.json found under {root}")
    rows.sort(key=lambda r: (r["slots"], r["level"], r["type"], r["scenario"]))
    cur = None
    for r in rows:
        key = (r["slots"], r["level"])
        if key != cur:
            cur = key
            print(f"\n=== slots={r['slots']}  load level L={r['level']} ===")
            print(f"{'type':9s} {'scen':>4s} {'n':>3s} {'done':>4s} {'mean':>6s} {'p95':>6s} "
                  f"{'max':>6s} {'>SLO%':>6s}  mean-SLO")
        print(f"{r['type']:9s} {'S'+str(r['scenario']):>4s} {r['n']:3d} {r['completed']:4d} "
              f"{fmt(r['mean'])} {fmt(r['p95'])} {fmt(r['max'])} {fmt(r['viol_pct'])}  "
              f"{'PASS' if r['slo_mean_pass'] else 'FAIL'}")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")

if __name__ == "__main__": main()