#!/usr/bin/env python3
"""compare_s3b.py -- S1 / S2 / S3b (and optionally plain S3) on identical arrivals.
usage: python3 compare_s3b.py BASE_DIR HOLD_DIR [HOLD_DIR ...] [--slots N]
BASE_DIR holds the S1/S2(/S3) sweep files; each HOLD_DIR holds scenario-3 runs made with
HOLD=T (label shown = directory name). Rows are matched by (slots, rho, seed); the schedule
digests must be identical or the row is flagged."""
import glob, json, re, sys
args=[a for a in sys.argv[1:]]; slots=None
if '--slots' in args:
    i=args.index('--slots'); slots=int(args[i+1]); del args[i:i+2]
if len(args)<2: sys.exit(__doc__)
def load(d):
    out={}
    for p in glob.glob(f'{d}/results_sweep_*.json'):
        m=re.search(r'rho([\d.]+)_scenario(\d)_slots(\d+)_seed(\d+)',p)
        if m: out[(int(m[3]),float(m[1]),int(m[4]),int(m[2]))]=json.load(open(p))
    return out
base=load(args[0]); holds={d:load(d) for d in args[1:]}
def _sc(h): return sorted({k[3] for k in h})[0] if h else 3   # the scenario number stored in that dir
cols=[('S1',base,1),('S2',base,2),('S3',base,3)]+[(d.rstrip('/').split('/')[-1],h,_sc(h)) for d,h in holds.items()]
def dig(r): return json.dumps(r['meta'].get('schedule_digests',r['meta'].get('schedule_digest')),sort_keys=True)
def cell(r):
    if r is None: return '      -      '
    pt=r['per_type']; c=pt['chat']; m=pt['moderate']; b=pt['batch']
    return f"{c['tail']['p95']:5.1f}s {100*c['slo_violation_rate']:3.0f}% {100*m['slo_violation_rate']:3.0f}% {100*b['slo_violation_rate']:3.0f}% {r['aggregate']['realized_busy_fraction']:.2f}"
print('each cell: chat-p95  chat-viol  mod-viol  batch-viol  busy-fraction')
keys=sorted({k[:3] for src in [base,*holds.values()] for k in src if slots is None or k[0]==slots})
print(f"{'slots rho seed':16s}"+''.join(f'| {n:^34s}' for n,_,_ in cols))
for k in keys:
    rs=[src.get((*k,sc)) for _,src,sc in cols]
    ds={dig(r) for r in rs if r is not None}
    flag='' if len(ds)<=1 else '  !! schedules differ'
    print(f'{k[0]:>3d} {k[1]:<4} {k[2]:>3d}     '+''.join(f'| {cell(r):34s}' for r in rs)+flag)