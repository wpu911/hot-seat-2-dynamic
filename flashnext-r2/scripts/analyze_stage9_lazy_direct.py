#!/usr/bin/env python3
"""Analyze Stage-9 lazy mmap vs direct-read A/B."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path


def med(xs):
    xs=[x for x in xs if isinstance(x,(int,float))]
    return statistics.median(xs) if xs else None

def pct(n,o):
    return None if not isinstance(n,(int,float)) or not isinstance(o,(int,float)) or o==0 else (n/o-1)*100

def collect(data, model):
    legs=[x for x in data.get('legs',[]) if x.get('model')==model]
    pp={}
    for k in sorted({k for l in legs for k in l.get('pp',{})}, key=int):
        pp[k]=med([r.get('pp') for l in legs for r in l.get('pp',{}).get(k,[])])
    tg={}; outs={}
    for w in ('zh','code','tool'):
        rows=[r for l in legs for r in l.get('tg',[]) if r.get('workload')==w]
        tg[w]=med([r.get('tg') for r in rows]); outs[w]=[r.get('content','') for r in rows]
    return {'legs':len(legs),'pp':pp,'tg':tg,'outs':outs}
def same(xs): return bool(xs) and len(set(xs))==1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('--baseline',required=True)
    ap.add_argument('--r2',required=True)
    ap.add_argument('--min-median-pp-gain',type=float,default=5.0)
    ap.add_argument('--max-tg-loss',type=float,default=3.0)
    a=ap.parse_args()
    d=json.loads(Path(a.result).read_text())
    b,r=collect(d,a.baseline),collect(d,a.r2)
    ppd={k:pct(r['pp'].get(k),b['pp'].get(k)) for k in sorted(set(b['pp'])|set(r['pp']),key=int)}
    tgd={w:pct(r['tg'].get(w),b['tg'].get(w)) for w in ('zh','code','tool')}
    exact={w:same(b['outs'][w]) and same(r['outs'][w]) and b['outs'][w][0]==r['outs'][w][0] for w in ('zh','code','tool')}
    ppm=med(list(ppd.values())); worst=min([x for x in tgd.values() if x is not None],default=None)
    passed=all(exact.values()) and ppm is not None and ppm>=a.min_median_pp_gain and (worst is None or worst>=-a.max_tg_loss)
    rep={'baseline':{k:v for k,v in b.items() if k!='outs'},'r2':{k:v for k,v in r.items() if k!='outs'},
         'delta_pct':{'pp':ppd,'tg':tgd},'bit_exact':exact,
         'gate':{'result':'PASS' if passed else 'FAIL','median_pp_gain_pct':ppm,'worst_tg_delta_pct':worst,
                 'min_median_pp_gain_pct':a.min_median_pp_gain,'max_tg_loss_pct':a.max_tg_loss}}
    for k in ppd: print(f"PP {k}: {b['pp'].get(k)} -> {r['pp'].get(k)} delta={ppd[k]}%")
    for w in tgd: print(f"TG {w}: {b['tg'].get(w)} -> {r['tg'].get(w)} delta={tgd[w]}% exact={exact[w]}")
    print('GATE',json.dumps(rep['gate'],ensure_ascii=False))
    out=Path(a.result).with_suffix('.lazy-direct-analysis.json'); out.write_text(json.dumps(rep,ensure_ascii=False,indent=2))
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if passed else 2)
if __name__=='__main__': main()
