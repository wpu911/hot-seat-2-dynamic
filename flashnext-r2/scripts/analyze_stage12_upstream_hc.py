#!/usr/bin/env python3
"""Gate merged upstream qwen4exp HC/norm improvements against production."""
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
    keys=sorted({k for l in legs for k in l.get('pp',{})}, key=lambda x:int(x))
    pp={}
    for k in keys:
        pp[k]=med([r.get('pp') for l in legs for r in l.get('pp',{}).get(k,[])])
    tg={}; outs={}
    for w in ('zh','code','tool'):
        rows=[r for l in legs for r in l.get('tg',[]) if r.get('workload')==w]
        tg[w]=med([r.get('tg') for r in rows]); outs[w]=[r.get('content','') for r in rows]
    return {'legs':len(legs),'pp':pp,'tg':tg,'outputs':outs}


def same(xs): return bool(xs) and len(set(xs))==1


def main():
    p=argparse.ArgumentParser(); p.add_argument('result'); p.add_argument('--baseline',required=True); p.add_argument('--r2',required=True)
    p.add_argument('--min-median-pp-gain',type=float,default=5.0); p.add_argument('--max-tg-loss',type=float,default=2.0)
    p.add_argument('--max-pp-loss',type=float,default=2.0); a=p.parse_args()
    d=json.loads(Path(a.result).read_text(encoding='utf-8')); b=collect(d,a.baseline); r=collect(d,a.r2)
    ppd={k:pct(r['pp'].get(k),b['pp'].get(k)) for k in sorted(set(b['pp'])|set(r['pp']),key=lambda x:int(x))}
    tgd={w:pct(r['tg'].get(w),b['tg'].get(w)) for w in ('zh','code','tool')}
    exact={}
    for w in ('zh','code','tool'):
        bo=b['outputs'].get(w,[]); ro=r['outputs'].get(w,[]); exact[w]=same(bo) and same(ro) and bo[0]==ro[0]
    ppv=[x for x in ppd.values() if isinstance(x,(int,float))]; tgv=[x for x in tgd.values() if isinstance(x,(int,float))]
    ppm=med(ppv); ppworst=min(ppv) if ppv else None; tgworst=min(tgv) if tgv else None
    ok=all(exact.values()) and ppm is not None and ppm>=a.min_median_pp_gain and (ppworst is None or ppworst>=-a.max_pp_loss) and (tgworst is None or tgworst>=-a.max_tg_loss)
    rep={'baseline':a.baseline,'r2':a.r2,'delta_pct':{'pp':ppd,'tg':tgd},'bit_exact':exact,'gate':{'result':'PASS' if ok else 'FAIL','median_pp_gain_pct':ppm,'worst_pp_delta_pct':ppworst,'worst_tg_delta_pct':tgworst}}
    print(json.dumps(rep,ensure_ascii=False,indent=2)); out=Path(a.result).with_suffix('.upstream-hc-analysis.json'); out.write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'ANALYSIS={out}'); raise SystemExit(0 if ok else 2)
if __name__=='__main__': main()
