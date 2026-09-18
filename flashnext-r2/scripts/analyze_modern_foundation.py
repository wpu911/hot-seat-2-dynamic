#!/usr/bin/env python3
"""Gate Sep18-upstream foundation before using it as the base for MTP/FR-Spec.

The foundation's job is compatibility first, speed second. It must preserve the
production deterministic outputs, MTP acceptance and throughput within tight
bounds. Any genuine speedup is recorded but is not required for PASS.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

WORKLOADS=("zh","code","tool")

def med(xs):
    xs=[x for x in xs if isinstance(x,(int,float))]
    return statistics.median(xs) if xs else None

def pct(new,old):
    if not isinstance(new,(int,float)) or not isinstance(old,(int,float)) or old==0: return None
    return (new/old-1.0)*100.0

def collect(data,model):
    legs=[x for x in data.get("legs",[]) if x.get("model")==model]
    pp={}
    for k in sorted({k for l in legs for k in l.get("pp",{})},key=lambda x:int(x)):
        pp[k]=med([r.get("pp") for l in legs for r in l.get("pp",{}).get(k,[])])
    tg={}; acc={}; out={}
    for w in WORKLOADS:
        rs=[r for l in legs for r in l.get("tg",[]) if r.get("workload")==w]
        tg[w]=med([r.get("tg") for r in rs])
        av=[]
        for r in rs:
            d,a=r.get("drafted"),r.get("accepted")
            if isinstance(d,(int,float)) and d>0 and isinstance(a,(int,float)): av.append(a/d)
        acc[w]=med(av); out[w]=[r.get("content","") for r in rs]
    return {"legs":len(legs),"pp":pp,"tg":tg,"acceptance":acc,"outputs":out}

def stable_equal(a,b):
    return bool(a) and bool(b) and len(set(a))==1 and len(set(b))==1 and a[0]==b[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--baseline",required=True)
    ap.add_argument("--foundation",required=True)
    ap.add_argument("--max-median-tg-loss",type=float,default=2.0)
    ap.add_argument("--max-workload-tg-loss",type=float,default=3.0)
    ap.add_argument("--max-median-pp-loss",type=float,default=3.0)
    ap.add_argument("--max-acceptance-drop-pp",type=float,default=2.0)
    args=ap.parse_args()
    data=json.loads(Path(args.result).read_text(encoding="utf-8"))
    b=collect(data,args.baseline); f=collect(data,args.foundation)
    td={w:pct(f["tg"].get(w),b["tg"].get(w)) for w in WORKLOADS}
    pd={k:pct(f["pp"].get(k),b["pp"].get(k)) for k in sorted(set(b["pp"])|set(f["pp"]),key=lambda x:int(x))}
    ad={}; exact={}
    for w in WORKLOADS:
        ba,fa=b["acceptance"].get(w),f["acceptance"].get(w)
        ad[w]=None if ba is None or fa is None else (fa-ba)*100.0
        exact[w]=stable_equal(b["outputs"].get(w,[]),f["outputs"].get(w,[]))
    tvals=[x for x in td.values() if isinstance(x,(int,float))]
    pvals=[x for x in pd.values() if isinstance(x,(int,float))]
    avals=[x for x in ad.values() if isinstance(x,(int,float))]
    tmed=med(tvals); tworst=min(tvals) if tvals else None; pmed=med(pvals); aworst=min(avals) if avals else None
    checks={
      "bit_exact":all(exact.values()),
      "median_tg":tmed is not None and tmed>=-args.max_median_tg_loss,
      "worst_tg":tworst is not None and tworst>=-args.max_workload_tg_loss,
      "median_pp":pmed is not None and pmed>=-args.max_median_pp_loss,
      "acceptance":aworst is not None and aworst>=-args.max_acceptance_drop_pp,
    }
    result="PASS" if all(checks.values()) else "FAIL"
    report={"baseline":args.baseline,"foundation":args.foundation,
      "baseline_metrics":{k:v for k,v in b.items() if k!="outputs"},
      "foundation_metrics":{k:v for k,v in f.items() if k!="outputs"},
      "tg_delta_pct":td,"pp_delta_pct":pd,"acceptance_delta_percentage_points":ad,
      "bit_exact":exact,"checks":checks,
      "gate":{"result":result,"median_tg_delta_pct":tmed,"worst_tg_delta_pct":tworst,
              "median_pp_delta_pct":pmed,"worst_acceptance_delta_pp":aworst}}
    for w in WORKLOADS:
        print(f"{w:>5} TG {b['tg'].get(w)} -> {f['tg'].get(w)} ({td[w]}%)  accΔ={ad[w]}pp exact={exact[w]}")
    for k in sorted(pd,key=lambda x:int(x)):
        print(f"PP {k:>5} {b['pp'].get(k)} -> {f['pp'].get(k)} ({pd[k]}%)")
    print(json.dumps(report["gate"],ensure_ascii=False,indent=2))
    out=Path(args.result).with_suffix(".foundation-analysis.json")
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if result=="PASS" else 2)

if __name__=="__main__": main()
