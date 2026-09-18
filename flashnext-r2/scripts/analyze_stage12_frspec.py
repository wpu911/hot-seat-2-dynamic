#!/usr/bin/env python3
"""Analyze full-vocab MTP sidecar vs FR-Spec reduced-vocab sidecar.

Promotion is intentionally strict because a smaller draft head can look fast while
quietly damaging speculative acceptance. The target model is unchanged, so greedy
outputs should remain identical for the controlled workloads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

WORKLOADS = ("zh", "code", "tool")


def med(xs):
    xs=[x for x in xs if isinstance(x,(int,float))]
    return statistics.median(xs) if xs else None


def pct(new, old):
    if not isinstance(new,(int,float)) or not isinstance(old,(int,float)) or old == 0:
        return None
    return (new/old-1.0)*100.0


def collect(data, model):
    legs=[x for x in data.get("legs",[]) if x.get("model")==model]
    pp={}
    for k in sorted({k for leg in legs for k in leg.get("pp",{})}, key=lambda x:int(x)):
        vals=[]
        for leg in legs:
            vals += [row.get("pp") for row in leg.get("pp",{}).get(k,[])]
        pp[k]=med(vals)

    tg={}; acc={}; outputs={}
    for w in WORKLOADS:
        tv=[]; av=[]; ov=[]
        for leg in legs:
            for row in leg.get("tg",[]):
                if row.get("workload") != w:
                    continue
                tv.append(row.get("tg"))
                ov.append(row.get("content",""))
                d,a=row.get("drafted"),row.get("accepted")
                if isinstance(d,(int,float)) and d>0 and isinstance(a,(int,float)):
                    av.append(a/d)
        tg[w]=med(tv); acc[w]=med(av); outputs[w]=ov
    return {"legs":len(legs),"pp":pp,"tg":tg,"acceptance":acc,"outputs":outputs}


def exact_pair(a,b):
    return bool(a) and bool(b) and len(set(a))==1 and len(set(b))==1 and a[0]==b[0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--frspec", required=True)
    ap.add_argument("--min-median-tg-gain", type=float, default=3.0)
    ap.add_argument("--max-workload-tg-loss", type=float, default=2.0)
    ap.add_argument("--max-acceptance-drop-pp", type=float, default=1.5)
    ap.add_argument("--max-median-pp-loss", type=float, default=3.0)
    args=ap.parse_args()

    data=json.loads(Path(args.result).read_text(encoding="utf-8"))
    b=collect(data,args.baseline); f=collect(data,args.frspec)

    tg_delta={w:pct(f["tg"].get(w),b["tg"].get(w)) for w in WORKLOADS}
    pp_delta={k:pct(f["pp"].get(k),b["pp"].get(k)) for k in sorted(set(b["pp"])|set(f["pp"]),key=lambda x:int(x))}
    acc_delta={}
    exact={}
    for w in WORKLOADS:
        ba,fa=b["acceptance"].get(w),f["acceptance"].get(w)
        acc_delta[w]=None if ba is None or fa is None else (fa-ba)*100.0
        exact[w]=exact_pair(b["outputs"].get(w,[]),f["outputs"].get(w,[]))

    tgv=[x for x in tg_delta.values() if isinstance(x,(int,float))]
    ppv=[x for x in pp_delta.values() if isinstance(x,(int,float))]
    av=[x for x in acc_delta.values() if isinstance(x,(int,float))]
    tg_med=med(tgv); tg_worst=min(tgv) if tgv else None
    pp_med=med(ppv); acc_worst=min(av) if av else None

    pass_exact=all(exact.values())
    pass_gain=tg_med is not None and tg_med >= args.min_median_tg_gain
    pass_tg=tg_worst is None or tg_worst >= -args.max_workload_tg_loss
    pass_pp=pp_med is None or pp_med >= -args.max_median_pp_loss
    pass_acc=acc_worst is not None and acc_worst >= -args.max_acceptance_drop_pp
    result="PASS" if all((pass_exact,pass_gain,pass_tg,pass_pp,pass_acc)) else "FAIL"

    report={
        "baseline":args.baseline,"frspec":args.frspec,
        "baseline_metrics":{k:v for k,v in b.items() if k!="outputs"},
        "frspec_metrics":{k:v for k,v in f.items() if k!="outputs"},
        "tg_delta_pct":tg_delta,"pp_delta_pct":pp_delta,
        "acceptance_delta_percentage_points":acc_delta,
        "bit_exact":exact,
        "gate":{
            "result":result,
            "median_tg_gain_pct":tg_med,
            "worst_tg_delta_pct":tg_worst,
            "median_pp_delta_pct":pp_med,
            "worst_acceptance_delta_pp":acc_worst,
            "min_median_tg_gain_pct":args.min_median_tg_gain,
            "max_workload_tg_loss_pct":args.max_workload_tg_loss,
            "max_acceptance_drop_pp":args.max_acceptance_drop_pp,
            "max_median_pp_loss_pct":args.max_median_pp_loss,
        },
    }

    print("TG / acceptance / exactness")
    for w in WORKLOADS:
        print(f"  {w:>5}: TG {b['tg'].get(w)} -> {f['tg'].get(w)} ({tg_delta[w]}%)  "
              f"acc {b['acceptance'].get(w)} -> {f['acceptance'].get(w)} "
              f"({acc_delta[w]} pp) exact={exact[w]}")
    print("PP")
    for k in sorted(pp_delta,key=lambda x:int(x)):
        print(f"  {k:>6}: {b['pp'].get(k)} -> {f['pp'].get(k)} ({pp_delta[k]}%)")
    print("GATE")
    print(json.dumps(report["gate"],ensure_ascii=False,indent=2))

    out=Path(args.result).with_suffix(".frspec-analysis.json")
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"ANALYSIS={out}")
    raise SystemExit(0 if result=="PASS" else 2)

if __name__=="__main__":
    main()
