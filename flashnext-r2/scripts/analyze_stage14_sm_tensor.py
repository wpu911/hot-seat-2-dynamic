#!/usr/bin/env python3
"""Analyze layer split vs tensor split on the same qwen4exp binary."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

W=("zh","code","tool")
def med(xs):
    xs=[x for x in xs if isinstance(x,(int,float))]; return statistics.median(xs) if xs else None
def pct(n,o):
    return None if not isinstance(n,(int,float)) or not isinstance(o,(int,float)) or o==0 else (n/o-1)*100
def collect(d,m):
    legs=[x for x in d.get("legs",[]) if x.get("model")==m]
    pp={}
    for k in sorted({k for l in legs for k in l.get("pp",{})},key=lambda x:int(x)):
        pp[k]=med([r.get("pp") for l in legs for r in l.get("pp",{}).get(k,[])])
    tg={}; outs={}; acc={}
    for w in W:
        rs=[r for l in legs for r in l.get("tg",[]) if r.get("workload")==w]
        tg[w]=med([r.get("tg") for r in rs]); outs[w]=[r.get("content","") for r in rs]
        a=[]
        for r in rs:
            dn,an=r.get("drafted"),r.get("accepted")
            if isinstance(dn,(int,float)) and dn>0 and isinstance(an,(int,float)): a.append(an/dn)
        acc[w]=med(a)
    return {"pp":pp,"tg":tg,"outputs":outs,"acceptance":acc,"legs":len(legs)}
def exact(a,b): return bool(a) and bool(b) and len(set(a))==1 and len(set(b))==1 and a[0]==b[0]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("result"); ap.add_argument("--layer",required=True); ap.add_argument("--tensor",required=True)
    ap.add_argument("--min-median-tg-gain",type=float,default=3.0); ap.add_argument("--max-workload-tg-loss",type=float,default=2.0)
    ap.add_argument("--max-median-pp-loss",type=float,default=5.0); ap.add_argument("--max-acceptance-drop-pp",type=float,default=2.0)
    a=ap.parse_args(); d=json.loads(Path(a.result).read_text(encoding="utf-8")); b=collect(d,a.layer); t=collect(d,a.tensor)
    td={w:pct(t["tg"].get(w),b["tg"].get(w)) for w in W}; pd={k:pct(t["pp"].get(k),b["pp"].get(k)) for k in sorted(set(b["pp"])|set(t["pp"]),key=lambda x:int(x))}
    ex={w:exact(b["outputs"].get(w,[]),t["outputs"].get(w,[])) for w in W}; ad={}
    for w in W:
        x,y=b["acceptance"].get(w),t["acceptance"].get(w); ad[w]=None if x is None or y is None else (y-x)*100
    tv=[x for x in td.values() if isinstance(x,(int,float))]; pv=[x for x in pd.values() if isinstance(x,(int,float))]; av=[x for x in ad.values() if isinstance(x,(int,float))]
    tm=med(tv); tw=min(tv) if tv else None; pm=med(pv); aw=min(av) if av else None
    checks={"exact":all(ex.values()),"tg_gain":tm is not None and tm>=a.min_median_tg_gain,"worst_tg":tw is not None and tw>=-a.max_workload_tg_loss,"pp":pm is not None and pm>=-a.max_median_pp_loss,"acceptance":aw is not None and aw>=-a.max_acceptance_drop_pp}
    result="PASS" if all(checks.values()) else "FAIL"
    report={"layer":a.layer,"tensor":a.tensor,"layer_metrics":{k:v for k,v in b.items() if k!="outputs"},"tensor_metrics":{k:v for k,v in t.items() if k!="outputs"},"tg_delta_pct":td,"pp_delta_pct":pd,"acceptance_delta_pp":ad,"bit_exact":ex,"checks":checks,"gate":{"result":result,"median_tg_gain_pct":tm,"worst_tg_delta_pct":tw,"median_pp_delta_pct":pm,"worst_acceptance_delta_pp":aw}}
    print(json.dumps(report["gate"],ensure_ascii=False,indent=2)); out=Path(a.result).with_suffix(".sm-tensor-analysis.json"); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(f"ANALYSIS={out}"); raise SystemExit(0 if result=="PASS" else 2)
if __name__=="__main__": main()
