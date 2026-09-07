#!/usr/bin/env python3
import argparse, hashlib, json, pathlib, statistics, time, urllib.request
p=argparse.ArgumentParser()
p.add_argument("--label",required=True)
p.add_argument("--port",type=int,default=5814)
p.add_argument("--base",type=int,default=32768)
p.add_argument("--suffix",type=int,default=8192)
p.add_argument("--repeats",type=int,default=2)
p.add_argument("--out",type=int,default=128)
a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent / "runs"
root.mkdir(exist_ok=True)
url=f"http://127.0.0.1:{a.port}"
def post(path,payload):
    req=urllib.request.Request(url+path,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=60))
def memory():
    for path in pathlib.Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
        total=int(path.read_text())
        if total>20*1024**3:
            used=int(path.with_name("mem_info_vram_used").read_text())
            return {"free_mib":(total-used)/1024**2,"used_mib":used/1024**2}
def complete(name,tokens,n,cache=True):
    slots=json.load(urllib.request.urlopen(url+"/slots",timeout=5))
    if any(s["is_processing"] for s in slots): raise RuntimeError("Lab server busy")
    payload={"prompt":tokens,"n_predict":n,"temperature":0,"seed":42,"cache_prompt":cache,"return_tokens":True,"stream":True,"ignore_eos":True}
    req=urllib.request.Request(url+"/completion",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    start=time.perf_counter(); arrivals=[]; output=[]; content=[]; final=None
    with urllib.request.urlopen(req,timeout=900) as response:
        for raw in response:
            if not raw.startswith(b"data: "): continue
            raw=raw[6:].strip()
            if raw==b"[DONE]":continue
            d=json.loads(raw)
            if "error" in d: raise RuntimeError(d["error"])
            if d.get("stop"):
                final=d
                continue
            new=d.get("tokens",[])
            if new:
                output.extend(new); arrivals.append(time.perf_counter()-start)
            content.append(d.get("content",""))
    elapsed=time.perf_counter()-start
    if final is None or not arrivals or len(output)!=n: raise RuntimeError(f"Incomplete response: {name}, output={len(output)}")
    itl=[y-x for x,y in zip(arrivals,arrivals[1:])]
    result={"label":a.label,"case":name,"input_tokens":len(tokens),"output_tokens":len(output),"prompt_sha256":hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),"ttft_s":arrivals[0],"wall_s":elapsed,"itl_mean_ms":statistics.mean(itl)*1000 if itl else None,"timings":final.get("timings"),"tokens_cached":final.get("tokens_cached"),"memory_after":memory(),"output_sha256":hashlib.sha256(json.dumps(output).encode()).hexdigest()}
    with (root/(a.label+".jsonl")).open("a") as f:f.write(json.dumps(result)+"\n")
    (root/(a.label+"-"+name+"-response.json")).write_text(json.dumps({"tokens":output,"content":"".join(content),"final":final},ensure_ascii=False))
    print(json.dumps(result),flush=True)
    return output
corpus=("本地推理性能报告：CPU 与 GPU 同时处理被路由选中的专家。缓存命中、内存带宽和数据传输需要分别计时。"
        "def route(tokens, experts): return [(token, experts[token % len(experts)]) for token in tokens]\n")*1200
pool=post("/tokenize",{"content":corpus,"add_special":False})["tokens"]
assert len(pool)>a.base+a.suffix+512
# Persist the exact token workload once for baseline/candidate reuse.
workload=pathlib.Path(__file__).resolve().parent / "workload.json"
if workload.exists():
    saved=json.loads(workload.read_text());pool=saved["tokens"]
else:workload.write_text(json.dumps({"tokens":pool}))
for rep in range(a.repeats):
    prefix=pool[:a.base]
    warm=complete(f"r{rep}-seed",prefix,8,False)
    # Reuse all evaluated output tokens; last sampled token has not been evaluated.
    history=prefix+warm[:-1]
    history+=pool[a.base:a.base+a.suffix]
    out=complete(f"r{rep}-large",history,a.out)
    history+=out[:-1]
    out=complete(f"r{rep}-tiny",history+pool[a.base+a.suffix:a.base+a.suffix+1],a.out)
(root/(a.label+".done")).write_text("ok\n")
