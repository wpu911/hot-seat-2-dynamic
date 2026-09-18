#!/usr/bin/env python3
"""Set -sm/--split-mode inside one llama-swap model cmd.

Edits only an experimental alias block, backs up the full config, and preserves
all other model arguments including --tensor-split, device order, MTP, HotSeat,
KV and context settings.
"""
from __future__ import annotations
import argparse, datetime as dt, os, re, shutil
from pathlib import Path

MODES=("none","layer","row","tensor")

def find_block(lines, alias):
    pat=re.compile(r"^(\s*)"+re.escape(alias)+r":\s*(?:#.*)?$")
    for i,line in enumerate(lines):
        m=pat.match(line.rstrip("\n"))
        if not m: continue
        ind=len(m.group(1)); j=i+1
        while j<len(lines):
            s=lines[j]
            if s.strip() and not s.lstrip().startswith("#") and len(s)-len(s.lstrip(" "))<=ind:
                break
            j+=1
        return i,j,ind
    raise RuntimeError(f"alias not found: {alias}")

def edit_cmd(block, model_indent, mode):
    lines=block.splitlines(True)
    cmd_i=None; tail=None
    rx=re.compile(rf"^(\s{{{model_indent+2}}})cmd:\s*(.*)$")
    for i,line in enumerate(lines):
        m=rx.match(line.rstrip("\n"))
        if m: cmd_i=i; tail=m.group(2); break
    if cmd_i is None: raise RuntimeError("model block has no cmd:")
    opt=re.compile(r"(?:(?<=\s)|^)(?:-sm|--split-mode)\s+(?:none|layer|row|tensor)\b")
    if tail and tail not in ("|",">","|-",">-"):
        raw=lines[cmd_i].rstrip("\n")
        hits=list(opt.finditer(raw))
        if len(hits)>1: raise RuntimeError("multiple split-mode options in inline cmd")
        raw=opt.sub(f"--split-mode {mode}",raw,count=1) if hits else raw+f" --split-mode {mode}"
        lines[cmd_i]=raw+"\n"
        return "".join(lines)
    end=cmd_i+1
    while end<len(lines):
        s=lines[end]
        if s.strip() and not s.lstrip().startswith("#") and len(s)-len(s.lstrip(" "))<=model_indent+2:
            break
        end+=1
    hits=[]
    for i in range(cmd_i+1,end):
        if opt.search(lines[i]): hits.append(i)
    if len(hits)>1: raise RuntimeError(f"multiple split-mode options in cmd lines {hits}")
    if hits:
        i=hits[0]; lines[i]=opt.sub(f"--split-mode {mode}",lines[i],count=1)
    else:
        lines.insert(end," "*(model_indent+4)+f"--split-mode {mode}\n")
    return "".join(lines)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="/app/share/llama_box/config/config-rocm714.yaml")
    ap.add_argument("--alias",required=True)
    ap.add_argument("--mode",choices=MODES,required=True)
    ap.add_argument("--backup-root",default="/app/share/backup")
    args=ap.parse_args()
    p=Path(args.config); lines=p.read_text(encoding="utf-8").splitlines(True)
    b0,b1,ind=find_block(lines,args.alias)
    old="".join(lines[b0:b1]); new=edit_cmd(old,ind,args.mode)
    if new==old: raise RuntimeError("split-mode edit made no change")
    stamp=dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    bd=Path(args.backup_root)/f"flashnext-r2-split-mode-before-{stamp}"
    bd.mkdir(parents=True,exist_ok=False); shutil.copy2(p,bd/p.name)
    lines[b0:b1]=[new]
    tmp=p.with_suffix(p.suffix+".splittmp"); tmp.write_text("".join(lines),encoding="utf-8"); os.replace(tmp,p)
    print(f"OK alias={args.alias}")
    print(f"OK split_mode={args.mode}")
    print(f"OK backup={bd}")
    return 0
if __name__=="__main__": raise SystemExit(main())
