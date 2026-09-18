# Flash Next R2 Stage 3：Gated DeltaNet fusion

## 目标

Stage 3 只验证 Qwen3.8 Flash Next 的 Gated DeltaNet 两类融合：

1. `GDN_PROLOG`：把 `sigmoid(beta)` 与 `softplus(alpha + dt) * A` 收进 gated-delta-net kernel；
2. `GDN_L2`：把 q/k 的 L2 normalize 收进同一个 GDN kernel。

Stage 3 不把 J-cap、Q8 dedup、HIP Graph 等新因素同时加入实验。

## 上游依据

固定来源：

```text
JohnTDI-cpu/llama.cpp-flash-next-rdna4
commit/tree: 185252d1edb27fde6b332908eb7c89a20cadc4bb
```

使用补丁：

```text
0008 E7  GDN prolog
0012 E7b GDN L2
0016 E7b FMA selector
```

固定 Git blob：

```text
0008: 2717c0af96ea14612ccf85835b1933ed6277d77c
0012: d9f7ba405caedd67d9b57b8e1d23e997d3292212
0016: b7417dc4fb105ec0665562b8b37d1c55e0a9bd61
```

上游总结中，这几项单独收益不算巨大，但会减少每 token 的 kernel launch 数。对于双 GPU、MTP、launch-bound 路径，仍值得单独验证。

## HIP 数值兼容

上游后续发现：在 gfx1201 上，`__fmul_rn` / `__fadd_rn` 的结果可能和普通 `a*b` / `a+b` 不同。

因此 Stage 3 在应用 0008/0012/0016 后，只对 GDN 路径做最小数值兼容迁移：

```text
prolog add/mul -> 普通 +/*
L2 scale       -> 普通 *
L2 accumulation -> GDN_L2_FMA=1，继续用 fmaf 匹配 stock norm
```

不把 upstream 0017 的 shared-expert 等无关修改一起带进来。

## Stage 3 binary 与 alias

源码 worktree：

```text
/app/share/llama_box/src/llama.cpp-flashnext-r2-gdn-20260918
```

runtime：

```text
/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-gdn
```

同一个 binary 建两个 llama-swap alias：

```text
qwen3.8-flash-next-r2-gdn-off:256k
qwen3.8-flash-next-r2-gdn-on:256k
```

二者保持：

```text
同一 binary
同一模型
同一 MTP
同一 HotSeat
同一 Dynamic KV
同一双卡参数
同一 HC_MODE
同一 JMAX
```

只切：

```text
OFF:
GGML_JOHNV8_GDN_PROLOG=0
GGML_JOHNV8_GDN_L2=0
GGML_JOHNV8_GDN_L2_FMA=1

ON:
GGML_JOHNV8_GDN_PROLOG=1
GGML_JOHNV8_GDN_L2=1
GGML_JOHNV8_GDN_L2_FMA=1
```

默认 `HC_MODE=1`，表示在 Stage 2 HC 通过后继续叠加；如果 Stage 2 实测 HC 失败，可以执行前设置：

```bash
HC_MODE=0 bash flashnext-r2/scripts/prepare_stage3_gdn.sh
```

从而保持 GDN OFF/ON 两侧的 HC 状态仍完全一致。

## 执行

```bash
bash flashnext-r2/scripts/prepare_stage3_gdn.sh
bash flashnext-r2/scripts/run_stage3_gdn_ab.sh
```

所有请求仍走真实 llama-swap：

```text
127.0.0.1:8090
```

默认 A/B：

```text
GDN OFF
→ unload
→ GDN ON
→ unload
→ GDN OFF
→ unload
→ GDN ON
```

## Gate

Stage 3 默认要求：

```text
中文 / 代码 / tool JSON 输出完全一致
median TG gain >= 1.0%
任何 TG workload 回退 <= 1.5%
median PP 回退 <= 2.0%
无 compute error
无 HIP page fault
无 0.1 t/s CPU fallback
```

GDN 属于小步 kernel fusion，因此门槛比 HC 低，但不能因为“理论上 bit-exact”就跳过输出一致性检查。理论这种东西最喜欢在凌晨两点钟跟硬件分手。

## 结果

默认：

```text
/app/share/openclaw_tools/logs/flashnext-r2-gdn-ab.json
/app/share/openclaw_tools/logs/flashnext-r2-gdn-ab.gdn-analysis.json
```

通过后才进入 Stage 4：Q8_1 activation quantization dedup。
