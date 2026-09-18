# Flash Next R2 Stage 4：Q8_1 activation quantization dedup

## 目标

在保持 HC、GDN、JMAX 完全一致的前提下，只测试：

```text
GGML_JOHNV8_Q8_DEDUP=0 / 1
```

这项优化复用同一个 graph_compute 内相同输入的 Q8_1 activation quantization 结果，避免同一 activation 被多个 mmvq 路径重复量化。

上游记录的典型共享输入包括：

```text
beta | alpha
gate_inp | gate_inp_shexp
indexer q | k
attn_qkv | attn_gate
```

上游预期能减少每 token 数十次 Q8_1 quantize，属于小而干净的 launch/quantization 优化，不应期待它单独把 TG 翻倍。

## 固定上游补丁

```text
0009-E6d-pamiec-podreczna-kwantyzacji-Q8_1-aktywacji-dla-.patch
Git blob: 5da75737593bb93de3932407249eff999bbb3f6f
```

## 独立环境

源码：

```text
/app/share/llama_box/src/llama.cpp-flashnext-r2-q8dedup-20260918
```

runtime：

```text
/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-q8dedup
```

alias：

```text
qwen3.8-flash-next-r2-q8-off:256k
qwen3.8-flash-next-r2-q8-on:256k
```

二者使用同一个 binary，只切 Q8 dedup 环境变量。

默认固定：

```text
HC_MODE=1
GDN_MODE=1
JMAX=0
```

如果前面的 HC 或 GDN Gate 没通过，可在准备 Stage 4 时显式改为 0，使两边仍保持相同基础：

```bash
HC_MODE=0 GDN_MODE=1 bash flashnext-r2/scripts/prepare_stage4_q8dedup.sh
```

## 执行

```bash
bash flashnext-r2/scripts/prepare_stage4_q8dedup.sh
bash flashnext-r2/scripts/run_stage4_q8dedup_ab.sh
```

所有 benchmark 仍走 llama-swap `127.0.0.1:8090`。

## Gate

默认：

```text
输出完全一致
median TG gain >= 0.5%
任何 TG workload 回退 <= 1.5%
median PP 回退 <= 2%
```

由于预期收益较小，如果第一次结果处在噪声带，应增加 `TG`、`REPEAT` 后复测，而不是拿 0.3% 波动写成“史诗级优化”。

例如：

```bash
TG=512 REPEAT=5 bash flashnext-r2/scripts/run_stage4_q8dedup_ab.sh
```
