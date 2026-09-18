# Flash Next R2 工作区

分支：`flashnext-r2-20260918`

目标：在不修改生产 `main`、不覆盖生产 runtime 的前提下，针对当前 `7900 XTX 24GB + R9700 32GB` 异构双卡继续优化 Qwen3.8 Flash Next。

## 当前已知生产基线

- 2026-09-11 官方 llama.cpp 底座：`b0dcb8192b201e402ec3eff524e55450f8070e3e`
- 当时自定义生产 HEAD：`dc2e27a19f322fb8958d02d1deadbe88243e1983`
- Flash Next 生产 runtime 后续修正版：`/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/20260911-02/bin`
- 历史单 7900 XTX warm：约 19 tok/s
- 当前双卡现场：约 9.5–10.6 tok/s，需要先恢复到不低于历史单卡水平

---

## Stage 1：MMQ `mul_mat_id` J-cap

### 目标

减少大 ubatch 下 MoE `mul_mat_id` 按整个 ubatch 选择过大 J tile 导致的空算。

来源：JohnTDI-cpu 的 RDNA4 Flash Next 实验 `e55_jcap.patch`。

核心开关：

```bash
GGML_JOHNV8_MMQ_ID_JMAX=32
```

公开实验显示它主要改善 PP，decode 基本不变，因此 Stage 1 不承担 TG 提速目标。

### 文件

```text
patches/0001-mmq-id-jmax.patch
scripts/prepare_stage1.sh
scripts/install_llamaswap_r2_alias.py
scripts/bench_llamaswap_ab.py
scripts/analyze_llamaswap_ab.py
```

### 运行方式

```bash
bash flashnext-r2/scripts/prepare_stage1.sh
python3 flashnext-r2/scripts/bench_llamaswap_ab.py
python3 flashnext-r2/scripts/analyze_llamaswap_ab.py \
  /app/share/openclaw_tools/logs/flashnext-r2-ab.json
```

### llama-swap 测试路径

不再单独起测试端口。所有 A/B 都走真实生产入口：

```text
client
  ↓
llama-swap :8090
  ↓
不同 alias
  ↓
对应 llama-server runtime
```

生产 alias 保持：

```text
qwen3.8-flash-next:256k
```

Stage 1 alias：

```text
qwen3.8-flash-next-r2:256k
```

基准默认采用：

```text
baseline → R2 → baseline → R2
```

并在切换时只卸载相关 Flash Next alias，不全局卸载其他模型。

---

## Stage 2：HyperConnection glue fusion

Stage 2 已准备，重点开始冲 TG。

公开 RDNA4 Flash Next 实验中，`hc-mix + scale_silu + hc-combine` 是收益最大的单类 kernel fusion 之一；文档报告无 MTP 时约 +9%，整套 fusion + HIP Graph 在双 GPU 场景约 +17%。这些数字只作为外部参考，不能直接当成本机结果。

### 原则

Stage 2 **不叠 Stage 1 J-cap**。

原因：

```text
Stage 1：主要改善 PP
Stage 2：目标是 TG
```

如果一起开，最后快了也分不清谁立功。人类已经发明了足够多不可复现实验，没必要再贡献一个。

### 上游补丁固定版本

来源仓库：

```text
JohnTDI-cpu/llama.cpp-flash-next-rdna4
```

固定 tree/commit：

```text
185252d1edb27fde6b332908eb7c89a20cadc4bb
```

补丁：

```text
johnv8/patches/seria-fuzje/
0004-fuzje-hc-combine-hc-mix-z-forka-c689018e4-f76552838.patch
```

固定 Git blob：

```text
17776143967955297e5a547e04cf209ed0d3d8e2
```

`prepare_stage2_hc.sh` 会下载该固定版本并用 `git hash-object` 校验，不接受上游悄悄变更后的同名文件。

### Stage 2 独立源码/runtime

默认：

```text
源码 worktree：
/app/share/llama_box/src/llama.cpp-flashnext-r2-hc-20260918

runtime：
/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-hc
```

从当前生产源码 HEAD 创建 detached worktree，不在生产树直接打 patch。

默认 ROCm10 多架构编译：

```text
gfx1100
gfx1201
```

### 同一 binary 做 OFF / ON

Stage 2 不用两个不同 build 做 A/B，而是**同一个 patched binary**创建两个 llama-swap alias，只切换环境变量：

```text
qwen3.8-flash-next-r2-hc-off:256k
  GGML_JOHNV8_HC_FUSE=0
  GGML_JOHNV8_MIX_FUSE=0
  GGML_JOHNV8_MMQ_ID_JMAX=0
```

对比：

```text
qwen3.8-flash-next-r2-hc-on:256k
  GGML_JOHNV8_HC_FUSE=1
  GGML_JOHNV8_MIX_FUSE=1
  GGML_JOHNV8_MMQ_ID_JMAX=0
```

这样避免把编译差异、J-cap 或生产 binary 差异误判成 HyperConnection 收益。

### Stage 2 文件

```text
scripts/prepare_stage2_hc.sh
scripts/run_stage2_hc_ab.sh
scripts/analyze_stage2_hc.py
```

`install_llamaswap_r2_alias.py` 也已扩展为支持多次：

```text
--env KEY=VALUE
```

因此后续 GDN、Q8 dedup、HIP Graph 等实验可以继续复用同一配置克隆器。

### Stage 2 执行

```bash
bash flashnext-r2/scripts/prepare_stage2_hc.sh
bash flashnext-r2/scripts/run_stage2_hc_ab.sh
```

完整测试仍走：

```text
127.0.0.1:8090
```

默认顺序：

```text
HC OFF
→ unload
→ HC ON
→ unload
→ HC OFF
→ unload
→ HC ON
```

### Stage 2 Gate

必须同时满足：

1. `temperature=0` 的中文、代码、tool JSON 输出完全一致；
2. 三类 TG 的中位提升默认至少 `+3%`；
3. 任一 TG workload 不得回退超过 `2%`；
4. PP 中位不得回退超过 `3%`；
5. 不出现 compute error、HIP page fault、0.1 t/s CPU fallback；
6. baseline/ON 各至少两轮。

结果写入：

```text
/app/share/openclaw_tools/logs/flashnext-r2-hc-ab.json
/app/share/openclaw_tools/logs/flashnext-r2-hc-ab.hc-analysis.json
```

---

## 后续顺序

Stage 2 出结果后再决定是否继续叠加：

```text
Stage 3
Gated DeltaNet prolog + L2 fusion

Stage 4
Q8_1 activation quantization dedup

Stage 5
HIP Graph 独立 A/B

Stage 6
MTP n-max = 2 / 3 / 4 重扫

Stage 7
QSA HIP TOP_K backend + context ladder

Stage 8
pooled-key incremental cache / sparse QSA
```

### 明确不做

目前不做：

```text
多 stream fork/join
```

上游 RDNA4 Flash Next 实验中，这条路线在 ROCm 7.2.4 反而让 decode 下降约 40%～50%，事件同步成本太高。除非 ROCm10 后续有充分证据改变结论，否则不浪费时间给 PCIe 和事件队列制造就业机会。

---

## 生产安全规则

任何阶段都遵守：

```text
生产源码不原地打 patch
生产 runtime 不覆盖
生产 alias 不修改
实验 alias 独立
通过 llama-swap :8090 做真实路径 A/B
先 bit-exact，再看速度
失败项不进入下一阶段组合
```

真正进入生产前，必须最终生成一个只包含“已通过 Gate 的优化项”的 clean runtime，再替换生产 alias 的 binary 路径，而不是把所有实验残骸一起端上桌。
