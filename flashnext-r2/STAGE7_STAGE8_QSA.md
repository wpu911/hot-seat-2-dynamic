# Flash Next R2：Stage 7 / Stage 8 QSA 长上下文优化

日期：2026-09-18

## 结论先行

当前不再重复给生产树打 ROCm `TOP_K` 补丁。

`ggml-org/llama.cpp` PR #27466 `ROCm: add radix TOP_K for long rows` 已于 2026-08-31 合并；本项目生产基线是 2026-09-11，因此首先验证生产源码/构建确实包含该路径，再继续 QSA。

Stage 7 和 Stage 8 分别测试两条互补路线：

```text
Stage 7：QSA gather sparse attention
  解决：已经选出约 2K token，却仍对整个 KV context 做 attention

Stage 8：incremental pooled-key cache
  解决：每个 token、每个 QSA layer 都重新 gather/pool/norm/rope 全部历史 block summary
```

它们都主要针对长上下文 decode，不应该用 4K/8K 的短测结果决定去留。

---

## Stage 7A：确认 ROCm 长行 TOP_K

脚本：

```text
scripts/verify_stage7_rocm_topk.sh
```

执行：

```bash
bash flashnext-r2/scripts/verify_stage7_rocm_topk.sh
```

检查内容：

1. 生产源码 HEAD；
2. ggml-cuda 是否包含 radix TOP_K 实现；
3. 是否还残留旧的 `>1024` CPU fallback 限制；
4. 能找到 `test-backend-ops` 时执行 TOP_K backend tests；
5. 记录当前生产 llama-server 版本和 sha256。

如果 `SOURCE_RADIX_TOPK` 不能确认，先停，不把另一个 TOP_K patch 生硬叠进去。

---

## Stage 7B：QSA gather sparse attention

来源：

```text
ggml-org/llama.cpp PR #28213
qwen4exp : gather-based sparse attention for QSA decode
```

固定 head：

```text
beed2f78ac42cf16710b763e6f3ba20665c6d233
```

原理：

```text
原路径：
QSA indexer 选出 top-k
→ 转成 full-context mask
→ Flash Attention 仍扫描整个 KV cache

新路径：
QSA indexer 选出 top-k
→ gather 选中的 K/V
→ 生成约 2K token 的紧凑 attention 输入
→ dense FA 只处理选中集合
```

上游测试报告的趋势是上下文越深收益越大，短上下文提升小，130K 时提升最大。

### 独立 runtime

```text
/app/share/llama_box/src/llama.cpp-flashnext-r2-qsa-gather-20260918
/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-qsa-gather
```

### llama-swap aliases

```text
qwen3.8-flash-next-r2-qsa-off:256k
  QWEN4EXP_QSA_GATHER=0

qwen3.8-flash-next-r2-qsa-on:256k
  QWEN4EXP_QSA_GATHER=1
```

两个 alias 使用完全相同 binary。

### 准备

```bash
bash flashnext-r2/scripts/prepare_stage7_qsa_gather.sh
```

### 长上下文 A/B

```bash
bash flashnext-r2/scripts/run_stage7_qsa_gather_ab.sh
```

默认深度：

```text
4K
16K
32K
64K
128K
```

第一次只跑 OFF/ON 探索；如果有明显收益，再：

```bash
ROUNDS=4 bash flashnext-r2/scripts/run_stage7_qsa_gather_ab.sh
```

即：

```text
OFF → ON → OFF → ON
```

### 正确性

每个长上下文 prompt 在约 15% 位置埋入唯一 marker：

```text
QSA_VERIFY_20260918_7B_7F3C
```

要求生成结果仍能检索到 marker。

这是为了防止“速度快了，因为模型已经没认真看上下文”这种极具互联网精神的优化。

默认 Gate：

```text
32K 以上 TG 中位收益 >= 5%
任一深上下文 TG 回退不得超过 3%
PP 中位回退不得超过 5%
所有深度 marker 检索正确
```

结果：

```text
/app/share/openclaw_tools/logs/flashnext-r2-qsa-ladder.json
/app/share/openclaw_tools/logs/flashnext-r2-qsa-ladder.qsa-analysis.json
```

---

## Stage 8：QSA incremental pooled-key cache

来源：

```text
ggml-org/llama.cpp PR #28699
qwen4exp: incremental pooled-key cache for the QSA indexer
```

固定 head：

```text
141f3f5646aa15e88d53198610a7540f4f4b0d71
```

当前该 PR 仍为 draft，因此必须独立 runtime、同 binary A/B，不进入生产树直接叠加。

原理：

```text
原路径：
每 token
→ 每 QSA layer
→ 从整个 indexer cache gather block members
→ mean pool
→ norm
→ rope
→ score

新路径：
首次/新完成 block
→ 只计算新增 block summary
→ 写 pooled-key cache

后续 decode
→ 直接读取已经完成的 block summary
```

对于 layer split，多 GPU 情况尤其要避免把 pooled rows 全放在第一张卡，否则会把优化变成跨卡传输套餐。该 PR 已按 indexer buffer type 分设备分配 pooled buffers。

### 独立 runtime

```text
/app/share/llama_box/src/llama.cpp-flashnext-r2-qsa-pool-20260918
/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-qsa-pool
```

### llama-swap aliases

```text
qwen3.8-flash-next-r2-pool-off:256k
  LLAMA_QSA_NO_POOLED_CACHE=1

qwen3.8-flash-next-r2-pool-on:256k
  LLAMA_QSA_NO_POOLED_CACHE 必须完全不存在
```

因此 `install_llamaswap_r2_alias.py` 已增加：

```text
--unset-env KEY
```

用于真正删除 presence-based kill switch，而不是写成空字符串。对 `getenv()` 来说空字符串仍然“存在”，这类小坑很适合浪费半天。

### 准备

```bash
bash flashnext-r2/scripts/prepare_stage8_qsa_pooled_cache.sh
```

### A/B

```bash
bash flashnext-r2/scripts/run_stage8_qsa_pooled_ab.sh
```

同样使用 4K / 16K / 32K / 64K / 128K context ladder 和 needle correctness gate。

上游公开结果是在 63K / 114K 深度约 +9% decode，Prefill 基本不变；本机不得直接套用该数字。

---

## 后续组合规则

Stage 7 和 Stage 8 必须先独立通过。

只有两者独立 PASS 后，才建立组合树：

```text
production HEAD
  + QSA gather
  + pooled-key cache
```

再测试：

```text
QSA gather ON
pooled cache ON
```

组合后的目标不是简单把两个百分比相加。二者优化不同的 O(context) 部分，但会改变总瓶颈比例，所以收益一定不是小学加法。

之后才与前面的 winner 合并：

```text
JMAX（若通过）
+ HC fusion（若通过）
+ GDN fusion（若通过）
+ Q8 dedup（若通过）
+ HIP Graph（若通过）
+ 最优 MTP n-max
+ QSA gather（若通过）
+ pooled-key cache（若通过）
```

最后生成 clean production runtime，而不是把所有实验 worktree 直接当生产版本。
