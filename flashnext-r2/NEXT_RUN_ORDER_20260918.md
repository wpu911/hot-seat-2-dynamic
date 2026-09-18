# Flash Next R2 当前执行顺序（2026-09-18）

> 本文件取代早期“Sep11 + 单独挑几个 patch”的路线。现在的主线是：**先把生产自定义层完整前移到 Sep18 upstream foundation，再在同一个现代基线上逐项 A/B。**

---

## 0. 不动生产

正式 alias 继续保持：

```text
qwen3.8-flash-next:256k
```

正式 runtime 在所有实验完成前不覆盖。

所有实验通过 llama-swap `127.0.0.1:8090` 跑，模型切换只使用单模型 unload，不使用全局 unload。

生产源码存在有意保留的 HotSeat / Dynamic KV / cached Large-PP 等本地修改，因此实验基线必须先冻结 exact production snapshot，禁止直接拿 live tree 的 `HEAD` 冒充生产。

---

## 1. 先建立 Modern Foundation

生产官方底座记录为：

```text
Sep11 upstream = b0dcb8192b201e402ec3eff524e55450f8070e3e
```

PR #28243 当前依赖的 Sep18 upstream base：

```text
911f6cdc8ab8a530b2bee09ee61471a6f3178eeb
```

不再采用：

```text
Sep11 production
+ HC patch
+ MTP patch
+ 若干零散 commit
```

而改成：

```text
Sep11 official base
        +
exact production custom overlay
        ↓ 提取用户自定义层

Sep18 official base 911f6cdc
        +
同一份 exact production custom overlay
        ↓
Modern Foundation
```

准备：

```bash
bash flashnext-r2/scripts/create_exact_prod_snapshot_repo.sh
bash flashnext-r2/scripts/prepare_modern_foundation.sh
```

生成：

```text
source:
/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918

runtime:
/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/r2-modern-foundation

alias:
qwen3.8-flash-next-r2-modern-foundation:256k
```

A/B：

```bash
bash flashnext-r2/scripts/run_modern_foundation_ab.sh
```

Gate：

```text
production Sep11 custom engine
vs
Sep18 upstream + exact same production custom overlay
```

只有 Modern Foundation 本身通过后，后续 Stage 才有意义。

### 1.1 先做 carry-over audit，禁止重复搬旧优化

```bash
bash flashnext-r2/scripts/verify_modern_foundation_carryover.sh
```

已经核对 Sep18 foundation 中存在：

```text
seq_pos_tok_le()
  → PLE predecessor 通过 per-sequence position index 直接查
  → 不再需要旧 #27977 的全 cache × 256 seq 扫描

qwen4exp block pooling
  → 已用小 r 的 slice + add

qwen4exp indexer head reduction
  → 已用 strided slice + add
  → 不再需要旧 #27977 的 transpose + sum_rows 改法

indexer KV cache
  → 已把 indexer cache 伪装成 MLA 形状从而不分配无用 V cache
  → #28330 的核心优化已经存在
```

因此旧 #27977 / #28330 **不得整包再次移植**。其中 QSA gather 仍是独立变量，继续走 Stage 7。

---

## 2. Stage 10：新版 Qwen3.8 Flash Next MTP #28243

当前 pin：

```text
PR:   ggml-org/llama.cpp#28243
base: 911f6cdc8ab8a530b2bee09ee61471a6f3178eeb
head: 53b1389d0bf98fa367e2a0ce0475008e762ebf28
commits: 12
files: 19
```

Stage 10 在 **Modern Foundation** 上仅叠加 `base..head` 的 PR28243 delta。

准备：

```bash
bash flashnext-r2/scripts/inspect_stage10_mtp_layout.sh
bash flashnext-r2/scripts/prepare_stage10_upstream_mtp.sh
```

生成：

```text
baseline:
qwen3.8-flash-next-r2-modern-foundation:256k

candidate:
qwen3.8-flash-next-r2-modern-mtp:256k
```

测试：

```bash
bash flashnext-r2/scripts/run_stage10_mtp_ab.sh
```

必须同时通过：

```text
普通 PP/TG A/B
MTP acceptance
确定性输出
cached Large-PP / high-LCP 分叉回归
MTP catch-up / rollback
```

特别保留 PR28243 的 memory-sharing 正确性修复：Qwen4Exp draft 可借 target embedding / LM head，但不共享 target KV / recurrent memory。不能因为存在 `ctx_other` 就跳过 draft catch-up / rollback。

---

## 3. Stage 14：ROCm TOP_K #28313，先 smoke 再决定是否烧 128K

PR #28313：

```text
ROCm: resolve TOP_K kernels
head: 93ceb53397b8885c55533bd680f6ff430418317e
changed files: 1
  ggml/src/ggml-cuda/top-k.cu
```

它对 HIP TOP_K 增加/重排了 small-case、n-ary 和 radix 路径。但 Flash Next QSA 的典型 `k≈2048` multi-row shape 并不会吃到所有 small-k 微基准的巨大增幅，因此先做低成本端到端 smoke，不拿上游 microbenchmark 给自己的机器开支票。

准备：

```bash
bash flashnext-r2/scripts/prepare_stage14_rocm_topk.sh
```

Stage 14 会注册四个 alias：

```text
HIP graphs ON:
qwen3.8-flash-next-r2-modern-mtp:256k
qwen3.8-flash-next-r2-rocm-topk:256k

HIP graphs OFF:
qwen3.8-flash-next-r2-topk-base-nograph:256k
qwen3.8-flash-next-r2-topk-nograph:256k
```

默认先跑 smoke：

```bash
bash flashnext-r2/scripts/run_stage14_rocm_topk_ab.sh
```

优先只看：

```text
32K
64K
```

若正常 graphs-ON 路径有稳定实质收益，再跑完整确认：

```bash
FULL=1 bash flashnext-r2/scripts/run_stage14_rocm_topk_ab.sh
```

完整 ladder：

```text
16K
32K
64K
128K
```

每个深度不仅测 TG，还必须通过 needle retrieval，避免 TOP_K 跑快了却选错 QSA cell。

Stage 14 三种结论：

```text
PASS
  graphs ON 的正常生产路径有安全的实质提升

HIP_GRAPH_INTERACTION
  graphs OFF 明显变快，但 graphs ON 没吃到收益
  先查 HIP Graph，不直接否定 TOP_K kernel

FAIL
  无实质提升或出现检索 / acceptance / PP 回退
```

#28313 上游 benchmark 明确关闭 HIP graphs，并提到 ROCm graph update 问题，因此 graph-off 微基准不能直接代表生产收益。

PR 讨论中还出现过 RDNA wave64 路线争议，作者后来回退安全 wave32。本轮只测 pinned PR head，不额外强开 wave64。

---

## 4. Stage 7：QSA Gather #28213

这条目前是长上下文 TG 的高优先级候选。

默认基线：

```text
Modern Foundation
+ PR28243 MTP
```

如果 Stage 14 TOP_K 已确认胜出，则把 Stage14 source + alias 一起作为 Stage7 base，不能只换其中一个。

准备：

```bash
bash flashnext-r2/scripts/prepare_stage7_qsa_gather.sh
```

同一 binary，仅切换：

```text
QWEN4EXP_QSA_GATHER=0
QWEN4EXP_QSA_GATHER=1
```

默认先跑 32K / 64K smoke：

```bash
bash flashnext-r2/scripts/run_stage7_qsa_gather_ab.sh
```

有收益后再跑：

```bash
FULL=1 bash flashnext-r2/scripts/run_stage7_qsa_gather_ab.sh
```

完整确认覆盖短上下文和 128K；每个深度必须 needle retrieval 正确，而且 TG 测试必须实际生成要求数量的 token，不能让十几个 token 的 early-EOS 冒充 TG128。

---

## 5. Stage 8：Incremental Pooled-Key Cache #28699

只在已经通过的 QSA gather 树上增加 pooled cache：

```text
Modern Foundation
+ MTP
+ QSA gather
+ pooled-key cache   <- 唯一新变量
```

准备与测试：

```bash
bash flashnext-r2/scripts/prepare_stage8_qsa_pooled.sh
bash flashnext-r2/scripts/run_stage8_qsa_pooled_ab.sh
bash flashnext-r2/scripts/run_stage8_rollback_stress.sh
```

同一 binary，QSA gather 两边都开；只有：

```text
OFF: LLAMA_QSA_NO_POOLED_CACHE=1
ON : unset LLAMA_QSA_NO_POOLED_CACHE
```

这里性能 PASS 还不够。必须额外过：

```text
MTP draft / acceptance
seq_rm rollback
cached prefix
checkpoint/state restore
长输出 deterministic prefix
needle retrieval
```

这一步恰好踩在以前最爱出妖怪的缓存/回滚交界处，所以不允许“跑得快就算了”。

---

## 6. Stage 11：PLE Direct Read #29030，使用真实高多样性语料

Stage 11 已经改为建立在 **Modern-MTP** 上，不再回到 Sep11 runtime：

```text
BASE_SRC:
/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-mtp-20260918

candidate:
/app/share/llama_box/src/llama.cpp-flashnext-r2-modern-lazy-direct-20260918
```

准备：

```bash
bash flashnext-r2/scripts/prepare_stage11_lazy_direct.sh
```

两个 alias 使用字节相同的真实 `llama-server` ELF，只由 wrapper 强制：

```text
qwen3.8-flash-next-r2-modern-lazy-mmap:256k
  --lazy-mode on

qwen3.8-flash-next-r2-modern-lazy-direct:256k
  --lazy-mode on-direct
```

若 source alias 显式含 `--no-mmap`，prepare 直接停止，不制造假 A/B。

测试：

```bash
bash flashnext-r2/scripts/run_stage11_lazy_direct_ab.sh
```

**不能再使用重复同一句话的 PP prompt。** #29030 优化的正是巨大 PLE 表的随机行读取，重复文本会反复命中少量 n-gram/PLE rows，可能把 mmap 的 page-fault 成本隐藏掉。

Stage11 专用 benchmark 现在会：

```text
读取最近本地 OpenClaw session 文本
→ 只在 localhost 上 tokenizer
→ 切出互不重叠的 512 / 2048 / 8192 token 窗口
→ 检查 unique 4-gram ratio，默认最低 0.70
→ 低多样性直接拒测
→ mmap/direct 交替 A/B
→ 结果只写指标与 hash，不写 session 文本/文件名
```

本地文本不足时只按有界 chunk 补合成语料，不再把 token deficit 错当“行数”一次生成数万行。

Gate：

```text
median PP gain >= 5%
任一 PP 回退不得超过 3%
TG 回退不得超过 2%
TG sanity 至少实际生成 32 token
同 binary 两种 lazy mode 输出一致
语料多样性达标
```

---

## 7. Stage 13：FR-Spec

新版 MTP 通过后，再缩 speculative draft vocabulary，不提前把 FR-Spec 与其他 kernel / I/O 变量揉成一锅。

准备：

```bash
bash flashnext-r2/scripts/prepare_stage13_frspec_modern.sh
```

比较：

```text
qwen3.8-flash-next-r2-modern-frspec-full:256k
vs
qwen3.8-flash-next-r2-modern-frspec-65k:256k
```

测试：

```bash
bash flashnext-r2/scripts/run_stage13_frspec_modern_ab.sh
```

重点：

```text
TG
MTP acceptance
输出一致性
sidecar RAM/VRAM
cached Large-PP
```

如果 65K vocabulary 造成 acceptance 明显下降，不因为单次 TG 好看就保留。

---

## 8. 最后才扫参数和 RDNA4 专项

代码路径确定后再扫：

```text
MTP n-max 2 / 3 / 4
JMAX 16 / 32 / 64
HIP Graph ON/OFF
Q8 activation dedup
GDN fusion
JohnTDI 特有 RDNA4 kernel（只在 gfx1201 路径上）
```

`gfx1201` 专用优化不能默认套给 `gfx1100`。当前双卡是异构 GPU，必须分别验证，不能因为两块卡都姓 AMD 就假装它们共享童年。

---

## 9. 明确不走 / 不重复的路线

```text
CUDA sparse FA #28770
  NVIDIA CUDA 路线，不直接套 HIP。

旧 #27977 整包
  predecessor/indexer-head 等核心优化已经进入 Sep18 foundation 的后续实现；
  只保留仍独立的 QSA gather Stage7。

#28330 indexer V-cache fix
  Sep18 foundation 已有等价实现，不重复移植。

旧 n-gram #27992
  foundation 已有 seq_pos_tok_le() per-sequence position index，不重复移植。

多 stream fork/join
  以前 ROCm 实测同步代价过高，不进当前主线。

wave64 强开
  #28313 讨论已暴露 HIP 维护/兼容风险，本轮只测安全 pinned head。
```

---

## 10. 最终组合顺序

不是把所有 PASS patch 一股脑堆起来。最终 clean runtime：

```text
Modern Foundation
    ↓
MTP winner
    ↓
ROCm TOP_K winner（只有 Stage14 真 PASS 才保留）
    ↓
QSA gather
    ↓
pooled-key cache
    ↓
PLE direct-read winner
    ↓
FR-Spec winner（如果成立）
    ↓
MTP/JMAX/Graph 参数 sweep
    ↓
gfx1201 专项 kernel A/B
    ↓
short + 32K + 64K + 128K
+ cached Large-PP
+ rollback/checkpoint
+ OpenClaw real path regression
    ↓
最后才切生产 qwen3.8-flash-next:256k
```

任何单项如果只是微基准更快、但 llama-swap 真实路径无收益，就不进入最终 runtime。毕竟我们优化的是模型，不是 benchmark 截图收藏夹。
