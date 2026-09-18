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

两者之间已经隔了大量 upstream 变化。因此不再采用：

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

Stage 10 不是再从 Sep11 源码硬套 patch，而是在 **Modern Foundation** 上仅叠加 `base..head` 的 PR28243 delta。

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

## 3. Stage 14：ROCm TOP_K #28313

当前 AMD 长上下文 QSA 的 TOP_K 路径仍值得单独优化。PR #28313：

```text
ROCm: resolve TOP_K kernels
head: 93ceb53397b8885c55533bd680f6ff430418317e
changed files: 1
  ggml/src/ggml-cuda/top-k.cu
```

该 PR 对 HIP TOP_K 增加/重排了 small-case、n-ary 和专用选择路径。上游 microbenchmark 在多种 shape 上报告明显降低 kernel 时间，但是否能转化成这台 `gfx1100 + gfx1201` 的 Flash Next TG 提升，必须本机端到端验证。

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

测试：

```bash
bash flashnext-r2/scripts/run_stage14_rocm_topk_ab.sh
```

长上下文 ladder 默认：

```text
16K
32K
64K
128K
```

每个深度不仅测 TG，还必须通过 needle retrieval，避免 TOP_K 跑快了却选错 QSA cell。

Stage 14 有三个结论：

```text
PASS
  graphs ON 的正常生产路径有安全的实质提升

HIP_GRAPH_INTERACTION
  graphs OFF 明显变快，但 graphs ON 没吃到收益
  此时先查 HIP Graph，不直接否定 TOP_K kernel

FAIL
  无实质提升或出现检索 / acceptance / PP 回退
```

之所以同时测 graph ON/OFF，是因为 #28313 上游 benchmark 明确关闭了 HIP graphs，且提到 ROCm graph update 存在问题。不能拿 graph-off 微基准直接宣布生产提速。

另外，PR 讨论中曾争论 RDNA wave64，作者随后表示回退到安全 wave32 路线。当前实验只跟随 pin 的 PR head，不自行添加 wave64 魔改。

---

## 4. Stage 13：FR-Spec

新版 MTP 通过后，再缩 speculative draft vocabulary，不提前把 FR-Spec 和 MTP runtime 改动混在一起。

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

## 5. PLE Direct Read

Flash Next 的 PLE 大表读取仍作为 PP 专项优化保留。

```bash
bash flashnext-r2/scripts/prepare_stage11_lazy_direct.sh
bash flashnext-r2/scripts/run_stage11_lazy_direct_ab.sh
```

主要看：

```text
PP
page fault / I/O 行为
TG 不回退
```

如果当前生产模型路径显式依赖 `--no-mmap`，不得制造一个不成立的 on vs on-direct A/B。

---

## 6. QSA Gather + Incremental Pooled-Key Cache

### QSA gather

```bash
bash flashnext-r2/scripts/prepare_stage7_qsa_gather.sh
bash flashnext-r2/scripts/run_stage7_qsa_gather_ab.sh
```

### pooled-key incremental cache

```bash
bash flashnext-r2/scripts/prepare_stage8_qsa_pooled.sh
bash flashnext-r2/scripts/run_stage8_qsa_pooled_ab.sh
python3 flashnext-r2/scripts/bench_stage8_rollback_stress.py
```

最终组合时，Stage 14 TOP_K winner 应作为 QSA 基础层，再叠加 gather / pooled cache 重跑完整 ladder。

不能只看 4K/16K。真正关注：

```text
32K
64K
128K
cached branch
rollback
checkpoint/state restore
```

---

## 7. 最后才扫参数

代码路径确定后再扫：

```text
MTP n-max 2 / 3 / 4
JMAX 16 / 32 / 64
HIP Graph ON/OFF
Q8 activation dedup
GDN fusion
JohnTDI 特有 RDNA4 kernel（只在 gfx1201 路径上）
```

`gfx1201` 专用优化不能默认套给 `gfx1100`。双卡异构环境必须分别验证。

---

## 8. 明确不走的路线

```text
CUDA sparse FA #28770
  当前实现仍是 NVIDIA CUDA 路线，HIP 不直接套。

旧 n-gram #27992
  若 seq_pos_tok_le() 已在 modern foundation，则不重复移植。

多 stream fork/join
  以前 ROCm 实测同步代价过高，不进当前主线。

wave64 强开
  #28313 讨论已暴露维护/兼容风险，本轮只测安全的 pinned PR head。
```

---

## 9. 最终组合顺序

不是把所有 PASS 的 patch 一股脑堆上去。最终 clean runtime 应按：

```text
Modern Foundation
    ↓
MTP winner
    ↓
ROCm TOP_K winner
    ↓
FR-Spec winner（如果成立）
    ↓
PLE winner
    ↓
QSA gather
    ↓
pooled-key cache
    ↓
MTP/JMAX/Graph 参数 sweep
    ↓
short + 32K + 64K + 128K
+ cached Large-PP
+ rollback/checkpoint
+ OpenClaw real path regression
    ↓
最后才切生产 qwen3.8-flash-next:256k
```

任何单项如果只是微基准更快、但 llama-swap 真实路径无收益，就不进入最终 runtime。毕竟我们优化的是模型，不是 benchmark 截图收藏夹。
