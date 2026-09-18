# Flash Next R2 下一轮执行顺序（2026-09-18）

## 0. 先锁死基线

生产源码允许存在未提交的 HotSeat 修改，因此以后所有需要重新编译的 Stage 必须通过：

```bash
bash flashnext-r2/scripts/with_exact_prod.sh <prepare-script>
```

`with_exact_prod.sh` 会先调用 `create_exact_prod_snapshot_repo.sh`，把当前生产 working tree（包括有意保留的 tracked/untracked HotSeat 源码修改）冻结成一个 clean git snapshot，再从该 snapshot 做实验。

**禁止直接拿生产目录的 `HEAD` 建 worktree 做 A/B。** 否则会悄悄漏掉本地 HotSeat 修改。

---

## 1. 零成本验证

先跑，不编译：

```bash
bash flashnext-r2/scripts/verify_stage7_rocm_topk.sh
bash flashnext-r2/scripts/verify_stage9_ngram_index.sh
bash flashnext-r2/scripts/inspect_stage10_mtp_layout.sh
```

目的：

- 确认 ROCm 长行 radix TOP_K 真正在生产源码中；
- 确认已经合并的 n-gram position index 存在，不重复移植旧 #27992；
- 确认当前 MTP draft GGUF 是否能被新版 #28243 直接加载。

---

## 2. 第一优先：Stage 12 官方 HC / RMSNorm 图优化

生产基线是 2026-09-11，官方以下两项在生产基线之后合并：

- #28896：qwen4exp rms_norm + mul fusion graph refactor；
- #28901：qwen4exp HC fused ops。

准备：

```bash
bash flashnext-r2/scripts/with_exact_prod.sh \
  flashnext-r2/scripts/prepare_stage12_upstream_hc.sh
```

测试：

```bash
bash flashnext-r2/scripts/run_stage12_upstream_hc_ab.sh
```

原因：它已经进入 upstream，改动比自定义 HC fork 更容易长期维护，而且同时有 PP/TG 收益潜力。

如果官方 Stage 12 通过，再与 Stage 2 JohnTDI HC 做直接比较，不把两套 HC 同时叠加。

---

## 3. 第二优先：Stage 10 新版 Qwen3.8 Flash Next MTP

前提：

1. Stage 12 已经完成并保留 HC-only alias：
   `qwen3.8-flash-next-r2-upstream-hc:256k`；
2. `inspect_stage10_mtp_layout.sh` 判定当前 draft 为兼容候选，或者已经重新生成兼容 #28243 的 draft GGUF。

准备：

```bash
bash flashnext-r2/scripts/with_exact_prod.sh \
  flashnext-r2/scripts/prepare_stage10_upstream_mtp.sh
```

Stage 10 的候选树不是“生产 + MTP”，而是：

```text
exact production
+ 与 Stage 12 完全相同的官方 HC commits
+ PR #28243 MTP delta
```

测试时比较：

```text
Stage 12 HC-only
vs
Stage 12 HC + PR #28243 MTP
```

这样 HC 不再是隐藏变量，真正只测 MTP。PR #28243 当前 pin：

```text
base 911f6cdc8ab8a530b2bee09ee61471a6f3178eeb
head 53b1389d0bf98fa367e2a0ce0475008e762ebf28
```

不要再从历史首个 parent 生成 compare patch。该分支后来 merge 了新版 master，从旧 parent 拉 patch 会把无关 upstream 更新一起塞进实验。

测试：

```bash
bash flashnext-r2/scripts/run_stage10_mtp_ab.sh
```

该脚本现在包含两层 Gate：

```text
普通 cache_prompt=false A/B
+
cached Large-PP / high-LCP branch regression
```

第二层专门防止 2026-09-11 出现过的：

```text
cached Large-PP
→ Borrow / Transit 驻留结构变化
→ MTP multi-row verification 大量落 CPU
→ TG 约 0.1 t/s
```

另外必须保留 PR #28243 中 `d1a92352` 的正确性修复：Qwen4Exp draft 可以通过 `ctx_other` 借 target embedding / LM head，但**不共享 target KV/recurrent memory**。只有 `gemma4-assistant` 走 memory-shared 判定。否则 draft catch-up / rollback 会被跳过，M-RoPE position 会出问题。

本 Stage：

- 不改 JMAX；
- 不改 `--spec-draft-n-max`；
- 不改 HotSeat env；
- baseline 与 candidate 使用相同官方 HC；
- 只比较现有 MTP 与 #28243 MTP 差异。

先确认新版 MTP 本体收益，再进入 MTP2/3/4 sweep。

---

## 4. 第三优先：Stage 11 PLE lazy direct read

#29030 针对 qwen4exp 巨型 PLE 表，把 mmap demand-fault 行读取改成：

```text
整批索引
→ 去重
→ 按文件位置排序
→ 多线程 positional read
→ staging
```

准备：

```bash
bash flashnext-r2/scripts/with_exact_prod.sh \
  flashnext-r2/scripts/prepare_stage11_lazy_direct.sh
```

测试：

```bash
bash flashnext-r2/scripts/run_stage11_lazy_direct_ab.sh
```

两个 alias 使用相同真实 ELF，只通过 wrapper 强制：

```text
--lazy-mode on
vs
--lazy-mode on-direct
```

主要观察 PP。TG 不允许明显回退。

如果生产 alias 显式使用 `--no-mmap`，prepare 会主动停止，不做伪 A/B。

---

## 5. 第四优先：Stage 7 / 8 长上下文 QSA

### Stage 7：QSA gather

```bash
bash flashnext-r2/scripts/with_exact_prod.sh \
  flashnext-r2/scripts/prepare_stage7_qsa_gather.sh

bash flashnext-r2/scripts/run_stage7_qsa_gather_ab.sh
```

### Stage 8：incremental pooled-key cache

```bash
bash flashnext-r2/scripts/with_exact_prod.sh \
  flashnext-r2/scripts/prepare_stage8_qsa_pooled.sh

bash flashnext-r2/scripts/run_stage8_qsa_pooled_ab.sh
python3 flashnext-r2/scripts/bench_stage8_rollback_stress.py
```

Stage 8 必须额外通过 MTP / rollback stress，不能只看吞吐。

---

## 6. 之后才做参数扫

只有前面的代码路径稳定后再扫：

```text
JMAX 16 / 32 / 64
MTP n-max 2 / 3 / 4
HIP Graph ON/OFF
Q8 activation dedup
GDN fusion
```

避免同时改变多个变量后再猜是谁提速。

---

## 7. 明确跳过的路线

### CUDA sparse FA #28770

当前实现源码明确：

```text
GGML_USE_HIP -> sparse FA return false / abort
```

因此现在不把 NVIDIA-only sparse FA 移植到 ROCm 生产实验。AMD 长上下文优先使用 Stage 7 QSA gather + Stage 8 pooled-key cache。

### 旧 n-gram #27992

若 Stage 9 确认 `seq_pos_tok_le()` 已存在，则不再移植旧 #27992。官方后续实现已把历史 token 查找改成 per-sequence position index。

### 多 stream fork/join

此前 ROCm 实测同步成本过高，不列入当前主线。

---

## 8. 最终合并原则

最终 production R2 不是“把所有 patch 堆一起”。正确过程：

```text
每个 Stage 独立 A/B
→ 通过 Gate
→ 检查彼此是否重叠/替代
→ 只保留 winner
→ 从 exact production snapshot 重建 clean combined tree
→ 重新跑短上下文 + 长上下文 + cached Large-PP + rollback/MTP 回归
→ 最后才切 qwen3.8-flash-next:256k 的 runtime
```

重点组合关系：

```text
官方 HC (#28896/#28901)  vs JohnTDI HC     二选一后再组合
新版 MTP (#28243)         在官方 HC 相同基线上与旧 MTP 比较
QSA gather + pooled cache                    可以按顺序叠加
lazy direct PLE                              与 TG kernel 优化正交，但需单独确认 PP 收益
```

生产 alias、生产 runtime 在所有实验完成前保持不变。
