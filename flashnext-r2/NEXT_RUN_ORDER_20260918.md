# Flash Next R2 当前执行顺序（更新至 2026-09-19）

> 当前原则：**不动生产、统一走 llama-swap :8090、先继承现代 upstream 已有优化，再只 A/B 仍然独立且有收益可能的变量。**

正式生产 alias 始终保持：

```text
qwen3.8-flash-next:256k
```

实验结束前不覆盖生产 runtime，不自动 promote。

---

## 0. 先冻结 exact production

生产源码包含有意保留的 HotSeat / Dynamic KV / cached Large-PP 等本地修改，不能用 `git worktree add HEAD` 冒充真实生产状态。

```bash
bash flashnext-r2/scripts/create_exact_prod_snapshot_repo.sh
```

稳定指针：

```text
/app/share/llama_box/src/llama.cpp-prod-exact-snapshots/current
```

---

## 1. Modern Foundation

路线：

```text
Sep11 official base b0dcb819...
+ exact production custom overlay
        ↓ forward-port
Sep18 official base 911f6cdc...
        ↓
Modern Foundation
```

```bash
bash flashnext-r2/scripts/prepare_modern_foundation.sh
bash flashnext-r2/scripts/verify_modern_foundation_carryover.sh
```

生成：

```text
source  /app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918
alias   qwen3.8-flash-next-r2-modern-foundation:256k
```

### 1.1 Native recurrent rollback 是硬门槛

Sep18 foundation 已经包含合并的 qwen4exp recurrent rollback（upstream #28123）。它不是普通微优化，而是 MTP 正确且高效工作的基础：

```text
LLM_ARCH_QWEN4EXP
→ llm_arch_supports_rs_rollback()
→ target n_rs_seq = draft.n_max
→ qwen4exp convolution state 每个 rollback slot 保留历史 plane
→ speculative reject/rollback 不再每轮序列化整个 recurrent state 到 host
```

每次 Modern Foundation 和最终组合都要验证：

```bash
SRC=/app/share/llama_box/src/llama.cpp-flashnext-modern-foundation-20260918 \
ALIAS=qwen3.8-flash-next-r2-modern-foundation:256k \
bash flashnext-r2/scripts/verify_qwen4exp_native_rs_rollback.sh
```

旧的“把 speculative checkpoint 搬到 GPU”的 #28118 路线**不采用**。原因不是它不快，而是 qwen4exp 已经有更正确的 native rollback，而且 #28118 仍有 multi-range hard abort / prompt-cache restore 风险。不要把已经消失的 checkpoint 再搬回 GPU，这属于优化界的还魂术。

---

## 2. Phase 1：MTP + ROCm TOP_K

统一入口：

```bash
bash flashnext-r2/scripts/run_phase1_real_ab.sh
```

顺序：

```text
exact snapshot
→ Modern Foundation
→ carry-over audit
→ native rollback audit
→ production vs foundation A/B
→ MTP #28243
→ cached Large-PP / high-LCP regression
→ ROCm TOP_K #28313 32K/64K smoke
```

MTP #28243 固定：

```text
base 911f6cdc8ab8a530b2bee09ee61471a6f3178eeb
head 53b1389d0bf98fa367e2a0ce0475008e762ebf28
12 commits / 19 files
```

TOP_K 只在真实 graphs-ON 路径有收益时保留。smoke PASS 后才跑：

```bash
TOPK_FULL=1 bash flashnext-r2/scripts/run_phase1_real_ab.sh
```

确认 16K / 32K / 64K / 128K。

---

## 3. Phase 2：长上下文 + PLE + FR-Spec

```bash
bash flashnext-r2/scripts/run_phase2_real_ab.sh
```

顺序：

```text
TOP_K full decision
→ QSA Gather #28213
→ Incremental pooled-key cache #28699
→ rollback/checkpoint stress
→ PLE direct-read #29030
→ FR-Spec
```

QSA Gather 同 binary 只切：

```text
QWEN4EXP_QSA_GATHER=0/1
```

先 32K / 64K smoke，通过后才完整 4K / 16K / 32K / 64K / 128K。

Pooled cache 同 binary 只切：

```text
OFF  LLAMA_QSA_NO_POOLED_CACHE=1
ON   unset LLAMA_QSA_NO_POOLED_CACHE
```

性能通过后仍必须跑 64K rollback stress，不能因为快了几个百分点就把缓存一致性献祭掉。

PLE direct-read 使用真实高多样性 OpenClaw session 窗口，AB/BA 交替，不再用重复句子伪造 PP：

```text
512 / 2048 / 8192 token
unique 4-gram ratio >= 0.70
结果只保留指标/hash，不保存 session 正文
```

---

## 4. Phase 3：只组合 winner

```bash
bash flashnext-r2/scripts/prepare_phase3_final_candidate.sh
bash flashnext-r2/scripts/run_phase3_final_validation.sh
```

组合规则：

```text
Modern MTP
+ TOP_K             仅 full PASS
+ QSA Gather        仅 full PASS
+ pooled cache      仅 full + rollback PASS
+ PLE direct-read   仅 PASS
+ FR-Spec           仅 PASS
```

生成：

```text
source  /app/share/llama_box/src/llama.cpp-flashnext-r2-final-pre-sweep-20260919
alias   qwen3.8-flash-next-r2-final-pre-sweep:256k
```

最终验证固定四关：

```text
短上下文 PP/TG + greedy exact
4K / 32K / 64K / 128K retrieval + TG
16K cached Large-PP / high-LCP
64K recurrent rollback / MTP deterministic prefix
```

Gate 0 会再次运行 native rollback audit，防止中间 semantic merge 把 #28123 的关键逻辑抹掉。

---

## 5. Phase 4：只扫仍然有意义的参数

```bash
bash flashnext-r2/scripts/run_phase4_mtp_graph_sweep.sh
```

现在只扫：

```text
MTP n-max 2 / 3 / 4
HIP Graph ON / OFF
```

每个 MTP winner 必须重新过 cached Large-PP。

### 已退休的旧参数

不再扫：

```text
手工 JMAX 16 / 32 / 64
旧 GGML_JOHNV8_Q8_DEDUP
```

原因：Modern Foundation 已有 upstream `ncols_opt` RDNA3/RDNA4 per-expert tile heuristic，以及 broadcast activation quantize/scatter dedup。重复再打一层只会让代码更有收藏价值，不会让 token 更快。

---

## 6. Phase 5：仅剩的 RDNA4/GDN microfusion

现代 upstream 已经有 qwen4exp HC fused ops 和 fused GDN，所以不再整包移植 JohnTDI stack。

只隔离测试仍然独立的：

```text
E7   GDN prolog
     sigmoid(beta)
     softplus(alpha + dt) * A
     收进 GDN kernel

E7b  q/k L2 norm 收进 GDN kernel

E7b2 FMA accumulation selector
```

准备：

```bash
bash flashnext-r2/scripts/prepare_phase5_gdn_microfusion.sh
bash flashnext-r2/scripts/run_phase5_gdn_microfusion_ab.sh
```

三个相同 binary alias：

```text
GDN OFF
PROLOG only
PROLOG + L2
```

先普通 PP/TG exact A/B。候选若胜出，还必须再过：

```text
16K cached Large-PP / high-LCP
64K rollback stress
```

`gfx1201` 的数值兼容修改继续保留，不能把针对 R9700 验过的 intrinsic 行为盲目套给 gfx1100。

---

## 7. Modern Foundation 已经有，不再重复搬的东西

```text
qwen4exp native recurrent rollback #28123
seq_pos_tok_le() predecessor position index
indexer V-cache suppression
qwen4exp modern HC fusion
modern fused GDN control/path
RDNA3/RDNA4 MoE ncols_opt
broadcast Q8 quantize/scatter dedup
```

因此以下旧路线退休：

```text
旧 #27977 整包
旧 n-gram #27992
#28330 重复移植
旧 JMAX patch 主线
旧 Q8 dedup patch 主线
#28118 on-device full checkpoint for qwen4exp
```

明确暂不走：

```text
CUDA sparse FA #28770
  NVIDIA 路线，不直接套 HIP。

-sm tensor #28569
  没有针对当前 gfx1100 + gfx1201 HIP 异构双卡完成验证，不混入当前主线。

多 stream fork/join
  过去 ROCm 同步代价过高。

wave64 强开
  不拿兼容性换 benchmark 截图。
```

---

## 8. 最终生产切换条件

只有最终候选同时满足：

```text
short TG 不回退
PP 不回退到不可接受范围
32K / 64K / 128K 长上下文收益成立
MTP acceptance 正常
native recurrent rollback 确认启用
cached Large-PP 不再出现 0.1 t/s 类退化
rollback / seq_rm / prefix reuse 正常
OpenClaw 实际 session 路径正常
```

才允许替换：

```text
qwen3.8-flash-next:256k
```

在此之前所有实验 alias 都只是实验品，生产继续睡它自己的觉。