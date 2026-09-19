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

旧的“把 speculative checkpoint 搬到 GPU”的 #28118 路线**不采用**。qwen4exp 已经有更正确的 native rollback，而且 #28118 仍有 multi-range hard abort / prompt-cache restore 风险。不要把已经消失的 checkpoint 再搬回 GPU，这属于优化界的还魂术。

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

截至 2026-09-19，#28243 仍为 open，因此继续固定 pin，不跟着远端分支漂移。

TOP_K smoke 后才决定是否跑 full：

```bash
TOPK_FULL=1 bash flashnext-r2/scripts/run_phase1_real_ab.sh
```

确认 16K / 32K / 64K / 128K。TOP_K 的 Graph ON/OFF 交互必须保留记录，因为 ROCm graph exec-update 上游修复目前仍未落地，不能把 Graph ON 当作天然优选。

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

正式 benchmark 前先做 runtime isolation：

```text
build-tree RUNPATH → $ORIGIN
ldd 无缺失
不得解析到临时 build-r2-* 目录
gfx1100 + gfx1201 均可见
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

## 5. Phase 4：MTP depth + HIP Graph

```bash
bash flashnext-r2/scripts/run_phase4_mtp_graph_sweep.sh
```

现在扫描：

```text
MTP n-max 1 / 2 / 3 / 4
HIP Graph ON / OFF
```

为什么补 n-max=1：较新的 qwen4exp MTP 外部测试显示，一些 interconnect-bound 多卡机器上 1-deep draft 反而最好；另一批机器则 3/4 更好。这个结果不能外推到本机，所以 gfx1100 + gfx1201 直接四档一起测。

MTP sweep 不再做三组 pairwise A/B，而是同 runtime 四 alias 镜像顺序：

```text
1 → 2 → 3 → 4 → 4 → 3 → 2 → 1
```

硬要求：

```text
fixed TG512
每个 sample 有 draft_n / draft_n_accepted
每个 arm warm-up 后才计时
输出重新 tokenize
默认前 128 generated tokens 与 n-max=2 anchor 一致
坏 arm 单独淘汰
非 anchor 至少 +1% median TG 才改 winner
```

MTP winner 还必须重新过 cached Large-PP，然后才进入 HIP Graph ON/OFF。

### 已退休的旧参数

不再扫：

```text
手工 JMAX 16 / 32 / 64
旧 GGML_JOHNV8_Q8_DEDUP
```

原因：Modern Foundation 已有 upstream `ncols_opt` RDNA3/RDNA4 per-expert tile heuristic，以及 broadcast activation quantize/scatter dedup。重复再打一层只会让代码更有收藏价值，不会让 token 更快。

---

## 6. Phase 4b：参数 winner 深水区复验

Phase 4 的 winner 不能直接喂给源码级 GDN 实验。先运行：

```bash
bash flashnext-r2/scripts/run_phase4b_param_validation.sh
```

固定三关：

```text
32K / 64K / 128K retrieval + TG
16K cached Large-PP / high-LCP
64K recurrent rollback + MTP
```

如果 Phase-4 参数 winner 任一关失败，则**自动回退实验基线**到已经通过 Phase-3 的：

```text
qwen3.8-flash-next-r2-final-pre-sweep:256k
```

记录：

```text
PARAM_ACCEPTED=NO
PHASE4B_WINNER_ALIAS=<safe fallback>
VALIDATION=PASS
```

这里 `VALIDATION=PASS` 表示已经得到安全可继续的 winner/fallback，不表示被拒绝的参数 arm 通过。要判断参数 arm 自己，读 `PARAM_ACCEPTED`。

---

## 7. Phase 5：仅剩的 GDN microfusion

现代 upstream 已经有 qwen4exp HC fused ops 和 fused GDN，所以不再整包移植 JohnTDI stack。

只隔离测试：

```text
E7   GDN prolog
E7b  q/k L2 norm 收进 GDN kernel
E7b2 FMA accumulation selector
```

入口改为：

```bash
bash flashnext-r2/scripts/prepare_phase5_from_phase4b.sh
bash flashnext-r2/scripts/run_phase5_verified.sh
```

`prepare_phase5_from_phase4b.sh` 只接受 Phase-4b 已确认的 winner/fallback。verified runner 会先归一实验 runtime RUNPATH、检查共享库和双卡，再做三个相同 binary alias：

```text
GDN OFF
PROLOG only
PROLOG + L2
```

候选若胜出，还必须过 16K cached Large-PP / high-LCP 与 64K rollback stress。

---

## 8. Phase 6：qwen4exp 双卡 Tensor Split #28569

当前异构双卡为 gfx1100 + gfx1201。此前 layer split 的 Flash Next 双卡吞吐没有恢复到历史单卡 18–19 t/s，因此把 upstream #28569 作为**独立实验变量**。

截至 2026-09-19，#28569 仍为 open，固定测试其当前 head，不把“有 PR”误写成“已经进 upstream”。它的核心变化是：

```text
允许 LLM_ARCH_QWEN4EXP 使用 --split-mode tensor
qwen4exp hc_init 后强制 ggml_build_forward_expand(gf, res_hc)
```

第一轮严格 1,1：

```bash
bash flashnext-r2/scripts/prepare_phase6_tensor_split.sh
bash flashnext-r2/scripts/run_phase6_verified.sh
```

verified 入口会先归一 layer/tensor runtime 的 RUNPATH，再确认同一个真实 ELF，仅 wrapper 区分：

```text
LAYER
  --split-mode layer

TENSOR 1,1
  --split-mode tensor
  --tensor-split 1,1
  --fit off
```

这里 `--fit off` 是硬要求。llama.cpp 当前没有为 SPLIT_MODE_TENSOR 实现 auto-fit。让一个“不支持”的自动功能替你管理两张不同容量显卡，属于典型的人类乐观主义。

Phase 6 四关：

```text
短上下文 exact + 非灾难性回退
4K / 32K / 64K / 128K retrieval + TG
16K cached Large-PP / high-LCP
64K recurrent rollback / MTP
```

只有 tensor 1,1 在深上下文 median TG 至少 +2% 且其余 gate 全过，才进入 Phase 6b。

### 8.1 Phase 6b：异构显存比例自动 sweep

因为 7900 XTX 与 R9700 容量不同，1,1 不是最终结论。Phase 6b 读取**候选 binary 自己的 `--list-devices` 输出**以及 alias 的实际 `--device` 顺序，不猜 ROCm0/ROCm1 谁是谁。

```bash
bash flashnext-r2/scripts/prepare_phase6b_tensor_ratio_sweep.sh
bash flashnext-r2/scripts/run_phase6b_verified.sh
```

对两张卡自动生成：

```text
EVEN  1,1
MID   等分与容量比例之间的中间比例
CAP   按总 VRAM 容量比例
```

若设备顺序是 24 GiB → 32 GiB，典型近似：

```text
EVEN  1,1
MID   13,15 左右
CAP   3,4 左右
```

若设备顺序反过来，比例自动反过来。使用**总 VRAM**而不是当时 free VRAM，因为 free VRAM 可能被其他已加载模型污染。

Phase 6b 先跑 MID/CAP 的 32K/64K smoke，选出正收益候选，再做：

```text
short exact A/B
32K / 64K / 128K 四轮确认
cached Large-PP
64K rollback
```

最终 ratio 若连 +1% 的重复长上下文 median TG 都站不住，就保留 1,1，不为统计噪声多养一套配置。

---

## 9. Modern Foundation 已经有，不再重复搬的东西

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

多 stream fork/join
  过去 ROCm 同步代价过高。

wave64 强开
  不拿兼容性换 benchmark 截图。
```

`-sm tensor #28569` 已移入 Phase 6，但仍是实验路线，绝不在 A/B 之前直接改生产。

---

## 10. 中断恢复

每次 Work/会话重新接手，只运行：

```bash
git pull
python3 flashnext-r2/scripts/report_r2_state.py
```

只认它输出的 `NEXT_ACTION`。当前状态机已经包含 Phase-4b，不会再出现“Phase-4 选了个短跑冠军，直接拿去叠下一层补丁”的跳关。

---

## 11. 最终生产切换条件

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
若 tensor split 胜出，其设备顺序和 tensor ratio 已明确记录
```

才允许替换：

```text
qwen3.8-flash-next:256k
```

在此之前所有实验 alias 都只是实验品，生产继续睡它自己的觉。
