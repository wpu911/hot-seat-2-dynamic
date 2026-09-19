# Flash Next R2：ChatGPT Work 中断恢复入口

本文件只解决一个问题：**Work 会话/额度中断后，从真实机器现状继续，不重跑已经通过的阶段，也不靠聊天记忆猜进度。**

生产 alias 始终是：

```text
qwen3.8-flash-next:256k
```

任何恢复流程都不得自动 promote。

## 1. 每次 Work 接手后的第一条命令

先拉取 R2 分支，然后只读生成状态报告：

```bash
git pull
python3 flashnext-r2/scripts/report_r2_state.py
```

报告会扫描 Modern Foundation、Phase-1 至 Phase-6b 的 manifest/summary、关键 llama-swap alias，并只输出一个保守的：

```text
NEXT_ACTION=...
```

不要凭“我记得上次跑过”继续。聊天记忆不是构建系统，谢天谢地。

## 2. Work 在 Modern Foundation 后中断

优先使用：

```bash
bash flashnext-r2/scripts/resume_phase1_after_work.sh
```

它会验证 Foundation runtime bundle、gfx1100+gfx1201、HotSeat/Modern carry-over、native recurrent rollback，并重新验证旧 A/B JSON 是否符合当前 schema；旧数据不可信则重跑 Foundation A/B，然后继续 MTP #28243、cached Large-PP 和 ROCm TOP_K smoke。

不会重新覆盖生产 runtime，也不会自动 promote。

## 3. Phase-2 / Phase-3

Phase-1 有有效 MTP/TOP_K 决策后：

```bash
bash flashnext-r2/scripts/run_phase2_real_ab.sh
```

Phase-2 完整结束后：

```bash
bash flashnext-r2/scripts/prepare_phase3_final_candidate.sh
bash flashnext-r2/scripts/run_phase3_final_validation.sh
```

Phase-3 validation 在大规模 benchmark 前先把实验 runtime 的 build-tree RUNPATH 归一到 `$ORIGIN`，随后硬检查：

```text
llama-server / llama-server.real
RUNPATH/RPATH
ldd
禁止依赖临时 build-r2-* 目录
gfx1100 + gfx1201 双卡可见
native recurrent rollback
```

然后才跑 short / 4K-128K / cached Large-PP / rollback。

## 4. Phase-4：MTP depth + HIP Graph

入口：

```bash
bash flashnext-r2/scripts/run_phase4_mtp_graph_sweep.sh
```

现在实际扫描：

```text
MTP n-max 1 / 2 / 3 / 4
HIP Graph ON / OFF
```

MTP 不再用三组两两 A/B，而是同一个真实 runtime、四个 llama-swap alias 做镜像顺序：

```text
1 -> 2 -> 3 -> 4 -> 4 -> 3 -> 2 -> 1
```

每个 leg 都重新加载、warm-up 后再测速，固定 TG512，并要求每个样本都有 `draft_n / draft_n_accepted`。输出会再 tokenize，默认保护前 128 个生成 token：四个 depth 必须与 n-max=2 anchor 的保护前缀一致。出现坏 arm 时只淘汰那个 arm，不让它把整个 sweep 一锅端掉。

选择规则不是“0.2% 也算赢”。非 anchor 默认至少比 n-max=2 的 median TG 快 1%，否则继续保留 n-max=2。外部机器有人测到 n-max=1 最好，也有人测到 3/4 最好，所以这台 gfx1100 + gfx1201 异构双卡自己测，别替别人继承甜点参数。

HIP Graph 仍然必须 A/B。当前 ROCm graph exec update 上游修复尚未落地，不能把 Graph ON 当成天然更快。

## 5. Phase-4b：参数 winner 深水区复验

Phase-4 只负责选参数，**不允许直接把 winner 送进源码级 GDN 实验**。先跑：

```bash
bash flashnext-r2/scripts/run_phase4b_param_validation.sh
```

固定三关：

```text
32K / 64K / 128K retrieval + TG
16K cached Large-PP / high-LCP
64K recurrent rollback + MTP
```

如果 Phase-4 参数 winner 在任何一关失败，不中断整个 R2 流程，而是明确记录：

```text
PARAM_ACCEPTED=NO
PHASE4B_WINNER_ALIAS=qwen3.8-flash-next-r2-final-pre-sweep:256k
```

也就是退回已经通过 Phase-3 的安全基线。`VALIDATION=PASS` 在这种情况下表示“安全 fallback 已确定”，不是“参数 candidate 通过”，判断 candidate 要看 `PARAM_ACCEPTED`。这种字段区分虽然少了点浪漫，但能防止以后的人类把 fallback 当冠军。

## 6. Phase-5：GDN microfusion

Phase-5 必须从 Phase-4b 的 winner 准备：

```bash
bash flashnext-r2/scripts/prepare_phase5_from_phase4b.sh
bash flashnext-r2/scripts/run_phase5_verified.sh
```

不要再直接从旧 Phase-4 summary 调 `prepare_phase5_gdn_microfusion.sh`。

`run_phase5_verified.sh` 会先：

```text
normalize staged runtime RUNPATH
检查共享库解析
禁止临时 build 目录依赖
确认 gfx1100 + gfx1201
```

然后才进入 GDN OFF / PROLOG / PROLOG+L2 A/B、cached Large-PP 和 rollback。

## 7. Phase-6：双卡 split 是重点

准备：

```bash
bash flashnext-r2/scripts/prepare_phase6_tensor_split.sh
```

测速使用：

```bash
bash flashnext-r2/scripts/run_phase6_verified.sh
```

入口现在会先归一 layer / tensor 两套 staged runtime 的 RUNPATH，再确认：

```text
layer / tensor wrapper 的 real ELF SHA256 完全一致
layer 强制 --split-mode layer
tensor 强制 --split-mode tensor --tensor-split 1,1 --fit off
两套 staged runtime 不依赖临时 build 目录
gfx1100 + gfx1201 均可见
```

然后才执行 short exact A/B、4K/32K/64K/128K、cached Large-PP、64K recurrent rollback。

`analyze_exact_ab.py` 要求 benchmark schema v3、固定 TG、完整 `predicted_n`，并要求每一个 TG sample 都存在 MTP `draft_n / draft_n_accepted`。缺 acceptance 不再算过。

## 8. Phase-6b：24G + 32G 异构比例

仅当 Phase-6 winner 是 `TENSOR_1x1` 才准备：

```bash
bash flashnext-r2/scripts/prepare_phase6b_tensor_ratio_sweep.sh
```

测速：

```bash
bash flashnext-r2/scripts/run_phase6b_verified.sh
```

verified 入口会先归一 EVEN/MID/CAP 三套 staged runtime，再验证三者使用同一个 real ELF，并且实际 ratio 与 `phase6b-ratios.env` 完全一致。

## 9. 当前 rollback / cached gate 的硬要求

最新版本统一为：

```text
ignore_eos=true
必须生成完整 requested tokens
必须看到 MTP draft_n / draft_n_accepted
cached branch 的 MTP acceptance 不得明显恶化
rollback stress 每一个 leg 都必须实际走 MTP
needle / deterministic protected prefix 必须正确
```

因此，旧的“没有 acceptance 但 TG 看起来挺快”结果全部视为不可信。快得连推测解码有没有工作都不知道，那不叫优化，叫计时器有想法。

## 10. 绝对不要自动做的事

```text
不要覆盖 qwen3.8-flash-next:256k
不要删除原 production runtime
不要把 upstream/NVIDIA/别人的 ROCm benchmark 当本机成绩
不要在缺少 MTP acceptance 时宣称某候选更快
不要拿提前 EOS 的十几个 token 冒充 TG512
不要因为实验 alias 显示 ready 就视为通过
不要因为 Work 会话中断而重新从 Phase-0 编译所有东西
```

所有最终判断只认：

```text
7900 XTX gfx1100 + R9700 gfx1201
真实 llama-swap :8090
本机 A/B
fixed-length TG
MTP acceptance
cached prefix
rollback
OpenClaw 实际 session
```

## 11. Work 最简接管指令

> 先更新 `flashnext-r2-20260918` 分支，然后运行 `python3 flashnext-r2/scripts/report_r2_state.py`。严格根据它输出的 `NEXT_ACTION` 从现有真实机器状态继续。所有 benchmark 统一走 llama-swap `127.0.0.1:8090`。不得自动替换生产 alias `qwen3.8-flash-next:256k`。Phase-4 后必须跑 `run_phase4b_param_validation.sh`，Phase-5 必须通过 `prepare_phase5_from_phase4b.sh` 接收验证后的 winner。Phase-5、Phase-6、Phase-6b 测速分别使用 `run_phase5_verified.sh`、`run_phase6_verified.sh`、`run_phase6b_verified.sh`。遇到 FAIL 时保留结果并退回上一安全 winner，不要为了让 gate 变绿而放宽正确性条件。
