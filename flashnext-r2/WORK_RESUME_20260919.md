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

Phase-3 validation 在大规模 benchmark 前会硬检查最终 runtime：

```text
llama-server / llama-server.real
RUNPATH/RPATH
ldd
禁止依赖临时 build-r2-* 目录
gfx1100 + gfx1201 双卡可见
native recurrent rollback
```

然后才跑 short / 4K-128K / cached Large-PP / rollback。

## 4. Phase-4 / Phase-5

Phase-4：

```bash
bash flashnext-r2/scripts/run_phase4_mtp_graph_sweep.sh
```

只扫：

```text
MTP n-max 2 / 3 / 4
HIP Graph ON / OFF
```

Phase-5 准备：

```bash
bash flashnext-r2/scripts/prepare_phase5_gdn_microfusion.sh
```

测速使用 runtime hard-audit 入口，不再直接跑裸 runner：

```bash
bash flashnext-r2/scripts/run_phase5_verified.sh
```

它会先验证 Phase-5 新编 binary 的共享库、临时 build 依赖和双卡可见性，再进入 GDN OFF / PROLOG / PROLOG+L2 A/B。

## 5. Phase-6：双卡 split 是重点

准备：

```bash
bash flashnext-r2/scripts/prepare_phase6_tensor_split.sh
```

测速使用：

```bash
bash flashnext-r2/scripts/run_phase6_verified.sh
```

它会先确认：

```text
layer / tensor wrapper 的 real ELF SHA256 完全一致
layer 强制 --split-mode layer
tensor 强制 --split-mode tensor --tensor-split 1,1 --fit off
两套 staged runtime 不依赖临时 build 目录
gfx1100 + gfx1201 均可见
```

然后才执行 short exact A/B、4K/32K/64K/128K、cached Large-PP、64K recurrent rollback。

`analyze_exact_ab.py` 当前要求 benchmark schema v3、固定 TG、完整 `predicted_n`，并要求每一个 TG sample 都存在 MTP `draft_n / draft_n_accepted`。缺 acceptance 不再算过。

## 6. Phase-6b：24G + 32G 异构比例

仅当 Phase-6 winner 是 `TENSOR_1x1` 才准备：

```bash
bash flashnext-r2/scripts/prepare_phase6b_tensor_ratio_sweep.sh
```

测速：

```bash
bash flashnext-r2/scripts/run_phase6b_verified.sh
```

它会验证 EVEN/MID/CAP 三个 wrapper 使用同一个 real ELF，并且实际 ratio 与 `phase6b-ratios.env` 完全一致。

## 7. 当前 rollback / cached gate 的硬要求

最新版本已经统一为：

```text
ignore_eos=true
必须生成完整 requested tokens
必须看到 MTP draft_n / draft_n_accepted
cached branch 的 MTP acceptance 不得明显恶化
rollback stress 每一个 leg 都必须实际走 MTP
needle / deterministic prefix 必须正确
```

因此，旧的“没有 acceptance 但 TG 看起来挺快”结果全部视为不可信。快得连推测解码有没有工作都不知道，那不叫优化，叫计时器有想法。

## 8. 绝对不要自动做的事

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

## 9. Work 最简接管指令

> 先更新 `flashnext-r2-20260918` 分支，然后运行 `python3 flashnext-r2/scripts/report_r2_state.py`。严格根据它输出的 `NEXT_ACTION` 从现有真实机器状态继续。所有 benchmark 统一走 llama-swap `127.0.0.1:8090`。不得自动替换生产 alias `qwen3.8-flash-next:256k`。Phase-5、Phase-6 和 Phase-6b 分别使用 `run_phase5_verified.sh`、`run_phase6_verified.sh` 和 `run_phase6b_verified.sh`。遇到 FAIL 时保留结果并退回上一 winner，不要为了让 gate 变绿而放宽正确性条件。
