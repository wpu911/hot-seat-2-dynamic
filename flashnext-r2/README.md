# Flash Next R2 工作区

分支：`flashnext-r2-20260918`

目标：在不修改生产 `main`、不覆盖生产 runtime 的前提下，针对当前 `7900 XTX 24GB + R9700 32GB` 异构双卡继续优化 Qwen3.8 Flash Next。

## 当前已知生产基线

- 2026-09-11 官方 llama.cpp 底座：`b0dcb8192b201e402ec3eff524e55450f8070e3e`
- 当时自定义生产 HEAD：`dc2e27a19f322fb8958d02d1deadbe88243e1983`
- Flash Next 生产 runtime 后续修正版：`/app/share/llm/Qwen3.8-Flash-Next-GGUF/runtime-text/20260911-02/bin`
- 历史单 7900 XTX warm：约 19 tok/s
- 当前双卡现场：约 9.5–10.6 tok/s，需要先恢复到不低于历史单卡水平

## Stage 1 已准备

### 1. MMQ mul_mat_id J-cap

文件：

`patches/0001-mmq-id-jmax.patch`

来源：JohnTDI-cpu 的 RDNA4 Flash Next 实验 `e55_jcap.patch`。该补丁默认关闭，仅设置：

```bash
GGML_JOHNV8_MMQ_ID_JMAX=32
```

时改变 MoE `mul_mat_id` 的 J tile 选择。

公开实验中主要提升 PP，不能预设它会提升 TG。R2 中必须按 `off / 64 / 32 / 16` 做 A/B。

### 2. 独立 worktree / build 脚本

文件：

`scripts/prepare_stage1.sh`

脚本原则：

- 从当前生产源码 HEAD 创建 detached worktree；
- 不修改生产源码；
- 不修改 llama-swap；
- 不替换生产 runtime；
- 只在 R2 worktree 应用 J-cap；
- 默认尝试 `gfx1100;gfx1201` 多架构 HIP build；
- 优先使用 `/opt/host-rocm/core-10.0`；
- patch 冲突立即停止，不强行改源码。

### 3. HTTP A/B benchmark

文件：

`scripts/bench_http.py`

对已经启动的测试 llama-server 做一致的：

- PP 512 / 2048 / 8192；
- 中文 TG；
- 代码 TG；
- tool JSON TG；
- temperature=0；
- 不复用 prompt cache；
- JSON 记录每轮结果。

## Stage 1 正确执行顺序

```text
生产 HEAD 只读确认
→ 创建 R2 worktree
→ J-cap patch
→ gfx1100+gfx1201 编译
→ baseline JMAX=off
→ JMAX=64
→ JMAX=32
→ JMAX=16
→ bit-exact / greedy token 一致性
→ PP/TG A/B
→ 不提升则撤销，不污染下一阶段
```

## 下一阶段

J-cap 结论出来后再进入：

1. R9700/gfx1201 HyperConnection + GDN bit-exact fusion；
2. HIP Graph 独立 A/B；
3. MTP2 / MTP3 / MTP4 重扫；
4. QSA HIP TOP_K backend 检查；
5. long-context pooled-key / sparse QSA。

不一次叠多个优化，避免跑快了却不知道是谁干的，跑慢了更不知道该骂谁。
