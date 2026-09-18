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

### 2. 独立 worktree / build / runtime 脚本

文件：

`scripts/prepare_stage1.sh`

脚本原则：

- 从当前生产源码 HEAD 创建 detached worktree；
- 不修改生产源码；
- 不替换生产 runtime；
- 只在 R2 worktree 应用 J-cap；
- 默认使用 ROCm10 编译 `gfx1100;gfx1201`；
- 生成独立 runtime：`runtime-text/r2-stage1/bin`；
- 自动从当前生产 Flash Next 配置块克隆 R2 llama-swap alias；
- production alias 原样保留；
- patch 冲突或 ROCm10 不存在就立即停止。

### 3. llama-swap alias 克隆器

文件：

`scripts/install_llamaswap_r2_alias.py`

默认：

```text
qwen3.8-flash-next:256k
    ↓ 原样克隆所有参数
qwen3.8-flash-next-r2:256k
```

它不会重新手写模型参数，而是从当前生产块直接复制，因此会保留：

- 主模型路径；
- MTP draft 路径；
- MTP 参数；
- 双卡 / tensor split；
- Dynamic KV；
- Static / Borrow / Transit；
- HotSeat；
- Large-PP / SPEC gate；
- 当前所有后续生产修复。

R2 第一轮只改变：

```text
llama-server binary -> r2-stage1/bin
GGML_JOHNV8_MMQ_ID_JMAX=32
```

修改配置前自动备份到：

```text
/app/share/backup/flashnext-r2-alias-before-<timestamp>
```

并可直接执行 `llama-swap -validate`。

### 4. 真正通过 llama-swap 的 A/B benchmark

文件：

`scripts/bench_llamaswap_ab.py`

不再单独起测试 llama-server 端口。

完整路径就是生产路径：

```text
benchmark / OpenClaw style request
        ↓
llama-swap :8090
        ↓
production alias 或 R2 alias
        ↓
对应 llama-server
```

默认顺序：

```text
baseline
→ unload
→ R2
→ unload
→ baseline
→ unload
→ R2
```

这样避免只测一轮后被：

- page cache；
- RAM 热状态；
- GPU 温度；
- expert residency；
- 首次加载状态

冒充成优化收益。

测试内容：

- PP 512 / 2048 / 8192；
- 中文 TG；
- 代码 TG；
- tool JSON TG；
- temperature=0；
- 每个 workload 多轮；
- llama-swap `/running`；
- 可用时保存 `/api/performance`；
- 完整结果写 JSON。

llama-swap 卸载统一使用：

```text
POST /api/models/unload
```

不依赖私有测试端口。

## Stage 1 正确执行顺序

```text
生产 HEAD 只读确认
→ 创建 R2 worktree
→ J-cap patch
→ ROCm10 gfx1100+gfx1201 编译
→ 建独立 r2-stage1 runtime
→ 从当前生产配置块克隆 R2 alias
→ llama-swap -validate
→ llama-swap watch-config 自动读入
→ baseline / R2 / baseline / R2
→ bit-exact / greedy token 一致性
→ PP/TG A/B
→ 不提升则撤销，不污染下一阶段
```

首次完整执行：

```bash
bash flashnext-r2/scripts/prepare_stage1.sh
python3 flashnext-r2/scripts/bench_llamaswap_ab.py
```

## 结果门槛

J-cap 默认启用必须满足：

1. 输出一致；
2. 无 compute error / HIP page fault；
3. 不触发 0.1 tok/s CPU fallback；
4. PP 有稳定收益；
5. TG 不得出现可重复退化；
6. 至少 baseline/R2 各跑两轮。

如果 `JMAX=32` 只改善 PP、不改变 TG，也可以保留为 PP 优化，但不会把它算作双卡 TG 修复成果。

## 下一阶段

J-cap 结论出来后再进入：

1. R9700/gfx1201 HyperConnection + GDN bit-exact fusion；
2. HIP Graph 独立 A/B；
3. MTP2 / MTP3 / MTP4 重扫；
4. QSA HIP TOP_K backend 检查；
5. long-context pooled-key / sparse QSA。

不一次叠多个优化，避免跑快了却不知道是谁干的，跑慢了更不知道该骂谁。
