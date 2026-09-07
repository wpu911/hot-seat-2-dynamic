# Latest: Ornith PP/TG balance upgrade

2026-09-07: [Design, measurements and reproduction](docs/2026-09-07-ornith-pptg-balanced.md) | [中文技术帖](docs/xiaohongshu-ornith-pptg-20260907.md)

On the documented Ornith Q8_0 replay (32k cached + 8k new tokens), PP improves from 152.2 to 1176.2 tok/s while main-test TG stays approximately unchanged. This is a workload-specific PP result, not a universal generation speedup.

Use the new `ornith-pptg-balanced-from-bebc9350.patch` on its stated clean upstream base. Older patches and documentation below are historical snapshots; do not stack the full patches.

# llama.cpp / llama-swap HotSeat 升级迁移备忘录
## Ornith 1.5 HotSeat V2 + Qwen3.5-122B R1
生成日期：2026-08-23

> 目的：以后升级 llama.cpp、ROCm 或 llama-swap 时，明确知道现在生产版本是怎么做出来的、改了什么、编译了什么、哪些 profile 能复用、哪些必须重采。
>
> 这不是模型使用说明，而是“保命用迁移文档”。以后别再靠回忆猜当时改了哪几个文件，人类记忆不适合做版本控制。

---

# 1. 当前整体关系

现在并不是两套完全无关的补丁。

版本关系是：

```text
官方 llama.cpp b10235
commit 221f0f6356efe2260023208365705ec5d5a7c8f5
        │
        ├─ 旧 HotSeat / HotExpert 基础设施
        │  包括：
        │  - expert profile
        │  - resident expert cache
        │  - dynamic-hybrid
        │  - full-layer bank
        │  - shadow bank
        │  - arena
        │  - runtime migration
        │
        ↓
HotSeat V2 baseline snapshot
816fb44ca
        │
        ├─ V2 Stage0 profiler
        ├─ V2 Stage1 route history + async worker
        ├─ V2 Stage2 transit prefetch
        ├─ V2 Stage2.1 self-repeat predictor
        └─ V2 Stage2.4 persistent two-entry transit mini-cache
        ↓
Ornith 1.5 生产版
commit 9263152dc558b30ed2cec6e531f54d83e50688f1
tag    ornith-v2-prod-20260823
branch hotseat-v2-ornith
        │
        ├─ Q122 R0
        └─ Q122 R1
        ↓
Qwen3.5-122B 稳定版
commit 42da2948beedcac288274399f62f46de438d8bca
tag    q122-r1-stable-20260823
branch hotseat-r1-q122-stable
```

**重要结论：**

1. Ornith 使用的是通用 HotSeat V2。
2. Qwen3.5-122B R1 是在 Ornith/HotSeat V2 生产树上继续叠加的专用补丁。
3. 以后升级新版 llama.cpp，最合理的迁移顺序是：
   - 先迁通用 HotSeat/HotExpert
   - 再迁 HotSeat V2
   - 验证 Ornith
   - 再叠 Q122 R1
4. 不要把 Q122 R1 直接往干净新版 llama.cpp 上拍，它依赖前面的 HotExpert 基础。

---

# 2. 当前源码和二进制位置

## 2.1 Ornith / 通用 HotSeat V2

源码：

```text
/app/share/llama_box/src/llama.cpp-b10235-hotexpert-v2
```

宿主机对应：

```text
/home/victor/ai_share/llama_box/src/llama.cpp-b10235-hotexpert-v2
```

Git：

```text
branch: hotseat-v2-ornith
commit: 9263152dc558b30ed2cec6e531f54d83e50688f1
tag:    ornith-v2-prod-20260823
```

生产二进制：

```text
/app/share/llama_box/src/llama.cpp-b10235-hotexpert-v2/build-hip-rocm714/bin/llama-server
```

这个 V2 二进制当前可服务：

- Ornith 1.5 Text
- Ornith 1.5 Vision
- Qwen3.6 Text
- Qwen3.6 Vision

它们架构同属 qwen35moe，具体 profile / predictor 各自独立。

---

## 2.2 Qwen3.5-122B R1

源码：

```text
/app/share/llama_box/src/llama.cpp-b10235-hotexpert-v2-q122
```

宿主机对应：

```text
/home/victor/ai_share/llama_box/src/llama.cpp-b10235-hotexpert-v2-q122
```

Git：

```text
branch: hotseat-r1-q122-stable
commit: 42da2948beedcac288274399f62f46de438d8bca
tag:    q122-r1-stable-20260823
```

生产二进制：

```text
/app/share/llama_box/src/llama.cpp-b10235-hotexpert-v2-q122/build-hip-rocm714/bin/llama-server
```

Q122 不要和 Ornith V2 二进制混用。两者虽然有共同祖先，但 Q122 R1 修改了 CPU/GPU 图拆分和 MMID 行为。

---

# 3. 官方基线

当前 HotSeat 系列真正的官方干净基线是：

```text
llama.cpp tag: b10235
commit:
221f0f6356efe2260023208365705ec5d5a7c8f5
```

commit 标题：

```text
metal : add SILU_BACK (#25982)
```

第一个自定义迁移 commit：

```text
4ac031da7
hotseat: port layer placement and profiling to b10235
```

所以以后需要生成“从干净官方 b10235 到当前 Ornith 生产版”的完整补丁，可用：

```bash
git -C /home/victor/ai_share/llama_box/src/llama.cpp-b10235-hotexpert-v2 \
  diff --binary \
  221f0f6356efe2260023208365705ec5d5a7c8f5..9263152dc558b30ed2cec6e531f54d83e50688f1 \
  > ornith-hotseat-full-from-b10235-base.patch
```

当前实测这份完整 patch：

```text
约 379,318 bytes
23 个文件
约 8,393 行新增
```

---

# 4. Ornith / 通用 HotSeat 的完整改动范围

从官方 b10235 到 Ornith V2 生产版，累计涉及 23 个文件：

```text
A  ggml/include/ggml-hotexpert.h
M  ggml/src/CMakeLists.txt
M  ggml/src/ggml-cuda/common.cuh
M  ggml/src/ggml-cuda/ggml-cuda.cu
A  ggml/src/ggml-cuda/hotexpert-arena.cu
A  ggml/src/ggml-cuda/hotexpert-arena.cuh
A  ggml/src/ggml-cuda/hotexpert-cache.cu
A  ggml/src/ggml-cuda/hotexpert-cache.cuh
A  ggml/src/ggml-cuda/hotexpert-descriptor.cu
A  ggml/src/ggml-cuda/hotexpert-descriptor.cuh
A  ggml/src/ggml-cuda/hotexpert-planner.cu
A  ggml/src/ggml-cuda/hotexpert-planner.cuh
A  ggml/src/ggml-cuda/hotexpert-profiler.cu
A  ggml/src/ggml-cuda/hotexpert-profiler.cuh
A  ggml/src/ggml-cuda/hotexpert-runtime.cu
A  ggml/src/ggml-cuda/hotexpert-runtime.cuh
M  ggml/src/ggml-cuda/mmq.cu
M  ggml/src/ggml-cuda/mmvq.cu
M  ggml/src/ggml-cuda/topk-moe.cu
A  ggml/src/ggml-hotexpert.cpp
M  src/llama-graph.cpp
M  src/llama-model-loader.cpp
M  tools/server/server.cpp
```

累计 stat：

```text
23 files changed
8393 insertions(+)
6 deletions(-)
```

---

# 5. HotSeat / HotExpert 基础层做了什么

## 5.1 目标

RX 7900 XTX 只有 24GB VRAM，而 Q8 MoE 模型远大于显存。

普通 CPU+GPU offload 的核心问题不是算力，而是：

```text
token 路由到专家
→ 专家权重不在 GPU
→ 从系统内存经 PCIe 访问/搬运
→ decode 被 PCIe miss 拖死
```

HotSeat 的思路不是把整个模型硬塞显存，而是：

```text
系统 RAM = 完整权重 backing store
GPU VRAM = 热专家 / 热层工作集
```

让常用 expert 常驻 GPU，只让少量冷 expert 走 host。

---

## 5.2 Full Layer Bank

一个 Full Layer 表示：

```text
该 MoE 层的全部专家都在 GPU
```

Ornith 1.5：

```text
40 个 MoE 层
256 experts / layer
Top-8 experts / token
```

每个完整 MoE expert layer 权重约：

```text
855,638,016 bytes
≈ 816 MiB
```

当前 Ornith Text：

```text
HOTSEAT_FULL_LAYER_COUNT=4
HOTSEAT_RUNTIME_INIT_FULL_LAYERS=0,1,2,3
HOTSEAT_FULL_SHADOW=1
```

含义：

```text
4 个逻辑 Full Layer
+ 1 个 shadow full bank
```

Full Layer 不是永久固定 0/1/2/3。

在 `dynamic-hybrid` 下，运行时可根据统计把 Full 身份迁移给更值得整层常驻的 layer。

---

## 5.3 Resident Expert Slots

非 Full 层不需要把 256 个专家全部放 GPU。

Ornith Text 当前：

```text
HOTSEAT_EXPERT_SLOTS=95
```

逻辑上：

```text
4 个 Full layers
剩余 36 层，每层 95 个 resident experts
```

Vision 因 mmproj 额外吃显存，当前使用：

```text
HOTSEAT_EXPERT_SLOTS=90
```

resident slots 是基础长期热专家池。

---

## 5.4 Spare / Shadow Bank

运行时更新不能直接覆盖正在使用的 bank。

所以使用：

- Full Shadow bank
- Top-N spare/pool bank

典型迁移过程：

```text
1. 新内容先复制到 shadow/spare
2. 等 copy 完成
3. graph boundary 原子切换映射
4. 旧 bank 再轮换为下一次 shadow/spare
```

这样正常推理时不需要：

- unload model
- rebuild llama_context
- reset KV
- runtime hipMalloc/hipFree

---

## 5.5 Dynamic Hybrid

核心环境变量：

```text
HOTSEAT_RUNTIME_MODE=dynamic-hybrid
```

该模式下：

- 完整权重仍以 host backing 为基础
- runtime 管理 full layer banks
- runtime 管理 top-N expert banks
- full layer 成员可以动态变化
- resident expert 集合可以动态变化
- prefill 的部分 MMQ 路径可映射到 full banks

它不是简单的静态 Top-N cache。

---

## 5.6 Arena

相关：

```text
HOTSEAT_AUTO_RESERVE_ARENA=1
HOTSEAT_ARENA_APPLY_PROFILE=...
HOTSEAT_AUTO_PROFILE_DIR=...
```

Arena 是固定地址的额外 GPU expert 区域。

用途：

- 根据实际 VRAM low-water 自动估计还能安全占多少
- 增加额外 resident experts
- 支撑部分 full-layer promotion
- 可 retire 冷专家
- 避免运行中频繁分配/释放 VRAM

Vision 当前额外：

```text
HOTSEAT_ARENA_TARGET_FREE_MB=600
HOTSEAT_ARENA_JITTER_MB=64
```

因为 mmproj 会额外占显存。

**如果以后升级 ROCm、llama.cpp、mmproj、batch/ubatch 或上下文参数，Arena profile 最好重新采。**

---

# 6. HotSeat V2 具体增加了什么

HotSeat V2 是在旧 HotExpert 基础上增加“短期预测 + transient cache”。

## 6.1 V2 commit 链

```text
8db135dbf
v2-stage0-profiler: low-overhead per-token HotExpert profiler

77f325fe0
v2-stage1: mapped routing history with async worker

82a7eb354
v2-stage2: double-buffered opportunistic transit prefetch

faae89ba8
v2-stage2.1: confidence-gated self-repeat predictor

9263152dc
v2-stage2.4: persistent two-entry transit mini-cache
```

最终生产：

```text
9263152dc558b30ed2cec6e531f54d83e50688f1
tag ornith-v2-prod-20260823
```

---

## 6.2 仅 V2 阶段修改的文件

从 `816fb44ca` V2 baseline 到 `9263152dc`：

```text
M ggml/src/ggml-cuda/common.cuh
M ggml/src/ggml-cuda/hotexpert-runtime.cu
M ggml/src/ggml-cuda/hotexpert-runtime.cuh
M ggml/src/ggml-cuda/mmvq.cu
```

仅 V2 patch 大约：

```text
66,217 bytes
```

生成：

```bash
git -C /home/victor/ai_share/llama_box/src/llama.cpp-b10235-hotexpert-v2 \
  diff --binary 816fb44ca..9263152dc \
  > ornith-hotseat-v2.patch
```

---

# 7. V2 Stage0：路由 profiler

V2 profiler 记录真实 decode token 的 expert 路由。

临时启用：

```text
HOTSEAT_V2_PROFILE=1
HOTSEAT_V2_PROFILE_CAPACITY=16384
HOTSEAT_V2_PROFILE_DIR=/some/path
```

主要生成：

```text
v2-token-profile.jsonl
v2-layer-profile.jsonl
```

layer profile 会记录：

```text
token
layer
expert_ids[Top8]
resident_hits
host_misses
estimated bytes
MMVQ timing
```

这个 profiler 用来训练 predictor，不应该长期在生产模式一直开。

---

# 8. V2 Stage1：每 token 路由历史 + worker

每个 decode token 把各层的 Top-8 expert id 写入固定 ring。

后台 worker 读取已经完成的 token 路由记录。

目的：

```text
Token N 已经知道用了哪些专家
↓
为 Token N+1 预测哪些 cold expert 很可能再次出现
```

---

# 9. V2 Stage2：Transit Prefetch

环境变量：

```text
HOTSEAT_V2_PREFETCH=1
HOTSEAT_V2_TRANSIT=1
HOTSEAT_V2_TRANSIT_SLOTS=2
```

每个非 Full layer 有少量 transient slots。

当前：

```text
2 slots
```

它和 90/95 resident slots 不一样。

resident：

```text
长期热点
```

transit：

```text
预测下一 token 很可能马上要用的短期热点
```

---

# 10. V2 Predictor

当前 Ornith Text：

```text
/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat-v2/
ornith15-self-repeat-predictor-v2.json
```

当前 Ornith Vision：

```text
/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat-v2/
ornith15-vision-self-repeat-predictor-v2.json
```

阈值：

```text
HOTSEAT_V2_PREDICTOR_MIN_PPM=300000
```

即：

```text
self-repeat probability >= 30%
```

才认为某 cold expert 值得做 next-token transit prefetch。

predictor 不是通用模型文件。

**同架构模型也不能共用 predictor。**

例如：

- Ornith predictor 不能给 Qwen3.6
- Text predictor 不应直接替代 Vision predictor

因为 expert id 的语义和实际路由统计不同。

---

# 11. Predictor 的实际统计方法

当前 predictor 是 `self_repeat_ppm`。

对每层、每个 expert：

```text
条件：
Token N 该 expert 被路由选中

统计：
Token N+1 同一层是否再次出现这个 expert
```

最后：

```text
score_ppm =
repeat_count / occurrence_count * 1,000,000
```

输出结构类似：

```json
{
  "version": 2,
  "kind": "self_repeat_ppm",
  "layers": [
    {
      "layer": 0,
      "samples": 83800,
      "scores_ppm": [ ... 256 values ... ]
    }
  ]
}
```

---

# 12. V2 Stage2.4：Persistent Transit

环境变量：

```text
HOTSEAT_V2_PERSISTENT_TRANSIT=1
```

普通 Transit 如果下一 token 还要同一个 expert，可能再次拷贝。

Persistent Transit 会：

1. 检查两个 transit banks
2. 如果目标 expert 已存在，直接复用
3. 刷新 LRU / generation
4. 不重复 H2D copy

这是当前生产 commit 的最后一层关键优化。

---

# 13. Ornith 当前生产配置

## 13.1 Text

```yaml
ornith-1.5-35b:256k:
  ttl: 1200
  env:
    - "LD_LIBRARY_PATH=/app/share/llama_box/src/llama.cpp-b10235-hotexpert-v2/build-hip-rocm714/bin:/opt/rocm/lib"
    - "HIP_VISIBLE_DEVICES=0"
    - "HOTSEAT_TENSOR_LAYERS=0"
    - "HOTSEAT_EXPERT_SLOTS=95"
    - "HOTSEAT_LAYER_PLAN_FILE=/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat/ornith15-hotseat-native-4.json"
    - "HOTSEAT_RUNTIME_MODE=dynamic-hybrid"
    - "HOTSEAT_RUNTIME_PROFILE=/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat/ornith15-hotexpert-profile-native-12288.json"
    - "HOTSEAT_RUNTIME_INIT_FULL_LAYERS=0,1,2,3"
    - "HOTSEAT_FULL_LAYER_COUNT=4"
    - "HOTSEAT_FULL_SHADOW=1"
    - "HOTSEAT_AUTO_RESERVE_ARENA=1"
    - "HOTSEAT_ARENA_APPLY_PROFILE=/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat/arena-text-baseline.json"
    - "HOTSEAT_AUTO_PROFILE_DIR=/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat/arena-text"
    - "HOTSEAT_RUNTIME_LOG=/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat/swaps.jsonl"
    - "HOTSEAT_V2_PREFETCH=1"
    - "HOTSEAT_V2_TRANSIT=1"
    - "HOTSEAT_V2_TRANSIT_SLOTS=2"
    - "HOTSEAT_V2_PERSISTENT_TRANSIT=1"
    - "HOTSEAT_V2_PREDICTOR_FILE=/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat-v2/ornith15-self-repeat-predictor-v2.json"
    - "HOTSEAT_V2_PREDICTOR_MIN_PPM=300000"
  cmd: >
    /app/share/llama_box/src/llama.cpp-b10235-hotexpert-v2/build-hip-rocm714/bin/llama-server
    --host 127.0.0.1
    --port ${PORT}
    --jinja
    -m /app/share/llm/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-A3B-abliterated.Q8_0.gguf
    -c 262144
    -ngl 999
    -t 6
    -tb 16
    -np 1
    -b 2048
    -ub 1024
    --cache-ram 0
    --no-mmap
    --mlock
```

---

## 13.2 Vision

Vision 仍使用同一个 V2 二进制，但：

```text
HOTSEAT_EXPERT_SLOTS=90
```

并使用独立 predictor：

```text
ornith15-vision-self-repeat-predictor-v2.json
```

以及：

```text
arena-vision-baseline.json
HOTSEAT_ARENA_TARGET_FREE_MB=600
HOTSEAT_ARENA_JITTER_MB=64
```

mmproj：

```text
/app/share/llm/Ornith-1.5-35B-A3B-GGUF/
mmproj-Ornith-1.5-35B-BF16.gguf
```

---

# 14. Ornith 关键 profile 文件

最重要的保留文件：

```text
/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat/
ornith15-hotexpert-profile-native-12288.json

/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat/
ornith15-hotseat-native-4.json

/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat/
arena-text-baseline.json

/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat/
arena-vision-baseline.json

/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat-v2/
ornith15-self-repeat-predictor-v2.json

/app/share/llm/Ornith-1.5-35B-A3B-GGUF/hotseat-v2/
ornith15-vision-self-repeat-predictor-v2.json
```

同一个模型、同一 GGUF、同样 routing 语义时，expert profile / predictor 通常可以继续用。

但是以下情况建议重采：

- 模型 GGUF 换了
- expert tensor 排布变化
- llama.cpp 模型加载逻辑大改
- MoE graph 大改
- profiler fingerprint 不一致
- Vision mmproj 换了
- 实际业务负载明显改变

Arena profile 对 VRAM 布局更敏感，升级后更应该重采。

---

# 15. Qwen3.5-122B 为什么不能直接照搬 Ornith

Qwen3.5-122B Q8：

```text
48 MoE layers
256 experts/layer
Top-8
```

但单 expert 大得多。

Q122：

```text
单 expert gate+up+down ≈ 9.56 MiB
完整 expert layer ≈ 2.45 GiB
```

Ornith：

```text
单 expert ≈ 3.19 MiB
完整 expert layer ≈ 816 MiB
```

也就是说 Q122 专家约是 Ornith 的 3 倍。

24GB 显存无法像 Ornith 那样维护：

```text
4 Full + 36×95 resident
```

如果硬套，bank 本身就能把 VRAM 吃爆。

所以 Q122 最终采用另一条 R1 路线。

---

# 16. Q122 R1 的核心思想

Q122 R1 不是 Dynamic Hybrid Full+95。

它做的是：

```text
CPU miss branch
+
GPU resident branch
+
最后合并
```

目标：

```text
GPU 只算缓存命中的 resident experts
CPU 只算 miss experts
```

避免：

```text
CPU 把所有 experts 又完整算一遍
+
GPU 再重复算 resident experts
```

这是一个 CPU/GPU MMID split。

---

# 17. Q122 R1 commit 链

在 Ornith V2 `9263152dc` 上：

```text
27598c5d2
q122-r0: CPU MMID resident-skip upper-bound probe

42da2948b
q122-r1: CPU miss plus GPU resident split prototype
```

最终：

```text
42da2948beedcac288274399f62f46de438d8bca
tag q122-r1-stable-20260823
```

---

# 18. Q122 R1 专用改动文件

相对 Ornith V2：

```text
M ggml/src/ggml-backend.cpp
M ggml/src/ggml-cpu/ggml-cpu.c
A ggml/src/ggml-cpu/q122-r0-rank.inc
M ggml/src/ggml-cuda/ggml-cuda.cu
M ggml/src/ggml-cuda/hotexpert-cache.cu
M ggml/src/ggml-cuda/mmvq.cu
M src/llama-graph.cpp
```

约：

```text
33,967 bytes patch
310 insertions
16 deletions
```

生成：

```bash
git -C /home/victor/ai_share/llama_box/src/llama.cpp-b10235-hotexpert-v2-q122 \
  diff --binary \
  9263152dc558b30ed2cec6e531f54d83e50688f1..42da2948beedcac288274399f62f46de438d8bca \
  > q122-r1.patch
```

---

# 19. Q122 R1 各文件作用

## `src/llama-graph.cpp`

主要做 graph 级拆分：

```text
原始 mul_mat_id
↓
GPU resident MMID branch
+
CPU miss MMID branch
↓
add / merge
```

并且 R1 只在：

```text
decode
ubatch.n_tokens == 1
非 warmup
指定 split layers
```

启用。

Prefill / warmup 保留原路径，减少 PP 被实验性 split 拖慢的风险。

---

## `ggml/src/ggml-backend.cpp`

处理 backend scheduler。

目的：

- 普通 host tensor 冷路径继续强制 CPU
- 明确命名的 `q122_r1_gpu_*` node 才允许走 GPU backend

否则 ROCm_Host 可见权重很容易被 scheduler 误认为“GPU也能处理”，最后所有 cold MMID 又被自动 offload，R1 就失去意义。

---

## `ggml/src/ggml-cpu/ggml-cpu.c`

CPU MMID 支持“跳过 resident expert”。

R0/R1 用静态 rank：

```text
q122-r0-rank.inc
```

在 CPU 分支里：

```text
resident expert → 输出清零 / skip
cold miss       → CPU 正常计算
```

R1 模式下只对指定 split layer 生效。

---

## `ggml/src/ggml-cpu/q122-r0-rank.inc`

固化了：

```text
48 layers × ranked hot experts
```

这是从 Q122 profile 得到的长期热点排序。

它不是 llama.cpp 上游文件，是 Q122 专用数据。

如果模型文件变化、路由分布变化，应重新生成。

---

## `ggml/src/ggml-cuda/hotexpert-cache.cu`

R1 对 HotExpert cache 做了专用处理：

- 只为指定 Q122 split layers 使用 cache
- resident-only 模式
- miss 用 nullptr sentinel
- GPU MMVQ 看到 nullptr 时不算该 expert
- host backing 仍保留给 CPU miss branch

---

## `ggml/src/ggml-cuda/mmvq.cu`

增加 resident-only miss 行为：

```text
expert ptr == nullptr
→ 这是 GPU miss
→ 输出 0
→ 不做 dot product
```

这样 GPU 分支只计算真正 resident 的专家。

---

## `ggml/src/ggml-cuda/ggml-cuda.cu`

R1 下禁止普通 cold host MMID 自动 offload。

只允许明确 graph-pinned 的 GPU resident branch 使用 GPU。

还包含 Q122 R1 的诊断路径。

---

# 20. Q122 当前生产配置

```yaml
qwen3.5-122b:256k:
  ttl: 1200
  env:
    - "LD_LIBRARY_PATH=/app/share/llama_box/src/llama.cpp-b10235-hotexpert-v2-q122/build-hip-rocm714/bin:/opt/rocm/lib"
    - "HIP_VISIBLE_DEVICES=0"

    - "HOTSEAT_TENSOR_LAYERS=3"

    - "Q122_R1=1"
    - "Q122_R1_SPLIT_LAYERS=3,21,22,23,24,25,26,27,28,29,30,34,35,37,43,44"
    - "Q122_R0_SKIP_SLOTS=21"
    - "Q122_R1_RESIDENT_ONLY=1"

    - "HOTEXPERT_CACHE=1"
    - "HOTEXPERT_CACHE_SLOTS=21"
    - "HOTEXPERT_CACHE_VRAM_BUDGET_MB=3400"
    - "HOTEXPERT_CACHE_PROFILE=/app/share/llm/Qwen3.5-122B-A10B-abliterated-Q8_0-GGUF/hotseat-v2/q122-static-cache-profile.json"

  cmd: >
    /app/share/llama_box/src/llama.cpp-b10235-hotexpert-v2-q122/build-hip-rocm714/bin/llama-server
    --host 127.0.0.1
    --port ${PORT}
    --jinja
    -m /app/share/llm/Qwen3.5-122B-A10B-abliterated-Q8_0-GGUF/Qwen3.5-122B-A10B-abliterated.Q8_0.gguf
    -c 262144
    -ngl 999
    -t 4
    -np 1
    -b 2048
    -ub 2048
    --cache-ram 0
    --no-mmap
    --mlock
```

---

# 21. Q122 当前 profile

生产用：

```text
/app/share/llm/Qwen3.5-122B-A10B-abliterated-Q8_0-GGUF/
hotseat-v2/q122-static-cache-profile.json
```

历史保留的有价值测试证据：

```text
hotseat-v2/stage1-profile/
hotseat-v2/r1-all45-ab-baseline/
hotseat-v2/r1-top21-best16/
```

最终生产：

```text
HOTEXPERT_CACHE_SLOTS=21
Q122_R1_SPLIT_LAYERS=
3,21,22,23,24,25,26,27,28,29,30,34,35,37,43,44
```

即：

```text
16 个 split layers
21 resident experts / selected layer
```

---

# 22. Q122 R1 为什么不是通用方案

R1 中有明显的 Q122 专用假设：

- 48 MoE layers
- 256 experts
- `q122-r0-rank.inc`
- Q122 layer list
- Q122 node naming
- Q122 env vars
- CPU/GPU MMID split 只针对这个图结构调过

因此：

**不要把 q122-r1.patch 当成通用 MoE 优化。**

如果升级后 Qwen3.5 graph 的 MMID shape、scale tensor、scheduler 行为变了，需要逐点验证。

---

# 23. 当前 ROCm 7.14 编译信息

两个生产 worktree 的 `CMakeCache.txt` 已核对，关键配置一致：

```text
CMAKE_BUILD_TYPE=Release
GGML_HIP=ON
GGML_CUDA=OFF
AMDGPU_TARGETS=gfx1100
CMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++
```

RX 7900 XTX：

```text
gfx1100
```

当前 ROCm 7.14 环境：

```bash
unset HSA_OVERRIDE_GFX_VERSION
export HIP_VISIBLE_DEVICES=0
```

**ROCm 7.14 不要再设置：**

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

当前环境已经可以原生识别 gfx1100。

---

# 24. 推荐的重编译命令

这是根据当前生产 `CMakeCache` 还原的等价构建命令：

```bash
SRC=/path/to/patched/llama.cpp
BUILD="$SRC/build-hip-rocm714"

unset HSA_OVERRIDE_GFX_VERSION
export HIP_VISIBLE_DEVICES=0

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++

cmake --build "$BUILD" \
  -j"$(nproc)" \
  --target llama-server
```

产物：

```text
build-hip-rocm714/bin/llama-server
```

升级 ROCm 后，HIP compiler path 可能变化，不要机械保留 `core-7.14`。

---

# 25. 当前 llama-swap 启动方式

容器内当前实际进程：

```text
/app/share/llama_box/bin/llama-swap
  -config /app/share/llama_box/config/config-rocm714.yaml
  -listen 0.0.0.0:8090
```

所以实际生产配置是：

```text
/app/share/llama_box/config/config-rocm714.yaml
```

宿主机：

```text
/home/victor/ai_share/llama_box/config/config-rocm714.yaml
```

不是优先看：

```text
config.yaml
```

升级 llama-swap 时最重要的是保住：

1. model alias
2. env 列表
3. 每模型独立 `cmd`
4. 指向正确 patched llama-server
5. TTL
6. `checkEndpoint`
7. `${PORT}` 行为

---

# 26. 升级 llama-swap 时要注意什么

HotSeat 主要改的是 llama.cpp，不需要 patch llama-swap 本体。

llama-swap 的职责只是：

```text
选择模型
设置环境变量
启动对应 llama-server 二进制
停止/卸载
健康检查
```

所以升级 llama-swap 后先确认新版 config schema 没改。

必须验证：

```bash
curl http://127.0.0.1:8090/v1/models
```

然后分别加载：

```text
ornith-1.5-35b:256k
qwen3.5-122b:256k
```

确认进程 cmdline 指向正确 binary。

不要只看 Web UI 显示 ready。

---

# 27. 以后升级新版 llama.cpp 的推荐流程

## Step 1：不要覆盖旧生产树

新建独立源码目录，例如：

```text
llama.cpp-new
```

旧的：

```text
llama.cpp-b10235-hotexpert-v2
llama.cpp-b10235-hotexpert-v2-q122
```

先全部保留。

---

## Step 2：先编译纯净新版

先确认官方新版在 ROCm 7.14 / 当前显卡上能正常编译和启动。

这样后面出错时能区分：

```text
上游坏了
vs
HotSeat patch 冲突
```

---

## Step 3：先迁通用 HotSeat，不碰 Q122

推荐迁移顺序：

```text
HotExpert core
→ descriptor / profiler / planner
→ resident cache
→ dynamic-hybrid runtime
→ arena
→ HotSeat V2 profiler
→ routing history
→ transit
→ predictor
→ persistent transit
```

如果新版源码和 b10235 差距不大，可以先尝试：

```bash
git apply --3way ornith-hotseat-full-from-b10235-base.patch
```

但是跨很多 llama.cpp commit 后，不要指望 8393 行 patch 无冲突。

出现 conflict 时按功能迁移，不要为了“patch clean”乱改新版上游代码。

---

## Step 4：先验证 Ornith

先只启用基础参数：

```text
4 Full
95 slots
dynamic-hybrid
arena
```

确认：

- 能加载
- 无 VRAM overflow
- PP 正常
- TG 正常
- host miss 路径正常

再启 V2：

```text
PREFETCH
TRANSIT
TRANSIT_SLOTS=2
PERSISTENT_TRANSIT
PREDICTOR
```

---

## Step 5：对比旧版

至少记录：

```text
模型加载时间
PP tok/s
TG tok/s
首字时间
VRAM peak
RAM
host miss
resident hit
是否有非法内存访问
```

不要只看一次 TG 峰值。

---

## Step 6：单独创建 Q122 worktree

确认新版 Ornith/通用 V2 稳定后，再基于这棵树创建：

```bash
git worktree add ../llama.cpp-new-q122 -b q122-r1-new
```

然后迁：

```text
q122-r1.patch
```

---

## Step 7：Q122 重点验证

Q122 必须特别检查：

1. `build_moe_ffn()` 的结构有没有变
2. `mul_mat_id` graph naming 是否还有效
3. backend scheduler API 是否变化
4. ROCm_Host buffer 判断是否变化
5. MMVQ expert pointer table 结构是否变化
6. expert scale tensor 是否仍可在 CPU+GPU merge 后只应用一次
7. decode-only 条件是否仍正确
8. prefill 是否仍走原路径

任何一项变了，都不能机械 cherry-pick。

---

# 28. 哪些文件能复用，哪些应该重采

| 项目 | 同模型升级 llama.cpp | 升级 ROCm | 换模型/GGUF |
|---|---|---|---|
| expert long profile | 通常可复用，但验 fingerprint | 通常可复用 | 必须重采 |
| V2 predictor | 通常可复用 | 通常可复用 | 必须重采 |
| Vision predictor | 通常可复用 | 通常可复用 | mmproj/模型变化建议重采 |
| Arena baseline | 建议重采 | **强烈建议重采** | 重采 |
| Full layer plan | 可先复用再验证 | 可复用 | 重算 |
| Q122 rank.inc | 同 GGUF 可先复用 | 可复用 | **必须重算** |
| Q122 split layers | 先复用再 benchmark | 先复用再 benchmark | 重做 |
| llama-server binary | **必须重编** | **必须重编** | 视架构支持 |
| llama-swap yaml | 可迁移 | 可迁移 | 更新模型项 |

---

# 29. 升级后最容易踩的坑

## 29.1 新版 llama.cpp 改 MoE graph

这是最大风险。

重点文件：

```text
src/llama-graph.cpp
```

特别是：

```text
build_moe_ffn
mul_mat_id
expert scale
gate/up/down
selected_experts
```

---

## 29.2 scheduler 行为变化

Q122 R1 尤其依赖：

```text
ggml-backend scheduler
```

如果新版改变 ROCm_Host tensor 的自动 offload 逻辑，可能导致：

```text
CPU miss branch 又被送 GPU
```

这样速度和显存都会异常。

---

## 29.3 CUDA/HIP MMVQ kernel 接口变化

重点：

```text
ggml/src/ggml-cuda/mmvq.cu
```

V2 transit 和 Q122 resident-only 都碰这里。

---

## 29.4 tensor name / shape 变化

HotExpert parser、Q122 layer parser、cache profile 都依赖 tensor 命名和 layer/expert layout。

升级后必须观察启动日志。

---

## 29.5 Arena 复用旧 low-water

如果新版本：

- graph 多占显存
- KV 行为变
- kernel workspace 变
- mmproj 变

继续套旧 Arena baseline 可能 OOM。

---

## 29.6 改了配置但没重启 llama_box_714

当前 llama-swap 配置读取是在容器/进程启动链路。

生产经验已经证明：

```text
修改 config-rocm714.yaml 后
→ 重启 llama_box_714
```

最稳。

别再花半小时研究为什么参数“明明写了却没生效”，那种谜题没有奖金。

---

# 30. 当前性能作为升级回归基准

这些数用于升级后判断有没有明显 regression，不作为绝对 benchmark。

## Ornith 1.5 Text

真实长输出近期：

```text
约 38–40 tok/s
```

特定热场景/benchmark 曾更高，约 50+。

生产回归建议以：

```text
长输出稳定 38–40
```

作为更现实基线。

---

## Ornith 1.5 Vision

HotSeat V2 同图 A/B：

旧版：

```text
31.92
35.05 tok/s
```

V2：

```text
44.56
43.61 tok/s
```

平均提升约：

```text
+31.6%
```

PP 基本不变：

```text
约 242 → 241 tok/s
```

---

## Qwen3.5-122B Q8

当前稳定版约：

```text
9–10 tok/s
```

历史最佳 R1 某 prompt：

```text
约 9.44 tok/s
```

固定 prompt 后续约：

```text
8.79–8.80 tok/s
```

不同 prompt routing 会显著影响速度。

**不要拿单一 prompt 的 9.44 当所有工作负载保证值。**

---

# 31. 一键检查当前生产 Git 状态

```bash
V2=/home/victor/ai_share/llama_box/src/llama.cpp-b10235-hotexpert-v2
Q122=/home/victor/ai_share/llama_box/src/llama.cpp-b10235-hotexpert-v2-q122

git -C "$V2" status --short
git -C "$V2" branch --show-current
git -C "$V2" rev-parse HEAD
git -C "$V2" tag --points-at HEAD

git -C "$Q122" status --short
git -C "$Q122" branch --show-current
git -C "$Q122" rev-parse HEAD
git -C "$Q122" tag --points-at HEAD
```

正确应看到：

```text
Ornith:
hotseat-v2-ornith
9263152dc558b30ed2cec6e531f54d83e50688f1
ornith-v2-prod-20260823

Q122:
hotseat-r1-q122-stable
42da2948beedcac288274399f62f46de438d8bca
q122-r1-stable-20260823
```

并且 worktree 应 clean。

---

# 32. 重新导出三种 patch 的命令

## 完整 HotSeat + V2

```bash
git -C /home/victor/ai_share/llama_box/src/llama.cpp-b10235-hotexpert-v2 \
  diff --binary \
  221f0f6356efe2260023208365705ec5d5a7c8f5..9263152dc558b30ed2cec6e531f54d83e50688f1 \
  > ornith-hotseat-full-from-b10235-base.patch
```

当前 SHA256：

```text
118f458d6ab050a9d376584d91c4ad907629b4d29922f766c4d42ca353fcaad5
```

---

## 仅 HotSeat V2 阶段

```bash
git -C /home/victor/ai_share/llama_box/src/llama.cpp-b10235-hotexpert-v2 \
  diff --binary \
  816fb44ca..9263152dc558b30ed2cec6e531f54d83e50688f1 \
  > ornith-hotseat-v2.patch
```

当前 SHA256：

```text
94019314f3de7377547dc10e4f610dd2756a92dc9cc665dda157b85bd3d8818f
```

---

## Q122 R1 专用

```bash
git -C /home/victor/ai_share/llama_box/src/llama.cpp-b10235-hotexpert-v2-q122 \
  diff --binary \
  9263152dc558b30ed2cec6e531f54d83e50688f1..42da2948beedcac288274399f62f46de438d8bca \
  > q122-r1.patch
```

当前 SHA256：

```text
93a6025c4e6bb8ad680db82251a50d4cfbe9188c6e7b03d02c38fc1d80976270
```

---

# 33. 最短迁移清单

如果以后只想看一屏，按这个：

```text
[ ] 保留当前两个 stable worktree + tags
[ ] 新版纯净 llama.cpp 先单独编译成功
[ ] 迁 HotExpert core
[ ] 迁 Dynamic Hybrid + Arena
[ ] 迁 HotSeat V2
[ ] 编译 gfx1100 ROCm
[ ] Ornith Text A/B
[ ] Ornith Vision A/B
[ ] Arena profile 重新确认
[ ] predictor fingerprint 确认
[ ] 建 Q122 独立 worktree
[ ] 迁 q122-r1
[ ] 检查 build_moe_ffn / scheduler / MMVQ
[ ] Q122 PP/TG A/B
[ ] 更新 config-rocm714.yaml 二进制路径
[ ] 重启 llama_box_714
[ ] curl 8090/v1/models
[ ] 正式 OpenClaw 调用验证
```

---

# 34. 最重要的三个版本锚点

以后如果文档其它东西都丢了，只记住这三个：

```text
官方基线:
221f0f6356efe2260023208365705ec5d5a7c8f5
llama.cpp b10235

Ornith / 通用 HotSeat V2:
9263152dc558b30ed2cec6e531f54d83e50688f1
tag ornith-v2-prod-20260823

Qwen3.5-122B R1:
42da2948beedcac288274399f62f46de438d8bca
tag q122-r1-stable-20260823
```

这三个 commit 能把当前整个优化链重新钉回去。

---

## 2026-09-07 production update

Latest AMD ROCm production work adds:

- DynamicKV low-water VRAM balancing
- Tiny / Normal / Large PP automatic routing
- Large-PP elastic-expert suspend and tail prefetch
- HotSeat V2 Large-PP support for Qwen3.6 text and vision
- Q122 Large-PP suspend hysteresis fix
- Ornith text/vision 4096 batch + Flash Attention where validated

Portable artifacts:

- `patches/hotseat-dynkv-largepp-tailprefetch-from-bebc9350.patch`
- `patches/20260907-working-delta-from-b96fdd6fc.patch`
- `config-snippets-20260907.yaml`
- `docs/2026-09-07-dynkv-tiny-largepp-tail-prefetch.md`
- `patch-manifest-20260907.txt`
- `SHA256SUMS-20260907.txt`

Public upstream llama.cpp base: `bebc9350ecc42a31ad119da1513998386671cf5b`.

Selected measured results:

- Ornith text warm large incremental PP: **141.15 tok/s**, TG **46.96 tok/s**
- Qwen3.6 text warm incremental PP: **859.31 tok/s**, TG **57.51 tok/s**
- Qwen3.6 vision fresh PP: **907.25 tok/s**, TG **55.62 tok/s**
- Ornith vision warm large incremental PP: **113.24 tok/s**, TG **50.03 tok/s**
- Ornith text direct 240K validation: PP **307.19 tok/s**, TG **35.09 tok/s**, with no KV grow/reserve failure

The full patch was generated directly from the public upstream base and passed `git apply --check` against that exact base before upload.