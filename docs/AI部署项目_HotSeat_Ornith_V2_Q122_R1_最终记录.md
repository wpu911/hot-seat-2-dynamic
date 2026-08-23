# HotSeat Ornith V2 与 Qwen3.5-122B R1 最终部署记录

更新日期：2026-08-23  
GitHub 仓库：<https://github.com/wpu911/hot-seat-2-dynamic>

## 1. 最终结论

当前生产环境保留两套独立的 llama-server 二进制：

| 模型 | 实现 | 生产源码树 | 生产二进制 |
|---|---|---|---|
| Ornith 1.5 Text / Vision、Qwen3.6 Text / Vision | 通用 HotSeat V2 | /app/share/llama_box/src/llama.cpp-b10235-hotexpert-v2 | build-hip-rocm714/bin/llama-server |
| Qwen3.5-122B Q8 | Q122 R1 专用 CPU/GPU MMID split | /app/share/llama_box/src/llama.cpp-b10235-hotexpert-v2-q122 | build-hip-rocm714/bin/llama-server |

两套二进制有共同祖先，但不能混用。Q122 R1 修改了 MoE 图拆分、backend scheduler、CPU resident-skip 和 GPU resident-only 路径。

## 2. 三个版本锚点

~~~text
官方 llama.cpp b10235:
221f0f6356efe2260023208365705ec5d5a7c8f5

Ornith / 通用 HotSeat V2:
9263152dc558b30ed2cec6e531f54d83e50688f1
tag: ornith-v2-prod-20260823
branch: hotseat-v2-ornith

Qwen3.5-122B R1:
42da2948beedcac288274399f62f46de438d8bca
tag: q122-r1-stable-20260823
branch: hotseat-r1-q122-stable
~~~

仓库中的三份补丁：

| 补丁 | 用途 | SHA256 |
|---|---|---|
| patches/ornith-hotseat-full-from-b10235-base.patch | 官方 b10235 到完整 HotExpert + HotSeat V2 | 118f458d6ab050a9d376584d91c4ad907629b4d29922f766c4d42ca353fcaad5 |
| patches/ornith-hotseat-v2.patch | V2 baseline 到 Ornith V2 生产版 | 94019314f3de7377547dc10e4f610dd2756a92dc9cc665dda157b85bd3d8818f |
| patches/q122-r1.patch | Ornith V2 生产树到 Q122 R1 | 93a6025c4e6bb8ad680db82251a50d4cfbe9188c6e7b03d02c38fc1d80976270 |

## 3. Ornith HotSeat V2

Ornith 1.5 为 40 层 MoE、每层 256 experts、Top-8。24GB RX 7900 XTX 无法容纳全部 Q8 专家，因此采用：

~~~text
系统内存：完整 GGUF backing store
GPU 显存：Full Layer + resident experts + arena + transit mini-cache
~~~

生产配置：

~~~text
HOTSEAT_RUNTIME_MODE=dynamic-hybrid
HOTSEAT_FULL_LAYER_COUNT=4
HOTSEAT_RUNTIME_INIT_FULL_LAYERS=0,1,2,3
HOTSEAT_FULL_SHADOW=1
HOTSEAT_EXPERT_SLOTS=95          # Text
HOTSEAT_EXPERT_SLOTS=90          # Vision
HOTSEAT_AUTO_RESERVE_ARENA=1
HOTSEAT_V2_PREFETCH=1
HOTSEAT_V2_TRANSIT=1
HOTSEAT_V2_TRANSIT_SLOTS=2
HOTSEAT_V2_PERSISTENT_TRANSIT=1
HOTSEAT_V2_PREDICTOR_MIN_PPM=300000
~~~

V2 增加的核心能力：

1. 每 token expert route profiler。
2. 固定 ring 的路由历史与异步 worker。
3. next-token transit prefetch。
4. self-repeat predictor。
5. 两项 persistent transit mini-cache，命中时复用，避免重复 H2D copy。

Text 与 Vision 必须使用各自的 predictor 和 arena profile。Ornith 与 Qwen3.6 即使同属 qwen35moe，也不能共享 predictor。

## 4. Qwen3.5-122B R1

Q122 单 expert 约 9.56 MiB，完整 expert layer 约 2.45 GiB，不能照搬 Ornith 的 4 Full + 95 slots。

R1 使用 decode-only 的 CPU/GPU MMID split：

~~~text
CPU miss branch
    +
GPU resident branch
    ↓
add / merge
~~~

启用条件：

- decode；
- ubatch.n_tokens == 1；
- 非 warmup；
- 仅指定 split layers。

生产配置：

~~~text
HOTSEAT_TENSOR_LAYERS=3
Q122_R1=1
Q122_R1_SPLIT_LAYERS=3,21,22,23,24,25,26,27,28,29,30,34,35,37,43,44
Q122_R0_SKIP_SLOTS=21
Q122_R1_RESIDENT_ONLY=1
HOTEXPERT_CACHE=1
HOTEXPERT_CACHE_SLOTS=21
HOTEXPERT_CACHE_VRAM_BUDGET_MB=3400
~~~

R1 是 Q122 专用实现，依赖 48 MoE layers、256 experts、静态 rank、指定 layer list 和当前 graph naming，不作为通用 MoE 补丁使用。

## 5. ROCm 7.14 编译

当前硬件目标：

~~~text
GPU: AMD Radeon RX 7900 XTX
AMDGPU target: gfx1100
ROCm: 7.14
~~~

关键规则：

~~~bash
unset HSA_OVERRIDE_GFX_VERSION
export HIP_VISIBLE_DEVICES=0
~~~

ROCm 7.14 已原生识别 gfx1100，不再设置 HSA_OVERRIDE_GFX_VERSION=11.0.0。

构建：

~~~bash
./rebuild-rocm714.sh /path/to/patched/llama.cpp
~~~

等价命令：

~~~bash
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

cmake --build "$BUILD" -j"$(nproc)" --target llama-server
~~~

## 6. llama-swap 生产关系

生产配置：

~~~text
宿主机：
/home/victor/ai_share/llama_box/config/config-rocm714.yaml

容器：
/app/share/llama_box/config/config-rocm714.yaml
~~~

启动：

~~~text
/app/share/llama_box/bin/llama-swap
  -config /app/share/llama_box/config/config-rocm714.yaml
  -listen 0.0.0.0:8090
~~~

修改配置后重启：

~~~bash
docker restart llama_box_714
curl -f http://127.0.0.1:8090/v1/models
~~~

必须检查进程实际 cmdline 是否指向对应 patched binary，不能只看 Web UI 的 ready。

## 7. 当前回归基线

| 模型 | 当前稳定基线 |
|---|---|
| Ornith 1.5 Text | 长输出约 38–40 tok/s；特定热场景曾达到 50+ |
| Ornith 1.5 Vision | V2 约 43.61–44.56 tok/s；旧版约 31.92–35.05 tok/s；平均提升约 31.6% |
| Ornith 1.5 Vision PP | 约 241 tok/s，与旧版约 242 tok/s 基本持平 |
| Qwen3.5-122B Q8 | 通常约 9–10 tok/s；固定 prompt 约 8.79–8.80 tok/s；历史单 prompt 约 9.44 tok/s |

Q122 对 prompt routing 敏感，不能把单一 prompt 的峰值当作所有负载保证。

## 8. 以后升级 llama.cpp 的顺序

1. 保留当前两个 stable worktree、branch 和 tag，不覆盖生产树。
2. 新建纯净新版 llama.cpp，先完成 ROCm 编译与原版启动验证。
3. 迁 HotExpert core、descriptor、profiler、planner、resident cache、dynamic-hybrid 和 arena。
4. 迁 HotSeat V2 profiler、route history、transit、predictor 和 persistent transit。
5. 先验证 Ornith Text，再验证 Ornith Vision。
6. 重新确认 Arena low-water；ROCm、graph、KV、mmproj 或 batch/ubatch 变化时重采。
7. 基于已稳定的通用 V2 树单独建立 Q122 worktree，再应用 q122-r1.patch。
8. 检查 build_moe_ffn()、mul_mat_id、scheduler、ROCm_Host 判断、MMVQ pointer table、scale tensor 和 decode-only 条件。
9. A/B 记录加载时间、PP、TG、首字、VRAM peak、RAM、resident hit 和 host miss。
10. 更新 config-rocm714.yaml 的 binary 路径，重启 llama_box_714，再用 llama-swap 与 OpenClaw 做完整验收。

跨越较多上游 commit 时，git apply --3way 只能作为起点。出现冲突应按功能迁移，不能为了让 patch clean 而破坏新版上游逻辑。

## 9. 可复用与应重采项目

| 项目 | 同模型升级 llama.cpp | 升级 ROCm | 换模型或 GGUF |
|---|---|---|---|
| expert long profile | 可先复用并验 fingerprint | 通常可复用 | 必须重采 |
| V2 predictor | 通常可复用 | 通常可复用 | 必须重采 |
| Vision predictor | 通常可复用 | 通常可复用 | 模型或 mmproj 变化建议重采 |
| Arena baseline | 建议重采 | 强烈建议重采 | 重采 |
| Full layer plan | 可先复用再验证 | 可复用 | 重算 |
| Q122 rank.inc | 同 GGUF 可先复用 | 可复用 | 必须重算 |
| Q122 split layers | 复用后重新 benchmark | 复用后重新 benchmark | 重做 |
| llama-server binary | 必须重编 | 必须重编 | 按新架构重编 |
| llama-swap YAML | 迁移并检查 schema | 可迁移 | 更新模型项 |

## 10. 回滚点

生产旧树不删除，回滚时只把 config-rocm714.yaml 中的模型 cmd 指回旧版 llama-server，然后重启 llama_box_714。

回滚后至少验证：

~~~bash
curl -f http://127.0.0.1:8090/v1/models
curl -f http://127.0.0.1:18789/
~~~

详细改动、文件清单、完整配置摘录、补丁生成方式和诊断说明以仓库根目录 README.md 为准。
