# Flash Next R2 集成计划（2026-09-18）

目标：在现有 HotSeat / Dynamic KV / Static+Borrow+Transit / MTP multirow 基础上，针对当前 7900 XTX 24GB + Radeon AI PRO R9700 32GB 异构双卡，优先移植已公开验证的新优化；生产 main 暂不修改。

## 当前基线

历史单 7900 XTX Flash Next 已达到约 19 tok/s。当前双卡现场约 9.5–10.6 tok/s，说明双卡路径存在明显回退，不能把“显存增加”直接视为性能增加。

现有生产技术：
- HotSeat CPU/GPU Hybrid Expert
- Static + Borrow
- Transit / predictor / persistent transit
- Dynamic KV
- Large-PP / TG 分离
- MTP multirow Hybrid
- MTP Transit
- cached Large-PP SPEC gate / sticky safety

## R2 第一阶段：先移植低风险、高确定性优化

### 1. 对齐 llama.cpp PR #28243 的 Qwen3.8-Flash-Next MTP 路径

重点：
- qwen4exp MTP support
- shared MTP embeddings / shared target memory
- 避免 draft 重复保存/加载 embedding、norm、lm_head
- 保留现有 MTP multirow HotSeat 逻辑

原则：先确认当前生产树是否已经等价包含；若已有，不重复打补丁。

### 2. R9700 / gfx1201 专用 bit-exact kernel fusion

参考 `JohnTDI-cpu/llama.cpp-flash-next-rdna4`：
- HyperConnection glue fusion
- hc-combine / hc-mix / hc-inject
- shared-expert tail fusion
- Q8_1 activation quantization dedup
- Gated DeltaNet prolog + l2norm fusion
- HIP Graph

只默认启用在 gfx1201。gfx1100 先维持当前稳定路径，除非 bit-exact gate + A/B benchmark 证明同样安全。

### 3. MoE PP MMQ column tile cap

候选：`GGML_JOHNV8_MMQ_ID_JMAX=32`

背景：Flash Next 512 experts，ubatch 很大时每个 expert 实际 token rows 很少；按整个 ubatch 选择 128-column tile 会浪费大量工作。

测试：
- JMAX=off / 64 / 32 / 16
- PP 2K / 8K / 32K
- 记录 PP、VRAM peak、GPU util、输出一致性

### 4. HIP Graph

单独 A/B，不与 kernel fusion 一次性混测。双卡尤其要检查 graph capture 与异构 device scheduling 是否引入等待。

## R2 第二阶段：重新做 MTP 参数扫描

现有历史值 `--spec-draft-n-max 2` 不视为当前双卡最优。

扫描：
- n-max = 2 / 3 / 4
- p-min = 默认 / 0
- prose / code / tool-call / OpenClaw 长会话四类负载

指标：
- TG
- acceptance
- verify rows
- CPU miss rows
- GPU resident/transit hit
- 每轮 verify wall time

## R2 第三阶段：QSA / long-context

单独做 context ladder：1K / 4K / 16K / 32K / 64K / 128K。

重点确认：
- TOP_K 是否在 HIP 长上下文回 CPU
- QSA pooled-key / sparse gather 是否存在 O(n_ctx) 重复工作
- 能否加入 persistent pooled-key incremental cache

这一阶段不与 HotSeat expert cache 同时改，避免混淆瓶颈。

## 验证门槛

任何默认启用项必须同时满足：
1. greedy temperature=0 短 prompt 输出 token 完全一致；
2. ~1.5K 长 prompt 输出 token 完全一致；
3. 真实 OpenClaw capture 回放通过；
4. 不出现 page fault / compute error / 0.1 tok/s 回退；
5. 至少 3 次 A/B，且每轮重新跑 baseline，避免温度漂移误判。

## 暂不做

- 不把 gfx1201 kernel 直接强开到 gfx1100；
- 不先改 Expert Cache 大结构；
- 不先做多 stream fork/join；公开 R9700 实验中该路线在 ROCm 上反而有明显回退；
- 不直接落生产 alias；所有 R2 先走独立 runtime / test port。

## 第一批目标

先做到：
- 恢复双卡 Flash Next 至少回到历史单卡 18–19 tok/s 档；
- 然后再判断 R9700 fusion + MTP3/4 是否能推进到 20+ tok/s；
- PP 不以牺牲 TG 为代价。
