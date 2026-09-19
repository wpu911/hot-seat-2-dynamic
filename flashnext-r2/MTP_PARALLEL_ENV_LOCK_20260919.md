# Flash Next R2：环境锁 + MTP 多 Slot 隔离 Gate

日期：2026-09-19

## 1. 为什么新增环境锁

R2 的 A/B 会持续很久，而 llama-swap、生产 binary、ROCm 动态库和内核任何一个在中途变化，都可能让后面的 0.5～2% 差异失去比较意义。

当前 upstream llama-swap 已经发布 v256，而本项目历史迁移记录停在 v255。R2 不在中途自动升级 llama-swap。升级可以做，但应该重新建立 baseline，而不是让版本变化偷偷混进同一组 A/B。

第一次正式跑 R2 前：

```bash
python3 flashnext-r2/scripts/lock_r2_environment.py --create
```

之后每个关键阶段可通过：

```bash
bash flashnext-r2/scripts/r2_env_guard.sh <原命令>
```

环境锁固定检查：

```text
llama-swap binary SHA256 + version
正式 qwen3.8-flash-next:256k alias block SHA256
正式 llama-server wrapper SHA256
正式 llama-server.real SHA256
kernel release
正式 runtime 的 ldd 实际解析到的 ROCm/HIP 库及 SHA256
```

实验 alias、实验 runtime 和实验 YAML block 可以继续新增，因此不会粗暴地 hash 整个 config。

如果环境发生变化，`--check` 直接 FAIL。不要用 `--replace` 把锁改到“重新一致”，除非明确决定结束旧 baseline、重新开始一轮受控测试。

## 2. 为什么新增 MTP 多 Slot Gate

截至 2026-09-19，llama.cpp upstream issue #28286 仍为 open：

> `draft-mtp + --parallel > 1` 在并发请求下可能出现跨 slot 内容污染。

报告的危险点不是乱码或 crash，而是输出仍然像正常文本，却混入另一个并发请求的上下文。issue 还明确记录：

```text
--parallel 1 未复现
关闭 HIP Graph 仍复现
低信息、重复 prompt 容易测不出来
高复杂度 prompt 才容易暴露
```

对应 issue：

https://github.com/ggml-org/llama.cpp/issues/28286

qwen4exp recurrent rollback 的基础支持 #28123 已经合并，但 #28286 比它更晚提交，而且目前仍未关闭。因此不能因为 native rollback 已启用，就自动宣布多 slot MTP 安全。

## 3. 本项目 Gate 怎么测

最终 R2 winner 在进入 OpenClaw 回归前运行：

```bash
python3 flashnext-r2/scripts/run_mtp_parallel_isolation.py
```

流程：

```text
只卸载当前实验 winner
通过 llama-swap :8090 冷加载 winner
读取 /upstream/<winner>/slots
```

如果只有 1 个 slot：

```text
MODE=SERIALIZED_SAFE
PASS
```

这不是证明 upstream 并发 bug 被修了，而是当前部署根本没有进入 `--parallel > 1` 风险条件。

如果有多个 slot，则默认最多 4 路并发，连续 3 轮。每路使用：

```text
完全不同的高信息量技术主题
独立随机 canary
4 个不可能自然碰撞的专属 signature token
大量不同 ref/check 数值
fixed-length TG384
MTP draft_n / draft_n_accepted
```

每个响应必须：

```text
生成完整 requested tokens
存在 MTP counters
出现自己的 canary/signature
绝不能出现其他 slot 的 canary/signature
```

任何 foreign signature 命中都直接 FAIL，不平均、不投票、不拿 TG 更快来抵消。

## 4. 最终顺序

现在生产前尾部顺序固定为：

```text
Phase-6 / 6b winner
→ MTP multi-slot isolation
→ OpenClaw Gateway real-session regression
→ read-only promotion review
→ explicit production promotion
```

`report_r2_state.py` 已把 MTP slot isolation 插在 OpenClaw 之前。旧的 slot-isolation 结果如果早于当前 winner，也会被视为 stale。

## 5. 生产原则

如果多 slot Gate FAIL，有两个方向：

```text
A. 保留当前性能 binary，但生产 alias 强制 --parallel 1，重新完整验证
B. 等/移植 upstream 真正修复，再恢复多 slot
```

不要让一个会偶发把 A 会话内容塞进 B 会话的推理服务，因为“平均 TG 更高”就进生产。那不是并发优化，是随机串台。
