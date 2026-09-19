# Flash Next R2 Phase-6b：异构双卡双向 Tensor Ratio Sweep

日期：2026-09-19

## 为什么不能只按显存容量分

本机是异构双卡：RX 7900 XTX 24GB + Radeon AI PRO R9700 32GB。

旧 Phase-6b 只从 `1,1` 向“显存容量比例”方向试：

```text
EVEN = 1,1
MID  = sqrt(VRAM0 / VRAM1)
CAP  = VRAM0 / VRAM1
```

这只能回答“更大的显存卡多分一点是否更好”，却不能回答“更快的卡多分一点是否更好”。对于异构卡，这两个方向完全可能相反。

AMD 官方规格本身就说明不能把容量当吞吐代理：RX 7900 XTX 是 24GB，但标称显存带宽最高 960 GB/s；R9700 是 32GB，但标称显存带宽最高 640 GB/s。实际 llama.cpp 的 Q8/MoE/QSA/HC/GDN kernel 还会受架构、ROCm kernel、PCIe 和同步开销影响，所以最终比例必须由本机 A/B 决定，而不是拿规格表直接拍脑袋。

参考：

- AMD RX 7900 XTX product page: https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html
- AMD Radeon AI PRO R9700 datasheet: https://www.amd.com/content/dam/amd/en/documents/partner-hub/radeon-pro/radeon-ai-pro-r9700-datasheet.pdf

## 新的五臂 ratio

`prepare_phase6b_tensor_ratio_sweep.sh` 现在根据真实 llama.cpp device order 和总 VRAM 自动生成：

```text
EVEN     1,1
MID      从 1,1 向容量比例走一半
CAP      容量比例
INV_MID  MID 的倒数
INV_CAP  CAP 的倒数
```

如果设备顺序是 24GB -> 32GB，典型近似：

```text
EVEN     1,1
MID      13,15
CAP      3,4
INV_MID  15,13
INV_CAP  4,3
```

因此两边都会真正试到，不预设 ROCm0 或 ROCm1 哪张更值得分更多 tensor。

## 测速顺序

`bench_tensor_ratio_sweep.py` 使用镜像顺序：

```text
EVEN -> MID -> CAP -> INV_MID -> INV_CAP
INV_CAP -> INV_MID -> CAP -> MID -> EVEN
```

每个 leg：

```text
只卸载这 5 个实验 alias
重新冷加载当前 alias
4K warm-up，不计分
32K fixed TG
64K fixed TG
检查 needle
检查完整 requested tokens
检查 MTP draft_n / draft_n_accepted
```

如果实验 alias 正在被别的请求使用，直接停止，不允许把并发请求污染后的数字当 benchmark。

某个极端 ratio 因 VRAM 或 compute error 起不来，只淘汰该 arm，不影响其他 ratio。

## 第一轮选择

`analyze_tensor_ratio_sweep.py` 以 `1,1` 为 anchor，默认要求：

```text
marker / fixed TG / MTP counters 全通过
median long TG >= 1,1
单一 depth 不得比 1,1 低超过 5%
median PP 不得低超过 12%
MTP acceptance 不得下降超过 5 percentage points
```

第一轮只是 exploratory smoke。即便赢了，也不能直接成为最终 winner。

## 最终确认

smoke 最好的非 1,1 arm 必须继续通过：

```text
short PP/TG exact A/B
32K / 64K / 128K repeated confirmation
median long TG 至少 +1%
cached Large-PP / high-LCP
64K recurrent rollback + MTP
```

任一关失败就回退 `1,1`。

## 生产约束

整个 Phase-6b：

```text
不覆盖 qwen3.8-flash-next:256k
不全局 unload llama-swap
所有 ratio 使用同一个 llama-server.real SHA256
只通过 wrapper 改 --tensor-split
强制 --fit off
最终仍需 OpenClaw Gateway regression
```

这轮的目标不是证明“7900 XTX 一定应该多分”或“R9700 一定应该多分”，而是把这个问题从规格猜测变成本机实测问题。异构双卡已经够麻烦了，没必要再让容量数字兼职性能预测师。
