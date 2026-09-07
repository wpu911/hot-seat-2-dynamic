# 2026-09-07 HotSeat / DynamicKV / PP pipeline update

This update targets llama.cpp on AMD ROCm with a 24 GiB RX 7900 XTX and large 256K contexts. It extends the earlier HotSeat V2 / Q122 R1 work with dynamic KV balancing, tiny/normal/large PP routing, and Large-PP tail prefetch.

## Public upstream base

- upstream llama.cpp commit: `bebc9350ecc42a31ad119da1513998386671cf5b`
- current custom HEAD before working-tree changes: `b96fdd6fc3f610b77c2389ee6da645dfd0c7018a`
- branch: `q122-qwen38-upgrade-bebc9350`

The full patch in `patches/hotseat-dynkv-largepp-tailprefetch-from-bebc9350.patch` is generated directly from the public upstream base to the current production working tree.

## Main changes

1. Growable sparse/VMM KV allocation for 256K contexts.
2. Dynamic expert/KV VRAM balancing with a low-water reserve.
3. Tiny incremental PP fast path for cached contexts:
   - cache >= 32K
   - uncached suffix <= 64 tokens
   - serialize one prompt token at a time so the Hybrid/HotSeat decode path is used.
4. Large PP detection:
   - uncached suffix >= 4096 tokens
   - temporarily retire elastic expert memory during multi-token prefill.
5. Large PP tail prefetch:
   - start when <= 8192 prompt tokens remain
   - H2D copies run on a non-blocking side stream
   - preserve PP headroom
   - publish restored expert tables only after prefetch completion
   - cancel speculative staging first if KV growth needs memory.
6. HotSeat V2 generic Large-PP suspend/prefetch/resume support for Qwen3.6 text/vision.
7. Q122 suspend hysteresis fix so a Large-PP suspend is not misread as a context shrink and immediately re-borrowed.

## Production routing policy

- Tiny PP: cached context >= 32768 and suffix <= 64 tokens.
- Normal PP: 65..4095 uncached tokens.
- Large PP: >= 4096 uncached tokens.
- Large PP tail prefetch starts at <= 8192 tokens remaining.

## Current production knobs

Ornith text:

```text
-b 4096
-ub 4096
-fa on
Q122_DYNKV_VRAM_RESERVE_MB=600
Q122_DYNKV_VRAM_LOW_MB=600
Q122_LARGE_PP_PREFETCH_HEADROOM_MB=2048
```

Vision models use a more conservative Large-PP headroom of 3072 MiB. Exact model blocks are in `config-snippets-20260907.yaml`.

## Observed results

### Ornith 1.5 text

- fresh 100K PP: about 388.84 tok/s
- warm large incremental PP, cached 95,914 + new 12,096: 141.15 tok/s
- TG after that request: 46.96 tok/s
- Large PP suspend freed about 5.7 GiB
- tail prefetch staged about 4.49 GiB before PP completed
- only about 1.43 GiB remained for resume-time H2D
- HIP free after resume: about 606 MiB
- Linux/sysfs free after a subsequent tiny request: about 406 MiB
- tiny cached request, 108K cache + 1 token: prompt evaluation about 91 ms, TG about 46.31 tok/s

### Qwen3.6 35B text

- fresh PP: about 971 tok/s
- warm incremental PP: about 859 tok/s
- TG: about 57.5 tok/s
- tail prefetch can stage about 14.2 GiB before PP completion

### Qwen3.6 35B vision

- fresh PP: about 907 tok/s
- TG: about 55.6 tok/s
- tail prefetch staged about 11.95 GiB with 3072 MiB PP headroom

### Ornith 1.5 vision

After the suspend-hysteresis fix and enabling 4096/4096 + FA:

- fresh PP: about 451.95 tok/s
- warm large incremental PP: about 113.24 tok/s
- TG: about 50.03 tok/s
- compared with the earlier warm incremental PP around 60.74 tok/s, this is roughly an 86% PP improvement.

## 240K context validation

Ornith text completed a direct 240K prompt without the old 128K reserve/grow failure:

- PP: about 307.19 tok/s
- TG: about 35.09 tok/s
- no `cannot reserve`, `failed to grow`, memory fault, or compute error.

The runtime `cudaMemGetInfo()` free value was observed to read roughly 180 MiB higher than Linux/sysfs on this setup, so production low-water values are deliberately conservative.

## Safety / rollback

Local backups used during development:

```text
/app/share/backup/ornith-largepp-tail-prefetch-20260907-094009
/app/share/backup/hotseat-v2-largepp-prefetch-20260907-095621
/app/share/backup/q122-largepp-suspend-hysteresis-20260907-100743
/app/share/backup/ornith-vision-pp4096-20260907-101812
```

The full patch and config snapshot in this repository are the portable migration artifacts. Do not rely on local backup paths on another machine.
