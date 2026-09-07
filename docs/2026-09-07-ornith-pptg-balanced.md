# Ornith PP/TG balance update (2026-09-07)

This update changes memory use and execution by inference phase. It builds on the existing CPU/GPU expert split and dynamic KV cache. The measured improvement is prompt processing (PP), while token generation (TG) stays approximately unchanged in the main Ornith replay.

## Scope and reproducibility

- Host: Ryzen 9 9950X, 192 GB RAM, RX 7900 XTX 24 GB, gfx1100, Linux, ROCm 10.0 toolchain.
- Model: Huihui Ornith-1.5-35B-A3B Q8_0, context capacity 262144.
- Main comparison: identical weights, routing, precision, b=4096, ub=4096, CPU settings and sampling.
- No SSD expert spill, new draft model, or reduction of weight/KV precision was introduced.
- Codex assisted implementation, packaging and writing; measurements are from the local host.

Public upstream base: `bebc9350ecc42a31ad119da1513998386671cf5b`.

Apply [the full patch](../patches/ornith-pptg-balanced-from-bebc9350.patch) to a CLEAN checkout of that base. It includes the existing HotSeat/DynamicKV foundation. **Do not stack it on older full patches.**

The [working delta](../patches/20260907-pptg-working-delta-from-b96fdd6fc.patch) is relative to custom commit `b96fdd6fc3f610b77c2389ee6da645dfd0c7018a` and also contains earlier uncommitted changes. Do not apply it on top of the previously published 20260907 working delta.

The full patch passed an index-only apply check on the public base; its resulting tracked tree matches current source exactly. See [manifest](../patch-manifest-ornith-pptg-20260907.json). Publication did not restart production services.

## Why the phases need different layouts

TG repeatedly runs small, single-token work. The existing hybrid path keeps useful experts on the GPU and computes selected misses on the CPU. Fixed and elastic expert residency avoids repeated transfers.

Large PP needs GPU workspace and batched matrix operations. The previous R1 placement rule could keep host expert tensors on the CPU even after elastic experts were suspended. Freeing VRAM alone did not make those operators run efficiently on the GPU.

Small follow-ups need another policy: unloading and restoring a large expert bank can cost more than simply using the warm decode path.

## Implemented changes

1. Opt-in Q8_0 batched MUL_MAT_ID placement exception for host expert tensors.
2. Copy an expert weight tensor into one reusable GPU staging allocation before existing MMQ execution. The tested allocation is 272 MiB: a tensor/bank buffer, not one individual expert or the entire model.
3. Release dedicated staging at the final PP boundary before budgeting the elastic decode bank. Do not strand it in a generic caching pool.
4. Disable HIP graph capture for PP subgraphs that use this reclaimable pointer; retain existing TG graph execution.
5. Group destination allocations, then submit restoration copies; prioritize high-utility layers. Retain asynchronously copied pointer tables until synchronization completes.
6. Route short text additions through the single-token hybrid path; account for checkpoint-induced batch splitting.
7. For Vision, serialize short text spans before an image while keeping image embeddings batched.

This does not implement new double-buffered overlap of expert transfers and GPU compute, or multi-token CPU/GPU expert splitting during PP.

## Phase policy

Ornith text uses the existing single-token path for up to 64 uncached tokens, including empty prompt cache. At least 65 uncached tokens triggers larger-PP expert suspension. Actual expert batches of 32 tokens or more can use Q8 GPU staging. These two thresholds measure different things: a request may be split into smaller batches by checkpoints.

Near PP completion, prefetch decode experts with workspace headroom. At the final boundary, release staging and synchronize/publish restored expert tables before decoding.

```text
Q122_PP_STREAM_GPU=1
Q122_PP_STREAM_MIN_TOKENS=32
Q122_PP_STREAM_RECLAIM=1
Q122_LARGE_PP_PIPELINED=1
Q122_INCREMENTAL_PP_MAX_TOKENS=64
Q122_INCREMENTAL_PP_MIN_CACHE=0
Q122_LARGE_PP_MIN_TOKENS=65
Q122_LARGE_PP_PREFETCH_TAIL_TOKENS=8192
Q122_LARGE_PP_PREFETCH_HEADROOM_MB=2048
```

Vision additionally uses Q122_MM_TEXT_CHUNK_SERIAL=1, a large-PP threshold of 129, and 3072 MiB prefetch headroom. [Example config blocks](../config-snippets-ornith-pptg-20260907.yaml) use placeholder model roots. Provide matching model-specific profiles; the YAML alone does not contain these data files.

Historical Q122/R1 names remain in code and switches. This evolves that implementation rather than removing all of its internals.

## Main Ornith measurements

Two rounds per version: seed with 32768 input tokens and generate 8; reuse 32775 evaluated tokens, append 8192 input tokens, generate 128. Then test a one-token follow-up.

| Metric | Before | After |
|---|---:|---:|
| PP for cached context plus 8192 input tokens | 152.193 tok/s | 1176.204 tok/s |
| Whole 8192+128 request | 56.808 s | 9.878 s |
| TG after large PP | 42.881 tok/s | 43.683 tok/s |
| TG after one-token follow-up | 45.960 tok/s | 45.071 tok/s |

PP improves about 7.73x in this workload; total request time falls about 82.6%. TG is approximately unchanged, including a 1.9% decrease in the follow-up measurement.

The workload repeats Chinese/code text. This is not a diverse benchmark or a guarantee for arbitrary prompts. All six main input/output hashes matched, but some separate short-request outputs differed. Unchanged weights do not imply universally bit-exact floating-point behavior.

### Long context and cancellation

A 240000-token input completed at PP 625.74 tok/s, with following TG 33.25 tok/s. Final free VRAM was 463.77 MiB. Minimum sampled free VRAM across the combined boundary/long-context check was 428.83 MiB at 0.2 s intervals. Shorter transient peaks are not excluded. Other tested shapes left 600-840 MiB unused; an always-400-to-600-MiB invariant was not demonstrated.

After cancelling and resubmitting a large suffix, the recorded case reused 249082 tokens and only processed 4 again. Cancellation waits for in-flight GPU work to complete. There was no old-version 240k comparison in this round, so no speedup factor at that length is claimed.

### Short-request tradeoff

The first 8-token request after load had TTFT 0.217 -> 1.013 s while total time was 1.512 -> 1.563 s. Initialization moved earlier. Other tested 32/64/128/512-token requests avoided older long stalls.

Later Qwen migration improved 4k PP but not every short request or small-image case. The main headline here is specifically the Ornith replay.

## Apply and build

```bash
git clone https://github.com/wpu911/hot-seat-2-dynamic.git
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp-ornith
git -C llama.cpp-ornith checkout bebc9350ecc42a31ad119da1513998386671cf5b
git -C llama.cpp-ornith apply --check ../hot-seat-2-dynamic/patches/ornith-pptg-balanced-from-bebc9350.patch
git -C llama.cpp-ornith apply ../hot-seat-2-dynamic/patches/ornith-pptg-balanced-from-bebc9350.patch
ROCM_ROOT=/opt/rocm bash hot-seat-2-dynamic/rebuild-ornith-rocm.sh "$PWD/llama.cpp-ornith"
```

Set ROCM_ROOT to the actual installation. The tested compiler was under /opt/host-rocm/core-10.0/lib/llvm/bin/clang++. Production used incremental compilation with frozen unchanged objects. The helper follows recorded CMake settings; a fresh full ROCm rebuild was not performed for this publication.

The recorded GGML_HIP_NO_VMM=ON build setting is separate from the custom dynamic-KV allocation mechanism.

## Replay

Use a dedicated idle server and its direct port. The script checks slots but clears prompt cache at each seed request; do not use an active conversation server.

```bash
python3 benchmarks/ornith-pptg-20260907/replay.py --label my-run --port 5814
```

Default is the recorded two-round 32k+8k workload. It writes to the ignored runs/ directory and does not run the 240k test. The [benchmark folder](../benchmarks/ornith-pptg-20260907/) includes exact token IDs and raw main, short, long and cancellation results. Token IDs require the matching tokenizer.

## Independent deployment

Each of the four local 35B entries now has private program/library copies under its weight directory: runtime-text/20260907-01/bin and runtime-vision/20260907-01/bin, separately under Ornith and Qwen roots. File inodes, internal symlinks and actual inference-library mappings were checked. RUNPATH is local to $ORIGIN; ROCm and system drivers remain shared. Isolation enables separate upgrades; it is not a speed optimization by itself.

This repository publishes patches, examples and measurements, not model weights. Upstream license text is retained in [licenses](../licenses/llama.cpp-LICENSE).
