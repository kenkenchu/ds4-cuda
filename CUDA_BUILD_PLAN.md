# CUDA Build Plan

This document tracks the port from the current Metal graph executor to a CUDA
executor optimized for NVIDIA DGX Spark / GB10.

Target platform:

- GPU: NVIDIA GB10, Blackwell, compute capability `sm_121`
- CUDA: 13.1 or newer
- CPU: ARM64 / aarch64
- Memory: 128 GB unified CPU/GPU memory
- OS: Ubuntu 24.04

## Current State

Completed:

- Linux builds compile and link against CUDA with `nvcc`.
- Default CUDA architecture is `sm_121`.
- `DS4_BACKEND_CUDA` exists in the public backend enum.
- CLI and server accept `--cuda` and `--backend cuda`.
- CUDA runtime initialization validates the selected CUDA device and reports
  memory.
- CUDA tensor/runtime helpers exist for allocation, views, copies, writes,
  reads, memsets, streams, and synchronization.
- CUDA primitive kernels exist for:
  - `add`
  - `repeat_hc`
  - `rms_norm_plain`
  - `embed_token_hc`
  - `embed_tokens_hc`
  - `rms_norm_weight`
  - `head_rms_norm`
  - `matmul_f16`
  - `matmul_f16_pair`
  - `matmul_f32`
  - `matmul_q8_0`
  - `matmul_q8_0_pair`
  - `hc_post`
  - `rope_tail`
  - `hc_pre_from_state`
- CUDA router selection kernels exist for hash-mode and top-k expert routing.
- Existing test target passes on Linux.

Not completed:

- CUDA generation and server sessions still fail explicitly instead of running
  the model.
- CUDA session allocation is not enabled.
- CUDA attention, MoE, and full decode/session orchestration are not
  implemented yet.
- CUDA prefill/decode correctness is not validated against CPU or official
  vectors.

Progress notes:

- The current CUDA work is a true device-side kernel layer, not a CPU fallback.
- The Metal implementation is being treated as a reference for model behavior,
  not as a required runtime target for CUDA work.
- The remaining gap is the graph/session executor and the last model-specific
  layer kernels needed to drive generation end-to-end.

## Porting Strategy

Keep the port incremental, but aim squarely at CUDA and Blackwell. Do not build
a backend-neutral abstraction layer. The existing Metal path is useful as a
reference implementation and should remain stable, but the new work should
create CUDA-native runtime, graph, and kernel code optimized for GB10.

The CUDA path should reuse DS4 model loading, tokenizer, public engine/session
APIs, tests, and CPU correctness references. It should not force Metal and CUDA
through a lowest-common-denominator runtime interface.

Do not hide missing CUDA work with CPU fallback. A CUDA backend failure should
be explicit until that stage is implemented and tested.

## Milestone 1: CUDA Runtime and Tensor Layer

Goal: add CUDA-native tensor and command utilities for the CUDA graph executor.
This layer is CUDA-specific and should be designed for GB10, not for backend
portability.

Tasks:

- Introduce CUDA-specific tensor names, for example `ds4_cuda_tensor`.
- Implement CUDA tensor allocation, view, free, bytes, host write, host read,
  device copy, memset, and synchronization helpers.
- Add CUDA stream ownership for prefill/decode work.
- Add lightweight launch/error-checking macros that include file, line, and
  kernel name.
- Use CUDA unified memory or managed allocations only where it is measurably
  better than explicit device allocation on GB10.
- Keep mmap model weights host-resident initially. Add CUDA weight staging only
  after kernel correctness is established.
- Do not rename or wrap Metal tensor code as part of this milestone.

Validation:

- `make test`
- Dedicated tensor tests for write/read/copy/view boundaries.
- Run under `cuda-memcheck` or compute-sanitizer for tensor tests.

Exit criteria:

- CUDA-owned activation/KV scratch tensors can be allocated and validated
  independently of the Metal graph.

Status:

- Done. CUDA tensor/runtime helpers are in place and covered by tests.
- The model mmap is still host-resident; CUDA kernels stage host weights to
  device memory as needed.

## Milestone 2: Primitive Kernel Bring-Up

Goal: port the smallest graph primitives and compare against CPU references.

Suggested order:

1. Embedding expansion into HC state.
2. Plain row operations: add, repeat, SwiGLU.
3. RMS norm: plain, weighted, row variants.
4. RoPE tail and FP8/F16 KV rounding.
5. F16 matmul.
6. Q8_0 matmul.
7. Router selection.
8. Attention raw prefill/decode.

Implementation notes:

- Prefer simple correct kernels first. Optimize after parity is stable.
- Use `__half` and vectorized loads where alignment permits.
- Compile for `sm_121`; do not optimize for older architectures unless needed.
- Keep CUDA kernel launch wrappers close to the DS4 operation shape, not to a
  generic backend abstraction.

Validation:

- Add per-primitive tests that compare CUDA output to CPU output.
- Use fixed deterministic inputs with max-absolute and relative error bounds.
- Run tests in both fast and `--quality` modes when the primitive has variants.

Exit criteria:

- Primitive tests pass for the first batch without depending on full model
  weights.

Status:

- In progress, with the initial and mid-level primitives implemented:
  - embedding
  - plain and weighted RMSNorm
  - head RMSNorm
  - Q8_0 and F16/F32 matmuls
  - HC post/pre glue
  - basic RoPE tail transform
- Still missing the attention and MoE kernels needed to run the model.
- Router score/top-k selection is now on CUDA and covered by tests.

Concrete next steps:

1. Finish the remaining layer-local CUDA kernels that the forward path needs:
   - routed expert gate/up matmuls
   - routed expert down matmul
   - shared expert Q8 path
   - fused shared-down + HC expand path
2. Port attention and KV cache behavior to CUDA:
   - raw SWA store/load
   - compressor state updates
   - ratio-4 compressed KV replay
   - indexer score computation and compressed-row masking
   - raw plus compressed attention for prefill and decode
3. Introduce CUDA session state and wire it into `ds4_session`:
   - CUDA graph/session struct
   - CUDA buffer allocation for context and prefill cap
   - CUDA prefill and one-token decode
   - checkpoint and session resume semantics
4. Route `ds4_engine_generate_argmax()` to the CUDA session path and keep
   logits readback on the host only for sampling and top-logprobs.
5. Add CUDA generation tests for short greedy decode, session extension,
   checkpoint rewrite/rebuild behavior, and payload save/load.

## Milestone 3: Quantized Matmul and MoE Kernels

Goal: port DS4-specific quantized expert paths.

Kernels:

- `IQ2_XXS` routed gate/up expert matmul.
- `Q2_K` routed down expert matmul.
- `Q4_K` routed expert path if q4 support is required.
- Shared expert Q8 path.
- Fused shared-down + HC expand path.

Optimization priorities for GB10:

- Maximize coalesced reads from quantized weight blocks.
- Avoid per-token heap allocation and CPU staging.
- Use persistent scratch tensors sized by prefill chunk.
- Tune block sizes for Blackwell occupancy, register pressure, and memory
  bandwidth rather than inheriting Metal threadgroup sizes.

Validation:

- Compare routed expert output against CPU for selected expert IDs and weights.
- Compare router top-k and weights exactly or within documented tolerance.
- Test q2 first because it is the 128 GB target quant.

Exit criteria:

- One full FFN block can run on CUDA and match CPU within tolerance.

Status:

- Not started as a full block yet.
- The basic Q8_0/F16/F32 projections and router selection are in place, but
  the routed expert matmuls and fused shared expert path are still missing.

## Milestone 4: Attention and Compressed KV

Goal: implement DS4 compressed KV behavior and attention.

Tasks:

- Raw SWA cache store/load.
- Compressor state update.
- Ratio-4 compressed KV replay.
- Indexer score computation and top-k mask.
- Mixed raw + compressed decode attention.
- Chunked prefill attention.

Optimization priorities:

- Keep raw and compressed KV resident on GPU.
- Use chunk sizes that fit GB10 memory pressure with 128 GB unified memory.
- Avoid materializing large attention masks when indexed top-k is sufficient.
- Revisit default prefill chunk size after profiling; `2048` is inherited from
  the Metal path and may not be optimal on GB10.

Validation:

- Short-context raw attention parity.
- Long-context compressed attention parity.
- Save/load session payload tests after CUDA tensors are wired.

Exit criteria:

- CUDA can prefill a prompt and produce logits matching CPU/Metal tolerance.

Status:

- Not started.
- This milestone is the main remaining model-behavior gap after the basic
  primitive kernels.

## Milestone 5: Full Decode Session

Goal: make `ds4_session_create`, `ds4_session_sync`, and `ds4_session_eval`
work on CUDA.

Tasks:

- Add CUDA graph state to `ds4_session`.
- Allocate CUDA graph buffers for context size and prefill cap.
- Implement prefill and one-token decode using CUDA primitives.
- Read logits back only when sampling/top-logprobs needs host access.
- Preserve existing checkpoint semantics.

Validation:

- CLI greedy generation for short prompts.
- Session extension without rebuild.
- Session rewind/invalidate paths.
- Disk KV save/load if server support is enabled.

Exit criteria:

- `./ds4 -p "..." --cuda -n 32 --nothink` generates tokens without CPU graph
  fallback.

Status:

- Not started.
- This is the first milestone that actually replaces the Metal graph/session
  execution path with CUDA-driven generation.

## Milestone 6: Server Enablement

Goal: run OpenAI/Anthropic-compatible server on CUDA.

Tasks:

- Enable CUDA session creation in server startup.
- Ensure disk KV persistence reads/writes CUDA tensor state safely.
- Verify long-prefix reuse with CUDA graph state.
- Add CUDA-specific startup logs for memory budget and context estimate.

Validation:

- `./ds4-server --cuda --ctx 100000`
- OpenAI-compatible streaming request.
- Anthropic-compatible `/v1/messages` request.
- KV disk cache cold save/load.

Exit criteria:

- Server can run sustained local agent traffic without graph rebuild bugs or
  memory growth.

## Milestone 7: Performance Tuning

Goal: optimize for GB10 after correctness is stable.

Focus areas:

- Prefill chunk size.
- Decode latency per token.
- Quantized expert throughput.
- Attention bandwidth and KV layout.
- Host/device synchronization frequency.
- Weight staging policy.
- Unified memory page migration behavior.

Measurements:

- Short prompt prefill tokens/sec.
- Long prompt prefill tokens/sec.
- Decode tokens/sec at 32K, 100K, and 300K context.
- Peak and steady-state CUDA memory.
- End-to-end server latency under single-client agent workload.

Tools:

- Nsight Systems for synchronization and host/device overlap.
- Nsight Compute for kernel occupancy and memory throughput.
- Compute Sanitizer for memory correctness.

Exit criteria:

- CUDA q2 path is usable within the 128 GB unified-memory envelope.
- Performance is competitive enough to replace CPU/debug usage on DGX Spark.

## Build Commands

Default Linux CUDA build:

```sh
make clean
make -j$(nproc)
```

Explicit GB10 build:

```sh
make clean
make -j$(nproc) CUDA_ARCH=sm_121 CUDA_HOME=/usr/local/cuda
```

Tests:

```sh
make test
```

CUDA smoke test after model is available:

```sh
./ds4 --cuda --inspect -m ds4flash.gguf
```

Expected behavior today: CUDA initializes, but generation/session execution is
not yet enabled.

## Guardrails

- Do not push CPU fallback behind `--cuda`.
- Keep CUDA failures explicit until the relevant stage is implemented.
- Preserve Metal behavior while porting.
- Keep CPU reference paths available for correctness comparison.
- Prefer correctness and deterministic tests before fused-kernel optimization.
- Do not tune for q4 before q2 works on the 128 GB target.
