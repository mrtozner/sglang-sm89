# Ada Lovelace (`sm_89`) vs. Blackwell (`sm_120`) Architecture & Kernel Analysis

This document compares NVIDIA Ada Lovelace (`sm_89`) workstation/consumer GPUs against Blackwell (`sm_120`) PCIe platforms for serving hybrid Gated DeltaNet + Full Attention models (`Qwen3.5` / `Qwen3.8-27B`) with speculative decoding (`DFlash2` / `MTP`) and constrained `xgrammar` generation, and outlines the clean-room `sglang-sm89` v0.2.0 architecture.

---

## 1. Hardware Matrix: `sm_89` Lineup vs. `sm_120` PCIe

| GPU | Arch | SMs | VRAM | Memory Bus & Peak BW | TDP | Dense FP8 E4M3 (FP32 accum) | Dense BF16/FP16 (FP32 accum) | Key Serving Profile |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **RTX 4090 (24 GB)** | `sm_89` | 128 | 24 GB GDDR6X | 384-bit, `1,008.1 GB/s` | 450 W | `~330 TFLOPS` | `~165 TFLOPS` *(1/2 rate)* | Fits `9B–14B` FP8 + `DFlash2`, or `27B` hybrid `W4A16` / `EXL3` at `c=1..4` |
| **RTX 4090 D (48 GB)** | `sm_89` | 114 | 48 GB GDDR6X | 384-bit, `1,008.1 GB/s` | 300 W | `294.2 TFLOPS` *(boost `362.5`)* | `147.1 TFLOPS` *(1/2 rate)* | Fits `27B` FP8 + FP8 `DFlash2` + `172K+` `fp8_e4m3` KV tokens at `c=8` |
| **RTX 6000 Ada** | `sm_89` | 142 | 48 GB GDDR6 ECC | 384-bit, `960.0 GB/s` | 300 W | `~728 TFLOPS` | `~91.1 TFLOPS` FP32 / full Tensor | Workstation ECC 48 GB; faster prefill (`142` SMs), slightly lower BW than GDDR6X |
| **NVIDIA L40S** | `sm_89` | 142 | 48 GB GDDR6 ECC | 384-bit, `864.0 GB/s` | 350 W | `~733 TFLOPS` | Full datacenter Tensor rate | Passive server 48 GB; high prefill MFU, lower decode BW (`864 GB/s`) |
| **RTX 5090 (32 GB)** | `sm_120` | 170 | 32 GB GDDR7 | 512-bit, `1,792.1 GB/s` | 575 W | Native FP8 + **NVFP4** | Native + **PDL** | `1.78x` decode BW vs 4090, but 32 GB squeezes `27B` KV (requires `NVFP4` KV) |
| **RTX PRO 6000 Blackwell** | `sm_120` | — | 96 GB GDDR7 ECC | 512-bit, `~1.6–1.8 TB/s` | ~600 W | Native FP8 + **NVFP4** | Native + **PDL** | Frontier PCIe capacity (`96 GB`); `rtx6kpro` / `b12x` / FlashInfer FP4 MoE stack |

### Architectural Implications for `sm_89`

1. **No Hardware `NVFP4` or Programmatic Dependent Launch (PDL):**
   - `cvt.rn.f16x2.e2m1x2` (hardware FP4 unpack) requires `__CUDA_ARCH__ >= 1000` (`sm_100`/`sm_120`).
   - `griddepcontrol.wait` / `launch_dependents` (PDL kernel tail overlap) requires `__CUDA_ARCH__ >= 900` (`sm_90+`).
   - Consequently, Blackwell-oriented engines (such as `b12x`-based CUTLASS FP4/MoE paths used in `rtx6kpro` PCIe workstation setups, or native `NVFP4` single-card engines) cannot run their 4-bit Tensor Core paths on `sm_89`.

2. **2× FP8 E4M3 Compute Advantage on Consumer `sm_89` (`294.2` vs `147.1` TFLOPS):**
   - On GeForce Ada (`RTX 4090` / `4090 D`), FP16/BF16 Tensor Cores with FP32 accumulation are throttled to **1/2 rate** (`147.1 TFLOPS` at 2.52 GHz on 114 SMs), whereas **FP8 E4M3 (`mma.m16n8k32.f32.e4m3.e4m3.f32`) runs at full `294.2 TFLOPS`** (`2.0×`).
   - Rather than relying on static pre-baked Block-FP8 JSON tables, `sglang_sm89/autotune_fp8.py` derives unique `[128, 128]` `(N, K)` shapes from the target model and runs live `triton.testing.do_bench` sweeps on the user's GPU (`BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages`), caching schedules under `${SGLANG_SM89_CONFIG_CACHE:-~/.cache/sglang-sm89/fp8_configs}` (with optional SRAM-constrained `<= 100 KiB` analytical fallback).
   - Meanwhile, `sglang_sm89/draft_fp8.py` (`QuantizedLinear8` & `DraftHeadFP8`) quantizes `DFlash2` draft linears and the draft `lm_head` *after* `_init_fused_kv_helper`, preserving fused KV materialization while unlocking Ada's `2×` FP8 Tensor Core path.

3. **Why 48 GB `sm_89` (`4090 D` 48 GB / `6000 Ada` / `L40S`) Excels at Lossless KV Capacity:**
   - On 32 GB `RTX 5090`, fitting a `27B` model + speculative drafter + `c=8` context requires compressing the KV cache to `NVFP4`.
   - On 48 GB `sm_89` (`+50%` VRAM), keeping the target in `FP8` + `DFlash2` in rowwise `FP8` (`draft_fp8.py`) + circular GDN ReplaySSM buffers (`gdn_pool_bridge.py` + `wy_gdn_sm89.cu` / `wy_gdn_verify.py`) retains **`172K+` lossless `fp8_e4m3` KV tokens**, while keeping `embed_tokens` (`~2.37 GiB`) resident in VRAM to avoid PCIe Gen4 `x4`/`x16` host-offload stalls (`~7.88 GB/s` on `x4` risers).

4. **Why 24 GB `RTX 4090` Requires Tuned Profiles:**
   - With `24 GiB` VRAM, a `27B` FP8 checkpoint (`~28.4 GiB` with dense heads) overflows single-GPU memory. 24 GB `RTX 4090` profiles (`profiles/rtx4090-24gb.yml`) target `9B–14B` FP8 models with `DFlash2` + `xgrammar`, or `27B` hybrid `W4A16` / `EXL3` checkpoints at tighter concurrency (`c=1..4`).

---

## 2. Grammar-Sparse Vocabulary Projection: Batch-Union GSP vs. `kTileV=64` Unified-64 Compaction

### Why Naive Batch-Union GSP Fails on Mixed-Density Batches
In speculative `TARGET_VERIFY` (`M = B × (draft + 1)`, e.g., `M = 8 × 8 = 64`), grammar constraints operate independently per request row `m ∈ [0, M)`:
- Approaches modeled on batch-union Grammar-Sparse Projection (e.g., `StealthEyeLLC/grammar-sparse-projection`) compute a single global bit-union across all rows $U = \bigvee_{m=0}^{M-1} M_m$ and launch a sparse projection over all tokens in $U$.
- **Failure mode:** In real `c=8` serving, batches frequently mix **sparse structural positions** (`56` rows inside JSON keys or SQL syntax where `|V_valid| ≈ 1..250`, `<0.1%` of `248,320`) with **dense positions** (`8` rows from an unconstrained request or inside a free-text JSON string `"answer": "..."` where `~85%` of tokens are valid).
- The global union $|U|$ immediately collapses to `~85–100%` of the vocabulary, forcing the naive batch-union kernel to execute `~6.8× V` effective weight reads (much slower than cuBLAS `torch.mm`).

### Solution: `kTileV=64` Unified-64 Compaction (`sparse_vocab_proj_sm89.cu`) + Roofline Dispatcher (`vocab_projection.py`)
1. **`kTileV = 64` Single-`uint64_t`-per-Tile Compaction (`M <= 64`):**
   - Each CTA (`128` threads = `4` warps, `__launch_bounds__(128, 4)`) processes `kTileV = 64` vocab rows (`3,880` CTAs for `V = 248,320`), where each batch row `m` contributes `2` `int32_t` mask words = **`1` `uint64_t` bitmask** (`tile_tail = (1ULL << rem) - 1ULL` on the final partial tile).
   - If the per-tile `uint64_t` union `s_tile_union == 0ULL`, inactive output slots are vector-filled with `-inf` and the CTA returns immediately with **zero** weight DRAM reads.
   - Otherwise, active vocab rows `[0..63]` are compacted via `__ballot_sync` across warps `0..1`, distributed across all `4` warps, and active batch rows `m ∈ [0, 64)` are compacted in a **single pass** via `uint64_t` warp ballot (avoiding two-pass `m_blk = {0, 32}` duplicate weight reads), streaming `uint4` weight loads by value (`#pragma unroll 1`, `__builtin_memcpy`) with **zero register spills** (`0 bytes stack frame, 0 bytes spill stores/loads`). Safe 32-lane chunked fallback handles `M > 64` (`M = 256, 257, 289`) and CUDA Graph replay cleanly.
2. **Sub-10-µs C++ Effective-Row Roofline Dispatcher (`select_sparse_vocab_proj_cpu`):**
   - Scanning `64`-row `uint64_t` tiles (`__builtin_popcountll`), the dispatcher computes exact effective weight-row reads:
     $$\mathrm{EffRows}(M, \mathcal{M}) = \sum_{c=0}^{\lceil V/64 \rceil - 1} \mathrm{popcnt}(u_c) \cdot \left\lceil \frac{a_c}{K_{\text{chunk}}} \right\rceil,$$
     exiting early as soon as $\mathrm{EffRows} > V \cdot \rho_{\max}$ and routing dense/mixed batches straight to cuBLAS `torch.mm`.

---

## 3. Overcoming Parallel Draft Independence: Coupled Resampling + Buffer Isolation + $O(1)$ SAM Grafting

### The Independence Limitation (`Weaver / Trees from Marginals`)
Parallel block drafters such as `DFlash2` predict marginal candidate distributions $q_j(v)$ simultaneously across positions $j = 1 \dots B-1$ conditioned on the committed prefix. As analyzed in *Weaver: Trees from Marginals*, factorizing $q(y_{1:B-1}) = \prod_j q_j(y_j)$ ignores intra-block conditional dependencies $P(y_j \mid y_{<j})$, degrading acceptance on longer blocks (`B = 8..16`). While *Weaver* solves this by training a ~57M-parameter autoregressive adapter over top-$K$ marginals, training/loading an extra adapter costs VRAM and latency.

### Zero-Training-Cost Solution (`coupled_grammar.py` + `sam_drafter.py`)
1. **Coupled Conditional Grammar Resampling + Eager/Graph Buffer Isolation (Theorems 1C & 1D):**
   - `sglang_sm89/coupled_grammar.py` walks the `xgrammar` PDA sequentially across $j = 0 \dots B_{\text{draft}}-1$ on CPU using the parallel drafter's top-$K$ candidates, enforcing hard structural dependencies via **maximal coupling** (Theorem 1C) on sampled (`T > 0`), greedy (`T = 0`), and mixed batches plus forced `|V_valid| == 1` jumps, guarded by `try...finally: rollback(accepted_cnt)`.
   - **Buffer Isolation (Theorem 1D):** Strictly isolates `worker._selector_sample` from stale `_draft_sampler` ring buffers during eager execution (`can_run_graph == False`), guaranteeing zero stale-probability cross-contamination (`TV = 0`).
2. **$O(1)$ Suffix Automaton (SAM) Tail Grafting (Theorem 5B, `sam_drafter.py`):**
   - Opt-in (`SGLANG_SM89_SAM_GRAFT=0` default): whenever a parallel draft tail slot has low confidence ($\max_v q_j(v) < \tau$) and the committed + drafted prefix matches a substring in the prompt or committed history (`O(1)` transition, gated by `xgrammar`), we graft the exact continuation token onto slot 0 via duplicate-free swap (`place_unique_at_slot0`), restoring multi-token continuity losslessly (`TV = 0`).
