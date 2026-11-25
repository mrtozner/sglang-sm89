# sglang-sm89

**Clean-room Ada Lovelace (`sm_89`) acceleration plugin for SGLang (`v0.5.23`)** — native `sm_89` CUDA kernels, a live GPU Block-FP8 autotuner, and `HookRegistry` runtime overlays for `RTX 4090`, `RTX 4090 D 48GB`, `RTX 6000 Ada`, and `L40S`, requiring **zero modifications** to upstream SGLang source files.

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

---

## Why `sglang-sm89` (`v0.2.0`)?

Modern hybrid linear-attention + GQA models (e.g., Qwen3.5 / Qwen3.8 Gated DeltaNet + GQA) paired with block-parallel speculative decoding (`DFLASH` / `DFlash2`) and CFG/JSON grammar enforcement (`xgrammar`) hit four Ada Lovelace (`sm_89`) bottlenecks out of the box:

1. **Missing `sm_89` Block-FP8 Triton GEMM Configs:** SGLang ships no `[128, 128]` Block-FP8 Triton schedules for `RTX 4090 D` (`114` SMs) and sibling Ada cards, emitting fallback warnings at launch. `sglang-sm89` profiles shapes live on the target GPU (`<= 100 KiB` Ada shared-memory budget) and symlinks those schedules into SGLang's loader path — zero pre-baked third-party config JSONs shipped.
2. **Unquantized BF16 `DFlash2` Draft Path + Blocked `ReplaySSM`:** Stock `DFLASH` keeps the 21 `DFlash2` draft linears and draft `lm_head` in BF16 (while `--speculative-draft-model-quantization w8a8_fp8` disables fused KV materialization), and SGLang blocks `GDN` `ReplaySSM` under `DFLASH`, inflating `intermediate_ssm_state_cache` and cutting KV capacity by `4.5×`.
3. **Grammar-Blind Speculation & Dense Verify `lm_head`:** Grammar-unaware speculative drafting triggers cascade rejections, while dense `248,320`-vocab BF16 logits waste memory bandwidth on constrained decode tokens (and naive batch-union projection slows down mixed structured + open-string batches).
4. **Triton GDN `ReplaySSM` Verify Overhead:** Sequential Triton chunk/recurrent verify under speculative batches leaves Ada Tensor Cores and shared memory underutilized on FP32 and short-token BF16 steps.

---

## Key Components

| Component | Module / Kernel | Role |
|---|---|---|
| **Live GPU Block-FP8 Autotuner** | [`sglang_sm89/autotune_fp8.py`](sglang_sm89/autotune_fp8.py) | Standalone Triton benchmark & cache linker (`python3 -m sglang_sm89.autotune_fp8`) deriving W8A8 `[128, 128]` schedules on the local `sm_89` GPU within Ada's `<= 100 KiB` SRAM limit. Zero static JSON files shipped. |
| **FP8 `DFlash2` Draft & Head** | [`sglang_sm89/draft_fp8.py`](sglang_sm89/draft_fp8.py) | Quantizes eligible `DFlash2` draft linears via `QuantizedLinear8` **after** fused-KV worker init and attaches `DraftHeadFP8` (keeping target verify BF16 lossless by default). |
| **GDN `ReplaySSM` Pool Bridge** | [`sglang_sm89/gdn_pool_bridge.py`](sglang_sm89/gdn_pool_bridge.py) | Enables `GDN` `ReplaySSM BF16` allocation under `DFLASH` via a bounded sentinel hook and routes `HybridLinearAttnBackend.update_mamba_state_after_mtp_verify` through `ReplaySSM` circular/fold commit (`4.50×` KV pool capacity). |
| **WY GDN `ReplaySSM` CUDA Kernel** | [`csrc/wy_gdn_sm89.cu`](csrc/wy_gdn_sm89.cu) + [`sglang_sm89/wy_gdn_verify.py`](sglang_sm89/wy_gdn_verify.py) | Clean-room `sm_89` WY chunk-parallel forward-substitution kernel (`K=V=128`, circular ring + flush support, CUDA-graph safe, zero `--use_fast_math`, `0` register spills) accelerating `_replayssm_target_verify`. |
| **Coupled Grammar Resampling** | [`sglang_sm89/coupled_grammar.py`](sglang_sm89/coupled_grammar.py) | Theorem 1C/1D overlap-safe `grammar_barrier()` + `\|V_valid\|==1` jump-forward + lossless eager & sampled coupled resampling with `_selector_sample` parity and `try...finally: rollback()` invariance. |
| **Unified-64 `kTileV=64` Sparse `lm_head`** | [`csrc/sparse_vocab_proj_sm89.cu`](csrc/sparse_vocab_proj_sm89.cu) + [`sglang_sm89/vocab_projection.py`](sglang_sm89/vocab_projection.py) | Single-pass `uint64_t` warp-ballot compaction (`M <= 64`) + `kTileV=64` tile early-exit + C++ SIMD/NEON roofline dispatcher (`select_sparse_vocab_proj`) avoiding mixed-density slowdowns. |
| **Optional $O(1)$ SAM Tail Drafter** | [`sglang_sm89/sam_drafter.py`](sglang_sm89/sam_drafter.py) | Opt-in (`SGLANG_SM89_SAM_GRAFT=0`) Suffix Automaton prompt continuation grafting on low-confidence tail draft slots. |

---

## Supported GPUs & Profiles

All Compose profiles bind exclusively to loopback (`127.0.0.1:30000:30000`) — pair with an SSH tunnel or local reverse proxy for remote access.

| GPU | VRAM | SMs | TDP | Architecture Notes | Compose Profile |
|---|---:|---:|---:|---|---|
| **RTX 4090 D 48GB** | 48 GB | 114 | 300 W | `1,008 GB/s` GDDR6X, `294.2 TFLOPS` FP8; fits 27B-FP8 + DFlash2 + `172K+` FP8 KV | [`profiles/rtx4090d-48gb.yml`](profiles/rtx4090d-48gb.yml) |
| **RTX 6000 Ada / L40S** | 48 GB | 142 | 300–350 W | Higher SM count (`c=12`, `ctx=16384` profile) | [`profiles/rtx6000ada-48gb.yml`](profiles/rtx6000ada-48gb.yml) |
| **RTX 4090** | 24 GB | 128 | 450 W | Tuned for 9B–14B FP8 or 4-bit hybrid models inside 24 GB | [`profiles/rtx4090-24gb.yml`](profiles/rtx4090-24gb.yml) |

See [`docs/RTX_4090_6000_ARCHITECTURE.md`](docs/RTX_4090_6000_ARCHITECTURE.md) and [`docs/THEORY_AND_PROOFS.md`](docs/THEORY_AND_PROOFS.md) for the hardware roofline comparison (`sm_89` vs `sm_120`), exact mathematical proofs, and executable verification scripts in [`proofs/`](proofs/).

---

## Quickstart

### 1. Build Docker Image & Generate Live FP8 Configs

```bash
docker build -t sglang-sm89:latest .

# Profile your Ada GPU once into ~/.cache/sglang-sm89/fp8_configs (no static JSONs shipped)
docker run --rm --gpus all \
  -v ~/.cache/sglang-sm89/fp8_configs:/root/.cache/sglang-sm89/fp8_configs \
  sglang-sm89:latest \
  python3 -m sglang_sm89.autotune_fp8 \
    --model Qwen/Qwen3.8-27B-FP8 \
    --cache-dir /root/.cache/sglang-sm89/fp8_configs

docker compose -f profiles/rtx4090d-48gb.yml up -d
```

### 2. Local / Container Wheel Install

Inside an SGLang `v0.5.23` (`sm_89` CUDA) environment:

```bash
pip install --no-cache-dir --no-build-isolation .
python3 -m sglang_sm89.autotune_fp8 --model Qwen/Qwen3.8-27B-FP8
```

SGLang automatically discovers `sglang_sm89.plugin:register` via the `sglang.srt.plugins` / `sglang.plugins` entry points at server launch.

---

## Environment Toggles

| Variable | Default | Description |
|---|---:|---|
| `SGLANG_SM89_ENABLE` | `1` | Master plugin enable/disable (`0` disables all hooks) |
| `SGLANG_SM89_FP8_CONFIGS` | `1` | Link cached/autotuned Block-FP8 `[128, 128]` schedules on startup |
| `SGLANG_SM89_SYNTHESIZE_UNTUNED_FP8` | `0` | Opt-in conservative Ada SRAM-bounded fallback when no autotuned cache exists |
| `SGLANG_SM89_DFLASH_FP8` | `1` | Rowwise FP8 `DFlash2` draft linears + `DraftHeadFP8` |
| `SGLANG_SM89_REPLAYSSM` | `1` | Unlock GDN `ReplaySSM BF16` allocation & commit under `DFLASH` |
| `SGLANG_SM89_COUPLED_GRAMMAR` | `1` | Coupled grammar resampling + `\|V_valid\|==1` jump-forward |
| `SGLANG_SM89_SPARSE_PROJ` | `1` | Unified-64 `kTileV=64` sparse vocab projection + C++ roofline dispatch |
| `SGLANG_SM89_WY_GDN` | `1` | Native `sm_89` WY GDN `ReplaySSM` verify CUDA kernel |
| `SGLANG_SM89_WY_GDN_MAX_BF16_TOKENS` | `32` | Auto-crossover token limit for BF16 WY GDN (`<=32` CUDA; `>32` Triton) |
| `SGLANG_SM89_FP8_VERIFY_HEAD` | `0` | Opt-in rowwise FP8 target verify `lm_head` (`0` preserves exact BF16 logits) |
| `SGLANG_SM89_SAM_GRAFT` | `0` | Opt-in $O(1)$ Suffix Automaton tail draft grafting |

---

## Measured Benchmarks (`NVIDIA GeForce RTX 4090 D 48GB`, `300 W`, `Qwen/Qwen3.8-27B-FP8`)

All measurements below were recorded on `NVIDIA GeForce RTX 4090 D` (`sm_89`, `300 W` TDP) serving `Qwen/Qwen3.8-27B-FP8` + `z-lab/Qwen3.8-27B-DFlash2` (`draft=8`, `kv_cache_dtype=fp8_e4m3`, `context_length=8192`, `--max-tokens 512`, `xgrammar 0.2.7`, read-only SQLite executable SQL + exact JSON ground-truth validation). Full raw artifacts live in [`benchmarks/results/`](benchmarks/results/).

### 1. Stock SGLang `0.5.23` vs `sglang-sm89 v0.2.0` ([`e2e_stock_vs_sm89_4090d.json`](benchmarks/results/e2e_stock_vs_sm89_4090d.json))

- **KV Pool Capacity (`max_total_num_tokens`):** `38,337` (Stock) → **`172,609`** (`sglang-sm89`, **`4.50×`**)
- **Block-FP8 Fallback Warnings:** `5` (Stock) → **`0`** (`sglang-sm89`, backed by live GPU autotuning in [`fp8_autotune_summary_4090d.json`](benchmarks/results/fp8_autotune_summary_4090d.json))
- **End-to-End Task Validity:** **`100.0%` valid (`4/4` at `c=1`, `8/8` at `c=8`) across every cell on both arms**

| Concurrency | Temp | Workload | Valid | `valid_tasks_per_s` (Stock → `sm89`) | `agg_tok_s` (Stock → `sm89`, Δ) | `per_stream_median` (`tok/s`) | `ttft_p50` (`ms`) | `tok/J` (Stock → `sm89`) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `c=1` | `0.0` | English | `4/4` | `0.162` → **`0.206`** (`+27.16%`) | `75.40` → **`92.50`** (**`+22.68%`**) | `71.13` → **`81.22`** | `103.75` → **`77.98`** | `0.2520` → **`0.3222`** |
| `c=1` | `0.0` | `xgrammar` JSON | `4/4` | `0.700` → **`0.981`** (`+40.14%`) | `83.47` → **`121.19`** (**`+45.19%`**) | `99.46` → **`137.81`** | `163.06` → **`109.41`** | `0.2940` → **`0.4453`** |
| `c=1` | `0.0` | SQL (exec) | `4/4` | `1.071` → **`1.631`** (**`+52.29%`**) | `111.41` → **`159.45`** (**`+43.12%`**) | `129.03` → **`192.31`** | `115.87` → **`103.26`** | `0.3741` → **`0.5504`** |
| `c=1` | `0.6` | English | `4/4` | `0.159` → **`0.194`** (`+22.01%`) | `73.33` → **`96.21`** (**`+31.20%`**) | `70.20` → **`97.84`** | `105.89` → **`92.79`** | `0.2449` → **`0.3217`** |
| `c=1` | `0.6` | `xgrammar` JSON | `4/4` | `0.744` → **`1.057`** (`+42.07%`) | `92.94` → **`118.10`** (**`+27.07%`**) | `102.64` → **`130.50`** | `125.40` → **`111.93`** | `0.3172` → **`0.4200`** |
| `c=1` | `0.6` | SQL (exec) | `4/4` | `0.958` → **`1.430`** (**`+49.27%`**) | `103.49` → **`144.74`** (**`+39.86%`**) | `116.39` → **`180.97`** | `117.10` → **`108.07`** | `0.3458` → **`0.4926`** |
| `c=8` | `0.0` | English | `8/8` | `0.806` → **`1.025`** (`+27.17%`) | `391.94` → **`500.42`** (**`+27.68%`**) | `66.44` → **`78.58`** | `410.73` → **`193.81`** | `1.3251` → **`1.6712`** |
| `c=8` | `0.0` | `xgrammar` JSON | `8/8` | **`3.129`** → `2.315` (`-26.01%`) | **`357.09`** → `263.31` (`-26.26%`)* | `84.38` → **`98.28`** | `732.76` → `2104.08`* | `1.2726` → **`1.3267`** |
| `c=8` | `0.0` | SQL (exec) | `8/8` | `4.397` → **`5.726`** (**`+30.23%`**) | `416.64` → **`542.55`** (**`+30.22%`**) | `113.53` → **`146.47`** | `391.86` → **`379.43`** | `1.3902` → **`1.8810`** |
| `c=8` | `0.6` | English | `8/8` | `0.840` → **`1.069`** (`+27.26%`) | `410.47` → **`520.24`** (**`+26.74%`**) | `62.40` → **`79.94`** | `251.88` → **`199.41`** | `1.3725` → **`1.7360`** |
| `c=8` | `0.6` | `xgrammar` JSON | `8/8` | `3.963` → **`4.311`** (`+8.78%`) | `455.76` → **`483.34`** (**`+6.05%`**) | `83.57` → **`99.63`** | `427.58` → `435.83` | `1.5238` → **`1.6137`** |
| `c=8` | `0.6` | SQL (exec) | `8/8` | `4.512` → **`5.375`** (`+19.13%`) | `434.26` → **`540.83`** (**`+24.54%`**) | `113.64` → **`135.57`** | `380.61` → **`384.96`** | `1.4523` → **`1.8149`** |

*\*On `c=8, temp=0.0, xgrammar JSON`, a cold first-batch schema compile spiked `ttft_p50` (`732.76 → 2104.08 ms`), reducing aggregate wall-clock `tok/s` (`357.09 → 263.31`) even as steady-state per-stream decode increased (`84.38 → 98.28 tok/s`) and `tok/J` rose (`1.2726 → 1.3267`).*

---

### 2. Paired `A1 → sglang-sm89 → A2` (`172,609` KV Baseline) & Component Ablation

To separate the contribution of **FP8 draft + `ReplaySSM`** (`38,337 → 172,609` KV capacity) from the incremental clean-room `v0.2.0` modules (`coupled_grammar`, `kTileV=64 sparse_vocab_proj`, `wy_gdn_cuda`), we ran a paired `Control A1 → sglang-sm89 v0.2.0 → Control A2` protocol where `A1` and `A2` already include autotuned FP8 + `ReplaySSM` (`172,609` KV in all three arms):

- **Paired E2E A/B/A ([`e2e_aba_baseline_vs_sm89_4090d.json`](benchmarks/results/e2e_aba_baseline_vs_sm89_4090d.json)):** **`100.0%` valid across all arms (`A1`, `B`, `A2`)**. On high-concurrency SQL (`c=8`), `sglang-sm89` edges above the best control arm at both `temp=0.0` (`542.55 tok/s`, `5.726 tasks/s` vs `A1=515.96 / A2=529.89`, **`+2.39%` agg / `+3.19%` `tasks/s`**) and `temp=0.6` (`540.83 tok/s`, `5.375 tasks/s` vs `A1=494.57 / A2=501.63`, **`+7.81%` agg / `+2.60%` `tasks/s`**). Across `c=1`, `c=8 eng`, and `c=8 json`, run-to-run variance and cold `xgrammar` schema compilation dominate (`-14.37%`..`-0.28%` agg `tok/s`; `c=8 json temp=0.0` hit the same `2104 ms` first-batch compile vs `A1/A2`, `-22.25%`).
- **2-Turn Agentic Tool-Calling A/B/A ([`agentic_aba_baseline_vs_sm89_4090d.json`](benchmarks/results/agentic_aba_baseline_vs_sm89_4090d.json)):** `6/6` schema valid and `6/6` exact match across every group on `A1`, `B`, and `A2` (lossless Theorem 1D eager & sampled TV parity). Notable regimes (`tool_call_decode` p50 `132.22–154.56 tok/s` on `sglang-sm89` vs `89.55–166.55` on `A1/A2`):

| Temp | Prefix Regime | Group | `A1` `traj_p50` (`ms`) | `sglang-sm89` `traj_p50` (`ms`) | `A2` `traj_p50` (`ms`) | vs Best Control | `sm89` `valid_tasks/s` |
|---:|---|---|---:|---:|---:|---:|---:|
| `0.0` | `volatile_first` | `initial` / `subsequent` | `972.3` / `975.3` | `997.3` / `980.7` | `976.0` / `978.6` | `0.975×` / `0.995×` | `0.990` / `1.020` |
| `0.0` | `catalog_first` | `initial`* | `668.7` | `1013.4`* | `669.0` | `0.660×`* | `0.987` |
| `0.0` | `catalog_first` | `subsequent` | `672.3` | **`361.6`** | `675.0` | **`1.859×`** | **`2.765`** |
| `0.6` | `volatile_first` | `initial` / `subsequent` | `1028.6` / `1073.5` | `1038.0` / **`989.3`** | `1079.6` / `1036.4` | `0.991×` / **`1.048×`** | `0.969` / `1.010` |
| `0.6` | `catalog_first` | `initial` / `subsequent` | `726.7` / `695.0` | **`700.7`** / **`686.7`** | `705.1` / `729.7` | **`1.006×`** / **`1.012×`** | `1.436` / `1.458` |

*\*`temp=0.0 catalog_first initial` incurred a cold radix prefix miss (`answer_ttft_p50` `426.2 ms` vs `93.7 ms`) before warming `subsequent` (`96.5 ms`, `1.859×`).*

- **4-Step Component Ablation ([`ablation_4090d.json`](benchmarks/results/ablation_4090d.json)):** Progressive toggle activation on top of `base_fp8_replayssm`:
  - `c=8, temp=0.6, sql`: `base_fp8_replayssm` `509.15 tok/s` (`5.189 tasks/s`) → `+coupled_grammar` `499.74 tok/s` (`5.112`) → `+sparse_vocab_proj` `362.63 tok/s` (cold first-batch `ttft_p50=1013.04 ms`, `136.86` per-stream) → `+wy_gdn_cuda` (all ON) **`540.83 tok/s`** (**`5.375 tasks/s`**, `1.8149 tok/J`).
  - `c=8, temp=0.0, sql`: `542.68` → `533.82` → `426.48` (first-batch `ttft_p50=776.62 ms`) → **`542.55 tok/s`** (`5.726 tasks/s`).
  - `c=1, temp=0.0, sql`: `155.93` → `153.89` → **`161.50`** (`1.656 tasks/s`) → `159.45 tok/s` (`1.631 tasks/s`).

---

### 3. Isolated Operator & Full-Vocab (`N=248,320`) Benchmarks (`0` `ptxas` Spills, `0` `nvcc` Warnings)

- **Full-Vocab Unified-64 `kTileV=64` Sparse `lm_head` (`N=248,320`, `5.10 GB` VRAM, [`isolated_gates_full_vocab_4090d.json`](benchmarks/results/isolated_gates_full_vocab_4090d.json)):**
  - `sparse_r1` (`M=1`): `2,606.08 us` → **`27.648 us`** (**`94.26×`**)
  - `sparse_r8` (`M=8`): `2,684.928 us` → **`165.888 us`** (**`16.19×`**)
  - `mixed_56struct_8str_r64` (`M=64`): `3,081.216 us` → **`420.864 us`** (**`7.32×`**)
  - `N=2,051` dirty tail-bit parity (`M=1..289`): `0` bad late rows (BF16 & FP8); `CUDAGraph` replay `max_diff=0.00391`.
- **C++ Roofline XGrammar Dispatch (`V=248,320`, [`mask_dispatch_qwen_4090d.json`](benchmarks/results/mask_dispatch_qwen_4090d.json), [`mask_dispatch_synthetic_4090d.json`](benchmarks/results/mask_dispatch_synthetic_4090d.json)):**
  - Real Qwen `xgrammar 0.2.7` tool-call masks (`4/4` `agree=true`): **`10.506–12.139 us`** `p50` (**`22.61×–33.57×`** vs NumPy `bit_count`, **`1,269.08×–5,347.27×`** vs naive scan).
  - Synthetic sparse/mixed/dense batch cases (`10/10` `agree=true`, `cpp_roofline` `p50` `7.489–81.776 us`, **`9.51×–34.11×`** vs NumPy `bit_count`, **`297.97×–12,388.41×`** vs naive scan).
- **WY GDN `ReplaySSM` Verify vs SGLang Triton (`K=V=128`, [`isolated_gates_4090d.json`](benchmarks/results/isolated_gates_4090d.json)):**
  - `14/14` correctness/parity cases PASS (`max_abs_err <= 0.000122`, `CUDAGraph` `max_abs_err=0.0`, `0` stack frame / `0` register spills).
  - **FP32 (`B=1..8`, `SL=6..16`):** **`16.384–65.536 us`** vs **`29.696–131.072 us`** (**`1.23×–2.00×`** across every timed shape: `B=8, SL=12` `2.00×`; `B=1, SL=6` `1.81×`; `B=2, SL=16` `1.79×`; `B=8, SL=8` `1.26×`).
  - **BF16 (`<=32` tokens routed via `SGLANG_SM89_WY_GDN_MAX_BF16_TOKENS=32`):** `B=1 SL=6/8` `17.408 vs 28.672 us` (**`1.65×`**), `B=2 SL=16` `22.528 vs 29.696 us` (**`1.32×`**), `B=8 SL=12` `58.368 vs 70.656 us` (**`1.21×`**), `B=4 SL=8` `31.744 vs 28.672 us` (`0.90×`), while `B=8 SL=8` (`46.080 vs 30.720 us`, `0.67×`) auto-routes to Triton above `32` tokens.
- **Live GPU Block-FP8 `[128, 128]` Autotune Summary ([`fp8_autotune_summary_4090d.json`](benchmarks/results/fp8_autotune_summary_4090d.json)):**
  - Tuned all `5` Qwen `3.8-27B` `[128, 128]` Block-FP8 shapes (`N=14336,K=5120`; `N=16384,K=5120`; `N=34816,K=5120`; `N=5120,K=17408`; `N=5120,K=6144`) across all `13` `M`-buckets (`1..4096`) directly on RTX 4090 D (`0` fallback buckets).

---

## Verification & Benchmarks

```bash
# 1. Run executable mathematical proofs (CPU)
for p in proofs/prove_*.py; do PYTHONPATH=. python3 "$p"; done

# 2. Run unit & CUDA gate suites
PYTHONPATH=. pytest -v tests/

# 3. Run live GPU Block-FP8 autotuning
python3 -m sglang_sm89.autotune_fp8 --model Qwen/Qwen3.8-27B-FP8 --cache-dir ~/.cache/sglang-sm89/fp8_configs

# 4. Run isolated Gate 1/2/3 & mask dispatch microbenchmarks
python3 benchmarks/run_isolated_gates.py
python3 benchmarks/run_isolated_gates.py --only-gate2 --full-vocab
python3 benchmarks/bench_mask_dispatch.py --vocab-size 248320

# 5. Run controlled E2E & agentic tool-calling benchmarks against a live endpoint
python3 benchmarks/bench_e2e_ab.py --help
python3 benchmarks/bench_agentic_tool_chain.py --dry-run
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
