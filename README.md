# sglang-sm89

**Ada Lovelace (`sm_89`) acceleration plugin for SGLang (`v0.5.23`)** (`v0.3.2`) — native `sm_89` CUDA/C++ kernels, a live GPU Block-FP8 autotuner, and `HookRegistry` runtime overlays for `RTX 4090`, `RTX 4090 D 48GB`, `RTX 6000 Ada`, and `L40S`, requiring **zero modifications** to upstream SGLang source files.

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

---

## Why `sglang-sm89` (`v0.3.2`)?

Modern hybrid linear-attention + GQA models (e.g., Qwen3.5 / Qwen3.8 Gated DeltaNet + GQA) paired with block-parallel speculative decoding (`DFLASH` / `DFlash2`) and CFG/JSON grammar enforcement (`xgrammar`) hit four Ada Lovelace (`sm_89`) bottlenecks out of the box:

1. **Missing `sm_89` Block-FP8 Triton GEMM Configs:** Stock SGLang `0.5.23` ships no `[128, 128]` Block-FP8 Triton schedules for `RTX 4090 D` (`114` SMs) and sibling Ada cards, emitting fallback warnings at launch. `sglang-sm89` profiles shapes live on the target GPU (`<= 100 KiB` Ada shared-memory budget) and symlinks those schedules into SGLang's loader path — zero pre-baked third-party config JSONs shipped.
2. **Unquantized BF16 `DFlash2` Draft Path + Blocked `ReplaySSM`:** Stock `DFLASH` keeps the 21 `DFlash2` draft linears and draft `lm_head` in BF16 (while `--speculative-draft-model-quantization w8a8_fp8` disables fused KV materialization), and SGLang blocks `GDN` `ReplaySSM` under `DFLASH`, inflating `intermediate_ssm_state_cache` and cutting KV capacity by `4.5×`.
3. **Grammar-Blind Speculation & Dense Verify `lm_head`:** Grammar-unaware speculative drafting triggers cascade rejections, while dense `248,320`-vocab BF16 logits waste memory bandwidth on constrained decode tokens (and naive batch-union projection slows down mixed structured + open-string batches). `v0.3.0` moves coupled resampling into a `21.02×`-faster C++ extension (`csrc/coupled_grammar_cpu.cpp`) and defers `lm_head` materialization via `LogitsProcessor._get_logits` + a C++ roofline dispatcher.
4. **Triton GDN `ReplaySSM` Verify Overhead:** Sequential Triton chunk/recurrent verify under speculative batches leaves Ada Tensor Cores and shared memory underutilized on FP32 and short-token BF16 steps. `v0.3.0` specializes `csrc/wy_gdn_sm89.cu` on `kMaxSpec` (`<8>` / `<16>`) with coalesced `uint4` (`128`-bit) `h0` loads and XOR-swizzled shared memory (`16,420 B`, `6 CTAs/SM`, `0` register spills).

---

## Key Components

| Component | Module / Kernel | Role |
|---|---|---|
| **Live GPU Block-FP8 Autotuner** | [`sglang_sm89/autotune_fp8.py`](sglang_sm89/autotune_fp8.py) | Standalone Triton benchmark & cache linker (`python3 -m sglang_sm89.autotune_fp8`) deriving W8A8 `[128, 128]` schedules on the local `sm_89` GPU within Ada's `<= 100 KiB` SRAM limit. Zero static JSON files shipped. |
| **FP8 `DFlash2` Draft & Head** | [`sglang_sm89/draft_fp8.py`](sglang_sm89/draft_fp8.py) | Quantizes eligible `DFlash2` draft linears via `QuantizedLinear8` **after** fused-KV worker init and attaches `DraftHeadFP8` (keeping target verify BF16 lossless by default). |
| **GDN `ReplaySSM` Pool Bridge** | [`sglang_sm89/gdn_pool_bridge.py`](sglang_sm89/gdn_pool_bridge.py) | Enables `GDN` `ReplaySSM BF16` allocation under `DFLASH` via a bounded sentinel hook and routes `HybridLinearAttnBackend.update_mamba_state_after_mtp_verify` through `ReplaySSM` circular/fold commit (`4.50×` KV pool capacity). |
| **WY GDN `ReplaySSM` CUDA Kernel** | [`csrc/wy_gdn_sm89.cu`](csrc/wy_gdn_sm89.cu) + [`sglang_sm89/wy_gdn_verify.py`](sglang_sm89/wy_gdn_verify.py) | `sm_89` WY chunk-parallel forward-substitution kernel (`K=V=128`, `<8>` / `<16>` `kMaxSpec` templates, coalesced `uint4` `h0` loads, XOR-swizzled smem, `0` spills, adaptive BF16 crossover `SGLANG_SM89_WY_GDN_MAX_BF16_TOKENS=32`). |
| **Fast C++ Coupled Grammar Resampling** | [`csrc/coupled_grammar_cpu.cpp`](csrc/coupled_grammar_cpu.cpp) + [`sglang_sm89/coupled_grammar.py`](sglang_sm89/coupled_grammar.py) | Theorem 1C/1D overlap-safe `grammar_barrier()` + `\|V_valid\|==1` jump-forward + native C++ (`coupled_refine_step_cpu`, `129.95 us`, `21.02×` vs Python) coupled resampling with `_selector_sample` parity, `SGLANG_SM89_MAX_SAMPLED_GRAMMAR_BS`, and `try...finally: rollback()` invariance. |
| **DFlash-Scoped `_get_logits` + `kTileV=64` Sparse `lm_head`** | [`csrc/sparse_vocab_proj_sm89.cu`](csrc/sparse_vocab_proj_sm89.cu) + [`sglang_sm89/vocab_projection.py`](sglang_sm89/vocab_projection.py) | Scopes `LogitsProcessor._get_logits` capture to strict `SpeculativeAlgorithm.DFLASH` (`is_dflash()`, excluding `DSPARK`/`EAGLE`/other workers) and sizes pre-allocated arenas dynamically via `get_batch_sizes_to_capture(...) * decode_num_tokens_per_req()` alongside `TpModelWorker.forward_batch_generation(..., is_verify=True)` TLS guards, then materializes lazily via `kTileV=64` warp-ballot compaction and `select_sparse_vocab_proj`. |
| **Optional $O(1)$ SAM Tail Drafter** | [`sglang_sm89/sam_drafter.py`](sglang_sm89/sam_drafter.py) | Opt-in (`SGLANG_SM89_SAM_GRAFT=0`) Suffix Automaton prompt continuation grafting on low-confidence tail draft slots. |

---

## Supported GPUs & Profiles

All Compose profiles bind exclusively to loopback (`127.0.0.1:30000:30000`) — pair with an SSH tunnel or local reverse proxy for remote access.

| GPU | VRAM | SMs | TDP | Architecture Notes | Compose Profile |
|---|---:|---:|---:|---|---|
| **RTX 4090 D 48GB** | 48 GB | 114 | 300 W | `1,008 GB/s` GDDR6X, `294.2 TFLOPS` FP8; fits 27B-FP8 + DFlash2 + `172K+` FP8 KV | [`profiles/rtx4090d-48gb.yml`](profiles/rtx4090d-48gb.yml) |
| **RTX 6000 Ada / L40S** | 48 GB | 142 | 300–350 W | Higher SM count (`c=12`, `ctx=16384` profile) | [`profiles/rtx6000ada-48gb.yml`](profiles/rtx6000ada-48gb.yml) |
| **RTX 4090** | 24 GB | 128 | 450 W | Tuned for 9B–14B FP8 or 4-bit hybrid models inside 24 GB | [`profiles/rtx4090-24gb.yml`](profiles/rtx4090-24gb.yml) |

See [`docs/RTX_4090_6000_ARCHITECTURE.md`](docs/RTX_4090_6000_ARCHITECTURE.md) and [`docs/THEORY_AND_PROOFS.md`](docs/THEORY_AND_PROOFS.md) for the hardware roofline comparison (`sm_89` vs `sm_120`), mathematical proofs, and executable verification scripts in [`proofs/`](proofs/).

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
| `SGLANG_SM89_COUPLED_GRAMMAR` | `1` | Coupled grammar resampling + `\|V_valid\|==1` jump-forward (`coupled_refine_step_cpu`) |
| `SGLANG_SM89_MAX_SAMPLED_GRAMMAR_BS` | `0` | Opt-in max batch size for sampled-batch coupled grammar refinement (`0` = unlimited / disabled in all published benchmarks; unbenchmarked opt-in cap when `> 0`, while greedy always runs) |
| `SGLANG_SM89_SPARSE_PROJ` | `1` | Strict `DFLASH` `LogitsProcessor._get_logits` (`get_batch_sizes_to_capture(...) * decode_num_tokens_per_req()` arena) + `kTileV=64` sparse vocab projection + C++ roofline dispatch |
| `SGLANG_SM89_WY_GDN` | `1` | Native `sm_89` WY GDN `ReplaySSM` verify CUDA kernel |
| `SGLANG_SM89_WY_GDN_MAX_BF16_TOKENS` | `32` | Adaptive BF16 crossover token limit (`<=32` CUDA `1.156×–1.800×`; `>32` Triton) |
| `SGLANG_SM89_FP8_VERIFY_HEAD` | `0` | Opt-in rowwise FP8 target verify `lm_head` (`0` preserves exact BF16 logits) |
| `SGLANG_SM89_SAM_GRAFT` | `0` | Opt-in $O(1)$ Suffix Automaton tail draft grafting |

---

## Measured Benchmarks (`NVIDIA GeForce RTX 4090 D 48GB`, `300 W`, `Qwen/Qwen3.8-27B-FP8`)

All measurements below were recorded at `v0.3.0` on `NVIDIA GeForce RTX 4090 D` (`sm_89`, `300.0 W` power limit) serving `Qwen/Qwen3.8-27B-FP8` + `z-lab/Qwen3.8-27B-DFlash2` (`draft=8`, `kv_cache_dtype=fp8_e4m3`, `context_length=8192`, `--max-tokens 512`, `xgrammar 0.2.7`, read-only SQLite executable SQL + exact JSON ground-truth validation, untimed schema pre-warming via `prewarm_workloads`; `v0.3.1`/`v0.3.2` scope `_get_logits` capture strictly to `SpeculativeAlgorithm.DFLASH` (`is_dflash()`) and dynamically size capture arenas via `get_batch_sizes_to_capture(...) * decode_num_tokens_per_req()` without altering these kernel/grammar paths, and `SGLANG_SM89_MAX_SAMPLED_GRAMMAR_BS` was `0` = unlimited in all published runs). All `12` raw result artifacts live in [`benchmarks/results/`](benchmarks/results/).

### 1. Stock SGLang `0.5.23` vs `sglang-sm89` ([`e2e_stock_vs_sm89_4090d.json`](benchmarks/results/e2e_stock_vs_sm89_4090d.json))

- **KV Pool Capacity (`max_total_num_tokens`):** `38,337` (Stock) → **`172,609`** (`sglang-sm89`, **`4.50×`**)
- **W8A8 Block-FP8 Fallback Warnings:** `5` (Stock) → **`0`** (`sglang-sm89`, backed by live GPU autotuning across all `5` model shapes in [`fp8_autotune_summary_4090d.json`](benchmarks/results/fp8_autotune_summary_4090d.json))
- **End-to-End Validity:** **`100.0%` valid (`4/4` `c=1`, `8/8` `c=8`) across all `12/12` cells on both arms** *(Note: Stock `0.5.23` runs without `--enable-linear-replayssm-spec` because upstream `0.5.23` raises `ValueError` on Qwen3.8 GDN with that flag).*

| Concurrency | Temp | Workload | Valid | `valid_tasks_per_s` (Stock → `sm89`, Δ) | `agg_tok_s` (Stock → `sm89`, Δ) | `per_stream_median` (`tok/s`) | `ttft_p50` (`ms`) | `tok/J` (Stock → `sm89`) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `c=1` | `0.0` | English | `4/4` | `0.169` → **`0.200`** (`+18.34%`) | `76.91` → **`91.15`** (**`+18.52%`**) | `71.59` → **`91.97`** | `82.37` → **`66.11`** | `0.2571` → **`0.3187`** |
| `c=1` | `0.0` | `xgrammar` JSON | `4/4` | `0.797` → **`1.140`** (`+43.04%`) | `94.87` → **`133.33`** (**`+40.54%`**) | `105.22` → **`148.75`** | `122.22` → **`114.25`** | `0.3184` → **`0.4550`** |
| `c=1` | `0.0` | SQL (exec) | `4/4` | `0.901` → **`1.653`** (**`+83.46%`**) | `88.09` → **`161.61`** (**`+83.46%`**) | `134.83` → **`205.60`** | `122.53` → **`105.32`** | `0.2953` → **`0.5431`** |
| `c=1` | `0.6` | English | `4/4` | `0.148` → **`0.216`** (`+45.95%`) | `70.64` → **`99.94`** (**`+41.48%`**) | `69.50` → **`95.59`** | `104.29` → **`79.86`** | `0.2359` → **`0.3341`** |
| `c=1` | `0.6` | `xgrammar` JSON | `4/4` | `0.753` → **`1.012`** (`+34.40%`) | `93.13` → **`120.92`** (**`+29.84%`**) | `99.04` → **`139.34`** | `124.50` → **`115.30`** | `0.3112` → **`0.4044`** |
| `c=1` | `0.6` | SQL (exec) | `4/4` | `1.240` → **`1.563`** (`+26.05%`) | `121.85` → **`158.27`** (**`+29.89%`**) | `139.56` → **`191.03`** | `117.14` → **`107.22`** | `0.4075` → **`0.5303`** |
| `c=8` | `0.0` | English | `8/8` | `0.852` → **`1.056`** (`+23.94%`) | `412.72` → **`504.73`** (**`+22.29%`**) | `65.67` → **`78.81`** | `183.33` → `189.25` | `1.3790` → **`1.6861`** |
| `c=8` | `0.0` | `xgrammar` JSON | `8/8` | `3.735` → **`4.149`** (`+11.08%`) | `434.24` → **`485.98`** (**`+11.92%`**) | `77.80` → **`103.47`** | `412.67` → `437.75` | `1.4638` → **`1.6344`** |
| `c=8` | `0.0` | SQL (exec) | `8/8` | `4.393` → **`5.577`** (`+26.95%`) | `417.30` → **`545.89`** (**`+30.81%`**) | `113.94` → **`141.76`** | `395.96` → `404.03` | `1.4059` → **`1.8355`** |
| `c=8` | `0.6` | English | `8/8` | `0.777` → **`1.027`** (`+32.18%`) | `389.98` → **`502.63`** (**`+28.89%`**) | `68.37` → **`73.75`** | `408.13` → **`196.56`** | `1.3177` → **`1.6740`** |
| `c=8` | `0.6` | `xgrammar` JSON | `8/8` | `4.122` → **`4.690`** (`+13.78%`) | `440.54` → **`528.78`** (**`+20.03%`**) | `83.67` → **`101.65`** | `413.07` → `431.61` | `1.4736` → **`1.7735`** |
| `c=8` | `0.6` | SQL (exec) | `8/8` | `4.404` → **`5.236`** (`+18.89%`) | `428.34` → **`511.86`** (**`+19.50%`**) | `112.63` → **`142.12`** | `377.00` → `389.93` | `1.4327` → **`1.7129`** |

---

### 2. Paired `A1 → sglang-sm89 → A2` (`172,609` KV Matched-Flag Baseline) ([`e2e_aba_baseline_vs_sm89_4090d.json`](benchmarks/results/e2e_aba_baseline_vs_sm89_4090d.json))

Both `Control A1` and `Control A2` already include autotuned FP8 + `DFlash2` FP8 + `ReplaySSM` (`172,609` KV in all three arms). Replacing the `v0.2.0` Python resampling loop with C++ `coupled_refine_step_cpu` ([`coupled_grammar_bench_4090d.json`](benchmarks/results/coupled_grammar_bench_4090d.json): **`129.95 us`** vs `2,731.37 us`, **`21.02×`**) narrowed the `c=8, temp=0.6` `xgrammar` JSON delta from `-13.2%` in `v0.2.0`, though `56` CPU `xgrammar.fill_vocab_mask` calls still add host latency at `bs=8` (an unbenchmarked opt-in cap `SGLANG_SM89_MAX_SAMPLED_GRAMMAR_BS` was added in `v0.3.1` with default `0` = unlimited, i.e., disabled across all published runs below): at `c=8, temp=0.6` JSON, `sglang-sm89` (`528.78 tok/s`, `4.690 tasks/s`) improved `+0.76%` vs `A1` (`524.77`) while trailing `A2` (`573.32 tok/s`, `4.874 tasks/s`) by **`-7.77%` `agg_tok_s` and `-3.78%` `valid_tasks_per_s` vs best control**. By contrast, `c=1, temp=0.0` greedy JSON gained **`+42.80%`** (`133.33` vs `93.27 / 93.37 tok/s`), and `temp=0.6` equal-cache agentic trajectories gained **`+8.85%`** (`catalog_first all_groups`) and **`+3.01%`** (`volatile_first all_groups`), with `100.0%` validity across all `A1`, `B`, and `A2` cells:

| Concurrency | Temp | Workload | `A1` `tok/s` (`tasks/s`) | `sglang-sm89` `tok/s` (`tasks/s`) | `A2` `tok/s` (`tasks/s`) | `B` vs Best Control |
|---:|---:|---|---:|---:|---:|---:|
| `c=1` | `0.0` | English | `99.64` (`0.214`) | `91.15` (`0.200`) | `102.92` (`0.226`) | `-11.44%` |
| `c=1` | `0.0` | `xgrammar` JSON | `93.27` (`0.790`) | **`133.33`** (**`1.140`**) | `93.37` (`0.781`) | **`+42.80%`** |
| `c=1` | `0.0` | SQL (exec) | `162.28` (`1.660`) | `161.61` (`1.653`) | `117.21` (`1.199`) | `-0.41%` |
| `c=1` | `0.6` | English | `96.03` (`0.199`) | **`99.94`** (**`0.216`**) | `95.35` (`0.186`) | **`+4.07%`** |
| `c=1` | `0.6` | `xgrammar` JSON | `132.86` (`1.071`) | `120.92` (`1.012`) | `117.68` (`0.943`) | `-8.99%` |
| `c=1` | `0.6` | SQL (exec) | `164.73` (`1.805`) | `158.27` (`1.563`) | `143.64` (`1.388`) | `-3.92%` |
| `c=8` | `0.0` | English | `543.29` (`1.103`) | `504.73` (`1.056`) | `540.35` (`1.107`) | `-7.10%` |
| `c=8` | `0.0` | `xgrammar` JSON | `518.60` (`4.480`) | `485.98` (`4.149`) | `496.61` (`4.328`) | `-6.29%` |
| `c=8` | `0.0` | SQL (exec) | `522.92` (`5.519`) | `545.89` (`5.577`) | `552.80` (`5.834`) | `-1.25%` |
| `c=8` | `0.6` | English | `540.04` (`1.122`) | `502.63` (`1.027`) | `509.93` (`1.044`) | `-6.93%` |
| `c=8` | `0.6` | `xgrammar` JSON | `524.77` (`4.870`) | `528.78` (`4.690`) | `573.32` (`4.874`) | `-7.77%` `tok/s` / `-3.78%` `tasks/s` *(+0.76% vs `A1`)* |
| `c=8` | `0.6` | SQL (exec) | `501.79` (`5.127`) | `511.86` (`5.236`) | `527.34` (`5.300`) | `-2.94%` *(beats `A1` `+2.01%`)* |

---

### 3. Equal-Cache 2-Turn Agentic Tool-Calling A/B/A ([`agentic_aba_baseline_vs_sm89_4090d.json`](benchmarks/results/agentic_aba_baseline_vs_sm89_4090d.json))

Executed with `POST /flush_cache` before every arm/timed sweep (`pairs=6`, `iterations=2`) under identical server flags, confirming **equal `cached_tokens_p50` across all 3 arms** (`0.0` cold / `64.0` warm / `32.0` combined on `volatile_first`; `640.0` / `608.0` / `608.0` on `catalog_first`) and **`12/12` schema-valid + `12/12` exact-match** on every arm (`A1`, `B`, `A2`):

| Temp | Prefix Regime | Group (`cached_p50`) | `A1` `traj_p50` (`vts`) | `sglang-sm89` `traj_p50` (`vts`) | `A2` `traj_p50` (`vts`) | `B` vs Best Control (`vts`) | `B` `call_tok/s` (`A1`/`B`/`A2`) |
|---:|---|---|---:|---:|---:|---:|---:|
| `0.0` | `volatile_first` | `initial_group` (`0.0`) | `944.5 ms` (`1.055`) | `995.0 ms` (`1.002`) | `945.8 ms` (`1.048`) | `-5.02%` | `130.37 / 132.62 / 149.79` |
| `0.0` | `volatile_first` | `subsequent_group` (`64.0`) | `958.3 ms` (`1.037`) | `984.2 ms` (`1.004`) | `959.9 ms` (`1.034`) | `-3.18%` | `129.66 / 131.96 / 134.40` |
| `0.0` | `volatile_first` | **`all_groups` (`32.0`)** | `955.4 ms` (`1.046`) | `990.5 ms` (`1.003`) | `959.0 ms` (`1.041`) | `-4.11%` | `130.01 / 132.51 / 135.78` |
| `0.0` | `catalog_first` | `initial_group` (`640.0`) | `621.1 ms` (`1.599`) | `648.0 ms` (`1.516`) | `624.1 ms` (`1.606`) | `-5.60%` | `129.94 / 151.21 / 130.32` |
| `0.0` | `catalog_first` | `subsequent_group` (`608.0`) | `670.3 ms` (`1.354`) | **`676.0 ms`** (**`1.466`**) | `672.0 ms` (`1.350`) | **`+8.27%`** | `149.99 / 132.28 / 129.90` |
| `0.0` | `catalog_first` | **`all_groups` (`608.0`)** | `650.6 ms` (`1.466`) | **`674.9 ms`** (**`1.490`**) | `639.6 ms` (`1.467`) | **`+1.57%`** | `130.47 / 132.86 / 130.19` |
| `0.6` | `volatile_first` | `initial_group` (`0.0`) | `1036.2 ms` (`0.900`) | `1030.1 ms` (`0.979`) | `1014.7 ms` (`0.984`) | `-0.51%` | `99.25 / 140.51 / 104.57` |
| `0.6` | `volatile_first` | `subsequent_group` (`64.0`) | `1093.3 ms` (`0.865`) | **`990.6 ms`** (**`1.002`**) | `1058.5 ms` (`0.942`) | **`+6.37%`** | `82.70 / 131.61 / 99.60` |
| `0.6` | `volatile_first` | **`all_groups` (`32.0`)** | `1050.7 ms` (`0.882`) | **`999.5 ms`** (**`0.991`**) | `1037.8 ms` (`0.962`) | **`+3.01%`** | `97.23 / 139.16 / 104.09` |
| `0.6` | `catalog_first` | `initial_group` (`640.0`) | `732.6 ms` (`1.395`) | **`648.9 ms`** (**`1.509`**) | `714.8 ms` (`1.398`) | **`+7.94%`** | `99.78 / 135.92 / 104.20` |
| `0.6` | `catalog_first` | `subsequent_group` (`608.0`) | `743.0 ms` (`1.259`) | **`715.6 ms`** (**`1.422`**) | `763.4 ms` (`1.295`) | **`+9.81%`** | `98.45 / 132.09 / 92.30` |
| `0.6` | `catalog_first` | **`all_groups` (`608.0`)** | `732.6 ms` (`1.323`) | **`680.4 ms`** (**`1.464`**) | `738.5 ms` (`1.345`) | **`+8.85%`** | `99.46 / 132.09 / 98.48` |

---

### 4. WY GDN `ReplaySSM` Microbenchmark (`<8>` / `<16>`, `uint4` Coalesced `h0`) & Isolated E2E A/B/A

- **Kernel Microbenchmark ([`wy_gdn_microbench_4090d.json`](benchmarks/results/wy_gdn_microbench_4090d.json)):** Zero `nvcc` warnings and **0 bytes stack frame / 0 spill stores / 0 spill loads** (`<BF16, BF16, BF16, false, 8>` compiles to `77` regs and `16,420 B` smem = `6 CTAs/SM`; `cos_sim_vs_triton >= 0.999995`, `max_abs_err <= 0.000122`):
  - **BF16 `<= 32` tokens (`B=1, 2, 4` at `SL=6, 8` and `B=1, 2` at `SL=16`):** **`15.360–23.040 us`** vs SGLang Triton **`26.624–27.648 us`** (**`1.156×–1.800×`**, e.g., `B=4, SL=8`: `16.384 vs 27.648 us` = `1.688×`).
  - **BF16 adaptive routing (`SGLANG_SM89_WY_GDN_MAX_BF16_TOKENS=32`):** Coalesced `uint4` loads improved `B=8, SL=8` from `36.864 us` to `32.768 us` (`0.844×` vs Triton `27.648 us`; `B=8, SL=6`: `30.720 vs 27.648 us` = `0.900×`; `B=4, SL=16`: `45.056 vs 31.744 us` = `0.705×`; `B=8, SL=16`: `64.512 vs 63.488 us` = `0.984×`). Retaining the default `<= 32` token threshold automatically routes those larger BF16 batches to Triton while keeping all `<= 32`-token wins on CUDA.
  - **FP32 (all 12 tested shapes `B=1..8`, `SL=6..16`):** **`15.360–70.656 us`** vs **`27.648–169.984 us`** (**`1.688×–2.588×`**, including `B=4, SL=6`: `2.588×`; `B=4, SL=8`: `2.444×`; `B=8, SL=16`: `2.406×`; `B=8, SL=8`: `2.145×`).
- **Dedicated `SGLANG_SM89_WY_GDN=0 → 1 → 0` E2E A/B/A across `c=1, 4, 8` ([`wy_gdn_isolated_aba_4090d.json`](benchmarks/results/wy_gdn_isolated_aba_4090d.json)):** `100.0%` valid across all `18` cells (`A1`, `B`, `A2`); live server wins on `c=4 t=0.0 json` (`325.70 vs 181.81 / 195.94 tok/s`, `+66.22%`), `c=8 t=0.6 eng` (`533.93 vs 483.87 / 503.61`, `+6.02%` / `+6.90% tasks/s`), `c=8 t=0.6 json` (`516.26 vs 483.11 / 492.98`, `+4.72%`), `c=1 t=0.6 sql` (`157.40 vs 153.12 / 154.59`, `+1.82%`), and `c=8 t=0.0 sql` (`534.50 vs 533.05 / 528.92`, `+0.27%`), with single-wave variance on `c=4 t=0.6 json/sql` (`-45.61%`/`-44.60%`) and `c=8 t=0.0 eng/json` (`-5.82%`/`-4.71%`).

---

### 5. Fresh 5-Arm Component Ablation ([`ablation_4090d.json`](benchmarks/results/ablation_4090d.json))

Five distinct server runs activating components cumulatively (`stock_0523 → plus_fp8_draft_and_autotune → plus_coupled_grammar → plus_sparse_vocab_proj → plus_wy_gdn_cuda`), `100.0%` valid across all `60` cells:

| Cell | `stock_0523` | `+fp8_autotune_replay` | `+coupled_grammar` | `+sparse_vocab_proj` | `+wy_gdn_cuda` (`all ON`) |
|---|---:|---:|---:|---:|---:|
| `c=1, t=0.0, json` | `100.67` (`0.855`) | `99.98` (`0.860`) | `120.99` (`1.017`) | `126.13` (`1.080`) | **`131.37`** (**`1.113`**) |
| `c=1, t=0.0, sql` | `122.62` (`1.254`) | `146.31` (`1.474`) | **`159.99`** (**`1.641`**) | `159.71` (`1.634`) | `153.30` (`1.474`) |
| `c=1, t=0.6, sql` | `117.44` (`1.186`) | `145.32` (`1.377`) | `151.65` (`1.469`) | `149.51` (`1.480`) | **`154.10`** (**`1.500`**) |
| `c=8, t=0.0, eng` | `421.87` (`0.858`) | `521.79` (`1.082`) | `539.47` (`1.095`) | `513.92` (`1.056`) | **`542.03`** (**`1.125`**) |
| `c=8, t=0.0, json`* | `472.68` (`4.244`) | **`501.94`** (**`4.384`**) | `488.65` (`4.194`) | `293.50` (`2.511`)* | `311.54` (`2.607`)* |
| `c=8, t=0.0, sql` | `380.62` (`3.986`) | `516.37` (`5.436`) | `524.72` (`5.545`) | **`535.48`** (**`5.652`**) | `529.44` (`5.544`) |
| `c=8, t=0.6, json` | `437.91` (`3.888`) | `417.01` (`4.024`) | `518.57` (`4.630`) | **`546.26`** (`4.615`) | `483.49` (`4.346`) |
| `c=8, t=0.6, sql` | `518.24` (`5.420`) | `489.64` (`4.680`) | **`537.88`** (`5.531`) | `503.15` (`5.167`) | `537.65` (**`5.550`**) |

*\*In the fresh ablation run, `c=8, t=0.0, json` on arms 4–5 encountered a single-wave cold first-batch TTFT outlier (`ttft_p50` `1508.8 / 1501.7 ms` vs `412.3 ms` on Stock) while steady-state `per_stream_median_tok_s` remained higher than Stock (`91.18 / 91.42` vs `83.66 tok/s`).*

---

### 6. Isolated Kernel, Full-Vocab (`N=248,320`), & C++ Coupled Grammar Gates

- **Full-Vocab Unified-64 `kTileV=64` Sparse `lm_head` (`N=248,320`, `5.10 GB` VRAM, [`isolated_gates_full_vocab_4090d.json`](benchmarks/results/isolated_gates_full_vocab_4090d.json)):**
  - `sparse_r1` (`M=1`): `2,605.056 us` → **`27.648 us`** (**`94.22×`**)
  - `sparse_r8` (`M=8`): `2,692.096 us` → **`164.864 us`** (**`16.33×`**)
  - `mixed_56struct_8str_r64` (`M=64`): `3,078.144 us` → **`419.840 us`** (**`7.33×`**)
  - `N=2,051` dirty tail-bit parity (`M=1..289`): `0` bad late rows (BF16 & FP8); `CUDAGraph` replay `max_diff=0.00391`.
- **C++ Roofline XGrammar Dispatch (`V=248,320`, [`mask_dispatch_qwen_4090d.json`](benchmarks/results/mask_dispatch_qwen_4090d.json), [`mask_dispatch_synthetic_4090d.json`](benchmarks/results/mask_dispatch_synthetic_4090d.json)):**
  - Real Qwen `xgrammar 0.2.7` tool-call masks (`4/4` `agree=true`): **`8.641–11.985 us`** `p50` (**`23.38×–33.19×`** vs NumPy `bit_count`, **`1,263.06×–8,862.73×`** vs naive scan).
  - Synthetic sparse/mixed/dense batch cases (`10/10` `agree=true`): **`7.420–82.095 us`** `p50` (**`9.51×–35.20×`** vs NumPy `bit_count`, **`296.32×–12,691.17×`** vs naive scan).
- **C++ Coupled Grammar Step ([`coupled_grammar_bench_4090d.json`](benchmarks/results/coupled_grammar_bench_4090d.json)):** `coupled_refine_step_cpu` (`bs=8, block_size=8, V=248,320`): **`129.95 us`** vs Python fallback **`2,731.37 us`** (**`21.02×`** refine speedup; `1,067.52 us` vs `3,706.37 us` total w/ mock mask fill).
- **Gate 1–3 Suite (`N=32,768`, `0.688 GB` VRAM, [`isolated_gates_4090d.json`](benchmarks/results/isolated_gates_4090d.json)):** `gate1_coupled_grammar`, `gate2_unified64_sparse_vocab_proj`, and `gate3_wy_gdn_sm89` all **`PASSED: True`**.

---

## Verification & Benchmarks

```bash
# 1. Run executable mathematical proofs (CPU)
for p in proofs/prove_*.py; do PYTHONPATH=. python3 "$p"; done

# 2. Run full unit & CUDA gate suites (43/43 on Ada GPU; CPU-only tests skip CUDA gates)
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

---

## Provenance & License

- **License:** Verbatim Apache License 2.0 ([`LICENSE`](LICENSE), `SPDX-License-Identifier: Apache-2.0`) and [`NOTICE`](NOTICE).
- **Attribution & Independence Verification:** Builds upon upstream [`sgl-project/sglang`](https://github.com/sgl-project/sglang) runtime interfaces (`HookRegistry`, `DFlashWorkerV2`, `LogitsProcessor`, Triton kernels) and Apache-2.0 prior art in [`dashportai/dashport-4090d-infer`](https://github.com/dashportai/dashport-4090d-infer) on `RTX 4090 D` Block-FP8 and `DFlash2` acceleration. Every module, CUDA/C++ source, test, benchmark script, and profile in `sglang-sm89` `v0.3.2` is verified with an automated normalized Longest Common Substring check (`MAX_LCS <= 4`, zero `>= 5`-line contiguous overlaps against public and internal overlay repositories) and zero static Block-FP8 config JSON files.
- **Git History Note:** Public `main` history was reset to a single release commit at `v0.2.0` (`07df620`), followed by fast-forward release commits for `v0.3.0` (`40194c7`), `v0.3.1` (`f7002bc`), and `v0.3.2`. An earlier unreferenced development commit (`a9157f3`) remains server-retained in GitHub object storage because the push OAuth token does not hold `delete_repo` scope; it contains zero secrets or credentials.
