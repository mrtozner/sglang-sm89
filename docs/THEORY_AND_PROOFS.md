# Mathematical Theory & Losslessness Proofs (`sglang-sm89`)

Every theorem below is accompanied by an executable CPU verification script in [`proofs/`](../proofs/) that checks both exact analytical identities and Monte Carlo / numerical parity.

---

## Theorem 1C & 1D: Lossless Coupled Conditional Resampling & Eager/Graph Buffer Isolation

**Script:** [`proofs/prove_coupled_grammar_resampling.py`](../proofs/prove_coupled_grammar_resampling.py)  
**Implementation:** `sglang_sm89/coupled_grammar.py`

### Setup (Theorem 1C)
At draft position $j$, let $C_j \subseteq V$ ($|C_j| = K$) be the parallel drafter's unique top-$K$ candidate set with draft probability distribution $q_j(v)$ ($\sum_{v \in C_j} q_j(v) = 1$, $q_j(v) = 0$ for $v \notin C_j$), and let $x_{\text{raw}} \sim q_j$ be the unconstrained draft draw. Let $M_j \in \{0,1\}^{|V|}$ be the `xgrammar` PDA bitmask for the speculative prefix up to $j-1$, inducing the constrained target distribution:
$$P_{\text{target}}(v) = \frac{p_{\text{raw}}(v)\,M_j(v)}{\sum_{u \in V} p_{\text{raw}}(u)\,M_j(u)}.$$
Define the legal candidate mass $Z_j = \sum_{v \in C_j} q_j(v)\,M_j(v)$.

### Resampling Rule & Proof of Exact Losslessness ($\mathrm{TV} = 0$)
1. **Forced single-token jump ($|\mathcal{V}_j| = 1$, unique $u$) or greedy ($T = 0$):** Place chosen legal token $u^*$ at slot 0 of `cand_row` (swapping if already present so candidate IDs remain strictly unique) with $\tilde{q}_j = \mathbf{1}_{u^*}$.
2. **Sampled ($T > 0$) with $Z_j > 0$:** Define $\tilde{q}_j(v) = \frac{q_j(v)\,M_j(v)}{Z_j}$. Apply maximal coupling:
   $$x' = \begin{cases} x_{\text{raw}} & \text{if } M_j(x_{\text{raw}}) = 1 \text{ and } \tilde{q}_j(x_{\text{raw}}) > 0, \\ x_{\text{res}} \sim \tilde{q}_j & \text{otherwise.} \end{cases}$$
   For any legal $v \in C_j$ ($M_j(v) = 1$):
   $$\mathbb{P}(x' = v) = q_j(v) + (1 - Z_j)\,\frac{q_j(v)}{Z_j} = q_j(v)\!\left(1 + \frac{1 - Z_j}{Z_j}\right) = \frac{q_j(v)}{Z_j} = \tilde{q}_j(v),$$
   and $\mathbb{P}(x' = v) = 0$ for $M_j(v) = 0$. Thus $x'$ is distributed **identically** as $\tilde{q}_j$ while maximizing retention $\mathbb{P}(x' = x_{\text{raw}}) = Z_j$.
3. **Sampled ($T > 0$) fallback ($Z_j = 0$):** Select any legal $v_{\text{first}}$ from $M_j$, setting $\tilde{q}_j = \mathbf{1}_{v_{\text{first}}}$.

Under speculative verification against $P_{\text{target}}$ using proposal $\tilde{q}_j$, accepting $x'$ with probability $\alpha(x') = \min\!\big(1, P_{\text{target}}(x')/\tilde{q}_j(x')\big)$ and otherwise drawing from $P_{\text{res}}(v) \propto \max(0, P_{\text{target}}(v) - \tilde{q}_j(v))$ yields:
$$\mathbb{P}(X_{\text{out}} = v) = \min(P_{\text{target}}(v), \tilde{q}_j(v)) + \max(0, P_{\text{target}}(v) - \tilde{q}_j(v)) = P_{\text{target}}(v) \qquad \forall v \in V.$$
Hence $\mathrm{TV}(P_{\text{out}}, P_{\text{target}}) = 0$ across greedy, sampled, and mixed batches, and `try...finally: rollback(accepted_cnt)` ensures `req.grammar` returns to the exact committed prefix before target verification.

### Eager vs. CUDA Graph Sampling Buffer Isolation Invariant (Theorem 1D)
In SGLang `DFlash2` (`dflash_worker_v2.py`), during **eager** draft execution (`draft_ran_graph == False`), `_propose_selector_block` allocates fresh `(candidate_ids, q_rows)` inside `worker._selector_sample`, while `worker._draft_sampler.candidate_out` and `q_out` retain **stale ring-buffer values** from an earlier CUDA Graph replay! Reading `sampler.candidate_out`/`q_out` ahead of `_selector_sample` (or on eager all-greedy batches where `_selector_sample is None`) corrupts the proposal distribution with stale tokens and probabilities ($\mathrm{TV} > 0$).

**Isolation Rule:** Let $S_{\text{sel}} = \text{worker.\_selector\_sample}$, $S_{\text{eager}} = \text{eager\_cand\_ids}$, and $g = \text{draft\_ran\_graph}$:
- **Rule A ($S_{\text{sel}} \neq \text{None}$ — all sampled & mixed batches, eager or graph):** Read and write **exclusively** $S_{\text{sel}}$; never read or mutate `sampler.candidate_out`/`q_out`. (In graph mode $S_{\text{sel}}$ already aliases the fresh `sampler` slice; in eager mode stale `sampler` buffers stay untouched.)
- **Rule B ($S_{\text{sel}} = \text{None}$ — all-greedy batches):** Read $S_{\text{eager}}$ if non-`None`; else read `sampler.candidate_out[:bs]` **iff** $g = \text{True}$; else `None` (mask-only jump-forward).

By induction over $t = 1 \dots B_{\text{draft}}$, coupling + isolation preserves the exact joint autoregressive constrained distribution $P_{\text{target}}(y_{1..T})$ over full multi-step trajectories.

**Verified Output (`prove_coupled_grammar_resampling.py`):**
- Single-step Case A ($Z = 0.5110$): `exact_TV = 1.04e-16`, `MC_proposal_TV(N=100k) = 0.00019`, `MC_verify_TV = 0.00354`.
- Theorem 1D poisoned stale-sampler isolation: eager sampled (`_selector_sample` vs poisoned `sampler`) and eager greedy (`eager_cand_ids` / `None` vs poisoned `sampler`) strictly isolated.
- **4-Step Sequential Draft Trajectory (`N = 80,000` trajectories, stateful grammar incl. `|V_valid|==1`):** `joint_TV = 0.00751 (< 0.01)`, `illegal_traj_mass = 0.0`.

---

## Theorem 3B: `kTileV=64` Mixed-Density Roofline Cost Model & Tail-Word Sanitization

**Script:** [`proofs/prove_mixed_density_roofline.py`](../proofs/prove_mixed_density_roofline.py)  
**Implementation:** `csrc/sparse_vocab_proj_sm89.{cuh,cu}`, `sglang_sm89/vocab_projection.py`

### Exact Effective Weight-Row Reads (`kTileV = 64`)
Partition the vocabulary $V$ into $C = \lceil V / 64 \rceil$ tiles of $64$ rows (`2 × uint32_t = 1 × uint64_t` bitmask per batch row `m`, where the final word $w = \lfloor V/32 \rfloor$ is sanitized by $(1 \ll (V \bmod 32)) - 1$ whenever $V \bmod 32 \neq 0$, and the final 64-row tile applies `(1ULL << rem) - 1ULL`):
- Let $u_c = \bigvee_{m=0}^{M-1} M_{m, c} \in \{0,1\}^{64}$ be the per-tile `uint64_t` bit union.
- Let $a_c = \sum_{m=0}^{M-1} \mathbf{1}[M_{m, c} \neq 0]$ be the active-`M` count inside tile $c$.

With $k$-chunking width $K_{\text{chunk}} = 8$:
$$\mathrm{EffRows}_{\text{Unified64}}(M, \mathcal{M}) = \sum_{c=0}^{C-1} \mathrm{popcnt}(u_c) \cdot \left\lceil \frac{a_c}{K_{\text{chunk}}} \right\rceil.$$
- **Comparison to Naive Batch-Union GSP:** Naive batch-union reads $\big(\sum_c \mathrm{popcnt}(u_c)\big) \cdot \lceil M / K_{\text{chunk}} \rceil$ rows, amplifying any dense request across all $\lceil M / K_{\text{chunk}} \rceil$ passes (`6.80× V` vs `0.85× V` on a mixed `56` sparse + `8` dense $M=64$ batch).
- **Comparison to Two-Pass `m_blk = {0, 32}`:** Splitting $M \le 64$ into two `32`-row passes reads $\sum_c \mathrm{popcnt}(u_c)\big(\lceil a_{c,0}/8 \rceil + \lceil a_{c,1}/8 \rceil\big)$, incurring $2\times$ redundant reads whenever active rows span both halves with $a_{c,0} + a_{c,1} \le 8$. Unified-64's `uint64_t` warp ballot eliminates this completely with zero register spills (`0 bytes` stack/spill).
- **Early-Exit Roofline Gate:** `select_sparse_vocab_proj_cpu` accumulates $\mathrm{EffRows}$ across `64`-row `uint64_t` tiles (`__builtin_popcountll`) and returns `false` (routing to cuBLAS `torch.mm`) immediately upon exceeding $V \cdot \rho_{\max}$.

**Verified Output (`prove_mixed_density_roofline.py`):**
- Tail-bit sanitization (`V = 2051`): exact `2051` valid bits (`0b111` tail word).
- Mixed `M = 64` (`56` sparse + `8` dense): `global_union = 85.0%`, `naive_union = 6.80x V`, `unified64 = 0.85x V` (routes to dense cuBLAS).
- Cross-half `M = 64` (`8` reqs split across `m_blk = {0, 32}`): `two_pass = 128` $\to$ `unified64 = 64` (`2.00x` fewer weight reads).
- Pure sparse `M = 64`: `unified64 = 0.0019x V` (routes to sparse CUDA kernel).

---

## Theorem 4A & 4B: Algebraic Chunk-Parallel WY Forward Substitution + Circular Ring `ReplaySSM`

**Script:** [`proofs/prove_wy_gdn_circular.py`](../proofs/prove_wy_gdn_circular.py)  
**Implementation:** `csrc/wy_gdn_sm89.{cuh,cu}`, `sglang_sm89/wy_gdn_verify.py`, `sglang_sm89/gdn_pool_bridge.py`

### Gated DeltaNet Recurrence & Circular Ring Folding (Theorem 4B)
For head $h_v$, $K = V = 128$, checkpoint $h_0 \in \mathbb{R}^{V \times K}$, decay $\alpha_t = \exp(g_t)$ ($g_t = -e^{A_{\log}}\,\mathrm{softplus}(a_t + \mathrm{dt\_bias})$), $\beta_t = \sigma(b_t)$, L2-normalized $k_t$, and scaled L2-normalized $q_t$:
$$S^{\text{dec}}_t = \alpha_t S_{t-1}, \qquad \delta_t = \beta_t\big(v_t - S^{\text{dec}}_t k_t\big), \qquad S_t = S^{\text{dec}}_t + \delta_t k_t^T, \qquad o_t = S_t q_t.$$
Given $W \le L$ committed entries in a power-of-2 circular ring `phys_c = (cache_base + c) & (L - 1)` with step prefix log-decays $g_{\text{prefix}}[c]$ and total log-decay $g_{\text{total}}$, the pre-verify state is $S_{\text{pre}} = e^{g_{\text{total}}} h_0 + \sum_{c=0}^{W-1} e^{g_{\text{total}} - g_{\text{prefix}}[c]} d_{\text{hist}}[c]\, k_{\text{hist}}[c]^T$. Projecting onto $Q, K \in \mathbb{R}^{T \times K}$ ($T = \text{spec\_len} \le 16$) avoids materializing $S_{\text{pre}}$ via a **single coalesced read of $h_0$**:
$$hw_q = e^{g_{\text{total}}}(h_0 Q^T) + \sum_{c=0}^{W-1} d_{\text{scaled}}[c]\,(k_{\text{hist}}[c]^T Q^T), \qquad hw_k = e^{g_{\text{total}}}(h_0 K^T) + \sum_{c=0}^{W-1} d_{\text{scaled}}[c]\,(k_{\text{hist}}[c]^T K^T),$$
where $d_{\text{scaled}}[c] = e^{g_{\text{total}} - g_{\text{prefix}}[c]} d_{\text{hist}}[c] \in \mathbb{R}^V$.

### WY Forward Substitution Without Matrix Inversion (Theorem 4A)
Let $G_s[t] = \sum_{u=0}^{t} g_s[u]$, $K_{\text{gram}}[i,j] = k_i^T k_j$, and $Q_{\text{gram}}[j,t] = k_j^T q_t$. Expanding $S_t = e^{G_s[t]} S_{\text{pre}} + \sum_{j=0}^{t} e^{G_s[t] - G_s[j]} \delta_j k_j^T$ into $\delta_t = \beta_t(v_t - e^{g_s[t]} S_{t-1} k_t)$ gives the unit lower-triangular system $(I_T + A) D^T = R^T$:
$$A_{i,j} = \begin{cases} \beta_i\, e^{G_s[i] - G_s[j]}\, K_{\text{gram}}[i,j] & i > j, \\ 0 & i \le j, \end{cases} \qquad R[:, t] = \beta_t\big(v_t - e^{G_s[t]} hw_k[:, t]\big).$$
Because $I_T + A$ is unit lower-triangular ($T \le 16$), $D_{\text{spec}} \in \mathbb{R}^{V \times T}$ is solved by direct forward substitution with **zero matrix inversion**:
$$D_{\text{spec}}[:, t] = R[:, t] - \sum_{j=0}^{t-1} A_{t,j}\, D_{\text{spec}}[:, j], \qquad \mathrm{out}[t, :] = e^{G_s[t]}\, hw_q[:, t] + \sum_{j=0}^{t} e^{G_s[t] - G_s[j]}\, Q_{\text{gram}}[j, t]\, D_{\text{spec}}[:, j].$$

**Verified Output (`prove_wy_gdn_circular.py`):** Across circular ring counts $W \in \{0, 19, 48\}$ (including wrap-around `cache_base = 53, 40` mod `L = 64`) and `BS ∈ {2, 6, 8, 12, 16}`: `f64_err = 1.73e-17..2.78e-17`, `f32_err = 7.45e-09..1.49e-08`.

---

## Theorem 5B: Losslessness of $O(1)$ SAM Tail-Draft Grafting

**Script:** [`proofs/prove_sam_draft_grafting.py`](../proofs/prove_sam_draft_grafting.py)  
**Implementation:** `sglang_sm89/sam_drafter.py`

When a parallel drafter has low confidence on slot $j$ ($\max_v q_j(v) < \tau$) and an $O(1)$ Suffix Automaton over prompt + committed tokens produces a valid continuation $u_{\text{sam}}$ matching length $\ge L_{\min}$ with $M_j(u_{\text{sam}}) = 1$, replacing slot $j$'s proposal with $\mathbf{1}_{u_{\text{sam}}}$ (or $(1-\epsilon)\mathbf{1}_{u_{\text{sam}}} + \epsilon q_j$) preserves **exact target distribution $\mathrm{TV}(P_{\text{out}}, P_{\text{target}}) = 0$** under speculative verification, while raising single-step acceptance whenever $P_{\text{target}}(u_{\text{sam}}) > \max_v q_j(v)$.

**Verified Output (`prove_sam_draft_grafting.py`):** `max_q = 0.093 < tau = 0.35`, `p_target(u_sam) = 0.7457`, `alpha_parallel = 0.2835 -> alpha_grafted = 0.7457`, `exact_TV(det) = 8.09e-17`, `exact_TV(mix) = 9.52e-17`, `MC_TV(N=100000) = 0.00397`.

---

## Ada `sm_89` INT4 $\to$ FP8 E4M3 Bijection & Roofline Summary

Every signed 4-bit integer $q \in \{-8, \dots, +7\}$ has at most 3 explicit mantissa bits and fits **bit-exactly** in `fp8_e4m3fn` (`err = 0.0`), enabling future `sm_89` `W4A8` kernels (`mma.m16n8k32.f32.e4m3.e4m3.f32`) to target Ada's full `294.2 TFLOPS` FP8 Tensor Cores (`2×` W4A16's `147.1 TFLOPS`) when paired with Block-128 Hadamard outlier rotation.
