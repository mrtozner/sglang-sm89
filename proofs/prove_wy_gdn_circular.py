#!/usr/bin/env python3
"""
Theorem 4A & 4B: Algebraic Chunk-Parallel WY Forward Substitution + Circular Ring ReplaySSM Equivalence.

Verifies exact equivalence between:
  (1) Step-by-step sequential Gated DeltaNet recurrence over W committed circular-ring
      history steps (including ring wrap-around) + BS speculative verify steps.
  (2) Theorem 4A/4B WY forward-substitution (I + A) D_T = R^T (no matrix inversion)
      with single-pass h0 @ [Q; K] and ring-history folding in projected D-space.

Matches SGLang ReplaySSM / sm_89 CUDA kernel contracts (K=V=128, L2-normalized q, k,
sigmoid beta, exp(-exp(A_log) * softplus(a + dt_bias)) decay, BF16/FP32 checkpoint).

Run:
  python3 proofs/prove_wy_gdn_circular.py
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def sequential_ring_and_verify_ref(
    h0: torch.Tensor,          # [V, K]
    k_hist: torch.Tensor,      # [W, K]
    d_hist: torch.Tensor,      # [W, V]
    g_hist: torch.Tensor,      # [W] step log-decays
    q_spec: torch.Tensor,      # [BS, K] (already L2-normed * scale)
    k_spec: torch.Tensor,      # [BS, K] (already L2-normed)
    v_spec: torch.Tensor,      # [BS, V]
    g_spec: torch.Tensor,      # [BS] step log-decays
    beta_spec: torch.Tensor,   # [BS]
) -> torch.Tensor:
    """Exact sequential reference: replay W committed steps onto h0, then verify BS steps."""
    S = h0.clone()
    W = k_hist.shape[0]
    # Note: in ReplaySSM circular buffer, S_pre = exp(sum g_hist) * h0 + sum_c exp(sum_{j>c} g_hist[j]) * d_hist[c] @ k_hist[c]^T
    for c in range(W):
        S = math.exp(float(g_hist[c])) * S + torch.outer(d_hist[c], k_hist[c])

    BS = q_spec.shape[0]
    outs = []
    for t in range(BS):
        alpha = torch.exp(g_spec[t])
        S_dec = alpha * S
        v_hat = S_dec @ k_spec[t]                       # [V]
        delta = beta_spec[t] * (v_spec[t] - v_hat)      # [V]
        S = S_dec + torch.outer(delta, k_spec[t])       # [V, K]
        outs.append(S @ q_spec[t])                      # [V]
    return torch.stack(outs, dim=0)


def wy_circular_forward_sub(
    h0: torch.Tensor,          # [V, K]
    d_cache: torch.Tensor,     # [L, V] circular buffer
    k_cache: torch.Tensor,     # [L, K] circular buffer
    g_prefix_cache: torch.Tensor,  # [L] cumulative log-decay prefix at slot c
    cache_base: int,
    write_pos: int,            # W <= L
    g_total: torch.Tensor,     # scalar sum_{c=0..W-1} g_hist[c]
    q_spec: torch.Tensor,      # [BS, K]
    k_spec: torch.Tensor,      # [BS, K]
    v_spec: torch.Tensor,      # [BS, V]
    g_spec: torch.Tensor,      # [BS]
    beta_spec: torch.Tensor,   # [BS]
) -> torch.Tensor:
    """Theorem 4A/4B WY verify with circular ring folding & direct forward substitution."""
    L, V = d_cache.shape
    BS = q_spec.shape[0]
    dtype = h0.dtype

    # 1. Single pass over checkpoint h0 for both Q and K -> [V, BS]
    hw_q = h0 @ q_spec.T                                # [V, BS]
    hw_k = h0 @ k_spec.T                                # [V, BS]

    # 2. Fold circular ring history c = 0..W-1 (with wrap-around modulo L)
    if write_pos > 0:
        b_total_decay = torch.exp(g_total)
        hw_q = b_total_decay * hw_q
        hw_k = b_total_decay * hw_k
        idx = [(cache_base + c) & (L - 1) for c in range(write_pos)]
        k_hist = k_cache[idx]                           # [W, K]
        d_hist = d_cache[idx]                           # [W, V]
        g_pref = g_prefix_cache[idx]                    # [W]
        replay_decay = torch.exp(g_total - g_pref)      # [W]
        d_scaled = d_hist * replay_decay[:, None]       # [W, V]
        hw_q = hw_q + d_scaled.T @ (k_hist @ q_spec.T)  # [V, BS]
        hw_k = hw_k + d_scaled.T @ (k_hist @ k_spec.T)  # [V, BS]

    # 3. Speculative block WY matrices
    G_s = torch.cumsum(g_spec, dim=0)                   # [BS]
    expG_s = torch.exp(G_s)                             # [BS]
    kk_mat = k_spec @ k_spec.T                          # [BS, BS]
    kq_mat = k_spec @ q_spec.T                          # [BS, BS]

    # Build strictly lower-triangular A[i, j] = beta_s[i] * exp(G_s[i] - G_s[j]) * kk_mat[i, j]
    A = torch.zeros((BS, BS), dtype=dtype)
    for i in range(1, BS):
        for j in range(i):
            A[i, j] = beta_spec[i] * torch.exp(G_s[i] - G_s[j]) * kk_mat[i, j]

    # RHS R[v, t] = beta_s[t] * (v_spec[t, v] - expG_s[t] * hw_k[v, t])
    R = beta_spec[None, :] * (v_spec.T - expG_s[None, :] * hw_k)  # [V, BS]

    # Direct forward substitution (I + A) D_T = R^T (zero matrix inversion!)
    D_spec = torch.empty((V, BS), dtype=dtype)
    for t in range(BS):
        acc = R[:, t].clone()
        for j in range(t):
            acc = acc - A[t, j] * D_spec[:, j]
        D_spec[:, t] = acc

    # Final outputs out[t, v] = expG_s[t] * hw_q[v, t] + sum_{j <= t} exp(G_s[t] - G_s[j]) * kq_mat[j, t] * D_spec[v, j]
    out = (expG_s[None, :] * hw_q).T.clone()            # [BS, V]
    for t in range(BS):
        for j in range(t + 1):
            w_tj = torch.exp(G_s[t] - G_s[j]) * kq_mat[j, t]
            out[t] = out[t] + w_tj * D_spec[:, j]
    return out


def main() -> None:
    torch.manual_seed(42)
    K = V = 128
    L = 64  # circular ring capacity (power of 2)

    for write_pos, cache_base in [(0, 0), (19, 53), (48, 40)]:  # includes wrap-around > 64
        for spec_len in [2, 6, 8, 12, 16]:
            h0 = torch.randn(V, K, dtype=torch.float64) * 0.1
            d_cache = torch.zeros(L, V, dtype=torch.float64)
            k_cache = torch.zeros(L, K, dtype=torch.float64)
            g_prefix_cache = torch.zeros(L, dtype=torch.float64)

            k_hist = F.normalize(torch.randn(write_pos, K, dtype=torch.float64), dim=-1) if write_pos > 0 else torch.empty(0, K, dtype=torch.float64)
            d_hist = torch.randn(write_pos, V, dtype=torch.float64) * 0.1 if write_pos > 0 else torch.empty(0, V, dtype=torch.float64)
            g_hist = -torch.rand(write_pos, dtype=torch.float64) * 0.08 if write_pos > 0 else torch.empty(0, dtype=torch.float64)
            g_pref = torch.cumsum(g_hist, dim=0) if write_pos > 0 else torch.empty(0, dtype=torch.float64)
            g_total = g_pref[-1] if write_pos > 0 else torch.tensor(0.0, dtype=torch.float64)

            for c in range(write_pos):
                phys = (cache_base + c) & (L - 1)
                k_cache[phys] = k_hist[c]
                d_cache[phys] = d_hist[c]
                g_prefix_cache[phys] = g_pref[c]

            q_spec = F.normalize(torch.randn(spec_len, K, dtype=torch.float64), dim=-1) / math.sqrt(K)
            k_spec = F.normalize(torch.randn(spec_len, K, dtype=torch.float64), dim=-1)
            v_spec = torch.randn(spec_len, V, dtype=torch.float64)
            g_spec = -torch.rand(spec_len, dtype=torch.float64) * 0.1
            beta_spec = torch.sigmoid(torch.randn(spec_len, dtype=torch.float64))

            out_ref = sequential_ring_and_verify_ref(
                h0, k_hist, d_hist, g_hist, q_spec, k_spec, v_spec, g_spec, beta_spec
            )
            out_wy_f64 = wy_circular_forward_sub(
                h0, d_cache, k_cache, g_prefix_cache, cache_base, write_pos, g_total,
                q_spec, k_spec, v_spec, g_spec, beta_spec,
            )
            err_f64 = float((out_ref - out_wy_f64).abs().max().item())
            assert err_f64 < 1e-12, f"FP64 WY circular error too high: {err_f64}"

            out_wy_f32 = wy_circular_forward_sub(
                h0.float(), d_cache.float(), k_cache.float(), g_prefix_cache.float(),
                cache_base, write_pos, g_total.float(),
                q_spec.float(), k_spec.float(), v_spec.float(), g_spec.float(), beta_spec.float(),
            )
            err_f32 = float((out_ref.float() - out_wy_f32).abs().max().item())
            assert err_f32 < 1e-5, f"FP32 WY circular error too high: {err_f32}"

            if spec_len in (8, 16):
                print(
                    f"[PASS] Theorem 4A/4B (W={write_pos:2d}, base={cache_base:2d}, BS={spec_len:2d}): "
                    f"f64_err={err_f64:.2e}, f32_err={err_f32:.2e}"
                )


if __name__ == "__main__":
    main()
