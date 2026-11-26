#!/usr/bin/env python3
"""Isolated GPU gates for sglang-sm89:

  Gate 1: Coupled Sampled/Mixed Grammar Theorem 1C + Eager/Graph Buffer Isolation
          (coupled_grammar)
  Gate 2: Unified-64 sparse_vocab_proj + M=256/257/289 + N=2051 tail-bit
          + CUDA Graph + C++ roofline dispatch (vocab_projection)
  Gate 3: WY GDN sm_89 vs SGLang Triton gdn_replayssm_spec_decode numerical
          parity, circular ring wrap-around, CUDA Graph replay, and latency
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)


def summarize_us(samples_us: list[float]) -> dict:
    ordered = sorted(samples_us)
    return {
        "median_us": round(statistics.median(samples_us), 3),
        "p95_us": round(ordered[(95 * len(samples_us) + 99) // 100 - 1], 3),
        "min_us": round(ordered[0], 3),
    }


def event_times_us(operation, iters: int, *, flush_buf: torch.Tensor | None = None, warmup: int = 5) -> dict:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, stop in events:
        if flush_buf is not None:
            flush_buf.zero_()
        start.record()
        operation()
        stop.record()
    torch.cuda.synchronize()
    return summarize_us([s.elapsed_time(e) * 1000.0 for s, e in events])


class _GateToolGrammar:
    """Alternating forced-token and two-choice stateful grammar with clean rollback."""

    def __init__(self) -> None:
        self._step_masks = ((1 << 13), (1 << 13) | (1 << 17))
        self.history = [9]

    def _current_mask(self) -> int:
        parity_idx = (len(self.history) - 1) & 1
        return self._step_masks[parity_idx]

    def is_terminated(self) -> bool:
        return False

    def fill_vocab_mask(self, mask: torch.Tensor, row: int) -> None:
        mask[row].zero_()
        mask[row, 0] = self._current_mask()

    def accept_token(self, token: int) -> None:
        allowed_bits = self._current_mask()
        if not (0 <= token < 32) or not ((allowed_bits >> token) & 1):
            raise AssertionError(f"Rejected illegal token {token} under mask {bin(allowed_bits)}")
        self.history.append(token)

    def rollback(self, count: int) -> None:
        if count > 0:
            self.history = self.history[:-count]


def run_gate1(iters: int) -> dict:
    from sglang_sm89.coupled_grammar import (
        _STEP_TLS,
        after_compute_candidates,
        around_dflash_forward_batch_generation,
        around_grammar_tree_from_linear_chain,
    )

    device = torch.device("cuda")
    fixtures = ((1, "greedy"), (1, "sampled"), (8, "greedy"), (8, "sampled"), (8, "mixed"))
    rows = []
    for batch_size, mode in fixtures:
        reqs = [SimpleNamespace(grammar=_GateToolGrammar()) for _ in range(batch_size)]
        temps = [
            0.0 if mode == "greedy" or (mode == "mixed" and i % 2 == 0) else 0.6
            for i in range(batch_size)
        ]
        sampling_info = SimpleNamespace(
            is_all_greedy=(mode == "greedy"),
            temperatures=torch.tensor(temps, device=device),
            top_ks=torch.zeros(batch_size, dtype=torch.int32, device=device),
        )
        batch = SimpleNamespace(has_grammar=True, reqs=reqs, sampling_info=sampling_info)

        # Include illegal tokens (11), forced token (13), and multi-choice slot duplication checks.
        ids_orig = torch.tensor([0, 11, 13, 11, 11, 11, 13, 11], device=device).repeat(batch_size, 1)
        cand_orig = torch.tensor([11, 13, 17, *range(19, 32)], device=device).repeat(batch_size, 7, 1)
        probs_orig = torch.tensor([0.65, 0.25, 0.10, *([0.0] * 13)], device=device).repeat(batch_size, 7, 1)
        ids = ids_orig.clone()
        candidates = cand_orig.clone()
        probs = probs_orig.clone()

        # Poison stale _draft_sampler buffers on eager checks to verify strict _selector_sample / eager isolation
        stale_cand = torch.full_like(cand_orig, 29)
        stale_probs = torch.full_like(probs_orig, 1.0 / 16.0)
        sampler = SimpleNamespace(candidate_out=stale_cand.clone(), q_out=stale_probs.clone())
        worker = SimpleNamespace(
            _draft_sampler=sampler,
            _selector_sample=(candidates, probs) if mode != "greedy" else None,
            target_worker=SimpleNamespace(model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=2051))),
        )
        barrier_calls = 0

        def barrier() -> None:
            nonlocal barrier_calls
            barrier_calls += 1

        def identity(_cls, t):
            return t

        def forward(_worker, _batch, grammar_barrier=None):
            after_compute_candidates(None, (candidates.reshape(batch_size * 7, -1), None))
            _STEP_TLS.draft_ran_graph = False
            return around_grammar_tree_from_linear_chain(identity, None, ids)

        def reset() -> None:
            ids.copy_(ids_orig)
            candidates.copy_(cand_orig)
            probs.copy_(probs_orig)
            sampler.candidate_out.copy_(stale_cand)
            sampler.q_out.copy_(stale_probs)
            torch.cuda.synchronize()

        for _ in range(5):
            reset()
            around_dflash_forward_batch_generation(forward, worker, batch, grammar_barrier=barrier)

        durations_us = []
        barrier_calls = 0
        for _ in range(iters):
            reset()
            t0 = time.perf_counter_ns()
            around_dflash_forward_batch_generation(forward, worker, batch, grammar_barrier=barrier)
            torch.cuda.synchronize()
            durations_us.append((time.perf_counter_ns() - t0) / 1000.0)

        rollback_exact = all(req.grammar.history == [9] for req in reqs)
        sorted_cand = candidates.sort(dim=-1).values
        duplicate_count = int((sorted_cand[..., 1:] == sorted_cand[..., :-1]).sum().item())
        q_sums_valid = torch.allclose(probs.sum(dim=-1), torch.ones((batch_size, 7), device=device), atol=1e-5)
        stale_untouched = torch.equal(sampler.candidate_out, stale_cand) and torch.equal(sampler.q_out, stale_probs)

        expected_greedy = torch.tensor([0, 13, 13, 13, 13, 13, 13, 13], device=device)
        per_row_ok = []
        for b in range(batch_size):
            if temps[b] == 0.0:
                per_row_ok.append(torch.equal(ids[b], expected_greedy))
            else:
                # Odd positions forced to 13, even positions allowed {13, 17}
                per_row_ok.append(all(int(ids[b, p].item()) in (13, 17) for p in range(1, 8)))
                # Coupled probabilities for choice steps preserve 0.25:0.10 ratio -> 5/7 : 2/7
                p13_idx = (candidates[b, 1] == 13).nonzero(as_tuple=False)[0, 0]
                p17_idx = (candidates[b, 1] == 17).nonzero(as_tuple=False)[0, 0]
                ratio_ok = abs(float(probs[b, 1, p13_idx]) - (0.25 / 0.35)) < 1e-4 and abs(
                    float(probs[b, 1, p17_idx]) - (0.10 / 0.35)
                ) < 1e-4
                per_row_ok.append(ratio_ok)

        correct = rollback_exact and duplicate_count == 0 and q_sums_valid and stale_untouched and all(per_row_ok)
        rows.append({
            "mode": mode,
            "batch": batch_size,
            "draft_block": int(ids.shape[1]),
            "correct": correct,
            "exact_rollback": rollback_exact,
            "duplicate_candidates": duplicate_count,
            "q_sums_valid": q_sums_valid,
            "stale_sampler_untouched": stale_untouched,
            "barrier_calls": barrier_calls,
            "first_row_ids": ids[0].tolist(),
            "timing": summarize_us(durations_us),
        })

    # Explicit eager-sampled & eager-greedy poisoned-buffer counterexample checks (Theorem 1D)
    for iso_mode in ("eager_sampled_poisoned", "eager_greedy_poisoned"):
        reqs = [SimpleNamespace(grammar=_GateToolGrammar())]
        is_greedy = iso_mode == "eager_greedy_poisoned"
        batch = SimpleNamespace(
            has_grammar=True,
            reqs=reqs,
            sampling_info=SimpleNamespace(
                is_all_greedy=is_greedy,
                temperatures=torch.tensor([0.0 if is_greedy else 0.6], device=device),
                top_ks=torch.zeros(1, dtype=torch.int32, device=device),
            ),
        )
        ids = torch.tensor([[0, 11]], device=device)
        fresh_cand = torch.tensor([[[17, 13, 19, 20]]], device=device)
        fresh_q = torch.tensor([[[0.9, 0.1, 0.0, 0.0]]], device=device)
        poison_cand = torch.tensor([[[29, 30, 31, 28]]], device=device)
        poison_q = torch.tensor([[[0.1, 0.9, 0.0, 0.0]]], device=device)
        sampler = SimpleNamespace(candidate_out=poison_cand.clone(), q_out=poison_q.clone())
        worker = SimpleNamespace(
            _draft_sampler=sampler,
            _selector_sample=None if is_greedy else (fresh_cand, fresh_q),
            target_worker=SimpleNamespace(model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=2051))),
        )

        def _fwd(_w, _b):
            after_compute_candidates(None, (fresh_cand.reshape(1, -1), None))
            _STEP_TLS.draft_ran_graph = False
            return around_grammar_tree_from_linear_chain(lambda _c, t: t, None, ids)

        around_dflash_forward_batch_generation(_fwd, worker, batch)
        if is_greedy:
            ok = int(ids[0, 1].item()) == 13 and torch.equal(sampler.candidate_out, poison_cand)
        else:
            p13 = (fresh_cand[0, 0] == 13).nonzero(as_tuple=False)
            ok = (
                int(ids[0, 1].item()) == 13
                and p13.numel() == 1
                and abs(float(fresh_q[0, 0, p13[0, 0]].item()) - 1.0) < 1e-6
                and torch.allclose(fresh_q.sum(dim=-1), torch.ones((1, 1), device=device), atol=1e-5)
                and torch.equal(sampler.candidate_out, poison_cand)
                and torch.equal(sampler.q_out, poison_q)
            )
        rows.append({"mode": iso_mode, "batch": 1, "draft_block": 2, "correct": ok})

    return {"gate": "gate1_coupled_grammar", "passed": all(r["correct"] for r in rows), "rows": rows}


def _make_mask(m: int, n: int, density: float, *, device: torch.device, dirty_tail: bool = False) -> torch.Tensor:
    words = (n + 31) // 32
    mask = torch.zeros((m, words), dtype=torch.int32, device="cpu")
    k_active = max(1, int(round(n * density)))
    rng = torch.Generator(device="cpu").manual_seed(1234 + m + n)
    if k_active >= n:
        mask.fill_(-1)
    else:
        for row_m in range(m):
            idx = torch.randperm(n, generator=rng)[:k_active]
            for v in idx.tolist():
                mask[row_m, v >> 5] |= 1 << (v & 31)
    if dirty_tail and (n & 31) != 0:
        mask[:, -1] |= (-1 << (n & 31))
    return mask.to(device)


def _reference_bf16_masked(hidden: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(hidden, weight.T)
    n = logits.shape[1]
    bit_pos = torch.arange(n, device=logits.device, dtype=torch.int32)
    selected_words = mask[:, bit_pos >> 5]
    allowed = ((selected_words >> (bit_pos & 31).unsqueeze(0)) & 1).bool()
    return logits.masked_fill(~allowed, float("-inf"))


def run_gate2(iters: int, *, n_main: int = 32768) -> dict:
    from sglang_sm89.vocab_projection import (
        ensure_extension_loaded,
        select_sparse_vocab_proj,
        sparse_vocab_bf16_lm_head,
        sparse_vocab_fp8_lm_head,
    )

    assert ensure_extension_loaded(), "sglang_sm89._C is required for Gate 2"
    device = torch.device("cuda")
    k = 5120

    # Part A: Tail-bit parity N=2051 across M in {1, 8, 64, 256, 257, 289} + dirty padding bits
    n_tail = 2051
    rng = torch.Generator(device="cpu").manual_seed(92033)
    w_int = torch.randint(-1, 2, (n_tail, k), generator=rng, dtype=torch.int8).to(device=device)
    w_bf16_tail = w_int.to(torch.bfloat16) * 0.125
    w_fp8_tail = w_int.to(torch.float8_e4m3fn)
    scale_tail = torch.full((n_tail,), 0.125, dtype=torch.float32, device=device)
    del w_int

    parity_rows = []
    for m in (1, 8, 64, 256, 257, 289):
        h = torch.randint(-1, 2, (m, k), generator=rng, dtype=torch.int8).to(device=device, dtype=torch.bfloat16)
        msk = _make_mask(m, n_tail, 0.01, device=device, dirty_tail=True)
        ref = _reference_bf16_masked(h, w_bf16_tail, msk)

        out_b = torch.full_like(ref, 23.0)
        t_b = event_times_us(lambda: sparse_vocab_bf16_lm_head(h, w_bf16_tail, msk, out=out_b), iters)
        bad_b = int((out_b != ref).sum().item())
        bad_late_b = int((out_b[256:] != ref[256:]).sum().item())

        out_f = torch.full_like(ref, 23.0)
        t_f = event_times_us(lambda: sparse_vocab_fp8_lm_head(h, w_fp8_tail, scale_tail, msk, out=out_f), iters)
        bad_f = int((out_f != ref).sum().item())
        bad_late_f = int((out_f[256:] != ref[256:]).sum().item())

        parity_rows.append({
            "n": n_tail,
            "m": m,
            "bf16_exact": bad_b == 0,
            "fp8_exact": bad_f == 0,
            "bad_late_rows": bad_b + bad_f if (bad_late_b or bad_late_f) else 0,
            "bf16_timing": t_b,
            "fp8_timing": t_f,
            "correct": (bad_b == 0) and (bad_f == 0),
        })

    del w_bf16_tail, w_fp8_tail, scale_tail
    torch.cuda.empty_cache()

    # Part B: Mixed-density 64-row Unified-64 check (56 structural + 8 dense rows) + C++ vs NumPy dispatch
    w_main_bf16 = torch.randn((n_main, k), dtype=torch.bfloat16, device=device) * 0.02
    words_main = (n_main + 31) // 32
    mix_cpu = torch.zeros((64, words_main), dtype=torch.int32)
    for r in range(56):
        tok = (r * 37) % n_main
        mix_cpu[r, tok >> 5] |= 1 << (tok & 31)
    for r in range(56, 64):
        idx = torch.randperm(n_main, generator=rng)[: max(1, n_main // 64)]
        for tok in idx.tolist():
            mix_cpu[r, tok >> 5] |= 1 << (tok & 31)

    cpp_dec = select_sparse_vocab_proj(mix_cpu, n_main, k_chunk=8, max_eff_row_ratio=0.55, force_numpy=False)
    np_dec = select_sparse_vocab_proj(mix_cpu, n_main, k_chunk=8, max_eff_row_ratio=0.55, force_numpy=True)
    disp_times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        select_sparse_vocab_proj(mix_cpu, n_main, k_chunk=8, max_eff_row_ratio=0.55, force_numpy=False)
        disp_times.append((time.perf_counter_ns() - t0) / 1000.0)
    cpp_dispatch_timing = summarize_us(disp_times)

    h64 = torch.randn((64, k), dtype=torch.bfloat16, device=device) * 0.1
    mix_gpu = mix_cpu.to(device)
    ref64 = _reference_bf16_masked(h64, w_main_bf16, mix_gpu)
    out64 = sparse_vocab_bf16_lm_head(h64, w_main_bf16, mix_gpu)
    active64 = torch.isfinite(ref64)
    mix_diff = float((out64[active64] - ref64[active64]).abs().max().item())
    mix_ok = (cpp_dec == np_dec) and torch.equal(torch.isfinite(out64), active64) and (mix_diff <= 0.05)

    # Part C: CUDA Graph capture + dynamic mask swap at M=64
    msk_cg = _make_mask(64, n_main, 0.005, device=device)
    out_cg = torch.empty((64, n_main), dtype=torch.bfloat16, device=device)
    sparse_vocab_bf16_lm_head(h64, w_main_bf16, msk_cg, out=out_cg)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_vocab_bf16_lm_head(h64, w_main_bf16, msk_cg, out=out_cg)
    msk_swap = _make_mask(64, n_main, 0.002, device=device)
    msk_cg.copy_(msk_swap)
    graph.replay()
    ref_cg = _reference_bf16_masked(h64, w_main_bf16, msk_swap)
    cg_diff = float((out_cg[torch.isfinite(ref_cg)] - ref_cg[torch.isfinite(ref_cg)]).abs().max().item())
    cg_ok = torch.equal(torch.isfinite(out_cg), torch.isfinite(ref_cg)) and (cg_diff <= 0.05)

    # Part D: Cold-DRAM microbenchmark (sparse M in {1, 8, 64} + mixed-64 Unified-64 vs dense matmul)
    flush_buf = torch.empty((48 * 1024 * 1024 // 4,), dtype=torch.float32, device=device)
    cold_bench = []
    for m in (1, 8, 64):
        hm = h64[:m]
        dense_t = event_times_us(lambda: torch.matmul(hm, w_main_bf16.T), iters, flush_buf=flush_buf)
        mskm = mix_gpu if m == 64 else _make_mask(m, n_main, 0.005, device=device)
        sp_t = event_times_us(
            lambda: sparse_vocab_bf16_lm_head(hm, w_main_bf16, mskm, out=out_cg[:m]),
            iters,
            flush_buf=flush_buf,
        )
        cold_bench.append({
            "case": "mixed_56struct_8str_r64" if m == 64 else f"sparse_r{m}",
            "m": m,
            "n": n_main,
            "dense_bf16_us": dense_t["median_us"],
            "sparse_bf16_us": sp_t["median_us"],
            "speedup": round(dense_t["median_us"] / sp_t["median_us"], 2),
        })

    del w_main_bf16, out_cg, out64, ref64, ref_cg, flush_buf
    torch.cuda.empty_cache()

    passed = all(r["correct"] for r in parity_rows) and mix_ok and cg_ok
    return {
        "gate": "gate2_unified64_sparse_vocab_proj",
        "passed": passed,
        "n_main": n_main,
        "tail_parity": parity_rows,
        "mixed_64_unified_ok": mix_ok,
        "mixed_64_max_diff": round(mix_diff, 5),
        "dispatch_cpp_numpy_agree": cpp_dec == np_dec,
        "cpp_dispatch_timing": cpp_dispatch_timing,
        "cudagraph_ok": cg_ok,
        "cudagraph_max_diff": round(cg_diff, 5),
        "cold_dram_bench": cold_bench,
    }


def _pytorch_replayssm_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    scale: float,
    max_cache_len: int,
    rawv_cache: torch.Tensor | None = None,
    rawk_cache: torch.Tensor | None = None,
    beta_cache: torch.Tensor | None = None,
    softplus_threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch sequential reference for ReplaySSM verify + flush fold."""
    total_tokens, h, k_dim = q.shape
    hv, v_dim = v.shape[1], v.shape[2]
    ratio = hv // h
    bs = int(ssm_state_indices.shape[0])
    out = torch.zeros((total_tokens, hv, v_dim), dtype=v.dtype, device=v.device)
    h0_out = checkpoint_state.clone()
    exact_raw = rawv_cache is not None and rawk_cache is not None and beta_cache is not None

    for bn in range(bs):
        bos = int(query_start_loc[bn].item())
        eos = int(query_start_loc[bn + 1].item())
        sidx = int(ssm_state_indices[bn].item())
        if sidx <= -1 or eos <= bos:
            continue
        ridx = int(replay_indices[bn].item())
        wp = int(write_pos[ridx].item()) if ridx >= 0 else 0
        cbase = int(cache_base[ridx].item()) if ridx >= 0 else 0
        flushed = bool(is_flush[ridx].item()) if ridx >= 0 else False

        for hv_i in range(hv):
            h_i = hv_i // ratio
            h_state = h0_out[sidx, hv_i].float().clone()  # [V, K]
            if wp > 0 and ridx >= 0:
                for c in range(wp):
                    phys = (cbase + c) & (max_cache_len - 1)
                    dec = math.exp(float(g_cache[ridx, hv_i, phys].item()))
                    if flushed and exact_raw:
                        kc = rawk_cache[ridx, h_i, phys].float()
                        kc = kc / math.sqrt(float((kc * kc).sum().item()) + 1e-6)
                        vc = rawv_cache[ridx, hv_i, phys].float()
                        beta_c = float(beta_cache[ridx, hv_i, phys].item())
                        delta_c = beta_c * (vc - dec * (h_state @ kc))
                        h_state = dec * h_state + torch.outer(delta_c, kc)
                    else:
                        kc = k_cache[ridx, h_i, phys].float()
                        dc = d_cache[ridx, hv_i, phys].float()
                        h_state = dec * h_state + torch.outer(dc, kc)
            if flushed:
                h0_out[sidx, hv_i] = h_state.to(h0_out.dtype)

            for t in range(eos - bos):
                qt = q[bos + t, h_i].float()
                kt = k[bos + t, h_i].float()
                qt = qt / math.sqrt(float((qt * qt).sum().item()) + 1e-6) * scale
                kt = kt / math.sqrt(float((kt * kt).sum().item()) + 1e-6)
                vt = v[bos + t, hv_i].float()

                x = float(a[bos + t, hv_i].item()) + float(dt_bias[hv_i].item())
                sp = math.log1p(math.exp(x)) if x <= softplus_threshold else x
                g_val = -math.exp(float(A_log[hv_i].item())) * sp
                beta_val = 1.0 / (1.0 + math.exp(-float(b[bos + t, hv_i].item())))

                dec = math.exp(g_val)
                v_hat = h_state @ kt
                delta = beta_val * (vt - dec * v_hat)
                h_state = dec * h_state + torch.outer(delta, kt)
                out[bos + t, hv_i] = (h_state @ qt).to(out.dtype)
    return out, h0_out


def run_gate3(iters: int) -> dict:
    import inspect
    from sglang_sm89.wy_gdn_verify import wy_gdn_replayssm_verify

    extra_verify_kw = {}
    if "force_cuda" in inspect.signature(wy_gdn_replayssm_verify).parameters:
        extra_verify_kw["force_cuda"] = True

    gdn_replayssm_spec_decode = None
    ref_backend = "pytorch_exact"
    for _mod in (
        "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode",
        "sglang.srt.layers.attention.linear.gdn_replayssm_spec_decode",
    ):
        try:
            _m = __import__(_mod, fromlist=["gdn_replayssm_spec_decode"])
            _fn = getattr(_m, "gdn_replayssm_spec_decode", None)
            if _fn is not None:
                gdn_replayssm_spec_decode = _fn
                ref_backend = "sglang_triton"
                break
        except Exception:
            continue

    device = torch.device("cuda")
    torch.manual_seed(77)
    H, HV, K, V, L = 16, 48, 128, 128, 64
    scale = K ** -0.5

    cases = []
    upstream_both_bug = False
    for ckpt_dtype in (torch.bfloat16, torch.float32):
        for B, spec_len, init_wp, base_off, do_flush in (
            (1, 6, 0, 0, False),
            (1, 8, 12, 58, False),  # circular wrap-around
            (4, 8, 5, 60, False),
            (8, 8, 3, 62, False),   # c=8, draft=8 prod shape
            (8, 12, 3, 62, False),
            (2, 16, 4, 0, False),
            (2, 8, 6, 61, True),
        ):
            num_slots = B + 2
            total_tokens = B * spec_len
            q = torch.randn((total_tokens, H, K), dtype=torch.bfloat16, device=device) * 0.5
            k = torch.randn((total_tokens, H, K), dtype=torch.bfloat16, device=device) * 0.5
            v = torch.randn((total_tokens, HV, V), dtype=torch.bfloat16, device=device) * 0.5
            a = torch.randn((total_tokens, HV), dtype=torch.float32, device=device) * 0.2
            b = torch.randn((total_tokens, HV), dtype=torch.float32, device=device) * 0.2
            A_log = torch.full((HV,), -0.5, dtype=torch.float32, device=device)
            dt_bias = torch.zeros((HV,), dtype=torch.float32, device=device)

            h0 = torch.randn((num_slots, HV, V, K), dtype=ckpt_dtype, device=device) * 0.05
            d_cache = torch.randn((num_slots, HV, L, V), dtype=torch.bfloat16, device=device) * 0.05
            k_cache = F.normalize(torch.randn((num_slots, H, L, K), dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16)
            g_cache = torch.full((num_slots, HV, L), -0.05, dtype=torch.float32, device=device)
            rawv_cache = torch.randn_like(d_cache) * 0.05 if do_flush else torch.zeros_like(d_cache)
            rawk_cache = k_cache.clone() if do_flush else torch.zeros_like(k_cache)
            beta_cache = torch.full((num_slots, HV, L), 0.5, dtype=torch.float32, device=device) if do_flush else None

            qsl = torch.arange(0, B + 1, dtype=torch.int32, device=device) * spec_len
            ssm_idx = torch.arange(B, dtype=torch.int32, device=device)
            if B >= 2:
                ssm_idx[-1] = -1  # test null slot handling
            rep_idx = torch.arange(B, dtype=torch.int32, device=device)
            wp = torch.full((num_slots,), init_wp, dtype=torch.int32, device=device)
            cbase = torch.full((num_slots,), base_off, dtype=torch.int32, device=device)
            flush_t = torch.full((num_slots,), 1 if do_flush else 0, dtype=torch.int8, device=device)

            out_cu = torch.empty((total_tokens, HV, V), dtype=torch.bfloat16, device=device)
            d_cu, k_cu, g_cu = d_cache.clone(), k_cache.clone(), g_cache.clone()
            rv_cu, rk_cu = rawv_cache.clone(), rawk_cache.clone()
            bt_cu = beta_cache.clone() if beta_cache is not None else None
            h0_cu, wp_cu = h0.clone(), wp.clone()

            wy_gdn_replayssm_verify(
                q, k, v, a, b, A_log, dt_bias, h0_cu,
                d_cu, k_cu, g_cu, rv_cu, rk_cu, bt_cu,
                qsl, ssm_idx, rep_idx, wp_cu, cbase, flush_t,
                out_cu, scale=scale, max_cache_len=L, max_spec_len=spec_len,
                launch_mode="both" if do_flush else "verify",
                **extra_verify_kw,
            )

            out_exact, h0_exact = _pytorch_replayssm_reference(
                q, k, v, a, b, A_log, dt_bias, h0.clone(),
                d_cache.clone(), k_cache.clone(), g_cache.clone(),
                qsl, ssm_idx, rep_idx, wp.clone(), cbase, flush_t,
                scale=scale, max_cache_len=L,
                rawv_cache=rawv_cache, rawk_cache=rawk_cache, beta_cache=beta_cache,
            )
            out_ref = out_exact
            h0_ref = h0_exact
            if gdn_replayssm_spec_decode is not None:
                out_tr = torch.empty_like(out_cu)
                d_ref, k_ref, g_ref = d_cache.clone(), k_cache.clone(), g_cache.clone()
                rv_ref, rk_ref = rawv_cache.clone(), rawk_cache.clone()
                bt_ref = beta_cache.clone() if beta_cache is not None else None
                h0_tr, wp_ref = h0.clone(), wp.clone()
                try:
                    gdn_replayssm_spec_decode(
                        q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
                        checkpoint_state=h0_tr,
                        d_cache=d_ref, k_cache=k_ref, g_cache=g_ref,
                        rawv_cache=rv_ref, rawk_cache=rk_ref, beta_cache=bt_ref,
                        query_start_loc=qsl, ssm_state_indices=ssm_idx,
                        replay_indices=rep_idx, write_pos=wp_ref, cache_base=cbase,
                        is_flush=flush_t, out=out_tr, scale=scale,
                        max_cache_len=L, max_spec_len=spec_len,
                        use_qk_l2norm_in_kernel=True,
                        null_block_id=-1,
                        launch_mode="both" if do_flush else "verify",
                    )
                    out_ref = out_tr
                    h0_ref = h0_tr
                except TypeError:
                    upstream_both_bug = True

            max_err = float((out_cu.float() - out_exact.float()).abs().max().item())
            triton_err = float((out_cu.float() - out_ref.float()).abs().max().item())
            state_err = float((h0_cu.float() - h0_exact.float()).abs().max().item()) if do_flush else 0.0
            null_ok = True if B < 2 else float(out_cu[(B - 1) * spec_len :].abs().max().item()) == 0.0

            wy_t = None
            triton_t = None
            speedup = None
            if not do_flush:
                wy_t = event_times_us(
                    lambda: wy_gdn_replayssm_verify(
                        q, k, v, a, b, A_log, dt_bias, h0_cu,
                        d_cu, k_cu, g_cu, rv_cu, rk_cu, bt_cu,
                        qsl, ssm_idx, rep_idx, wp_cu, cbase, flush_t,
                        out_cu, scale=scale, max_cache_len=L, max_spec_len=spec_len,
                        launch_mode="verify", **extra_verify_kw,
                    ),
                    iters,
                )
                if gdn_replayssm_spec_decode is not None:
                    triton_t = event_times_us(
                        lambda: gdn_replayssm_spec_decode(
                            q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
                            checkpoint_state=h0_ref,
                            d_cache=d_ref, k_cache=k_ref, g_cache=g_ref,
                            rawv_cache=rv_ref, rawk_cache=rk_ref, beta_cache=bt_ref,
                            query_start_loc=qsl, ssm_state_indices=ssm_idx,
                            replay_indices=rep_idx, write_pos=wp_ref, cache_base=cbase,
                            is_flush=flush_t, out=out_ref, scale=scale,
                            max_cache_len=L, max_spec_len=spec_len,
                            use_qk_l2norm_in_kernel=True,
                        null_block_id=-1,
                        launch_mode="verify",
                        ),
                        iters,
                    )
                    speedup = round(triton_t["median_us"] / wy_t["median_us"], 2)

            cases.append({
                "ckpt_dtype": str(ckpt_dtype).replace("torch.", ""),
                "B": B,
                "spec_len": spec_len,
                "write_pos": init_wp,
                "cache_base": base_off,
                "is_flush": do_flush,
                "max_abs_err": round(max_err, 6),
                "triton_tf32_abs_err": round(triton_err, 6),
                "flush_state_abs_err": round(state_err, 6),
                "null_slot_zero": null_ok,
                "correct": (max_err <= 0.02) and (triton_err <= 0.035) and (state_err <= 0.02) and null_ok,
                "wy_timing": wy_t,
                "triton_timing": triton_t,
                "speedup": speedup,
            })

    # CUDA Graph capture + replay parity at B=4, spec_len=8, ckpt=bfloat16
    cg_b, cg_s = 4, 8
    cg_slots = cg_b + 2
    cg_toks = cg_b * cg_s
    q_cg = torch.randn((cg_toks, H, K), dtype=torch.bfloat16, device=device) * 0.5
    k_cg = torch.randn((cg_toks, H, K), dtype=torch.bfloat16, device=device) * 0.5
    v_cg = torch.randn((cg_toks, HV, V), dtype=torch.bfloat16, device=device) * 0.5
    a_cg = torch.randn((cg_toks, HV), dtype=torch.float32, device=device) * 0.2
    b_cg = torch.randn((cg_toks, HV), dtype=torch.float32, device=device) * 0.2
    A_cg = torch.full((HV,), -0.5, dtype=torch.float32, device=device)
    dt_cg = torch.zeros((HV,), dtype=torch.float32, device=device)
    h0_cg = torch.randn((cg_slots, HV, V, K), dtype=torch.bfloat16, device=device) * 0.05
    d_cg = torch.randn((cg_slots, HV, L, V), dtype=torch.bfloat16, device=device) * 0.05
    kcache_cg = F.normalize(torch.randn((cg_slots, H, L, K), dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16)
    g_cg = torch.full((cg_slots, HV, L), -0.05, dtype=torch.float32, device=device)
    rv_cg = torch.zeros_like(d_cg)
    rk_cg = torch.zeros_like(kcache_cg)
    bt_cg = torch.zeros((cg_slots, HV, L), dtype=torch.float32, device=device)
    qsl_cg = torch.arange(0, cg_b + 1, dtype=torch.int32, device=device) * cg_s
    ssm_cg = torch.arange(cg_b, dtype=torch.int32, device=device)
    rep_cg = torch.arange(cg_b, dtype=torch.int32, device=device)
    wp_cg = torch.full((cg_slots,), 4, dtype=torch.int32, device=device)
    cbase_cg = torch.full((cg_slots,), 60, dtype=torch.int32, device=device)
    flush_cg = torch.zeros((cg_slots,), dtype=torch.int8, device=device)
    out_cg = torch.empty((cg_toks, HV, V), dtype=torch.bfloat16, device=device)

    wy_gdn_replayssm_verify(
        q_cg, k_cg, v_cg, a_cg, b_cg, A_cg, dt_cg, h0_cg,
        d_cg, kcache_cg, g_cg, rv_cg, rk_cg, bt_cg,
        qsl_cg, ssm_cg, rep_cg, wp_cg, cbase_cg, flush_cg,
        out_cg, scale=scale, max_cache_len=L, max_spec_len=cg_s,
        launch_mode="verify",
    )
    out_eager = out_cg.clone()
    out_cg.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wy_gdn_replayssm_verify(
            q_cg, k_cg, v_cg, a_cg, b_cg, A_cg, dt_cg, h0_cg,
            d_cg, kcache_cg, g_cg, rv_cg, rk_cg, bt_cg,
            qsl_cg, ssm_cg, rep_cg, wp_cg, cbase_cg, flush_cg,
            out_cg, scale=scale, max_cache_len=L, max_spec_len=cg_s,
            launch_mode="verify",
        )
    graph.replay()
    cg_err = float((out_cg.float() - out_eager.float()).abs().max().item())
    cg_ok = cg_err == 0.0

    return {
        "gate": "gate3_wy_gdn_sm89",
        "reference_backend": ref_backend,
        "upstream_launch_mode_both_bug": upstream_both_bug,
        "cudagraph_ok": cg_ok,
        "cudagraph_max_abs_err": cg_err,
        "passed": cg_ok and all(c["correct"] for c in cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("all", "1", "2", "3"), default="all")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--fraction", type=float, default=0.025, help="CUDA memory fraction if co-resident")
    parser.add_argument("--n-main", type=int, default=32768, help="Gate 2 main vocab size (<=65536 co-resident)")
    parser.add_argument("--full-vocab", action="store_true", help="Override --n-main to 248320 (dedicated GPU)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise SystemExit("Gate runner requires an SM89 (compute capability 8.9) CUDA GPU.")

    torch.set_num_threads(1)
    torch.manual_seed(18073)
    n_main = 248320 if args.full_vocab else args.n_main
    if args.fraction < 1.0 and not args.full_vocab:
        torch.cuda.set_per_process_memory_fraction(args.fraction, 0)

    results = []
    if args.gate in ("all", "1"):
        results.append(run_gate1(args.iters))
        torch.cuda.empty_cache()
    if args.gate in ("all", "2"):
        results.append(run_gate2(args.iters, n_main=n_main))
        torch.cuda.empty_cache()
    if args.gate in ("all", "3"):
        results.append(run_gate3(args.iters))
        torch.cuda.empty_cache()

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "max_torch_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "passed": all(r["passed"] for r in results),
        "gates": results,
    }
    encoded = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
