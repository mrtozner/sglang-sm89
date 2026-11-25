"""Parity, CUDA Graph, and latency tests for sm_89 WY GDN ReplaySSM verify."""

from __future__ import annotations

import math
import time

import pytest
import torch
import torch.nn.functional as F

from sglang_sm89.wy_gdn_verify import (
    _can_use_wy_gdn_fast_path,
    around_gdn_replayssm_spec_decode,
    register_hooks,
    wy_gdn_replayssm_verify,
)


def _sequential_replayssm_reference(
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
    rawv_cache: torch.Tensor | None,
    rawk_cache: torch.Tensor | None,
    beta_cache: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    scale: float,
    max_cache_len: int,
    launch_mode: str = "verify",
    softplus_threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact sequential ReplaySSM reference matching SGLang launch_mode semantics."""
    total_tokens, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    ratio = HV // H
    B = int(ssm_state_indices.shape[0])

    out = torch.zeros((total_tokens, HV, V), dtype=v.dtype, device=v.device)
    h0_ref = checkpoint_state.clone()
    d_ref = d_cache.clone()
    k_ref = k_cache.clone()
    g_ref = g_cache.clone()
    wp_ref = write_pos.clone()

    store_exact_raw = (rawv_cache is not None) and (rawk_cache is not None) and (beta_cache is not None)
    store_d_res = (not store_exact_raw) and (rawv_cache is not None)
    store_k_res = (not store_exact_raw) and (rawk_cache is not None)

    for bn in range(B):
        bos = int(query_start_loc[bn].item())
        eos = int(query_start_loc[bn + 1].item())
        spec_len = eos - bos
        if spec_len <= 0:
            continue
        sidx = int(ssm_state_indices[bn].item())
        if sidx <= -1:
            out[bos:eos].zero_()
            continue

        ridx = int(replay_indices[bn].item())
        wp = int(wp_ref[ridx].item())
        cbase = int(cache_base[ridx].item())
        flushed = bool(is_flush[ridx].item() != 0)

        if flushed and launch_mode == "verify":
            continue
        if (not flushed) and launch_mode == "flush":
            continue

        for hv_i in range(HV):
            h_i = hv_i // ratio
            h_state = h0_ref[sidx, hv_i].float().clone()
            if wp > 0:
                if flushed and store_exact_raw:
                    for c in range(wp):
                        phys = (cbase + c) & (max_cache_len - 1)
                        dec = math.exp(float(g_cache[ridx, hv_i, phys].item()))
                        k_raw = rawk_cache[ridx, h_i, phys].float()
                        k_norm = k_raw / math.sqrt(float((k_raw * k_raw).sum().item()) + 1e-6)
                        v_raw = rawv_cache[ridx, hv_i, phys].float()
                        beta_c = float(beta_cache[ridx, hv_i, phys].item())
                        h_state = dec * h_state
                        delta = beta_c * (v_raw - (h_state @ k_norm))
                        h_state = h_state + torch.outer(delta, k_norm)
                else:
                    for c in range(wp):
                        phys = (cbase + c) & (max_cache_len - 1)
                        dec = math.exp(float(g_cache[ridx, hv_i, phys].item()))
                        kc = k_cache[ridx, h_i, phys].float()
                        if store_k_res:
                            kc = kc + rawk_cache[ridx, h_i, phys].float()
                        dc = d_cache[ridx, hv_i, phys].float()
                        if store_d_res:
                            dc = dc + rawv_cache[ridx, hv_i, phys].float()
                        h_state = dec * h_state + torch.outer(dc, kc)

            if flushed:
                h0_ref[sidx, hv_i] = h_state.to(h0_ref.dtype)
                wp_eff = 0
            else:
                wp_eff = wp

            for t in range(spec_len):
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

                if wp_eff + t < max_cache_len:
                    phys_s = (cbase + wp_eff + t) & (max_cache_len - 1)
                    d_ref[ridx, hv_i, phys_s] = delta.to(d_ref.dtype)
                    g_ref[ridx, hv_i, phys_s] = g_val
                    if hv_i == h_i * ratio:
                        k_ref[ridx, h_i, phys_s] = kt.to(k_ref.dtype)

    return out, h0_ref, d_ref, k_ref, g_ref, wp_ref


def _make_replay_inputs(
    *,
    B: int,
    spec_lens: list[int],
    H: int = 16,
    HV: int = 48,
    L: int = 64,
    ckpt_dtype: torch.dtype = torch.bfloat16,
    act_dtype: torch.dtype = torch.bfloat16,
    ab_dtype: torch.dtype = torch.float32,
    write_pos_vals: list[int] | None = None,
    cache_base_vals: list[int] | None = None,
    flush_vals: list[int] | None = None,
    null_last: bool = False,
    with_beta_cache: bool = False,
    with_residuals: bool = True,
    split_qkv_view: bool = False,
    device: torch.device,
    seed: int = 1234,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    K = V = 128
    num_slots = B + 2
    total_tokens = sum(spec_lens)

    def _randn(shape, dtype, std=0.5):
        return (torch.randn(shape, generator=g, dtype=torch.float32) * std).to(device=device, dtype=dtype)

    if split_qkv_view:
        qkv = _randn((total_tokens, H * K * 2 + HV * V), act_dtype, std=0.4)
        q_flat, k_flat, v_flat = qkv.split((H * K, H * K, HV * V), dim=-1)
        q = q_flat.view(total_tokens, H, K)
        k = k_flat.view(total_tokens, H, K)
        v = v_flat.view(total_tokens, HV, V)
    else:
        q = _randn((total_tokens, H, K), act_dtype, std=0.4)
        k = _randn((total_tokens, H, K), act_dtype, std=0.4)
        v = _randn((total_tokens, HV, V), act_dtype, std=0.4)

    a = _randn((total_tokens, HV), ab_dtype, std=0.25)
    b = _randn((total_tokens, HV), ab_dtype, std=0.25)
    A_log = torch.full((HV,), -0.5, dtype=torch.float32, device=device)
    dt_bias = _randn((HV,), torch.float32, std=0.1)

    h0 = _randn((num_slots, HV, V, K), ckpt_dtype, std=0.05)
    d_cache = _randn((num_slots, HV, L, V), act_dtype, std=0.05)
    k_raw_init = F.normalize(torch.randn((num_slots, H, L, K), generator=g, dtype=torch.float32), dim=-1)
    k_cache = k_raw_init.to(device=device, dtype=act_dtype)
    g_cache = torch.full((num_slots, HV, L), -0.04, dtype=torch.float32, device=device)

    if with_beta_cache:
        rawv_cache = _randn((num_slots, HV, L, V), act_dtype, std=0.2)
        rawk_cache = _randn((num_slots, H, L, K), act_dtype, std=0.4)
        beta_cache = torch.full((num_slots, HV, L), 0.35, dtype=torch.float32, device=device)
    else:
        rawv_cache = torch.zeros_like(d_cache) if with_residuals else None
        rawk_cache = (
            (k_raw_init.to(device=device) - k_cache.float()).to(act_dtype)
            if with_residuals
            else None
        )
        beta_cache = None

    qsl_list = [0]
    for sl in spec_lens:
        qsl_list.append(qsl_list[-1] + sl)
    qsl = torch.tensor(qsl_list, dtype=torch.int32, device=device)

    ssm_idx = torch.arange(B, dtype=torch.int32, device=device)
    if null_last and B >= 1:
        ssm_idx[-1] = -1
    rep_idx = torch.arange(B, dtype=torch.int64, device=device)

    wp = torch.zeros((num_slots,), dtype=torch.int32, device=device)
    cbase = torch.zeros((num_slots,), dtype=torch.int32, device=device)
    flush_t = torch.zeros((num_slots,), dtype=torch.int8, device=device)
    for idx in range(B):
        if write_pos_vals is not None:
            wp[idx] = write_pos_vals[idx]
        if cache_base_vals is not None:
            cbase[idx] = cache_base_vals[idx]
        if flush_vals is not None:
            flush_t[idx] = flush_vals[idx]

    return dict(
        q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        checkpoint_state=h0, d_cache=d_cache, k_cache=k_cache, g_cache=g_cache,
        rawv_cache=rawv_cache, rawk_cache=rawk_cache, beta_cache=beta_cache,
        query_start_loc=qsl, ssm_state_indices=ssm_idx, replay_indices=rep_idx,
        write_pos=wp, cache_base=cbase, is_flush=flush_t,
        scale=K ** -0.5, max_cache_len=L, max_spec_len=max(spec_lens),
    )


def test_cpu_fallback_and_guard_logic() -> None:
    """Verify specialization guards cleanly reject CPU / unsupported shapes and use fallback."""
    inputs = _make_replay_inputs(B=2, spec_lens=[6, 6], H=4, HV=8, L=64, device=torch.device("cpu"))
    assert not _can_use_wy_gdn_fast_path(
        **{k: v for k, v in inputs.items() if k != "scale"},
        out=None,
        launch_mode="verify",
        use_qk_l2norm_in_kernel=True,
        null_block_id=-1,
    )
    called = {"count": 0}

    def _fake_fallback():
        called["count"] += 1
        return torch.zeros_like(inputs["v"])

    out = wy_gdn_replayssm_verify(**inputs, fallback_fn=_fake_fallback)
    assert called["count"] == 1
    assert out.shape == inputs["v"].shape

    # Also test around_gdn_replayssm_spec_decode positional arg binding and non-verify passthrough.
    def _dummy_spec(
        q, k, v, a, b, A_log, dt_bias, checkpoint_state,
        d_cache, k_cache, g_cache, rawv_cache=None, rawk_cache=None, beta_cache=None,
        query_start_loc=None, ssm_state_indices=None, replay_indices=None,
        write_pos=None, cache_base=None, is_flush=None, out=None,
        scale=None, max_cache_len=64, max_spec_len=16,
        launch_mode="verify", use_qk_l2norm_in_kernel=True, null_block_id=-1,
    ):
        return f"upstream:{launch_mode}:{q.shape[0]}"

    res_commit = around_gdn_replayssm_spec_decode(
        _dummy_spec,
        inputs["q"], inputs["k"], inputs["v"], inputs["a"], inputs["b"],
        inputs["A_log"], inputs["dt_bias"], inputs["checkpoint_state"],
        inputs["d_cache"], inputs["k_cache"], inputs["g_cache"],
        launch_mode="commit",
    )
    assert res_commit == f"upstream:commit:{inputs['q'].shape[0]}"

    res_verify_pos = around_gdn_replayssm_spec_decode(
        _dummy_spec,
        inputs["q"], inputs["k"], inputs["v"], inputs["a"], inputs["b"],
        inputs["A_log"], inputs["dt_bias"], inputs["checkpoint_state"],
        inputs["d_cache"], inputs["k_cache"], inputs["g_cache"],
        None, None, None,
        inputs["query_start_loc"], inputs["ssm_state_indices"], inputs["replay_indices"],
        inputs["write_pos"], inputs["cache_base"], inputs["is_flush"],
    )
    assert res_verify_pos == f"upstream:verify:{inputs['q'].shape[0]}"

    # Verify register_hooks registers GDNAttnBackend and kernels.ops.attention.linear without breaking early.
    recorded: list[tuple[str, object, str]] = []

    class _FakeHookType:
        AROUND = "AROUND"

    class _FakeHookRegistry:
        @staticmethod
        def register(target: str, fn: object, hook_type: str) -> None:
            recorded.append((target, fn, hook_type))

    register_hooks(_FakeHookRegistry, _FakeHookType)
    targets = {t for t, _, _ in recorded}
    assert "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend._replayssm_target_verify" in targets
    assert "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode.gdn_replayssm_spec_decode" in targets

    # Verify SGLANG_SM89_WY_GDN_MAX_BF16_TOKENS (default 32) threshold routing logic
    big_inputs = _make_replay_inputs(
        B=8, spec_lens=[8] * 8, H=4, HV=8, L=64, ckpt_dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert not _can_use_wy_gdn_fast_path(
        **{k: v for k, v in big_inputs.items() if k != "scale"},
        out=None,
        launch_mode="verify",
        use_qk_l2norm_in_kernel=True,
        null_block_id=-1,
        force_cuda=False,
        has_fallback=True,
    )

    # Verify upstream_fn fallback never passes unsupported 'softplus_threshold' kwarg
    import sglang_sm89.wy_gdn_verify as _wgv

    def _strict_upstream(
        q, k, v, a, b, A_log, dt_bias, checkpoint_state,
        d_cache, k_cache, g_cache, rawv_cache=None, rawk_cache=None, beta_cache=None,
        query_start_loc=None, ssm_state_indices=None, replay_indices=None,
        write_pos=None, cache_base=None, is_flush=None, out=None,
        scale=None, max_cache_len=64, max_spec_len=16,
        launch_mode="verify", use_qk_l2norm_in_kernel=True, null_block_id=-1,
    ):
        return out if out is not None else torch.zeros_like(v)

    orig_resolver = _wgv._resolve_upstream_gdn_replayssm
    _wgv._resolve_upstream_gdn_replayssm = lambda: (None, _strict_upstream)
    try:
        out_strict = wy_gdn_replayssm_verify(**big_inputs, force_cuda=False)
        assert out_strict.shape == big_inputs["v"].shape
    finally:
        _wgv._resolve_upstream_gdn_replayssm = orig_resolver


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("ckpt_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "B,spec_lens,wp_vals,base_vals,flush_vals,null_last,split_view,launch_mode,with_beta_cache",
    [
        (1, [2], [0], [0], [0], False, False, "verify", False),
        (1, [6], [0], [0], [0], False, True, "verify", False),
        (2, [8, 8], [19, 0], [53, 0], [0, 0], False, False, "verify", False),
        (4, [12, 12, 12, 12], [12, 48, 5, 0], [58, 40, 60, 0], [0, 0, 0, 0], True, True, "verify", False),
        (8, [16] * 8, [0, 10, 22, 45, 7, 15, 31, 0], [0, 50, 48, 30, 60, 52, 40, 0], [0, 0, 1, 0, 1, 0, 0, 0], True, False, "both", False),
        (2, [8, 8], [6, 4], [61, 0], [1, 0], False, False, "both", True),
    ],
)
def test_wy_gdn_cuda_parity(
    ckpt_dtype: torch.dtype,
    B: int,
    spec_lens: list[int],
    wp_vals: list[int],
    base_vals: list[int],
    flush_vals: list[int],
    null_last: bool,
    split_view: bool,
    launch_mode: str,
    with_beta_cache: bool,
) -> None:
    """Exact numerical parity across B=1..8, spec_len=2..16, wrap-around, flush, null_block_id, BF16/FP32."""
    device = torch.device("cuda")
    inputs = _make_replay_inputs(
        B=B,
        spec_lens=spec_lens,
        H=16,
        HV=48,
        L=64,
        ckpt_dtype=ckpt_dtype,
        act_dtype=torch.bfloat16,
        ab_dtype=torch.float32,
        write_pos_vals=wp_vals,
        cache_base_vals=base_vals,
        flush_vals=flush_vals,
        null_last=null_last,
        with_beta_cache=with_beta_cache,
        with_residuals=True,
        split_qkv_view=split_view,
        device=device,
        seed=100 + B + sum(spec_lens),
    )

    out_ref, h0_ref, d_ref, k_ref, g_ref, wp_ref = _sequential_replayssm_reference(
        q=inputs["q"], k=inputs["k"], v=inputs["v"], a=inputs["a"], b=inputs["b"],
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        checkpoint_state=inputs["checkpoint_state"],
        d_cache=inputs["d_cache"], k_cache=inputs["k_cache"], g_cache=inputs["g_cache"],
        rawv_cache=inputs["rawv_cache"], rawk_cache=inputs["rawk_cache"],
        beta_cache=inputs["beta_cache"],
        query_start_loc=inputs["query_start_loc"],
        ssm_state_indices=inputs["ssm_state_indices"],
        replay_indices=inputs["replay_indices"],
        write_pos=inputs["write_pos"], cache_base=inputs["cache_base"],
        is_flush=inputs["is_flush"],
        scale=inputs["scale"], max_cache_len=inputs["max_cache_len"],
        launch_mode=launch_mode,
    )

    out_cu = torch.zeros_like(inputs["v"])
    wy_gdn_replayssm_verify(**inputs, out=out_cu, launch_mode=launch_mode, force_cuda=True)
    torch.cuda.synchronize()

    max_out_err = float((out_cu.float() - out_ref.float()).abs().max().item())
    tol = 0.015 if ckpt_dtype == torch.bfloat16 else 0.01
    assert max_out_err <= tol, f"out max_err={max_out_err} > {tol}"

    if null_last:
        bos_last = int(inputs["query_start_loc"][-2].item())
        assert float(out_cu[bos_last:].abs().max().item()) == 0.0

    assert torch.equal(inputs["write_pos"], wp_ref)
    max_h0_err = float((inputs["checkpoint_state"].float() - h0_ref.float()).abs().max().item())
    assert max_h0_err <= tol, f"checkpoint_state fold max_err={max_h0_err}"
    max_g_err = float((inputs["g_cache"] - g_ref).abs().max().item())
    assert max_g_err <= 1e-5, f"g_cache max_err={max_g_err}"
    max_k_err = float((inputs["k_cache"].float() - k_ref.float()).abs().max().item())
    assert max_k_err <= 0.015, f"k_cache max_err={max_k_err}"
    max_d_err = float((inputs["d_cache"].float() - d_ref.float()).abs().max().item())
    assert max_d_err <= 0.02, f"d_cache max_err={max_d_err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_wy_gdn_cudagraph_replay_and_latency() -> None:
    """Verify CUDA Graph capture/replay with dynamic input mutation + measure latency."""
    device = torch.device("cuda")
    inputs = _make_replay_inputs(
        B=8,
        spec_lens=[8] * 8,
        H=16,
        HV=48,
        L=64,
        ckpt_dtype=torch.bfloat16,
        write_pos_vals=[0, 8, 16, 24, 32, 40, 48, 0],
        cache_base_vals=[0, 58, 40, 30, 20, 10, 5, 0],
        flush_vals=[0, 0, 0, 0, 0, 0, 0, 0],
        device=device,
        seed=999,
    )
    out_cg = torch.empty_like(inputs["v"])

    # Snapshot initial mutable caches so direct vs graph replay start from identical state
    init_d = inputs["d_cache"].clone()
    init_k = inputs["k_cache"].clone()
    init_g = inputs["g_cache"].clone()
    init_rv = inputs["rawv_cache"].clone() if inputs["rawv_cache"] is not None else None
    init_rk = inputs["rawk_cache"].clone() if inputs["rawk_cache"] is not None else None

    # Warmup
    wy_gdn_replayssm_verify(**inputs, out=out_cg, force_cuda=True)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wy_gdn_replayssm_verify(**inputs, out=out_cg, force_cuda=True)

    # Restore clean initial caches, mutate q/k/v in place, and replay CUDA Graph
    inputs2 = _make_replay_inputs(
        B=8,
        spec_lens=[8] * 8,
        H=16,
        HV=48,
        L=64,
        ckpt_dtype=torch.bfloat16,
        write_pos_vals=[0, 8, 16, 24, 32, 40, 48, 0],
        cache_base_vals=[0, 58, 40, 30, 20, 10, 5, 0],
        flush_vals=[0, 0, 0, 0, 0, 0, 0, 0],
        device=device,
        seed=2025,
    )
    inputs["d_cache"].copy_(init_d)
    inputs["k_cache"].copy_(init_k)
    inputs["g_cache"].copy_(init_g)
    if init_rv is not None:
        inputs["rawv_cache"].copy_(init_rv)
    if init_rk is not None:
        inputs["rawk_cache"].copy_(init_rk)
    inputs["q"].copy_(inputs2["q"])
    inputs["k"].copy_(inputs2["k"])
    inputs["v"].copy_(inputs2["v"])
    graph.replay()
    torch.cuda.synchronize()

    # Build matching reference inputs (seed=999 caches/gates + seed=2025 q,k,v)
    inputs_direct = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }
    inputs_direct["d_cache"].copy_(init_d)
    inputs_direct["k_cache"].copy_(init_k)
    inputs_direct["g_cache"].copy_(init_g)
    if init_rv is not None:
        inputs_direct["rawv_cache"].copy_(init_rv)
    if init_rk is not None:
        inputs_direct["rawk_cache"].copy_(init_rk)
    out_direct = torch.empty_like(inputs["v"])
    wy_gdn_replayssm_verify(**inputs_direct, out=out_direct, force_cuda=True)
    torch.cuda.synchronize()
    assert float((out_cg.float() - out_direct.float()).abs().max().item()) <= 1e-3

    # Latency benchmark (CUDA events)
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    iters = 30
    times_ms = []
    for _ in range(iters):
        start_ev.record()
        graph.replay()
        end_ev.record()
        torch.cuda.synchronize()
        times_ms.append(start_ev.elapsed_time(end_ev))
    times_ms.sort()
    median_us = times_ms[iters // 2] * 1000.0
    print(f"[wy_gdn_sm89] B=8 spec_len=8 HV=48 CUDAGraph median={median_us:.1f} us")
    assert median_us > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_wy_gdn_bf16_crossover_upstream_fallback() -> None:
    """Verify B=8 SL=8 (64 tokens > 32) BF16 crossover cleanly delegates to upstream without TypeError."""
    from sglang_sm89.wy_gdn_verify import _resolve_upstream_gdn_replayssm

    _, upstream_fn = _resolve_upstream_gdn_replayssm()
    if upstream_fn is None:
        pytest.skip("Upstream SGLang gdn_replayssm_spec_decode not installed")

    device = torch.device("cuda")
    inputs = _make_replay_inputs(
        B=8,
        spec_lens=[8] * 8,
        H=16,
        HV=48,
        L=64,
        ckpt_dtype=torch.bfloat16,
        write_pos_vals=[0, 3, 6, 9, 12, 15, 18, 21],
        cache_base_vals=[0, 58, 50, 40, 30, 20, 10, 0],
        flush_vals=[0] * 8,
        device=device,
        seed=777,
    )
    out_fb = torch.empty_like(inputs["v"])
    wy_gdn_replayssm_verify(**inputs, out=out_fb, force_cuda=False)
    torch.cuda.synchronize()
    assert torch.isfinite(out_fb).all()
