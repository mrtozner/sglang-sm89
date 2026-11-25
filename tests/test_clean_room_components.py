"""Unit tests for clean-room v0.2.0 autotuner, draft_fp8, gdn_pool_bridge, and profiles."""

from __future__ import annotations

import json
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_sm89.autotune_fp8 import (
    MAX_ADA_SMEM_BYTES,
    config_filename,
    derive_fp8_shapes_from_config,
    ensure_sm89_fp8_configs_installed,
    enumerate_sm89_candidates,
    synthesize_analytical_sm89_schedule,
)
from sglang_sm89.draft_fp8 import (
    QuantizedLinear8,
    attach_draft_head_fp8,
    quantize_draft_backbone,
    quantize_weight_rowwise_fp8,
)
from sglang_sm89.gdn_pool_bridge import (
    around_build_hybrid_req_pool,
    around_update_mamba_state_after_mtp_verify,
)


def test_zero_static_json_configs_in_repo():
    pkg_dir = pathlib.Path(__file__).resolve().parents[1] / "sglang_sm89"
    assert not list(pkg_dir.rglob("*.json")), "Static FP8 JSON templates must not be packaged"


def test_derive_shapes_and_analytical_sram_limit():
    cfg = {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    shapes = derive_fp8_shapes_from_config(cfg)
    assert (34816, 5120) in shapes
    assert (5120, 17408) in shapes
    assert all(n % 128 == 0 and k % 128 == 0 for n, k in shapes)

    for m_val in (1, 8, 64, 512, 4096):
        cands = enumerate_sm89_candidates(m_val)
        assert cands
        assert all(c.estimated_smem_bytes() <= MAX_ADA_SMEM_BYTES for c in cands)

    sched = synthesize_analytical_sm89_schedule(14336, 5120, sm_count=114)
    assert set(sched.keys()) == {
        "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096"
    }


def test_ensure_configs_links_user_cache_and_synthesizes_opt_in(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    target_dir = tmp_path / "sglang_configs"
    cache_dir.mkdir()
    target_dir.mkdir()

    dev = "NVIDIA_GeForce_RTX_4090_D"
    fname = config_filename(14336, 5120, dev)
    (cache_dir / fname).write_text(json.dumps({"1": {"BLOCK_SIZE_M": 16}}), encoding="utf-8")

    linked = ensure_sm89_fp8_configs_installed(
        cache_dir=cache_dir, target_dir=target_dir, device_name=dev
    )
    assert linked == 1
    assert (target_dir / fname).exists()

    # Empty cache with analytical synthesis opt-in
    cache2 = tmp_path / "cache2"
    target2 = tmp_path / "sglang_configs2"
    target2.mkdir()
    monkeypatch.setenv("SGLANG_SM89_SYNTHESIZE_UNTUNED_FP8", "1")
    monkeypatch.setenv("SGLANG_SM89_ANALYTICAL_SHAPES", "14336x5120")
    synthesized = ensure_sm89_fp8_configs_installed(
        cache_dir=cache2, target_dir=target2, device_name=dev
    )
    assert synthesized == 1
    assert (target2 / fname).is_symlink() or (target2 / fname).is_file()


def test_draft_fp8_quantize_roundtrip():
    torch.manual_seed(42)
    w = torch.randn((32, 64), dtype=torch.bfloat16)
    w_t, sc = quantize_weight_rowwise_fp8(w)
    assert w_t.shape == (64, 32)
    assert w_t.dtype == torch.float8_e4m3fn
    assert sc.shape == (32, 1)
    recon = w_t.T.float() * sc
    cos = torch.nn.functional.cosine_similarity(w.float().flatten(), recon.flatten(), dim=0)
    assert float(cos) > 0.998

    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = SimpleNamespace()
            self.self_attn.qkv_proj = nn.Linear(64, 96, bias=False, dtype=torch.bfloat16)
            self.self_attn.o_proj = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
            self.mlp = SimpleNamespace()
            self.mlp.gate_up_proj = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
            self.mlp.down_proj = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)

    class DummyDraft(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
            self.layers = nn.ModuleList([DummyBlock()])

    dm = DummyDraft()
    n_rep, saved = quantize_draft_backbone(dm)
    assert n_rep == 5 and saved > 0
    assert isinstance(dm.fc, QuantizedLinear8)
    assert dm.fc.in_features == 64 and dm.fc.input_size == 64
    assert dm.fc.out_features == 64 and dm.fc.output_size == 64
    assert not hasattr(dm.fc, "weight")
    assert dm.layers[0].self_attn.qkv_proj.out_features == 96
    assert dm.layers[0].self_attn.qkv_proj.output_size == 96
    assert not hasattr(dm.layers[0].self_attn.qkv_proj, "weight")
    head_mod = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
    assert attach_draft_head_fp8(dm, head_mod) == 128


def test_gdn_pool_bridge_sentinel_and_circular_order(monkeypatch):
    # Mock SGLang modules for around_build_hybrid_req_pool
    kv_mod = types.ModuleType("sglang.srt.mem_cache.kv_cache_configurator")
    kv_mod.get_exec = lambda: SimpleNamespace(
        mamba=SimpleNamespace(enable_linear_replayssm_spec=True)
    )
    kv_mod.get_spec = lambda: SimpleNamespace(
        speculative_algorithm="DFLASH", speculative_num_draft_tokens=8
    )
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache", types.ModuleType("sglang.srt.mem_cache"))
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.kv_cache_configurator", kv_mod)

    cfg = SimpleNamespace(hybrid_kda_config=None, hybrid_gdn_config=object())
    observed = {}

    def fake_orig(c):
        observed["sentinel_was_set"] = c.hybrid_kda_config is not None
        return "pool_ok"

    assert around_build_hybrid_req_pool(fake_orig, cfg) == "pool_ok"
    assert observed["sentinel_was_set"] is True
    assert cfg.hybrid_kda_config is None

    # Mock ReplaySSM circular update order (commit_spec -> commit_circular -> conv_scatter)
    call_order = []
    spec_utils_mod = types.ModuleType("sglang.srt.speculative.spec_utils")
    spec_utils_mod.pp_spec_stable_rows_enabled = lambda: False

    decode_mod = types.ModuleType("sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode")
    decode_mod.commit_gdn_replayssm_spec = lambda *a, **kw: call_order.append(("spec", kw))
    decode_mod.commit_gdn_replayssm_circular = lambda **kw: call_order.append(("circular", kw))

    scatter_mod = types.ModuleType("sglang.kernels.ops.mamba.mamba_state_scatter_triton")
    scatter_mod.fused_conv_window_scatter_with_mask = lambda *a, **kw: call_order.append(("conv", len(a)))

    monkeypatch.setitem(sys.modules, "sglang.srt.speculative", types.ModuleType("sglang.srt.speculative"))
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative.spec_utils", spec_utils_mod)
    monkeypatch.setitem(
        sys.modules, "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode", decode_mod
    )
    monkeypatch.setitem(
        sys.modules, "sglang.kernels.ops.mamba.mamba_state_scatter_triton", scatter_mod
    )

    spec_state = SimpleNamespace(
        temporal=torch.zeros((1, 4, 2, 4, 4), dtype=torch.bfloat16),
        replayssm_d=torch.zeros((1, 4, 2, 16, 4), dtype=torch.bfloat16),
        replayssm_k=torch.zeros((1, 4, 1, 16, 4), dtype=torch.bfloat16),
        replayssm_g=torch.zeros((1, 4, 2, 16), dtype=torch.float32),
        replayssm_rawv=None,
        replayssm_rawk=None,
        conv=[torch.zeros((4, 4, 4))],
        intermediate_conv_window=[torch.zeros((4, 8, 4, 4))],
    )
    mamba_pool = SimpleNamespace(
        replayssm_is_kda=False,
        replayssm_spec_fold=False,
        replayssm_cache_base=torch.zeros((4,), dtype=torch.int32),
        replayssm_spec_write_pos=torch.zeros((4,), dtype=torch.int32),
        replayssm_is_flush=torch.zeros((4,), dtype=torch.int8),
        linear_replayssm_cache_len=16,
    )
    req_pool = SimpleNamespace(
        mamba_pool=mamba_pool,
        speculative_num_draft_tokens=8,
        get_speculative_mamba2_params_all_layers=lambda: spec_state,
    )
    linear_backend = SimpleNamespace(
        req_to_token_pool=req_pool,
        forward_metadata=SimpleNamespace(mamba_cache_indices=torch.arange(4, dtype=torch.int64)),
        _translate_mamba_indices=lambda x: x,
    )
    backend = SimpleNamespace(linear_attn_backend=linear_backend)

    around_update_mamba_state_after_mtp_verify(
        lambda *_a, **_kw: pytest.fail("Should not delegate to original_fn"),
        backend,
        last_correct_step_indices=torch.tensor([2, 3], dtype=torch.int64),
        mamba_track_indices=torch.tensor([0, 1], dtype=torch.int64),
        mamba_steps_to_track=torch.tensor([1, -1], dtype=torch.int64),
        model=None,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
    )

    assert [c[0] for c in call_order] == ["spec", "circular", "conv", "conv"]
    assert call_order[0][1]["fold_every_commit"] is True
    assert call_order[0][1]["null_block_id"] == -1
    assert call_order[1][1]["null_block_id"] == -1


def test_profiles_use_loopback_ports_only():
    prof_dir = pathlib.Path(__file__).resolve().parents[1] / "profiles"
    files = sorted(prof_dir.glob("*.yml"))
    assert len(files) == 3
    for pf in files:
        text = pf.read_text(encoding="utf-8")
        assert "127.0.0.1:30000:30000" in text
        assert "'30000:30000'" not in text and '"30000:30000"' not in text
