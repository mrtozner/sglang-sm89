"""CUDA parity, tail-word N=2051, M>64, and CUDA Graph tests for sparse_vocab_proj_sm89."""

from __future__ import annotations

import os

import pytest
import torch

from sglang_sm89.vocab_projection import (
    select_sparse_vocab_proj,
    sparse_vocab_bf16_lm_head,
    sparse_vocab_fp8_lm_head,
)


def _make_sparse_or_mixed_mask(
    m: int, n: int, density: float, *, device: torch.device, poison_tail: bool = False
) -> torch.Tensor:
    words = (n + 31) // 32
    k_active = min(n, max(1, int(round(n * density))))
    gen = torch.Generator(device="cpu").manual_seed(9100 + m)
    if k_active == n:
        mask = torch.full((m, words), -1, dtype=torch.int32)
    else:
        mask = torch.zeros((m, words), dtype=torch.int32)
        for row_idx in range(m):
            cols = torch.randperm(n, generator=gen)[:k_active]
            w_idx = cols >> 5
            bits = (1 << (cols & 31)).to(torch.int32)
            for c_pos in range(k_active):
                mask[row_idx, int(w_idx[c_pos])] |= int(bits[c_pos])
    if poison_tail and (n & 31) != 0:
        mask[:, -1] |= -1 << (n & 31)
    return mask.to(device)


def _dense_masked_reference(hidden: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(hidden, weight.T)
    n = logits.shape[1]
    cols = torch.arange(n, device=logits.device, dtype=torch.int32)
    words = mask[:, cols >> 5]
    allowed = ((words >> (cols & 31).unsqueeze(0)) & 1).bool()
    return logits.masked_fill(~allowed, float("-inf"))


def _verify_vocab_suite(n: int, m_sizes: tuple[int, ...], *, poison_tail: bool = False) -> None:
    device = torch.device("cuda")
    k_dim = 5120
    w_bf16 = torch.randn((n, k_dim), dtype=torch.bfloat16, device=device) * 0.02
    w_f32 = w_bf16.float()
    w_scale = (w_f32.abs().amax(dim=1) / 448.0).clamp_min(1e-12).contiguous()
    w_fp8 = (w_f32 / w_scale.unsqueeze(1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    del w_f32

    for m in m_sizes:
        hidden = torch.randn((m, k_dim), dtype=torch.bfloat16, device=device) * 0.1
        for density in (0.005, 0.05):
            mask = _make_sparse_or_mixed_mask(m, n, density, device=device, poison_tail=poison_tail)
            ref = _dense_masked_reference(hidden, w_bf16, mask)
            active = torch.isfinite(ref)

            out_b = sparse_vocab_bf16_lm_head(hidden, w_bf16, mask)
            assert torch.equal(torch.isfinite(out_b), active), f"BF16 active mismatch M={m} N={n}"
            assert (out_b[active] - ref[active]).abs().max().item() <= 0.05

            out_f = sparse_vocab_fp8_lm_head(hidden, w_fp8, w_scale, mask)
            assert torch.equal(torch.isfinite(out_f), active), f"FP8 active mismatch M={m} N={n}"
            assert (out_f[active] - ref[active]).abs().max().item() <= 0.15

    if 64 in m_sizes:
        words = (n + 31) // 32
        mix_cpu = torch.zeros((64, words), dtype=torch.int32)
        for m in range(56):
            tok = (m * 37) % n
            mix_cpu[m, tok >> 5] |= 1 << (tok & 31)
        for m in range(56, 64):
            for tok in torch.randperm(n)[: max(1, n // 3)].tolist():
                mix_cpu[m, tok >> 5] |= 1 << (tok & 31)
        mix_gpu = mix_cpu.to(device)
        h64 = torch.randn((64, k_dim), dtype=torch.bfloat16, device=device) * 0.1
        ref64 = _dense_masked_reference(h64, w_bf16, mix_gpu)
        out64 = sparse_vocab_bf16_lm_head(h64, w_bf16, mix_gpu)
        assert torch.equal(torch.isfinite(out64), torch.isfinite(ref64))
        assert (out64[torch.isfinite(ref64)] - ref64[torch.isfinite(ref64)]).abs().max().item() <= 0.05

        assert select_sparse_vocab_proj(
            mix_cpu, n, k_chunk=8, max_eff_row_ratio=0.55, force_numpy=False
        ) == select_sparse_vocab_proj(
            mix_cpu, n, k_chunk=8, max_eff_row_ratio=0.55, force_numpy=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparse_vocab_proj_tail_2051_dirty_bf16_fp8() -> None:
    torch.manual_seed(2026)
    _verify_vocab_suite(2051, (1, 8, 64, 256, 257, 289), poison_tail=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparse_vocab_proj_large_vocab_bf16_fp8() -> None:
    torch.manual_seed(2026)
    full = os.getenv("RUN_FULL_VOCAB", "0") == "1"
    n = 248320 if full else 16384
    _verify_vocab_suite(n, (1, 8, 48, 64, 256, 257, 289), poison_tail=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparse_vocab_proj_cudagraph_m64() -> None:
    torch.manual_seed(2026)
    full = os.getenv("RUN_FULL_VOCAB", "0") == "1"
    n = 248320 if full else 16384
    k_dim = 5120
    device = torch.device("cuda")

    w_bf16 = torch.randn((n, k_dim), dtype=torch.bfloat16, device=device) * 0.02
    h_static = torch.randn((64, k_dim), dtype=torch.bfloat16, device=device) * 0.1
    mask_static = _make_sparse_or_mixed_mask(64, n, 0.01, device=device)
    out_static = torch.empty((64, n), dtype=torch.bfloat16, device=device)

    sparse_vocab_bf16_lm_head(h_static, w_bf16, mask_static, out=out_static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_vocab_bf16_lm_head(h_static, w_bf16, mask_static, out=out_static)

    mask_next = _make_sparse_or_mixed_mask(64, n, 0.005, device=device)
    mask_static.copy_(mask_next)
    graph.replay()
    ref_next = _dense_masked_reference(h_static, w_bf16, mask_next)
    assert torch.equal(torch.isfinite(out_static), torch.isfinite(ref_next))
    cg_diff = (out_static[torch.isfinite(ref_next)] - ref_next[torch.isfinite(ref_next)]).abs().max().item()
    assert cg_diff <= 0.05, f"CUDAGraph M=64 max_diff={cg_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_cudagraph_bs16_eager24_replay16_arena_isolation() -> None:
    """Real CUDA Graph bs=16 (128 rows) -> Eager 24 (192 rows) -> Graph replay keeps captured ptr."""
    from types import SimpleNamespace
    from sglang_sm89.coupled_grammar import _STEP_TLS
    from sglang_sm89.vocab_projection import (
        _VERIFY_TLS,
        around_capture_decode_graph,
        around_target_get_logits,
        around_tp_worker_forward_batch_generation,
        clear_target_verify_tls,
        materialize_deferred_target_logits,
    )

    class _Mode:
        def is_target_verify(self) -> bool:
            return True

    class LogitsProcessor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_fp32_lm_head = False
            self.use_fp32_lm_head_output = False
            self.final_logit_softcapping = None
            self.logit_scale = None
            self.do_tensor_parallel_all_gather = False
            self.tp_size = 1

    device = torch.device("cuda")
    n_vocab, hidden_dim = 256, 64
    weight = torch.zeros((n_vocab, hidden_dim), dtype=torch.bfloat16, device=device)
    weight[0, 0] = 1.0
    weight[1, 1] = 1.0
    lm_head = SimpleNamespace(weight=weight)
    lp = LogitsProcessor()
    model = torch.nn.Module()
    model.lp = lp
    model.lm_head = lm_head
    runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_dflash=lambda: True, name="DFLASH"),
        model=model,
        model_config=SimpleNamespace(vocab_size=n_vocab, hidden_size=hidden_dim),
        server_args=SimpleNamespace(cuda_graph_max_bs_decode=16, speculative_num_draft_tokens=8),
        decode_num_tokens_per_req=lambda: 8,
    )
    tp_w = SimpleNamespace(model_runner=runner)
    dflash_w = SimpleNamespace(_target_worker=tp_w, _target_logits_processor=lp)
    meta = SimpleNamespace(
        forward_mode=_Mode(), extend_return_logprob=False, next_token_logits_buffer=None
    )

    def _orig_get_logits(_self, h, lm_h, _meta, embedding_bias=None, use_logits_buffer=True):
        return torch.matmul(h, lm_h.weight.T)

    h_static = torch.zeros((128, hidden_dim), dtype=torch.bfloat16, device=device)
    graph = torch.cuda.CUDAGraph()
    captured_slot: list[torch.Tensor] = []

    def _capture_real_graph(_mr) -> None:
        around_target_get_logits(_orig_get_logits, lp, h_static, lm_head, meta, use_logits_buffer=False)
        with torch.cuda.graph(graph):
            slot = around_target_get_logits(
                _orig_get_logits, lp, h_static, lm_head, meta, use_logits_buffer=False
            )
            captured_slot.append(slot)

    clear_target_verify_tls()
    around_capture_decode_graph(_capture_real_graph, runner)
    assert lp._sm89_graph_verify_captured is True
    captured_ptr = lp._sm89_graph_hidden_arena.data_ptr()

    def _make_h(rows: int, a: float, b: float) -> torch.Tensor:
        h = torch.zeros((rows, hidden_dim), dtype=torch.bfloat16, device=device)
        h[:, 0] = a
        h[:, 1] = b
        return h

    mask_128 = torch.zeros((128, n_vocab // 32), dtype=torch.int32, device=device)
    mask_128[:, 0] = 0b11

    try:
        # 1. Eager bs=24 (192 rows) producing [9.0, 11.0] without touching _sm89_graph_hidden_arena.
        h_eager = _make_h(192, 9.0, 11.0)
        _STEP_TLS.stack.append((dflash_w, SimpleNamespace(), None))
        eager_out_holder: list[torch.Tensor] = []

        def _eager_24(_tp, **_kw):
            slot = around_target_get_logits(
                _orig_get_logits, lp, h_eager, lm_head, meta, use_logits_buffer=False
            )
            eager_out_holder.append(slot)
            return SimpleNamespace(
                can_run_cuda_graph=False,
                logits_output=SimpleNamespace(next_token_logits=slot),
            )

        around_tp_worker_forward_batch_generation(_eager_24, tp_w, is_verify=True)
        materialize_deferred_target_logits()
        _STEP_TLS.stack.pop()
        clear_target_verify_tls()
        assert torch.equal(
            eager_out_holder[0][:, :2],
            torch.tensor([[9.0, 11.0]], dtype=torch.bfloat16, device=device).expand(192, 2),
        )
        assert lp._sm89_graph_hidden_arena.data_ptr() == captured_ptr

        # 2. Real CUDA Graph replay (128 rows) producing [2.0, 3.0] via both dense & sparse CUDA kernel.
        h_static.copy_(_make_h(128, 2.0, 3.0))
        for use_sparse in (False, True):
            captured_slot[0].fill_(123.0)
            _STEP_TLS.stack.append((dflash_w, SimpleNamespace(), None))

            def _replay_16(_tp, **_kw):
                graph.replay()
                return SimpleNamespace(
                    can_run_cuda_graph=True,
                    logits_output=SimpleNamespace(next_token_logits=captured_slot[0]),
                )

            around_tp_worker_forward_batch_generation(_replay_16, tp_w, is_verify=True)
            if use_sparse:
                assert _VERIFY_TLS.pending is not None
                _VERIFY_TLS.pending["vocab_mask_gpu"] = mask_128
                _VERIFY_TLS.pending["use_sparse"] = True
            materialize_deferred_target_logits()
            _STEP_TLS.stack.pop()
            clear_target_verify_tls()

            assert torch.equal(
                captured_slot[0][:, :2],
                torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16, device=device).expand(128, 2),
            )
            if use_sparse:
                assert torch.all(torch.isneginf(captured_slot[0][:, 2:]))
        assert lp._sm89_graph_hidden_arena.data_ptr() == captured_ptr
    finally:
        while _STEP_TLS.stack:
            _STEP_TLS.stack.pop()
        clear_target_verify_tls()


def run_all() -> None:
    assert torch.cuda.is_available()
    test_sparse_vocab_proj_tail_2051_dirty_bf16_fp8()
    test_sparse_vocab_proj_large_vocab_bf16_fp8()
    test_sparse_vocab_proj_cudagraph_m64()
    print("sparse_vocab_proj_sm89 GPU parity & CUDA Graph PASS")


if __name__ == "__main__":
    run_all()
