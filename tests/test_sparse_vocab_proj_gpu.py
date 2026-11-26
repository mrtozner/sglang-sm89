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


def run_all() -> None:
    assert torch.cuda.is_available()
    test_sparse_vocab_proj_tail_2051_dirty_bf16_fp8()
    test_sparse_vocab_proj_large_vocab_bf16_fp8()
    test_sparse_vocab_proj_cudagraph_m64()
    print("sparse_vocab_proj_sm89 GPU parity & CUDA Graph PASS")


if __name__ == "__main__":
    run_all()
