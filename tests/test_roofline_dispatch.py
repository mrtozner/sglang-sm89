"""CPU/C++ tests for the kTileV=64 uint64 sparse vocab projection roofline selector."""

from __future__ import annotations

import math

import pytest
import torch

from sglang_sm89.vocab_projection import (
    ensure_extension_loaded,
    select_sparse_vocab_proj,
)


def test_tail_bitmask_2051_ignores_padding_bits() -> None:
    vocab_size = 2051
    words = (vocab_size + 31) // 32  # 65
    mask = torch.zeros((64, words), dtype=torch.int32)
    mask[:, 0] = 0b111
    # Poison unused padding bits [3..31] in word 64 across all rows.
    mask[:, -1] = -1 << (vocab_size & 31)
    snapshot = mask.clone()

    assert select_sparse_vocab_proj(mask, vocab_size, k_chunk=8, max_eff_row_ratio=0.55, force_numpy=True)
    # Must never mutate caller's CPU bitmask tensor.
    assert torch.equal(mask, snapshot)

    if ensure_extension_loaded():
        assert select_sparse_vocab_proj(mask, vocab_size, k_chunk=8, max_eff_row_ratio=0.55, force_numpy=False)
        assert torch.equal(mask, snapshot)


def test_mixed_64_tile_cost_model() -> None:
    vocab_size = 4096
    words = vocab_size // 32
    mask = torch.zeros((64, words), dtype=torch.int32)
    # 56 sparse tokens -> 10 bits active in tile 0; 8 dense tokens -> 100% active across all tiles.
    mask[:56, 0] = (1 << 10) - 1
    mask[56:, :] = -1

    # Compare against explicit per-64-row tile calculation.
    u64 = mask.numpy().view("uint64")
    eff = 0
    for tile_idx in range(vocab_size // 64):
        col = u64[:, tile_idx]
        active_m = int((col != 0).sum())
        if active_m > 0:
            pop_u = int(__import__("numpy").bitwise_or.reduce(col)).bit_count()
            eff += pop_u * math.ceil(active_m / 8)

    expected_tight = eff <= int(vocab_size * 0.55)
    got_np = select_sparse_vocab_proj(mask, vocab_size, k_chunk=8, max_eff_row_ratio=0.55, force_numpy=True)
    assert got_np is expected_tight

    # Lower budget triggers dense GEMM fallback.
    assert not select_sparse_vocab_proj(mask, vocab_size, k_chunk=8, max_eff_row_ratio=0.50, force_numpy=True)

    if ensure_extension_loaded():
        assert (
            select_sparse_vocab_proj(mask, vocab_size, k_chunk=8, max_eff_row_ratio=0.55, force_numpy=False)
            == got_np
        )
        assert not select_sparse_vocab_proj(
            mask, vocab_size, k_chunk=8, max_eff_row_ratio=0.50, force_numpy=False
        )


def test_boundary_full_vocab_248320() -> None:
    vocab_size = 248320
    words = (vocab_size + 31) // 32

    for m in (1, 8, 64):
        sparse_mask = torch.zeros((m, words), dtype=torch.int32)
        for r in range(m):
            tok = (r * 131) % vocab_size
            sparse_mask[r, tok >> 5] |= 1 << (tok & 31)
        assert select_sparse_vocab_proj(sparse_mask, vocab_size, force_numpy=True)
        if ensure_extension_loaded():
            assert select_sparse_vocab_proj(sparse_mask, vocab_size, force_numpy=False)

    dense_64 = torch.full((64, words), -1, dtype=torch.int32)
    assert not select_sparse_vocab_proj(dense_64, vocab_size, force_numpy=True)
    if ensure_extension_loaded():
        assert not select_sparse_vocab_proj(dense_64, vocab_size, force_numpy=False)

    with pytest.raises(ValueError):
        select_sparse_vocab_proj(torch.zeros((1, 4), dtype=torch.int32), 256, force_numpy=True)
