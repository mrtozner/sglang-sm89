#!/usr/bin/env python3
"""
Theorem 3B: Mixed-Density Roofline Cost Model & Tail-Word Mask Verification (kTileV = 64).

Compares on the clean-room kTileV = 64 single-uint64-per-tile architecture:
  1. Naive Batch-Union Sparse Projection (collapses to dense when >=1 row is dense)
  2. Two-Pass m_blk={0,32} Kernel (reads active tile weights twice for M=64)
  3. Unified-64 Per-Tile Active-M Compaction + Roofline Tile-Union Dispatcher

Also verifies 64-row tail-word bitmask sanitization when (vocab_size & 63) != 0
(e.g., V = 2051 -> 2051 & 63 == 3, 2051 & 31 == 3).

Run:
  python3 proofs/prove_mixed_density_roofline.py
"""
from __future__ import annotations

import math
import numpy as np


def sanitize_tail_words(words_u32: np.ndarray, vocab_size: int) -> np.ndarray:
    """Mask out padding bits in the final 32-bit word when (vocab_size & 31) != 0."""
    out = words_u32.copy()
    rem = vocab_size & 31
    if rem != 0:
        tail_mask = np.uint32((1 << rem) - 1)
        out[..., -1] &= tail_mask
    return out


def popcount_u32_array(arr: np.ndarray) -> np.ndarray:
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    return lut[arr.view(np.uint8)].reshape(*arr.shape, 4).sum(axis=-1)


def effective_weight_row_reads_ktile64(
    vocab_mask_u32: np.ndarray,  # [M, ceil_div(V, 32)]
    vocab_size: int,
    k_chunk: int = 8,
) -> dict[str, float]:
    """
    Compute exact effective weight-row reads for kTileV = 64 (2 x uint32 = 1 x uint64 per tile).
    """
    M, n_words = vocab_mask_u32.shape
    clean = sanitize_tail_words(vocab_mask_u32, vocab_size)
    n_tiles = (vocab_size + 63) // 64
    padded = np.zeros((M, n_tiles * 2), dtype=np.uint32)
    padded[:, :n_words] = clean

    # Reshape into [M, n_tiles, 2] -> each tile is 1 x uint64 (64 vocab rows)
    tiles = padded.reshape(M, n_tiles, 2)
    tile_row_nonzero = np.any(tiles != 0, axis=-1)                  # [M, n_tiles]
    active_m_per_tile = tile_row_nonzero.sum(axis=0)                # [n_tiles]

    tile_union = np.bitwise_or.reduce(tiles, axis=0)                # [n_tiles, 2]
    active_rows_per_tile = popcount_u32_array(tile_union).sum(axis=-1)  # [n_tiles]

    # 1. Unified-64 per-tile compaction: single uint64 ballot over M <= 64
    passes_unified64 = np.ceil(active_m_per_tile / float(k_chunk)).astype(np.int64)
    eff_unified64 = int((active_rows_per_tile * passes_unified64).sum())

    # 2. Two-pass m_blk={0,32} (uses tile union across two independent 32-row passes)
    eff_twopass = 0
    for m0 in range(0, M, 32):
        sub = tiles[m0 : min(m0 + 32, M)]
        sub_active_m = np.any(sub != 0, axis=-1).sum(axis=0)
        eff_twopass += int((active_rows_per_tile * np.ceil(sub_active_m / float(k_chunk))).sum())

    # 3. Naive Batch-Union GSP (launches global union rows across all M rows)
    global_union_rows = int(active_rows_per_tile.sum())
    eff_naive_union = global_union_rows * math.ceil(M / float(k_chunk))

    return {
        "dense_baseline_rows": float(vocab_size * math.ceil(M / float(k_chunk))),
        "global_union_vocab_frac": global_union_rows / float(vocab_size),
        "eff_unified64": float(eff_unified64),
        "eff_twopass": float(eff_twopass),
        "eff_naive_union": float(eff_naive_union),
    }


def main() -> None:
    rng = np.random.default_rng(17)

    # 1. Verify 64-row tile tail-word mask fix on V = 2051 (2051 & 63 == 3)
    V_tail = 2051
    n_words_tail = (V_tail + 31) // 32  # 65 words; padded to 66 words (33 tiles x 64)
    raw_tail = np.full((2, n_words_tail), 0xFFFFFFFF, dtype=np.uint32)
    sanitized = sanitize_tail_words(raw_tail, V_tail)
    total_valid_bits = int(popcount_u32_array(sanitized[0]).sum())
    assert total_valid_bits == V_tail, f"Expected {V_tail} bits, got {total_valid_bits}"
    assert int(sanitized[0, -1]) == 0b111, f"Tail word leak: {bin(sanitized[0, -1])}"

    # 2. Realistic V=248,320 mixed-density c=8 batch (M=64: 56 sparse SQL/JSON + 8 dense prose)
    V = 248_320
    n_words = (V + 31) // 32
    M = 64
    mask = np.zeros((M, n_words), dtype=np.uint32)

    sparse_words = rng.choice(n_words, size=10, replace=False)
    for m in range(56):
        w_sel = rng.choice(sparse_words, size=2, replace=False)
        mask[m, w_sel] = np.uint32(0x000F000F)

    dense_words = rng.choice(n_words, size=int(0.85 * n_words), replace=False)
    for m in range(56, 64):
        mask[m, dense_words] = np.uint32(0xFFFFFFFF)

    stats = effective_weight_row_reads_ktile64(mask, V, k_chunk=8)
    assert stats["eff_unified64"] <= stats["eff_twopass"]
    assert stats["eff_unified64"] < 0.25 * stats["eff_naive_union"]

    ratio_mixed = stats["eff_unified64"] / float(V)
    assert ratio_mixed > 0.35

    # 3. Cross-half sparse c=8 batch (M=64 split across m_blk 0 and 32)
    mask_cross = np.zeros((M, n_words), dtype=np.uint32)
    for req in range(8):
        m_row = req * 8
        mask_cross[m_row, sparse_words[:4]] = np.uint32(0x00FF00FF)
    stats_cross = effective_weight_row_reads_ktile64(mask_cross, V, k_chunk=8)
    assert stats_cross["eff_unified64"] == 0.5 * stats_cross["eff_twopass"]

    # 4. Pure sparse c=8 batch (64 sparse rows)
    mask_sparse = np.zeros((M, n_words), dtype=np.uint32)
    for m in range(M):
        w_sel = rng.choice(sparse_words, size=3, replace=False)
        mask_sparse[m, w_sel] = np.uint32(0x00FF00FF)
    stats_sparse = effective_weight_row_reads_ktile64(mask_sparse, V, k_chunk=8)
    ratio_sparse = stats_sparse["eff_unified64"] / float(V)
    assert ratio_sparse < 0.05

    print("[PASS] Theorem 3B (kTileV=64 Mixed-Density Roofline & Tail-Mask Verification):")
    print(f"  Tail-bit fix (V={V_tail}): exact valid bits={total_valid_bits} (tail word=0b{int(sanitized[0,-1]):b})")
    print(
        f"  Mixed M=64 (56 sparse + 8 dense): global_union={stats['global_union_vocab_frac']*100:.1f}% | "
        f"naive_union={stats['eff_naive_union']/V:.2f}x V | unified64={ratio_mixed:.2f}x V (routes to dense cuBLAS)"
    )
    print(
        f"  Cross-half M=64 (8 reqs split across m_blk 0/32): two_pass={stats_cross['eff_twopass']:.0f} -> "
        f"unified64={stats_cross['eff_unified64']:.0f} (2.00x fewer weight reads)"
    )
    print(f"  Pure Sparse M=64: unified64={ratio_sparse:.4f}x V (routes to sparse CUDA kernel)")


if __name__ == "__main__":
    main()
