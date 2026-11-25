#!/usr/bin/env python3
"""C++ vs NumPy vs naive mask dispatch microbenchmark on tool-call and mixed-density masks."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
from sglang_sm89.vocab_projection import (  # noqa: E402
    ensure_extension_loaded,
    select_sparse_vocab_proj,
)

DEFAULT_VOCAB_SIZE = 248_320
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def union_active_row_fraction(vocab_mask: torch.Tensor, vocab_size: int) -> tuple[float, int, int]:
    """Compute (active_tile_fraction, active_pairs, active_tiles) across 64-row uint64 tiles."""
    arr = vocab_mask.detach().contiguous().cpu().numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    m_rows = int(arr.shape[0])
    n_words = (vocab_size + 31) // 32
    words_u32 = arr[:, :n_words].view(np.uint32).copy()
    rem_bits = vocab_size & 31
    if rem_bits != 0:
        words_u32[:, n_words - 1] &= np.uint32((1 << rem_bits) - 1)
    n_tiles = (vocab_size + 63) // 64
    even_words = n_tiles * 2
    if n_words < even_words:
        padded = np.zeros((m_rows, even_words), dtype=np.uint32)
        padded[:, :n_words] = words_u32
    else:
        padded = np.ascontiguousarray(words_u32[:, :even_words])
    tiles_u64 = padded.view(np.uint64)
    tile_union = np.bitwise_or.reduce(tiles_u64, axis=0)
    active_tiles = int(np.count_nonzero(tile_union))
    active_pairs = int(np.count_nonzero(tiles_u64))
    return (active_tiles / max(1, n_tiles)), active_pairs, active_tiles


def naive_mask_dispatch(
    vocab_mask_cpu: torch.Tensor,
    vocab_size: int,
    k_chunk: int = 8,
    max_eff_row_ratio: float = 0.55,
) -> bool:
    """Naive Python/NumPy per-row full-bitmask scan baseline (kTileV=64) without early exit."""
    arr = vocab_mask_cpu.detach().contiguous().numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    m_rows = int(arr.shape[0])
    n_words = (vocab_size + 31) // 32
    words_u32 = arr[:, :n_words].view(np.uint32).copy()
    rem_bits = vocab_size & 31
    if rem_bits != 0:
        words_u32[:, n_words - 1] &= np.uint32((1 << rem_bits) - 1)
    n_tiles = (vocab_size + 63) // 64
    padded = np.zeros((m_rows, n_tiles * 2), dtype=np.uint32)
    padded[:, :n_words] = words_u32
    tiles = padded.reshape(m_rows, n_tiles, 2)

    eff_weight_rows = 0
    for c in range(n_tiles):
        active_m = 0
        union = np.zeros(2, dtype=np.uint32)
        for m in range(m_rows):
            row_tile = tiles[m, c]
            if int(row_tile[0] | row_tile[1]) != 0:
                active_m += 1
                union |= row_tile
        if active_m > 0:
            pop_u = int(_POPCOUNT_LUT[union.view(np.uint8)].sum())
            eff_weight_rows += pop_u * ((active_m + k_chunk - 1) // k_chunk)
    return float(eff_weight_rows) <= float(vocab_size) * float(max_eff_row_ratio)


def _fill_random_sparse(rng: np.random.Generator, rows: int, vocab_size: int, density: float) -> np.ndarray:
    n_words = (vocab_size + 31) // 32
    mat = np.zeros((rows, n_words), dtype=np.uint32)
    k_active = max(1, int(round(vocab_size * density)))
    for r in range(rows):
        tokens = rng.choice(vocab_size, size=k_active, replace=False)
        np.bitwise_or.at(mat[r], tokens // 32, np.uint32(1) << (tokens % 32).astype(np.uint32))
    return mat


def build_synthetic_workloads(rng: np.random.Generator, vocab_size: int):
    n_words = (vocab_size + 31) // 32
    for rows in (8, 64):
        sparse = _fill_random_sparse(rng, rows, vocab_size, 0.002)
        yield f"sparse_r{rows}", torch.from_numpy(sparse.view(np.int32))

        for name, row_idx in ((f"mixed_first_r{rows}", 0), (f"mixed_last_r{rows}", rows - 1)):
            mixed = sparse.copy()
            mixed[row_idx].fill(np.iinfo(np.uint32).max)
            yield name, torch.from_numpy(mixed.view(np.int32))

        dense = rng.integers(-(2**31), 2**31, size=(rows, n_words), dtype=np.int32)
        yield f"dense_r{rows}", torch.from_numpy(dense)

    # Mixed-density 64-row batches: 56 sparse structural rows + 8 dense localized string rows
    mixed_local = _fill_random_sparse(rng, 64, vocab_size, 0.001)
    # Dense ASCII/identifier subset in lower quarter of vocabulary (contiguous tiles)
    sub_vocab = max(128, vocab_size // 4)
    mixed_local[56:, : (sub_vocab + 31) // 32] = _fill_random_sparse(rng, 8, sub_vocab, 0.50)
    yield "mixed_56struct_8str_local_r64", torch.from_numpy(mixed_local.view(np.int32))

    mixed_85 = _fill_random_sparse(rng, 64, vocab_size, 0.002)
    mixed_85[56:] = _fill_random_sparse(rng, 8, vocab_size, 0.85)
    yield "mixed_56struct_8str_85pct_r64", torch.from_numpy(mixed_85.view(np.int32))


def build_xgrammar_workloads(tokenizer_path: str, vocab_size: int):
    import xgrammar as xgr

    if tokenizer_path == "ascii":
        vocabulary = [chr(index) for index in range(32, 127)]
        token_ids = {token: index for index, token in enumerate(vocabulary)}
        tokenizer_info = xgr.TokenizerInfo(
            [*vocabulary, "<eos>"], vocab_size=vocab_size, stop_token_ids=[len(vocabulary)]
        )

        def encode(text: str) -> list[int]:
            return [token_ids[ch] for ch in text]
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)

        def encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

    compiler = xgr.GrammarCompiler(tokenizer_info)
    fixtures = (
        (
            "tool_call",
            {"tool": {"enum": ["lookup_account", "lookup_invoice"]}, "args": {"type": "string"}},
            {"tool": "lookup_account", "args": "ACCT-0042"},
        ),
        (
            "tool_followup",
            {"status": {"enum": ["ok", "retry"]}, "summary": {"type": "string"}},
            {"status": "ok", "summary": "Completed lookup across two shards."},
        ),
    )
    for name, properties, value in fixtures:
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        compiled = compiler.compile_json_schema(json.dumps(schema))
        matcher = xgr.GrammarMatcher(compiled)
        tokens = encode(json.dumps(value, separators=(",", ":")))
        if len(tokens) < 8:
            raise RuntimeError(f"{name} produced fewer than 8 trace tokens")
        trace = xgr.allocate_token_bitmask(len(tokens), vocab_size)
        for index, tok in enumerate(tokens):
            matcher.fill_next_token_bitmask(trace, index)
            if not matcher.accept_token(tok):
                raise RuntimeError(f"{name} rejected token at index {index}")

        starts = np.linspace(0, len(tokens) - 8, 8, dtype=np.int64)
        windows = [trace[s : s + 8] for s in starts]
        yield f"{name}_r8", windows[len(windows) // 2]
        yield f"{name}_r64", torch.cat(windows, dim=0)


def _time_mean_us(fn, mask: torch.Tensor, inner: int) -> float:
    t0 = time.perf_counter_ns()
    for _ in range(inner):
        fn(mask)
    return (time.perf_counter_ns() - t0) / (inner * 1000.0)


def _summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "p50_us": round(statistics.median(samples), 3),
        "p95_us": round(ordered[(95 * len(samples) + 99) // 100 - 1], 3),
    }


def run_case(
    name: str,
    mask: torch.Tensor,
    *,
    vocab_size: int,
    k_chunk: int,
    max_eff_row_ratio: float,
    repeats: int,
    inner: int,
    has_cpp: bool,
) -> dict:
    ops = {
        "numpy_roofline": lambda m: select_sparse_vocab_proj(
            m, vocab_size, k_chunk=k_chunk, max_eff_row_ratio=max_eff_row_ratio, force_numpy=True
        ),
        "naive_scan": lambda m: naive_mask_dispatch(
            m, vocab_size, k_chunk=k_chunk, max_eff_row_ratio=max_eff_row_ratio
        ),
    }
    if has_cpp:
        ops["cpp_roofline"] = lambda m: select_sparse_vocab_proj(
            m, vocab_size, k_chunk=k_chunk, max_eff_row_ratio=max_eff_row_ratio, force_numpy=False
        )

    decisions = {arm: fn(mask) for arm, fn in ops.items()}
    if len(set(decisions.values())) != 1:
        raise RuntimeError(f"Dispatch disagreement on {name}: {decisions}")

    for fn in ops.values():
        _time_mean_us(fn, mask, inner)

    arm_names = list(ops.keys())
    timings: dict[str, list[float]] = {arm: [] for arm in arm_names}
    for rep in range(repeats):
        order = arm_names if rep % 2 == 0 else list(reversed(arm_names))
        for arm in order:
            timings[arm].append(_time_mean_us(ops[arm], mask, inner))

    summaries = {arm: _summarize(timings[arm]) for arm in arm_names}
    active_frac, active_pairs, active_tiles = union_active_row_fraction(mask, vocab_size)
    row = {
        "case": name,
        "shape": list(mask.shape),
        "vocab_size": vocab_size,
        "route": "sparse" if decisions["numpy_roofline"] else "dense",
        "agree": True,
        "active_pairs": active_pairs,
        "active_tiles": active_tiles,
        "union_active_tile_fraction": round(active_frac, 6),
        "timings": summaries,
    }
    if has_cpp:
        cpp_p50 = summaries["cpp_roofline"]["p50_us"]
        row["cpp_vs_numpy_speedup"] = round(summaries["numpy_roofline"]["p50_us"] / cpp_p50, 2)
        row["cpp_vs_naive_speedup"] = round(summaries["naive_scan"]["p50_us"] / cpp_p50, 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--k-chunk", type=int, default=8)
    parser.add_argument("--max-eff-row-ratio", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument(
        "--xgrammar-tokenizer",
        default=None,
        help="Optional HF tokenizer path or 'ascii' to benchmark real XGrammar tool masks",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    has_cpp = ensure_extension_loaded()
    rng = np.random.default_rng(args.seed)
    if args.xgrammar_tokenizer:
        workloads = build_xgrammar_workloads(args.xgrammar_tokenizer, args.vocab_size)
        scope = "XGrammar tool-call packed bitmask dispatch"
    else:
        workloads = build_synthetic_workloads(rng, args.vocab_size)
        scope = "Synthetic sparse, dense, and mixed-density 64-row bitmask dispatch"

    cases = [
        run_case(
            name,
            mask,
            vocab_size=args.vocab_size,
            k_chunk=args.k_chunk,
            max_eff_row_ratio=args.max_eff_row_ratio,
            repeats=args.repeats,
            inner=args.inner,
            has_cpp=has_cpp,
        )
        for name, mask in workloads
    ]

    report = {
        "scope": scope,
        "cpp_extension_loaded": has_cpp,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "config": {
            "vocab_size": args.vocab_size,
            "k_chunk": args.k_chunk,
            "max_eff_row_ratio": args.max_eff_row_ratio,
            "repeats": args.repeats,
            "inner": args.inner,
            "seed": args.seed,
            "xgrammar_tokenizer": (
                Path(args.xgrammar_tokenizer).name if args.xgrammar_tokenizer else None
            ),
        },
        "cases": cases,
    }
    if args.xgrammar_tokenizer:
        report["xgrammar"] = version("xgrammar")
        tok_json = Path(args.xgrammar_tokenizer) / "tokenizer.json"
        if tok_json.is_file():
            report["tokenizer_sha256"] = hashlib.sha256(tok_json.read_bytes()).hexdigest()

    encoded = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
