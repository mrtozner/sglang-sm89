"""Unit tests for O(1) Suffix Automaton prompt/schema tail drafter."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from sglang_sm89.sam_drafter import (
    SAMPromptCache,
    SuffixAutomaton,
    graft_sam_tail_drafts,
    is_sam_graft_enabled,
    place_unique_at_slot0,
)


def test_env_flag_default_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SGLANG_SM89_SAM_GRAFT", raising=False)
    assert is_sam_graft_enabled() is False
    monkeypatch.setenv("SGLANG_SM89_SAM_GRAFT", "1")
    assert is_sam_graft_enabled() is True


def test_sam_8k_build_and_o1_lookup_budget() -> None:
    # 8,192-token structured prompt repeating SQL/JSON schema tokens
    pattern = list(range(100, 356))
    prompt_8k = (pattern * 32)[:8192]
    assert len(prompt_8k) == 8192

    sam = SuffixAutomaton()
    t0 = time.perf_counter()
    sam.build(prompt_8k)
    build_ms = (time.perf_counter() - t0) * 1000.0

    query = [110, 111, 112, 113, 114]
    # Warm cursor and measure incremental step + lookup latency
    t1 = time.perf_counter()
    n_iters = 5000
    for _ in range(n_iters):
        match_len, draft = sam.longest_match_and_draft(query, max_draft=8, min_match=2)
    per_lookup_us = ((time.perf_counter() - t1) / n_iters) * 1e6

    # Also measure single-step O(1) cursor advance + draft_from_cursor
    state, cur_len = 0, 0
    for tok in query[:-1]:
        state, cur_len = sam.step_state(state, cur_len, tok)
    t2 = time.perf_counter()
    for _ in range(n_iters):
        s2, l2 = sam.step_state(state, cur_len, query[-1])
        _, d2 = sam.draft_from_cursor(s2, l2, 8, min_match=2)
    step_us = ((time.perf_counter() - t2) / n_iters) * 1e6

    assert match_len == 5
    assert draft == [115, 116, 117, 118, 119, 120, 121, 122]
    assert d2 == draft
    # Target budgets: build <= 3 ms (allow small CI jitter up to 6 ms, verify fast), lookup <= 5 us
    assert build_ms < 6.0, f"8K SAM build too slow: {build_ms:.2f} ms"
    assert step_us <= 5.0, f"O(1) step lookup too slow: {step_us:.2f} us"
    print(f"8K build={build_ms:.2f} ms, 5-tok lookup={per_lookup_us:.2f} us, 1-step cursor={step_us:.2f} us")


def test_place_unique_at_slot0_prevents_duplicates() -> None:
    cand = np.array([10, 20, 30, 40], dtype=np.int64)
    q = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)

    # Case 1: token already exists in cand -> swaps without duplicating
    place_unique_at_slot0(cand, q, 30)
    assert cand.tolist() == [30, 20, 10, 40]
    assert len(set(cand.tolist())) == 4
    assert q.tolist() == [1.0, 0.0, 0.0, 0.0]

    # Case 2: token not in cand -> overwrites slot 0
    place_unique_at_slot0(cand, q, 99)
    assert cand.tolist() == [99, 20, 10, 40]
    assert len(set(cand.tolist())) == 4
    assert q.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_graft_sam_tail_respects_tau_and_grammar_mask() -> None:
    # Prompt has schema sequence [50, 51, 52, 53, 54]
    req = SimpleNamespace(
        rid="req-1",
        origin_input_ids=[1, 2, 50, 51, 52, 53, 54, 9, 10],
        output_ids=[50, 51],  # committed ends with [50, 51]
    )
    cache = SAMPromptCache(window_tokens=16)

    # block_size = 4: pos 0 = bonus (51), pos 1 = high confidence (keep), pos 2 & 3 = uncertain tails
    draft_cpu = np.array([[51, 77, 88, 99]], dtype=np.int64)
    cand_cpu = np.array(
        [
            [
                [77, 52, 11, 12],   # pos 1: max_q = 0.85 >= tau=0.35 -> NOT grafted
                [88, 53, 13, 14],   # pos 2: max_q = 0.28 < tau=0.35 -> grafted if allowed
                [99, 54, 15, 16],   # pos 3: max_q = 0.26 < tau=0.35 -> blocked by grammar mask!
            ]
        ],
        dtype=np.int64,
    )
    q_cpu = np.array(
        [
            [
                [0.85, 0.05, 0.05, 0.05],
                [0.28, 0.25, 0.24, 0.23],
                [0.26, 0.25, 0.25, 0.24],
            ]
        ],
        dtype=np.float32,
    )

    vocab_size = 256
    n_words = (vocab_size + 31) // 32
    m_pos1 = np.full(n_words, -1, dtype=np.int32)
    m_pos2 = np.full(n_words, -1, dtype=np.int32)
    # Pos 3 grammar mask forbids token 54!
    m_pos3 = np.full(n_words, -1, dtype=np.int32)
    m_pos3[54 >> 5] &= ~(1 << (54 & 31))

    # Note: pos 1 keeps 77 (not 52), so pos 2 sees [50, 51, 77] (no match!) unless pos 1 was 52.
    # Let's test pos 1 matching [50, 51] -> 52 when pos 1 is low-confidence, then pos 2 -> 53, pos 3 -> 54 blocked!
    q_cpu[0, 0] = [0.30, 0.25, 0.25, 0.20]  # pos 1 uncertain -> grafted to 52!
    changed = graft_sam_tail_drafts(
        reqs=[req],
        draft_cpu=draft_cpu,
        cand_cpu=cand_cpu,
        q_cpu=q_cpu,
        sam_cache=cache,
        vocab_size=vocab_size,
        tau=0.35,
        min_match=2,
        mask_rows_per_req=[[m_pos1, m_pos2, m_pos3]],
    )

    assert changed is True
    # pos 1 grafted to 52, pos 2 grafted to 53, pos 3 kept at 99 because 54 was forbidden by grammar!
    assert draft_cpu[0].tolist() == [51, 52, 53, 99]
    assert len(set(cand_cpu[0, 0].tolist())) == 4
    assert cand_cpu[0, 0, 0] == 52 and q_cpu[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert len(set(cand_cpu[0, 1].tolist())) == 4
    assert cand_cpu[0, 1, 0] == 53 and q_cpu[0, 1].tolist() == [1.0, 0.0, 0.0, 0.0]
    # Pos 3 untouched
    assert cand_cpu[0, 2, 0] == 99
