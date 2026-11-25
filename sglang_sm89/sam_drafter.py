"""O(1)-per-token Suffix Automaton (SAM) prompt/schema lookup tail drafter for sglang-sm89.

Provides a zero-GPU, lossless speculative tail-grafting path gated behind
``SGLANG_SM89_SAM_GRAFT=0`` (opt-in). When a parallel drafter has low confidence
on a tail slot (``max(q) < tau``) and the committed prefix matches a substring in the
prompt or committed output history, the SAM proposes the exact continuation token if it
satisfies the active grammar mask.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Iterable, Optional, Sequence

import numpy as np


def is_sam_graft_enabled() -> bool:
    """Return True iff SGLANG_SM89_SAM_GRAFT is explicitly enabled."""
    val = os.getenv("SGLANG_SM89_SAM_GRAFT", "0").strip().lower()
    return val not in ("0", "false", "no", "off", "")


class SuffixAutomaton:
    """Linear-time O(N) build, O(1)-amortized transition Suffix Automaton over token IDs."""

    __slots__ = ("next_edges", "link", "length", "first_endpos", "last", "tokens")

    def __init__(self) -> None:
        self.next_edges: list[dict[int, int]] = [{}]
        self.link: list[int] = [-1]
        self.length: list[int] = [0]
        self.first_endpos: list[int] = [-1]
        self.last: int = 0
        self.tokens: list[int] = []

    def extend(self, tok: int) -> None:
        """Append one token ID to the automaton in amortized O(1) time."""
        next_edges = self.next_edges
        link = self.link
        length = self.length
        first_endpos = self.first_endpos

        pos = len(self.tokens)
        self.tokens.append(tok)

        cur = len(next_edges)
        next_edges.append({})
        link.append(0)
        length.append(length[self.last] + 1)
        first_endpos.append(pos)

        p = self.last
        while p != -1:
            edges_p = next_edges[p]
            if tok in edges_p:
                break
            edges_p[tok] = cur
            p = link[p]

        if p == -1:
            link[cur] = 0
        else:
            q = next_edges[p][tok]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(next_edges)
                next_edges.append(next_edges[q].copy())
                link.append(link[q])
                length.append(length[p] + 1)
                first_endpos.append(first_endpos[q])
                while p != -1:
                    edges_p = next_edges[p]
                    if edges_p.get(tok) != q:
                        break
                    edges_p[tok] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self.last = cur

    def build(self, seq: Sequence[int]) -> None:
        """Extend the automaton with a full sequence of token IDs in O(len(seq))."""
        next_edges = self.next_edges
        link = self.link
        length = self.length
        first_endpos = self.first_endpos
        a_next = next_edges.append
        a_link = link.append
        a_len = length.append
        a_end = first_endpos.append
        tokens = self.tokens
        pos = len(tokens)
        tokens.extend(seq)
        last = self.last

        was_gc = gc.isenabled()
        if was_gc:
            gc.disable()
        try:
            for tok in seq:
                cur = len(next_edges)
                a_next({})
                a_link(0)
                a_len(length[last] + 1)
                a_end(pos)
                pos += 1

                p = last
                while p != -1:
                    ep = next_edges[p]
                    if tok in ep:
                        break
                    ep[tok] = cur
                    p = link[p]

                if p != -1:
                    q = next_edges[p][tok]
                    if length[p] + 1 == length[q]:
                        link[cur] = q
                    else:
                        clone = len(next_edges)
                        a_next(next_edges[q].copy())
                        a_link(link[q])
                        a_len(length[p] + 1)
                        a_end(first_endpos[q])
                        while p != -1:
                            ep = next_edges[p]
                            if ep.get(tok) != q:
                                break
                            ep[tok] = clone
                            p = link[p]
                        link[q] = clone
                        link[cur] = clone
                last = cur
        finally:
            if was_gc:
                gc.enable()

        self.last = last

    def step_state(self, state: int, cur_len: int, tok: int) -> tuple[int, int]:
        """Advance a query cursor (state, cur_len) by one token in O(1) amortized time."""
        next_edges = self.next_edges
        link = self.link
        length = self.length
        while state != 0 and tok not in next_edges[state]:
            state = link[state]
            cur_len = length[state]
        nxt = next_edges[state].get(tok)
        if nxt is not None:
            return nxt, cur_len + 1
        return 0, 0

    def _resolve_continuation_state(
        self, state: int, cur_len: int, min_match: int
    ) -> tuple[int, int, int]:
        """Fall back along suffix links if the longest match ends only at len(tokens)-1."""
        n_toks = len(self.tokens)
        first_endpos = self.first_endpos
        link = self.link
        length = self.length
        s = state
        mlen = cur_len
        while s > 0 and mlen >= min_match:
            start = first_endpos[s] + 1
            if start < n_toks:
                return mlen, s, start
            s = link[s]
            mlen = length[s]
        return cur_len, state, -1

    def match_suffix(
        self, query_suffix: Sequence[int], *, min_match: int = 2
    ) -> tuple[int, int, int]:
        """Walk query_suffix and return (match_len, state, next_pos) for continuation."""
        state = 0
        cur_len = 0
        next_edges = self.next_edges
        link = self.link
        length = self.length
        for tok in query_suffix:
            while state != 0 and tok not in next_edges[state]:
                state = link[state]
                cur_len = length[state]
            nxt = next_edges[state].get(tok)
            if nxt is not None:
                state = nxt
                cur_len += 1
            else:
                state = 0
                cur_len = 0
        if cur_len < min_match:
            return cur_len, state, -1
        return self._resolve_continuation_state(state, cur_len, min_match)

    def draft_from_cursor(
        self,
        state: int,
        cur_len: int,
        max_draft: int,
        *,
        min_match: int = 2,
    ) -> tuple[int, list[int]]:
        """Emit up to max_draft continuation tokens directly from an existing query cursor."""
        if cur_len < min_match or max_draft <= 0:
            return cur_len, []
        mlen, _, start = self._resolve_continuation_state(state, cur_len, min_match)
        if start < 0:
            return mlen, []
        end = min(len(self.tokens), start + max_draft)
        return mlen, self.tokens[start:end]

    def longest_match_and_draft(
        self,
        query_suffix: Sequence[int],
        max_draft: int,
        *,
        min_match: int = 2,
    ) -> tuple[int, list[int]]:
        """Walk query_suffix and emit up to max_draft continuation tokens."""
        cur_len, state, start = self.match_suffix(query_suffix, min_match=min_match)
        if start < 0 or max_draft <= 0:
            return cur_len, []
        end = min(len(self.tokens), start + max_draft)
        if start >= end:
            return cur_len, []
        return cur_len, self.tokens[start:end]


class SAMPromptCache:
    """Per-request Suffix Automaton cache indexed by request ID."""

    def __init__(self, *, window_tokens: int = 16) -> None:
        self.window_tokens = int(window_tokens)
        self._entries: dict[Any, tuple[SuffixAutomaton, int]] = {}

    def sync_request(self, req_id: Any, committed_tokens: Sequence[int]) -> SuffixAutomaton:
        """Build or incrementally extend the SAM for req_id to match committed_tokens."""
        entry = self._entries.get(req_id)
        n_toks = len(committed_tokens)
        if entry is None:
            sam = SuffixAutomaton()
            # Index up to all except the final token first? No, indexing full committed
            # sequence is fine since match_suffix walks the trailing query window and
            # requires first_endpos + 1 < len(tokens) (never pointing past the end).
            if n_toks > 0:
                sam.build(committed_tokens)
            self._entries[req_id] = (sam, n_toks)
            return sam

        sam, indexed_len = entry
        if n_toks < indexed_len:
            # Request ID was reused or retracted: rebuild cleanly.
            sam = SuffixAutomaton()
            if n_toks > 0:
                sam.build(committed_tokens)
            self._entries[req_id] = (sam, n_toks)
            return sam

        if n_toks > indexed_len:
            sam.build(committed_tokens[indexed_len:])
            self._entries[req_id] = (sam, n_toks)
        return sam

    def evict_missing(self, active_req_ids: Iterable[Any]) -> None:
        """Drop SAM state for finished requests."""
        active = set(active_req_ids)
        for key in list(self._entries.keys()):
            if key not in active:
                del self._entries[key]


def place_unique_at_slot0(cand_row: Any, q_row: Any, tok: int) -> None:
    """Place ``tok`` at slot 0 of ``cand_row`` without creating duplicate candidate IDs."""
    n = int(len(cand_row))
    swap_idx = -1
    for idx in range(n):
        if int(cand_row[idx]) == tok:
            swap_idx = idx
            break
    if swap_idx > 0:
        tmp = int(cand_row[0])
        cand_row[0] = tok
        cand_row[swap_idx] = tmp
    elif swap_idx < 0:
        cand_row[0] = tok

    q_row[:] = 0.0
    q_row[0] = 1.0


def _is_bitmask_allowed(mask_row: Optional[np.ndarray], tok: int, vocab_size: int) -> bool:
    if mask_row is None:
        return True
    if tok < 0 or tok >= vocab_size:
        return False
    w_idx = tok >> 5
    if w_idx >= mask_row.shape[0]:
        return False
    return bool((int(mask_row[w_idx]) >> (tok & 31)) & 1)


def graft_sam_tail_drafts(
    *,
    reqs: Sequence[Any],
    draft_cpu: Any,
    cand_cpu: Optional[Any],
    q_cpu: Optional[Any],
    sam_cache: SAMPromptCache,
    vocab_size: int,
    tau: float = 0.35,
    min_match: int = 2,
    mask_rows_per_req: Optional[Sequence[Optional[Sequence[Optional[np.ndarray]]]]] = None,
) -> bool:
    """Losslessly graft O(1) SAM prompt continuations onto low-confidence tail draft slots.

    For each request ``i`` and draft position ``pos = 1..block_size-1``:
      - Computes the SAM cursor over the trailing committed window + ``draft_cpu[i, :pos]``.
      - Only grafts when ``q_cpu[i, pos - 1].max() < tau`` (or when ``q_cpu`` is None),
        the SAM match length is ``>= min_match``, a valid continuation token ``u_sam`` exists,
        and ``u_sam`` passes ``mask_rows_per_req[i][pos - 1]`` (if provided).
      - Updates ``cand_cpu[i, pos - 1]`` & ``q_cpu[i, pos - 1]`` via ``place_unique_at_slot0``
        so candidate IDs remain duplicate-free and target verification remains 100% lossless.
    """
    bs = int(draft_cpu.shape[0])
    block_size = int(draft_cpu.shape[1])
    if bs == 0 or block_size <= 1:
        return False

    modified = False
    window = sam_cache.window_tokens

    for i in range(bs):
        req = reqs[i]
        req_id = getattr(req, "rid", i)
        committed = getattr(req, "origin_input_ids", None)
        output_ids = getattr(req, "output_ids", None)
        if committed is None:
            continue
        seq = list(committed)
        if output_ids:
            seq.extend(output_ids)
        if len(seq) < min_match:
            continue

        sam = sam_cache.sync_request(req_id, seq)
        tail_ctx = seq[-window:] if len(seq) > window else seq
        # Seed cursor from trailing committed prefix, excluding the bonus token at pos 0
        # if it is already the last element of seq.
        prefix_for_cursor = tail_ctx[:-1] if tail_ctx and tail_ctx[-1] == int(draft_cpu[i, 0]) else tail_ctx
        state, cur_len = 0, 0
        for tok in prefix_for_cursor:
            state, cur_len = sam.step_state(state, cur_len, int(tok))

        req_masks = mask_rows_per_req[i] if mask_rows_per_req is not None else None

        for pos in range(1, block_size):
            prev_tok = int(draft_cpu[i, pos - 1])
            state, cur_len = sam.step_state(state, cur_len, prev_tok)

            if cur_len < min_match:
                continue

            if q_cpu is not None:
                max_q = float(q_cpu[i, pos - 1].max())
                if max_q >= tau:
                    continue

            _, cont = sam.draft_from_cursor(state, cur_len, 1, min_match=min_match)
            if not cont:
                continue
            u_sam = int(cont[0])
            if u_sam < 0 or u_sam >= vocab_size:
                continue

            mask_row = req_masks[pos - 1] if req_masks is not None else None
            if not _is_bitmask_allowed(mask_row, u_sam, vocab_size):
                continue

            if cand_cpu is not None and q_cpu is not None:
                place_unique_at_slot0(cand_cpu[i, pos - 1], q_cpu[i, pos - 1], u_sam)

            if int(draft_cpu[i, pos]) != u_sam:
                draft_cpu[i, pos] = u_sam
                modified = True

    return modified
