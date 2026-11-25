"""Unit tests, eager/graph stale-buffer counterexamples, and 4-step rejection TV tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_sm89.coupled_grammar import (
    _STEP_TLS,
    after_compute_candidates,
    after_draft_runner_forward,
    around_dflash_forward_batch_generation,
    around_grammar_tree_from_linear_chain,
)


class _BitmaskGrammar:
    """Minimal stateful multi-step grammar matching XGrammar fill_vocab_mask / accept / rollback."""

    def __init__(self, vocab_size: int, allowed_fn) -> None:
        self.vocab_size = vocab_size
        self.allowed_fn = allowed_fn
        self.history: list[int] = []

    def is_terminated(self) -> bool:
        return len(self.history) >= 4

    def fill_vocab_mask(self, bitmask: torch.Tensor, row_idx: int) -> None:
        bitmask[row_idx].zero_()
        for tok in self.allowed_fn(tuple(self.history)):
            bitmask[row_idx, tok >> 5] |= 1 << (tok & 31)

    def accept_token(self, tok: int) -> None:
        self.history.append(int(tok))

    def rollback(self, n: int) -> None:
        if n <= 0:
            raise AssertionError("rollback(0) is invalid in XGrammar")
        del self.history[-n:]


class _DummyDraftRunner:
    def __init__(self, can_run_graph: bool) -> None:
        self.can_run_graph = can_run_graph

    def forward(self):
        return SimpleNamespace(can_run_graph=self.can_run_graph)


def _make_worker(vocab_size: int, *, can_run_graph: bool) -> SimpleNamespace:
    target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=vocab_size))
    )
    sampler = SimpleNamespace(candidate_out=None, q_out=None)
    return SimpleNamespace(
        target_worker=target_worker,
        draft_model_runner=_DummyDraftRunner(can_run_graph=can_run_graph),
        _draft_sampler=sampler,
        _selector_sample=None,
    )


def _linear_tree_passthrough(_cls, verify_ids_2d: torch.Tensor):
    return verify_ids_2d


def test_stale_sampler_counterexample_eager_sampled() -> None:
    """Eager sampled batches must read/write _selector_sample exclusively and ignore stale sampler."""
    vocab_size = 128
    worker = _make_worker(vocab_size, can_run_graph=False)

    # Poison stale _draft_sampler buffers from an earlier CUDA Graph step.
    stale_cand = torch.tensor([[[99, 98]]], dtype=torch.int64)
    stale_q = torch.tensor([[[0.1, 0.9]]], dtype=torch.float32)
    worker._draft_sampler.candidate_out = stale_cand.clone()
    worker._draft_sampler.q_out = stale_q.clone()

    for legal_tokens, expected_q in (
        ((10, 11, 12), [0.9, 0.1]),
        ((10, 12, 13), [1.0, 0.0]),
    ):
        fresh_cand = torch.tensor([[[10, 11]]], dtype=torch.int64)
        fresh_q = torch.tensor([[[0.9, 0.1]]], dtype=torch.float32)
        worker._selector_sample = (fresh_cand, fresh_q)

        grammar = _BitmaskGrammar(vocab_size, lambda _hist, lt=legal_tokens: lt)
        batch = SimpleNamespace(
            has_grammar=True,
            reqs=[SimpleNamespace(grammar=grammar)],
            sampling_info=SimpleNamespace(
                is_all_greedy=False,
                temperatures=torch.tensor([1.0]),
                top_ks=torch.tensor([2]),
            ),
        )
        verify_ids = torch.tensor([[1, 11]], dtype=torch.int64)

        def _eager_step(w, _batch):
            res = w.draft_model_runner.forward()
            after_draft_runner_forward(res, w.draft_model_runner)
            after_compute_candidates(fresh_cand)
            around_grammar_tree_from_linear_chain(_linear_tree_passthrough, object, verify_ids)
            return verify_ids

        around_dflash_forward_batch_generation(_eager_step, worker, batch)

        assert torch.allclose(worker._selector_sample[1][0, 0], torch.tensor(expected_q))
        assert torch.equal(worker._selector_sample[0][0, 0], torch.tensor([10, 11]))
        assert torch.unique(worker._selector_sample[0][0, 0]).numel() == 2
        assert torch.allclose(worker._draft_sampler.q_out, stale_q)
        assert torch.equal(worker._draft_sampler.candidate_out, stale_cand)
        assert grammar.history == []


def test_stale_sampler_counterexample_eager_all_greedy() -> None:
    """Eager all-greedy batches must use eager_cand_ids and never touch stale sampler.candidate_out."""
    vocab_size = 128
    worker = _make_worker(vocab_size, can_run_graph=False)
    worker._selector_sample = None

    stale_cand = torch.tensor([[[77, 78]]], dtype=torch.int64)
    worker._draft_sampler.candidate_out = stale_cand.clone()

    grammar = _BitmaskGrammar(vocab_size, lambda _hist: (42, 77, 90))
    batch = SimpleNamespace(
        has_grammar=True,
        reqs=[SimpleNamespace(grammar=grammar)],
        sampling_info=SimpleNamespace(is_all_greedy=True),
    )
    verify_ids = torch.tensor([[1, 0]], dtype=torch.int64)
    eager_cand = torch.tensor([[[42, 43]]], dtype=torch.int64)

    def _eager_greedy_step(w, _batch):
        res = w.draft_model_runner.forward()
        after_draft_runner_forward(res, w.draft_model_runner)
        after_compute_candidates(eager_cand)
        around_grammar_tree_from_linear_chain(_linear_tree_passthrough, object, verify_ids)
        return verify_ids

    around_dflash_forward_batch_generation(_eager_greedy_step, worker, batch)
    assert int(verify_ids[0, 1].item()) == 42
    assert torch.equal(worker._draft_sampler.candidate_out, stale_cand)

    # Without eager_cand_ids on eager step, still must never touch stale candidate 77 -> falls back to lowest legal (42).
    verify_ids2 = torch.tensor([[1, 0]], dtype=torch.int64)
    grammar2 = _BitmaskGrammar(vocab_size, lambda _hist: (15, 77))
    batch2 = SimpleNamespace(
        has_grammar=True,
        reqs=[SimpleNamespace(grammar=grammar2)],
        sampling_info=SimpleNamespace(is_all_greedy=True),
    )

    def _eager_greedy_no_cand(w, _batch):
        res = w.draft_model_runner.forward()
        after_draft_runner_forward(res, w.draft_model_runner)
        around_grammar_tree_from_linear_chain(_linear_tree_passthrough, object, verify_ids2)
        return verify_ids2

    around_dflash_forward_batch_generation(_eager_greedy_no_cand, worker, batch2)
    assert int(verify_ids2[0, 1].item()) == 15


def test_graph_all_greedy_uses_sampler_candidate_out() -> None:
    """When draft runs via CUDA Graph with all-greedy batch, sampler.candidate_out is fresh."""
    vocab_size = 128
    worker = _make_worker(vocab_size, can_run_graph=True)
    worker._selector_sample = None
    worker._draft_sampler.candidate_out = torch.tensor([[[55, 56]]], dtype=torch.int64)

    grammar = _BitmaskGrammar(vocab_size, lambda _hist: (10, 55))
    batch = SimpleNamespace(
        has_grammar=True,
        reqs=[SimpleNamespace(grammar=grammar)],
        sampling_info=SimpleNamespace(is_all_greedy=True),
    )
    verify_ids = torch.tensor([[1, 0]], dtype=torch.int64)

    def _graph_step(w, _batch):
        res = w.draft_model_runner.forward()
        after_draft_runner_forward(res, w.draft_model_runner)
        around_grammar_tree_from_linear_chain(_linear_tree_passthrough, object, verify_ids)
        return verify_ids

    around_dflash_forward_batch_generation(_graph_step, worker, batch)
    assert int(verify_ids[0, 1].item()) == 55
    assert _STEP_TLS.draft_ran_graph is None


def test_mixed_batch_forced_fallback_tail_padding_and_rollback() -> None:
    """Mixed batch handles |V|==1 jump-forward, Z<=1e-12 fallback, tail padding, and rollback."""
    vocab_size = 67  # odd tail padding in word 2
    worker = _make_worker(vocab_size, can_run_graph=True)

    def g0_rules(hist):
        if len(hist) == 0:
            return (19,)  # forced deterministic |V| == 1
        return (2, 4, 6)

    def g1_rules(_hist):
        return (5, 7, 66)

    g0 = _BitmaskGrammar(vocab_size, g0_rules)
    g1 = _BitmaskGrammar(vocab_size, g1_rules)
    cand = torch.tensor(
        [
            [[10, 19, 20, 21], [1, 2, 3, 4]],
            [[1, 2, 3, 4], [5, 7, 8, 9]],
        ],
        dtype=torch.int64,
    )
    q = torch.tensor(
        [
            [[0.4, 0.3, 0.2, 0.1], [0.5, 0.2, 0.2, 0.1]],
            [[0.5, 0.3, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]],
        ],
        dtype=torch.float32,
    )
    worker._selector_sample = (cand, q)
    batch = SimpleNamespace(
        has_grammar=True,
        reqs=[SimpleNamespace(grammar=g0), SimpleNamespace(grammar=g1)],
        sampling_info=SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.tensor([0.0, 0.8]),
            top_ks=torch.tensor([4, 4]),
        ),
    )
    verify_ids = torch.tensor([[0, 10, 1], [0, 1, 7]], dtype=torch.int64)

    def _step(w, _b):
        res = w.draft_model_runner.forward()
        after_draft_runner_forward(res, w.draft_model_runner)
        around_grammar_tree_from_linear_chain(_linear_tree_passthrough, object, verify_ids)

    around_dflash_forward_batch_generation(_step, worker, batch)

    assert int(verify_ids[0, 1].item()) == 19
    assert int(verify_ids[0, 2].item()) == 2
    assert int(verify_ids[1, 1].item()) == 5
    for row in range(2):
        for slot in range(2):
            assert torch.unique(cand[row, slot]).numel() == 4
    assert g0.history == [] and g1.history == []


def _4step_allowed(hist: tuple[int, ...]) -> tuple[int, ...]:
    step = len(hist)
    if step == 0:
        return (1, 2)
    if step == 1:
        return (5,) if hist[0] == 1 else (5, 6)
    if step == 2:
        return (2, 8) if hist[1] == 5 else (1, 4)
    return (3, 9) if (hist[2] % 2 == 0) else (2, 7)


def _target_dist_np(hist: tuple[int, ...], vocab_size: int = 10) -> np.ndarray:
    step = len(hist)
    prev = hist[-1] if hist else 0
    logits = np.array(
        [0.4 + 0.25 * ((step + 1) * ((t + prev) % 5)) for t in range(vocab_size)],
        dtype=np.float64,
    )
    allowed = _4step_allowed(hist)
    masked = np.full(vocab_size, -1e18, dtype=np.float64)
    masked[list(allowed)] = logits[list(allowed)]
    ex = np.exp(masked - np.max(masked))
    return ex / ex.sum()


def _draft_proposal_np(hist: tuple[int, ...], vocab_size: int = 10, top_k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Deliberately shifted proposal placing mass on BOTH illegal and legal tokens within top_k=6."""
    allowed = list(_4step_allowed(hist))
    illegal = [t for t in range(vocab_size) if t not in allowed]
    cands = (allowed + illegal)[:top_k]
    raw_weights = np.array([1.0 + 0.3 * ((len(hist) + idx) % 4) for idx in range(top_k)], dtype=np.float64)
    q_vals = raw_weights / raw_weights.sum()
    return np.array(cands, dtype=np.int64), q_vals


def _fast_draw_cdf(u: float, probs: np.ndarray) -> int:
    acc = 0.0
    n = probs.shape[0]
    for i in range(n - 1):
        acc += float(probs[i])
        if u < acc:
            return i
    return n - 1


def _exact_4step_target_table() -> dict[tuple[int, int, int, int], float]:
    table: dict[tuple[int, int, int, int], float] = {}
    p0 = _target_dist_np(())
    for t0 in _4step_allowed(()):
        p1 = _target_dist_np((t0,))
        for t1 in _4step_allowed((t0,)):
            p2 = _target_dist_np((t0, t1))
            for t2 in _4step_allowed((t0, t1)):
                p3 = _target_dist_np((t0, t1, t2))
                for t3 in _4step_allowed((t0, t1, t2)):
                    table[(t0, t1, t2, t3)] = float(p0[t0] * p1[t1] * p2[t2] * p3[t3])
    return table


@pytest.mark.parametrize("regime", ["eager_sampled", "graph_sampled", "eager_mixed"])
def test_four_step_sequential_rejection_tv_eager_graph_mixed(regime: str) -> None:
    """Monte Carlo N>=40,000 4-step trajectory TV < 0.01 for eager/graph/mixed batches."""
    torch.manual_seed(20260 + hash(regime) % 1000)
    rng = np.random.default_rng(777 + len(regime))
    vocab_size = 10
    block_size = 5
    top_k = 6
    is_eager = regime in ("eager_sampled", "eager_mixed")
    is_mixed = regime == "eager_mixed"
    chunk_bs = 4000
    n_trajectories = 56000 if is_mixed else 40000  # 42,000 sampled in mixed >= 40,000

    # Precompute target/draft tables by history tuple for fast batch assembly.
    target_cache: dict[tuple[int, ...], np.ndarray] = {}
    draft_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}

    def get_target(h: tuple[int, ...]) -> np.ndarray:
        if h not in target_cache:
            target_cache[h] = _target_dist_np(h, vocab_size)
        return target_cache[h]

    def get_draft(h: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        if h not in draft_cache:
            draft_cache[h] = _draft_proposal_np(h, vocab_size, top_k)
        return draft_cache[h]

    # Exact greedy 4-step trajectory when mixed batch has greedy rows:
    expected_greedy: list[int] = []
    for _ in range(4):
        expected_greedy.append(int(np.argmax(get_target(tuple(expected_greedy)))))
    expected_greedy_tuple = tuple(expected_greedy)

    worker = _make_worker(vocab_size, can_run_graph=(not is_eager))
    poison_cand = torch.full((chunk_bs, block_size - 1, top_k), 9, dtype=torch.int64)
    poison_q = torch.full((chunk_bs, block_size - 1, top_k), 1.0 / top_k, dtype=torch.float32)

    empirical_counts: dict[tuple[int, int, int, int], int] = {}
    sampled_total = 0

    for _chunk_start in range(0, n_trajectories, chunk_bs):
        histories: list[list[int]] = [[] for _ in range(chunk_bs)]
        grammars = [_BitmaskGrammar(vocab_size, _4step_allowed) for _ in range(chunk_bs)]
        is_greedy_row = [(is_mixed and (i % 4 == 0)) for i in range(chunk_bs)]
        active = list(range(chunk_bs))

        while active:
            m = len(active)
            verify_ids = torch.zeros((m, block_size), dtype=torch.int64)
            cand_t = torch.zeros((m, block_size - 1, top_k), dtype=torch.int64)
            q_t = torch.zeros((m, block_size - 1, top_k), dtype=torch.float32)

            # Generate unconstrained draft linear chains across remaining horizon per item.
            u_draws = rng.random((m, block_size - 1))
            for row_i, req_idx in enumerate(active):
                sim_hist = list(histories[req_idx])
                if sim_hist:
                    verify_ids[row_i, 0] = sim_hist[-1]
                for slot in range(block_size - 1):
                    h_key = tuple(sim_hist) if len(sim_hist) < 4 else ()
                    cands, probs = get_draft(h_key)
                    cand_t[row_i, slot] = torch.from_numpy(cands)
                    q_t[row_i, slot] = torch.from_numpy(probs.astype(np.float32))
                    if is_greedy_row[req_idx]:
                        tok = int(cands[0])
                    else:
                        tok = int(cands[_fast_draw_cdf(float(u_draws[row_i, slot]), probs)])
                    verify_ids[row_i, slot + 1] = tok
                    if len(sim_hist) < 4:
                        sim_hist.append(tok)

            if is_eager:
                worker._draft_sampler.candidate_out = poison_cand[:m].clone()
                worker._draft_sampler.q_out = poison_q[:m].clone()
            else:
                worker._draft_sampler.candidate_out = cand_t
                worker._draft_sampler.q_out = q_t
            worker._selector_sample = (cand_t, q_t)

            reqs_slice = [SimpleNamespace(grammar=grammars[idx]) for idx in active]
            temps = torch.tensor([0.0 if is_greedy_row[idx] else 1.0 for idx in active], dtype=torch.float32)
            batch = SimpleNamespace(
                has_grammar=True,
                reqs=reqs_slice,
                sampling_info=SimpleNamespace(
                    is_all_greedy=False,
                    temperatures=temps,
                    top_ks=torch.full((m,), top_k, dtype=torch.int32),
                ),
            )

            def _sim_step(w, _b):
                res = w.draft_model_runner.forward()
                after_draft_runner_forward(res, w.draft_model_runner)
                if is_eager:
                    after_compute_candidates(cand_t)
                around_grammar_tree_from_linear_chain(_linear_tree_passthrough, object, verify_ids)

            around_dflash_forward_batch_generation(_sim_step, worker, batch)

            if is_eager:
                assert torch.equal(worker._draft_sampler.candidate_out, poison_cand[:m])
                assert torch.equal(worker._draft_sampler.q_out, poison_q[:m])

            # Execute SGLang target verification (p / q acceptance + residual resampling).
            cand_np = worker._selector_sample[0].numpy()
            q_np = worker._selector_sample[1].numpy().astype(np.float64)
            v_np = verify_ids.numpy()
            u_accept = rng.random((m, block_size - 1))
            u_res = rng.random((m, block_size - 1))

            next_active: list[int] = []
            for row_i, req_idx in enumerate(active):
                hist = histories[req_idx]
                g = grammars[req_idx]
                slot = 0
                while len(hist) < 4 and slot < (block_size - 1):
                    p_masked = get_target(tuple(hist))
                    if is_greedy_row[req_idx]:
                        y_true = int(np.argmax(p_masked))
                        draft_tok = int(v_np[row_i, slot + 1])
                        if draft_tok == y_true:
                            hist.append(draft_tok)
                            g.accept_token(draft_tok)
                            slot += 1
                            continue
                        hist.append(y_true)
                        g.accept_token(y_true)
                        break

                    q_eff = np.zeros(vocab_size, dtype=np.float64)
                    for k_i in range(top_k):
                        q_eff[int(cand_np[row_i, slot, k_i])] += q_np[row_i, slot, k_i]
                    d_tok = int(v_np[row_i, slot + 1])
                    q_d = q_eff[d_tok]
                    alpha = 1.0 if q_d > 0.0 and p_masked[d_tok] >= q_d else (p_masked[d_tok] / q_d if q_d > 0.0 else 0.0)
                    if float(u_accept[row_i, slot]) < alpha:
                        hist.append(d_tok)
                        g.accept_token(d_tok)
                        slot += 1
                    else:
                        residual = np.maximum(0.0, p_masked - q_eff)
                        residual /= residual.sum()
                        y_res = _fast_draw_cdf(float(u_res[row_i, slot]), residual)
                        hist.append(y_res)
                        g.accept_token(y_res)
                        break

                if len(hist) < 4:
                    next_active.append(req_idx)
                else:
                    traj = (hist[0], hist[1], hist[2], hist[3])
                    if is_greedy_row[req_idx]:
                        assert traj == expected_greedy_tuple
                    else:
                        empirical_counts[traj] = empirical_counts.get(traj, 0) + 1
                        sampled_total += 1
            active = next_active

    exact_table = _exact_4step_target_table()
    assert sampled_total >= 40000
    assert all(traj in exact_table for traj in empirical_counts), "Emitted illegal 4-step trajectory!"

    tv = 0.5 * sum(
        abs((empirical_counts.get(traj, 0) / sampled_total) - p_exact)
        for traj, p_exact in exact_table.items()
    )
    assert tv < 0.01, f"4-step TV={tv:.5f} >= 0.01 for regime={regime} (sampled_total={sampled_total})"
