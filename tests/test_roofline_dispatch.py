"""CPU/C++ tests for the kTileV=64 uint64 sparse vocab projection roofline selector."""

from __future__ import annotations

import math
from types import SimpleNamespace

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


def test_get_logits_deferral_state_machine_graph_prefill_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_logits arms on DFlash graph capture and survives interleaved eager prefill."""
    from types import SimpleNamespace
    import sglang_sm89.vocab_projection as vp
    from sglang_sm89.coupled_grammar import _STEP_TLS
    from sglang_sm89.vocab_projection import (
        around_capture_decode_graph,
        around_target_get_logits,
        around_tp_worker_forward_batch_generation,
        before_apply_dflash_verify_logits_adjustments,
        clear_target_verify_tls,
    )

    class _Mode:
        def __init__(self, verify: bool) -> None:
            self._v = verify

        def is_target_verify(self) -> bool:
            return self._v

    class LogitsProcessor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_fp32_lm_head = False
            self.use_fp32_lm_head_output = False
            self.final_logit_softcapping = None
            self.logit_scale = None
            self.do_tensor_parallel_all_gather = False
            self.tp_size = 1

    torch.manual_seed(17)
    weight = torch.randn((16, 8), dtype=torch.bfloat16)
    lm_head = SimpleNamespace(weight=weight)
    lp = LogitsProcessor()
    model = torch.nn.Module()
    model.lp = lp
    model.lm_head = lm_head
    model_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
        model=model,
        model_config=SimpleNamespace(vocab_size=16, hidden_size=8),
        num_tokens_per_req=4,
    )
    ext_buf = torch.zeros((8, 16), dtype=torch.bfloat16)

    def _orig_get_logits(_self, h, lm_h, _meta, embedding_bias=None, use_logits_buffer=True):
        return h @ lm_h.weight.T

    # Step 1: DFlash capture arms deferral cleanly
    h_verify = torch.randn((4, 8), dtype=torch.bfloat16)
    meta_verify = SimpleNamespace(
        forward_mode=_Mode(True),
        extend_return_logprob=False,
        next_token_logits_buffer=ext_buf,
    )

    def _cap_fn(_mr):
        monkeypatch.setattr(vp, "_is_cuda_capturing", lambda _dev: True)
        try:
            out_v = around_target_get_logits(_orig_get_logits, lp, h_verify, lm_head, meta_verify)
            assert out_v.data_ptr() == ext_buf[:4].data_ptr()
        finally:
            monkeypatch.setattr(vp, "_is_cuda_capturing", lambda _dev: False)

    around_capture_decode_graph(_cap_fn, model_runner)
    assert getattr(lp, "_sm89_verify_armed", False) is True

    # Step 2: Interleaved eager prefill (not verify) runs original_fn and must NOT disarm verify
    h_prefill = torch.randn((2, 8), dtype=torch.bfloat16)
    meta_prefill = SimpleNamespace(
        forward_mode=_Mode(False),
        extend_return_logprob=False,
        next_token_logits_buffer=None,
    )
    out_p = around_target_get_logits(_orig_get_logits, lp, h_prefill, lm_head, meta_prefill)
    assert torch.allclose(out_p, h_prefill @ weight.T)
    assert getattr(lp, "_sm89_verify_armed", False) is True

    # Step 3: Verify CUDA Graph replay executes without re-running _get_logits Python
    tp_w = SimpleNamespace(model_runner=model_runner)
    dflash_w = SimpleNamespace(_target_worker=tp_w, _target_logits_processor=lp)
    _STEP_TLS.stack.append((dflash_w, SimpleNamespace(), None))
    try:
        def _replay_forward(_tp_w, **_kw):
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=ext_buf[:4]))

        around_tp_worker_forward_batch_generation(_replay_forward, tp_w, is_verify=True)
        before_apply_dflash_verify_logits_adjustments()
        assert torch.allclose(ext_buf[:4], h_verify @ weight.T)
    finally:
        _STEP_TLS.stack.pop()
        clear_target_verify_tls()


def _make_23_fixture(n_vocab: int = 64, hidden_dim: int = 8):
    class _Mode:
        def __init__(self, verify: bool) -> None:
            self._v = verify

        def is_target_verify(self) -> bool:
            return self._v

    class LogitsProcessor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_fp32_lm_head = False
            self.use_fp32_lm_head_output = False
            self.final_logit_softcapping = None
            self.logit_scale = None
            self.do_tensor_parallel_all_gather = False
            self.tp_size = 1

    weight = torch.zeros((n_vocab, hidden_dim), dtype=torch.bfloat16)
    weight[0, 0] = 2.0
    weight[1, 1] = 3.0
    hidden = torch.zeros((2, hidden_dim), dtype=torch.bfloat16)
    hidden[:, 0] = 1.0
    hidden[:, 1] = 1.0
    ext_buf = torch.full((4, n_vocab), 123.0, dtype=torch.bfloat16)
    lm_head = SimpleNamespace(weight=weight)
    lp = LogitsProcessor()
    model = torch.nn.Module()
    model.lp = lp
    model.lm_head = lm_head
    meta = SimpleNamespace(
        forward_mode=_Mode(True),
        extend_return_logprob=False,
        next_token_logits_buffer=ext_buf,
    )

    def _orig_get_logits(_self, h, lm_h, _meta, embedding_bias=None, use_logits_buffer=True):
        return h @ lm_h.weight.T

    return lp, model, lm_head, weight, hidden, ext_buf, meta, _orig_get_logits


def test_codex_p1_direct_target_verify_outside_dflash_computes_logits() -> None:
    """Direct LogitsProcessor._get_logits TARGET_VERIFY call outside DFlash must compute [2, 3]."""
    from sglang_sm89.vocab_projection import (
        _VERIFY_TLS,
        around_target_get_logits,
        clear_target_verify_tls,
    )

    clear_target_verify_tls()
    lp, _model, lm_head, _w, hidden, ext_buf, meta, orig_fn = _make_23_fixture()
    out = around_target_get_logits(orig_fn, lp, hidden, lm_head, meta)

    assert torch.equal(out[:, :2], torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.bfloat16))
    assert not torch.equal(out[:, :2], torch.full((2, 2), 123.0, dtype=torch.bfloat16))
    assert not hasattr(lp, "_sm89_verify_armed")
    assert not hasattr(lp, "_sm89_eager_verify_bypassed")
    assert _VERIFY_TLS.pending is None


def test_codex_p1_eagle_capture_and_runtime_target_verify_computes_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EAGLE (is_dflash_family()==False) capture and runtime TARGET_VERIFY must compute [2, 3]."""
    from types import SimpleNamespace
    import sglang_sm89.vocab_projection as vp
    from sglang_sm89.vocab_projection import (
        _VERIFY_TLS,
        around_capture_decode_graph,
        around_target_get_logits,
        around_tp_worker_forward_batch_generation,
        clear_target_verify_tls,
    )

    clear_target_verify_tls()
    lp, model, lm_head, _w, hidden, ext_buf, meta, orig_fn = _make_23_fixture()
    eagle_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_dflash_family=lambda: False),
        model=model,
        model_config=SimpleNamespace(vocab_size=64, hidden_size=8),
        num_tokens_per_req=4,
    )
    eagle_tp = SimpleNamespace(model_runner=eagle_runner)

    def _eagle_capture(_mr):
        monkeypatch.setattr(vp, "_is_cuda_capturing", lambda _dev: True)
        try:
            out_cap = around_target_get_logits(orig_fn, lp, hidden, lm_head, meta)
            assert torch.equal(out_cap[:, :2], torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.bfloat16))
        finally:
            monkeypatch.setattr(vp, "_is_cuda_capturing", lambda _dev: False)

    around_capture_decode_graph(_eagle_capture, eagle_runner)
    assert not hasattr(lp, "_sm89_verify_hidden_arena")
    assert not hasattr(lp, "_sm89_verify_armed")

    def _eagle_forward(_tp, **_kw):
        out_run = around_target_get_logits(orig_fn, lp, hidden, lm_head, meta)
        return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=out_run))

    res = around_tp_worker_forward_batch_generation(_eagle_forward, eagle_tp, is_verify=True)
    assert torch.equal(
        res.logits_output.next_token_logits[:, :2],
        torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.bfloat16),
    )
    assert torch.equal(ext_buf[:, :2], torch.full((4, 2), 123.0, dtype=torch.bfloat16))
    assert _VERIFY_TLS.pending is None
    assert _VERIFY_TLS.active_dflash_lp is None


def test_codex_p1_dflash_draft_worker_is_verify_computes_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DFlash draft_worker (tp_worker is not target_worker) with is_verify=True must compute [2, 3]."""
    from types import SimpleNamespace
    import sglang_sm89.vocab_projection as vp
    from sglang_sm89.coupled_grammar import _STEP_TLS
    from sglang_sm89.vocab_projection import (
        _VERIFY_TLS,
        around_capture_decode_graph,
        around_target_get_logits,
        around_tp_worker_forward_batch_generation,
        clear_target_verify_tls,
    )

    clear_target_verify_tls()
    target_lp, target_model, _, _, _, _, _, _ = _make_23_fixture()
    draft_lp, draft_model, draft_lm_head, _w, hidden, ext_buf, meta, orig_fn = _make_23_fixture()

    draft_runner = SimpleNamespace(
        is_draft_worker=True,
        spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
        model=draft_model,
        model_config=SimpleNamespace(vocab_size=64, hidden_size=8),
        num_tokens_per_req=4,
    )

    def _draft_capture(_mr):
        monkeypatch.setattr(vp, "_is_cuda_capturing", lambda _dev: True)
        try:
            out_cap = around_target_get_logits(orig_fn, draft_lp, hidden, draft_lm_head, meta)
            assert torch.equal(out_cap[:, :2], torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.bfloat16))
        finally:
            monkeypatch.setattr(vp, "_is_cuda_capturing", lambda _dev: False)

    around_capture_decode_graph(_draft_capture, draft_runner)
    assert not hasattr(draft_lp, "_sm89_verify_hidden_arena")
    assert not hasattr(draft_lp, "_sm89_verify_armed")

    target_tp = SimpleNamespace(
        model_runner=SimpleNamespace(model=target_model)
    )
    draft_tp = SimpleNamespace(model_runner=draft_runner)
    dflash_w = SimpleNamespace(_target_worker=target_tp, _target_logits_processor=target_lp)

    _STEP_TLS.stack.append((dflash_w, SimpleNamespace(), None))
    try:
        def _draft_verify_fwd(_tp, **_kw):
            out_run = around_target_get_logits(orig_fn, draft_lp, hidden, draft_lm_head, meta)
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=out_run))

        res = around_tp_worker_forward_batch_generation(_draft_verify_fwd, draft_tp, is_verify=True)
        assert torch.equal(
            res.logits_output.next_token_logits[:, :2],
            torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.bfloat16),
        )
        assert torch.equal(ext_buf[:, :2], torch.full((4, 2), 123.0, dtype=torch.bfloat16))
        assert not hasattr(draft_lp, "_sm89_verify_armed")
        assert _VERIFY_TLS.pending is None
    finally:
        _STEP_TLS.stack.pop()
        clear_target_verify_tls()


def test_valid_dflash_capture_and_verify_eager_and_graph_dense_and_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid DFlash capture + eager/graph TARGET_VERIFY defers and materializes [2, 3] (dense & sparse)."""
    import sys
    import types
    from types import SimpleNamespace
    import sglang_sm89.vocab_projection as vp
    from sglang_sm89.coupled_grammar import _STEP_TLS
    from sglang_sm89.vocab_projection import (
        _VERIFY_TLS,
        around_build_grammar_vocab_mask,
        around_capture_decode_graph,
        around_target_get_logits,
        around_tp_worker_forward_batch_generation,
        before_apply_dflash_verify_logits_adjustments,
        before_dflash_accept_block,
        clear_target_verify_tls,
        register_hooks,
    )

    clear_target_verify_tls()
    lp, model, lm_head, _w, hidden, ext_buf, meta, orig_fn = _make_23_fixture()
    model_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
        model=model,
        model_config=SimpleNamespace(vocab_size=64, hidden_size=8),
        num_tokens_per_req=4,
    )
    target_tp = SimpleNamespace(model_runner=model_runner)
    dflash_w = SimpleNamespace(_target_worker=target_tp, _target_logits_processor=lp)

    # 1. Valid DFlash decode graph capture allocates >=64 rows and defers during capture.
    def _valid_capture(_mr):
        monkeypatch.setattr(vp, "_is_cuda_capturing", lambda _dev: True)
        try:
            out_cap = around_target_get_logits(orig_fn, lp, hidden, lm_head, meta)
            assert out_cap.data_ptr() == ext_buf[:2].data_ptr()
            assert torch.equal(out_cap[:, :2], torch.full((2, 2), 123.0, dtype=torch.bfloat16))
        finally:
            monkeypatch.setattr(vp, "_is_cuda_capturing", lambda _dev: False)

    around_capture_decode_graph(_valid_capture, model_runner)
    assert lp._sm89_verify_hidden_arena.shape[0] >= 64
    assert lp._sm89_verify_logits_arena.shape[0] >= 64
    assert lp._sm89_verify_armed is True
    assert _VERIFY_TLS.capturing_dflash_lp is None

    # 2. CUDA Graph replay + Dense materialization -> [2, 3].
    _STEP_TLS.stack.append((dflash_w, SimpleNamespace(), None))
    try:
        ext_buf.fill_(123.0)

        def _graph_replay(_tp, **_kw):
            lp._sm89_verify_hidden_arena[:2].copy_(hidden)
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=ext_buf[:2]))

        around_tp_worker_forward_batch_generation(_graph_replay, target_tp, is_verify=True)
        assert _VERIFY_TLS.pending is not None
        assert torch.equal(ext_buf[:2, :2], torch.full((2, 2), 123.0, dtype=torch.bfloat16))
        before_apply_dflash_verify_logits_adjustments()
        assert torch.equal(
            ext_buf[:2, :2], torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.bfloat16)
        )
    finally:
        _STEP_TLS.stack.pop()
        clear_target_verify_tls()

    # 3. Eager TARGET_VERIFY + Sparse materialization (both real GPU kernel if present & CPU mock).
    mask_cpu = torch.zeros((2, 2), dtype=torch.int32)
    mask_cpu[:, 0] = 0b11  # tokens 0 and 1 active
    spec_pkg = types.ModuleType("sglang.srt.speculative")
    spec_utils_mod = types.ModuleType("sglang.srt.speculative.spec_utils")
    spec_utils_mod.generate_token_bitmask = lambda _reqs, _t0, _t1, _v: (mask_cpu, "stub_grammar")
    spec_utils_mod.GrammarMask = lambda g, m: SimpleNamespace(grammar=g, mask=m)
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative", spec_pkg)
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative.spec_utils", spec_utils_mod)

    sparse_calls: list[bool] = []
    monkeypatch.setattr(vp, "ensure_extension_loaded", lambda: True)
    monkeypatch.setattr(vp, "select_sparse_vocab_proj", lambda *_a, **_k: True)

    def _mock_sparse_bf16(h, w, m, out=None):
        sparse_calls.append(True)
        torch.mm(h, w.T, out=out)
        active = ((m[:, 0].unsqueeze(1) >> torch.arange(64, device=m.device)) & 1).bool()
        out.masked_fill_(~active, float("-inf"))
        return out

    monkeypatch.setattr(vp, "sparse_vocab_bf16_lm_head", _mock_sparse_bf16)

    _STEP_TLS.stack.append((dflash_w, SimpleNamespace(), None))
    try:
        ext_buf.fill_(123.0)

        def _eager_verify(_tp, **_kw):
            out_e = around_target_get_logits(orig_fn, lp, hidden, lm_head, meta)
            assert torch.equal(out_e[:, :2], torch.full((2, 2), 123.0, dtype=torch.bfloat16))
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=out_e))

        around_tp_worker_forward_batch_generation(_eager_verify, target_tp, is_verify=True)
        gm = around_build_grammar_vocab_mask(
            lambda *_a, **_k: None,
            reqs=[],
            tree=SimpleNamespace(resolve=lambda: (None, None)),
            sampling_info=SimpleNamespace(vocab_size=64, grammar_mask="stale"),
            device=torch.device("cpu"),
        )
        assert gm.grammar == "stub_grammar"
        assert _VERIFY_TLS.pending is not None and _VERIFY_TLS.pending["use_sparse"] is True
        before_dflash_accept_block()
        assert len(sparse_calls) == 1
        assert torch.equal(
            ext_buf[:2, :2], torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.bfloat16)
        )
        assert torch.all(torch.isneginf(ext_buf[:2, 2:]))
    finally:
        _STEP_TLS.stack.pop()
        clear_target_verify_tls()

    # 4. Verify register_hooks registers capture_decode_graph + TpModelWorker AROUND hooks and
    # DFlashWorkerV2 guard does not double-push _STEP_TLS.stack when wrapped twice.
    registered: list[tuple[str, object, str]] = []

    class _FakeHR:
        @staticmethod
        def register(path, fn, htype):
            registered.append((path, fn, htype))

    class _FakeHT:
        AROUND = "AROUND"
        BEFORE = "BEFORE"
        AFTER = "AFTER"

    register_hooks(_FakeHR, _FakeHT)
    by_path = {p: (fn, ht) for p, fn, ht in registered}
    assert by_path[
        "sglang.srt.model_executor.model_runner_components.cuda_graph_setup.capture_decode_graph"
    ][1] == "AROUND"
    assert by_path["sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation"][1] == "AROUND"
    step_fn = by_path[
        "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.forward_batch_generation"
    ][0]
    from sglang_sm89.coupled_grammar import around_dflash_forward_batch_generation

    depths: list[int] = []
    around_dflash_forward_batch_generation(
        lambda w, _b: step_fn(lambda _w, _bb: depths.append(len(_STEP_TLS.stack)), w, _b),
        dflash_w,
        SimpleNamespace(),
    )
    assert depths == [1]


def test_max_sampled_grammar_bs_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SGLANG_SM89_MAX_SAMPLED_GRAMMAR_BS skips sampled bs>cap before fill_vocab_mask."""
    from types import SimpleNamespace
    from sglang_sm89.coupled_grammar import (
        _STEP_TLS,
        apply_coupled_conditional_resampling,
        around_grammar_tree_from_linear_chain,
    )

    class _CountingGrammar:
        def __init__(self) -> None:
            self.fill_calls = 0

        def is_terminated(self) -> bool:
            return False

        def fill_vocab_mask(self, bitmask: torch.Tensor, row_idx: int) -> None:
            self.fill_calls += 1
            bitmask[row_idx].zero_()
            bitmask[row_idx, 0] = 0b10  # token 1 only

        def accept_token(self, _tok: int) -> None:
            pass

        def rollback(self, _n: int) -> None:
            pass

    bs = 8
    grammars = [_CountingGrammar() for _ in range(bs)]
    reqs = [SimpleNamespace(grammar=g) for g in grammars]
    draft = torch.full((bs, 4), 2, dtype=torch.int64)
    cand = torch.tensor([[[1, 2]]], dtype=torch.int64).repeat(bs, 3, 1)
    q = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32).repeat(bs, 3, 1)

    # Cap at 4: sampled batch bs=8 skips cleanly before fill_vocab_mask.
    monkeypatch.setenv("SGLANG_SM89_MAX_SAMPLED_GRAMMAR_BS", "4")
    d_c, c_c, q_c = apply_coupled_conditional_resampling(
        reqs=reqs,
        draft_cpu=draft.clone(),
        cand_cpu=cand.clone(),
        q_cpu=q.clone(),
        greedy_flags=[False] * bs,
        vocab_size=64,
    )
    assert (d_c, c_c, q_c) == (False, False, False)
    assert sum(g.fill_calls for g in grammars) == 0

    # Also verify hook bails out before D2H copy / fill_vocab_mask when sampled bs > cap.
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=64))),
        _draft_sampler=None,
        _selector_sample=(cand.clone(), q.clone()),
    )
    batch_sampled = SimpleNamespace(
        has_grammar=True,
        reqs=reqs,
        sampling_info=SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.ones(bs),
            top_ks=torch.full((bs,), 2, dtype=torch.int32),
        ),
    )
    verify_ids = draft.clone()
    _STEP_TLS.stack.append((worker, batch_sampled, None))
    try:
        around_grammar_tree_from_linear_chain(lambda _c, v: v, object, verify_ids)
    finally:
        _STEP_TLS.stack.pop()
    assert sum(g.fill_calls for g in grammars) == 0
    assert torch.equal(verify_ids, draft)

    # All-greedy batch at bs=8 still runs even with SGLANG_SM89_MAX_SAMPLED_GRAMMAR_BS=4.
    d_g, _c_g, _q_g = apply_coupled_conditional_resampling(
        reqs=reqs,
        draft_cpu=draft.clone(),
        cand_cpu=None,
        q_cpu=None,
        greedy_flags=[True] * bs,
        vocab_size=64,
    )
    assert d_g is True
    assert sum(g.fill_calls for g in grammars) > 0

    # Default "0" (unlimited) runs bs=8 sampled cleanly.
    monkeypatch.delenv("SGLANG_SM89_MAX_SAMPLED_GRAMMAR_BS", raising=False)
    for g in grammars:
        g.fill_calls = 0
    d_u, _c_u, q_u = apply_coupled_conditional_resampling(
        reqs=reqs,
        draft_cpu=draft.clone(),
        cand_cpu=cand.clone(),
        q_cpu=q.clone(),
        greedy_flags=[False] * bs,
        vocab_size=64,
    )
    assert d_u is True and q_u is True
    assert sum(g.fill_calls for g in grammars) > 0
