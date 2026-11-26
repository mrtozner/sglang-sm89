"""Eager/CUDA-Graph-isolated Coupled Conditional Resampling for SGLang DFlash2 + XGrammar."""

from __future__ import annotations

import logging
import os
import random
import threading
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class _CoupledContextTLS(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[Tuple[Any, Any, Optional[Callable[[], None]]]] = []
        self.eager_cand_ids: Optional[torch.Tensor] = None
        self.draft_ran_graph: Optional[bool] = None
        self.mask_row: Optional[torch.Tensor] = None
        self.draft_pin: Optional[torch.Tensor] = None
        self.cand_pin: Optional[torch.Tensor] = None
        self.q_pin: Optional[torch.Tensor] = None
        self.h2d_event: Optional[Any] = None


_STEP_TLS = _CoupledContextTLS()
_HOST_RNG = random.Random()
_CPP_REFINE_FN: Optional[Callable[..., Tuple[int, bool, bool]]] = None
_CPP_REFINE_CHECKED = False


def _get_cpp_refine():
    global _CPP_REFINE_CHECKED, _CPP_REFINE_FN
    if _CPP_REFINE_CHECKED:
        return _CPP_REFINE_FN
    _CPP_REFINE_CHECKED = True
    try:
        import sglang_sm89._C as _c_mod

        _CPP_REFINE_FN = getattr(_c_mod, "coupled_refine_step_cpu", None)
    except Exception:
        _CPP_REFINE_FN = None
    return _CPP_REFINE_FN


def _borrow_bitmask_row(vocab_words: int) -> torch.Tensor:
    buf = _STEP_TLS.mask_row
    if buf is None or buf.shape != (1, vocab_words):
        buf = torch.full((1, vocab_words), -1, dtype=torch.int32)
        _STEP_TLS.mask_row = buf
    return buf


def _borrow_pinned_buf(
    attr: str, shape: Tuple[int, ...], dtype: torch.dtype, *, pin: bool
) -> torch.Tensor:
    buf = getattr(_STEP_TLS, attr)
    if (
        buf is None
        or buf.dtype != dtype
        or len(buf.shape) != len(shape)
        or any(buf.shape[i] < shape[i] for i in range(len(shape)))
        or (len(shape) >= 2 and buf.shape[1:] != shape[1:])
    ):
        alloc_shape = (max(int(shape[0]), 64), *shape[1:])
        buf = torch.empty(alloc_shape, dtype=dtype, pin_memory=pin)
        setattr(_STEP_TLS, attr, buf)
    slices = tuple(slice(0, s) for s in shape)
    return buf[slices]


def _clear_padding_bits(mask_words_np: np.ndarray, vocab_size: int) -> None:
    """Zero unused trailing bits beyond vocab_size in the final int32 word."""
    n_words = (vocab_size + 31) // 32
    rem = vocab_size & 31
    if rem != 0 and mask_words_np.shape[0] >= n_words:
        u32 = mask_words_np.view(np.uint32)
        u32[n_words - 1] &= np.uint32((1 << rem) - 1)


def is_vocab_bit_set(mask_words_np: np.ndarray, token_id: int, vocab_size: int) -> bool:
    """Return True iff 0 <= token_id < vocab_size has its bit set in mask_words_np."""
    if token_id < 0 or token_id >= vocab_size:
        return False
    w_idx = token_id >> 5
    if w_idx >= mask_words_np.shape[0]:
        return False
    return bool((int(mask_words_np[w_idx]) >> (token_id & 31)) & 1)


def find_single_legal_token(mask_words_np: np.ndarray, vocab_size: int) -> int:
    """Return the sole legal token ID when |V_valid| == 1, or -1 otherwise."""
    n_words = (vocab_size + 31) // 32
    u32 = mask_words_np[:n_words].view(np.uint32)
    nz_indices = np.flatnonzero(u32)
    if nz_indices.size == 0:
        return -1

    rem = vocab_size & 31
    last_word_mask = ((1 << rem) - 1) if rem != 0 else 0xFFFFFFFF
    sole_token = -1

    for idx_np in nz_indices:
        w_idx = int(idx_np)
        w_val = int(u32[w_idx])
        if w_idx == n_words - 1:
            w_val &= last_word_mask
        if w_val == 0:
            continue
        if sole_token >= 0 or (w_val & (w_val - 1)) != 0:
            return -1
        sole_token = (w_idx << 5) + (w_val.bit_length() - 1)

    return sole_token if (0 <= sole_token < vocab_size) else -1


def find_lowest_legal_token(mask_words_np: np.ndarray, vocab_size: int) -> int:
    """Return the smallest token ID allowed by mask_words_np, or -1 if empty."""
    n_words = (vocab_size + 31) // 32
    u32 = mask_words_np[:n_words].view(np.uint32)
    rem = vocab_size & 31
    last_word_mask = ((1 << rem) - 1) if rem != 0 else 0xFFFFFFFF

    for idx_np in np.flatnonzero(u32):
        w_idx = int(idx_np)
        w_val = int(u32[w_idx])
        if w_idx == n_words - 1:
            w_val &= last_word_mask
        if w_val != 0:
            low_bit = w_val & -w_val
            tok = (w_idx << 5) + (low_bit.bit_length() - 1)
            return tok if tok < vocab_size else -1
    return -1


def _put_unique_at_front(
    cand_row: Optional[torch.Tensor],
    q_row: Optional[torch.Tensor],
    token_id: int,
) -> None:
    """Move or write token_id into slot 0 without duplicating it, and make q_row one-hot."""
    if cand_row is not None:
        hits = (cand_row == token_id).nonzero(as_tuple=False)
        if hits.numel() > 0:
            pos = int(hits[0, 0].item())
            if pos != 0:
                displaced = int(cand_row[0].item())
                cand_row[0] = token_id
                cand_row[pos] = displaced
        else:
            cand_row[0] = token_id
    if q_row is not None:
        q_row.zero_()
        q_row[0] = 1.0


def resolve_batch_greedy_flags(
    sampling_info: Any, bs: int, reqs: Optional[list[Any]] = None
) -> list[bool]:
    """Resolve per-request greedy flags with step-level caching keyed on req identities."""
    if sampling_info is None or getattr(sampling_info, "is_all_greedy", True):
        return [True] * bs

    temps = getattr(sampling_info, "temperatures", None)
    top_ks = getattr(sampling_info, "top_ks", None)
    temp_ptr = int(temps.data_ptr()) if isinstance(temps, torch.Tensor) else 0
    topk_ptr = int(top_ks.data_ptr()) if isinstance(top_ks, torch.Tensor) else 0

    cache_sig = None
    if reqs is not None:
        cache_sig = (bs, temp_ptr, topk_ptr, tuple(id(r) for r in reqs[:bs]))
        if getattr(sampling_info, "_sm89_greedy_cache_sig", None) == cache_sig:
            return sampling_info._sm89_greedy_cache_flags

    if isinstance(temps, torch.Tensor) and isinstance(top_ks, torch.Tensor):
        # Coalesce both temperature and top_k predicates into a single tiny D2H sync.
        greedy_t = (temps.view(-1)[:bs] < 1e-5) | (top_ks.view(-1)[:bs] == 1)
        flags = [bool(x) for x in greedy_t.detach().to("cpu").tolist()]
    else:
        temp_vals = (
            temps.view(-1)[:bs].detach().cpu().tolist()
            if isinstance(temps, torch.Tensor)
            else [1.0] * bs
        )
        top_k_vals = (
            top_ks.view(-1)[:bs].detach().cpu().tolist()
            if isinstance(top_ks, torch.Tensor)
            else [0] * bs
        )
        flags = [
            (float(temp_vals[i]) < 1e-5) or (int(top_k_vals[i]) == 1)
            for i in range(bs)
        ]

    if cache_sig is not None:
        sampling_info._sm89_greedy_cache_sig = cache_sig
        sampling_info._sm89_greedy_cache_flags = flags
    return flags


def _python_refine_slot(
    mask_np: np.ndarray,
    draft_row: torch.Tensor,
    cand_row: Optional[torch.Tensor],
    q_row: Optional[torch.Tensor],
    pos: int,
    vocab_size: int,
    is_greedy: bool,
    allow_forced_single: bool,
) -> Tuple[int, bool, bool]:
    cur_tok = int(draft_row[pos].item())
    d_chg = False
    q_chg = False

    forced_tok = find_single_legal_token(mask_np, vocab_size) if allow_forced_single else -1
    if forced_tok >= 0:
        if forced_tok != cur_tok:
            draft_row[pos] = forced_tok
            d_chg = True
        if cand_row is not None or q_row is not None:
            _put_unique_at_front(cand_row, q_row, forced_tok)
            if q_row is not None:
                q_chg = True
        return forced_tok, d_chg, q_chg

    if is_greedy or cand_row is None or q_row is None:
        if is_vocab_bit_set(mask_np, cur_tok, vocab_size):
            picked = cur_tok
        else:
            picked = -1
            if cand_row is not None:
                for idx in range(int(cand_row.shape[0])):
                    cand_tok = int(cand_row[idx].item())
                    if is_vocab_bit_set(mask_np, cand_tok, vocab_size):
                        picked = cand_tok
                        break
            if picked < 0:
                picked = find_lowest_legal_token(mask_np, vocab_size)
            if picked < 0:
                return -1, False, False
            draft_row[pos] = picked
            d_chg = True

        if cand_row is not None or q_row is not None:
            _put_unique_at_front(cand_row, q_row, picked)
            if q_row is not None:
                q_chg = True
        return picked, d_chg, q_chg

    k_width = int(cand_row.shape[0])
    allowed_flags = [
        is_vocab_bit_set(mask_np, int(cand_row[idx].item()), vocab_size)
        for idx in range(k_width)
    ]
    legal_t = torch.tensor(allowed_flags, dtype=torch.bool)
    q_legal = torch.where(legal_t, q_row.clamp_min(0.0), torch.zeros_like(q_row))
    z_mass = float(q_legal.sum().item())

    if z_mass > 1e-12:
        q_cond = q_legal / z_mass
        q_row.copy_(q_cond)
        q_chg = True

        cur_hits = (cand_row == cur_tok).nonzero(as_tuple=False)
        keep_existing = False
        if is_vocab_bit_set(mask_np, cur_tok, vocab_size) and cur_hits.numel() > 0:
            hit_idx = int(cur_hits[0, 0].item())
            if float(q_cond[hit_idx].item()) > 0.0:
                keep_existing = True

        if keep_existing:
            picked = cur_tok
        else:
            drawn_idx = int(torch.multinomial(q_cond, num_samples=1).item())
            picked = int(cand_row[drawn_idx].item())
            draft_row[pos] = picked
            d_chg = True
        return picked, d_chg, q_chg

    fallback_tok = find_lowest_legal_token(mask_np, vocab_size)
    if fallback_tok < 0:
        return -1, False, False
    _put_unique_at_front(cand_row, q_row, fallback_tok)
    draft_row[pos] = fallback_tok
    return fallback_tok, True, True


def apply_coupled_conditional_resampling(
    *,
    reqs: list[Any],
    draft_cpu: torch.Tensor,
    cand_cpu: Optional[torch.Tensor],
    q_cpu: Optional[torch.Tensor],
    greedy_flags: Optional[list[bool]],
    vocab_size: int,
    allow_forced_single: bool = True,
) -> Tuple[bool, bool, bool]:
    """Refine drafted positions 1..block_size-1 via Coupled Conditional Resampling.

    Returns ``(draft_changed, cand_changed, q_changed)`` so callers only upload mutated buffers.
    """
    bs, block_size = int(draft_cpu.shape[0]), int(draft_cpu.shape[1])
    max_sampled_bs = int(os.getenv("SGLANG_SM89_MAX_SAMPLED_GRAMMAR_BS", "0"))
    if max_sampled_bs > 0 and bs > max_sampled_bs and any(not g for g in (greedy_flags or ())):
        return False, False, False
    vocab_words = (vocab_size + 31) // 32
    mask_buf = _borrow_bitmask_row(vocab_words)
    mask_row = mask_buf[0]
    mask_np = mask_row.numpy()

    cpp_refine = _get_cpp_refine()
    use_cpp = (
        cpp_refine is not None
        and draft_cpu.is_contiguous()
        and (cand_cpu is None or cand_cpu.is_contiguous())
        and (q_cpu is None or q_cpu.is_contiguous())
    )
    rng_random = _HOST_RNG.random

    draft_changed = False
    cand_changed = False
    q_changed = False

    if use_cpp:
        mask_buf.fill_(-1)

    for b in range(bs):
        grammar = getattr(reqs[b], "grammar", None)
        if grammar is None or grammar.is_terminated():
            continue

        is_greedy = True if (greedy_flags is None or b >= len(greedy_flags)) else greedy_flags[b]
        draft_b = draft_cpu[b]
        cand_b = cand_cpu[b] if cand_cpu is not None else None
        q_b = q_cpu[b] if q_cpu is not None else None
        accepted = 0
        try:
            for pos in range(1, block_size):
                if grammar.is_terminated():
                    break
                slot = pos - 1
                cand_row = cand_b[slot] if cand_b is not None else None
                q_row = q_b[slot] if q_b is not None else None

                if use_cpp:
                    grammar.fill_vocab_mask(mask_buf, 0)
                    picked, d_c, q_c = cpp_refine(
                        mask_row,
                        draft_b,
                        cand_row,
                        q_row,
                        pos,
                        vocab_size,
                        is_greedy,
                        allow_forced_single,
                        rng_random(),
                        True,
                    )
                    if picked < 0:
                        break
                    if d_c:
                        draft_changed = True
                    if q_c:
                        q_changed = True
                        if cand_row is not None:
                            cand_changed = True
                else:
                    mask_buf.fill_(-1)
                    grammar.fill_vocab_mask(mask_buf, 0)
                    _clear_padding_bits(mask_np, vocab_size)
                    picked, d_c, q_c = _python_refine_slot(
                        mask_np,
                        draft_b,
                        cand_row,
                        q_row,
                        pos,
                        vocab_size,
                        is_greedy,
                        allow_forced_single,
                    )
                    if picked < 0:
                        break
                    if d_c:
                        draft_changed = True
                    if q_c:
                        q_changed = True
                        if cand_row is not None:
                            cand_changed = True

                grammar.accept_token(picked)
                accepted += 1
        finally:
            if accepted > 0:
                grammar.rollback(accepted)

    return draft_changed, cand_changed, q_changed


def _ensure_draft_runner_forward_hook(worker: Any) -> None:
    runner = getattr(worker, "draft_model_runner", None)
    if runner is None or getattr(runner, "_sm89_fwd_observed", False):
        return
    cls_fwd = getattr(type(runner), "forward", None)
    if getattr(cls_fwd, "_sm89_coupled_fwd_hooked", False):
        return
    orig_bound = getattr(runner, "forward", None)
    if orig_bound is None:
        return

    def _observed_forward(*f_args, **f_kwargs):
        res = orig_bound(*f_args, **f_kwargs)
        after_draft_runner_forward(res, runner, *f_args, **f_kwargs)
        return res

    runner.forward = _observed_forward
    runner._sm89_fwd_observed = True


def after_compute_candidates(result, *args, **kwargs):
    """Capture eager DFlash2DraftModel.compute_candidates IDs inside active step."""
    if _STEP_TLS.stack and result is not None:
        cand = result[0] if isinstance(result, (tuple, list)) else result
        if isinstance(cand, torch.Tensor):
            _STEP_TLS.eager_cand_ids = cand
    return result


def after_draft_runner_forward(result, runner_self, *args, **kwargs):
    """Record whether worker.draft_model_runner ran under CUDA Graph."""
    if _STEP_TLS.stack:
        active_worker = _STEP_TLS.stack[-1][0]
        if getattr(active_worker, "draft_model_runner", None) is runner_self:
            _STEP_TLS.draft_ran_graph = bool(getattr(result, "can_run_graph", False))
    return result


after_draft_runner_forward._sm89_coupled_fwd_hooked = True


def _reshape_eager_candidates(
    eager_cand: torch.Tensor, bs: int, num_draft_pos: int
) -> Optional[torch.Tensor]:
    if eager_cand.dim() == 3:
        if eager_cand.shape[0] >= bs and eager_cand.shape[1] >= num_draft_pos:
            return eager_cand[:bs, :num_draft_pos]
        return None
    if eager_cand.dim() == 2:
        if eager_cand.shape[0] == bs and eager_cand.shape[1] == num_draft_pos:
            return eager_cand.unsqueeze(-1)
        if eager_cand.shape[0] >= bs * num_draft_pos:
            return eager_cand[: bs * num_draft_pos].reshape(bs, num_draft_pos, -1)
        return None
    if eager_cand.dim() == 1 and eager_cand.numel() >= bs * num_draft_pos:
        return eager_cand[: bs * num_draft_pos].reshape(bs, num_draft_pos, -1)
    return None


def around_dflash_forward_batch_generation(original_fn, worker, *args, **kwargs):
    """Isolate eager vs CUDA Graph sampling buffers across each DFlash step."""
    batch = kwargs["batch"] if "batch" in kwargs else args[0]
    grammar_barrier = kwargs.get("grammar_barrier", None)
    _ensure_draft_runner_forward_hook(worker)

    saved_state = (_STEP_TLS.eager_cand_ids, _STEP_TLS.draft_ran_graph)
    _STEP_TLS.eager_cand_ids = None
    _STEP_TLS.draft_ran_graph = None
    ctx_stack = _STEP_TLS.stack
    ctx_stack.append((worker, batch, grammar_barrier))
    try:
        out = original_fn(worker, *args, **kwargs)
    finally:
        ctx_stack.pop()
        _STEP_TLS.eager_cand_ids, _STEP_TLS.draft_ran_graph = saved_state
    return out


def _extract_active_grammar_context(
    verify_ids_2d: torch.Tensor,
) -> Optional[Tuple[Any, Any, list[Any], int, int]]:
    if not _STEP_TLS.stack:
        return None
    worker, batch, grammar_barrier = _STEP_TLS.stack[-1]
    if not getattr(batch, "has_grammar", False):
        return None
    if grammar_barrier is not None:
        grammar_barrier()
    reqs = getattr(batch, "reqs", None)
    bs = len(reqs) if reqs is not None else 0
    num_draft_pos = int(verify_ids_2d.shape[1]) - 1
    if bs <= 0 or num_draft_pos <= 0:
        return None
    has_live_grammar = any(
        getattr(req, "grammar", None) is not None and not req.grammar.is_terminated()
        for req in reqs[:bs]
    )
    if not has_live_grammar:
        return None
    return worker, batch, reqs, bs, num_draft_pos


def around_grammar_tree_from_linear_chain(original_fn, cls, verify_ids_2d: torch.Tensor):
    """Refine drafted tokens & proposal distributions using coalesced pinned D2H + C++ kernel."""
    ctx = _extract_active_grammar_context(verify_ids_2d)
    if ctx is None:
        return original_fn(cls, verify_ids_2d)
    worker, batch, reqs, bs, num_draft_pos = ctx

    sampler = getattr(worker, "_draft_sampler", None)
    selector_sample = getattr(worker, "_selector_sample", None)

    cand_gpu: Optional[torch.Tensor] = None
    q_gpu: Optional[torch.Tensor] = None
    sel_cand_target: Optional[torch.Tensor] = None
    sel_q_target: Optional[torch.Tensor] = None

    if selector_sample is not None:
        sel_cand_target, sel_q_target = selector_sample[0], selector_sample[1]
        cand_gpu = sel_cand_target
        q_gpu = sel_q_target
    else:
        if _STEP_TLS.eager_cand_ids is not None:
            cand_gpu = _reshape_eager_candidates(_STEP_TLS.eager_cand_ids, bs, num_draft_pos)
        elif (
            _STEP_TLS.draft_ran_graph is True
            and sampler is not None
            and getattr(sampler, "candidate_out", None) is not None
        ):
            cand_gpu = sampler.candidate_out[:bs]

    use_sampled = selector_sample is not None and q_gpu is not None
    vocab_size = int(worker.target_worker.model_runner.model_config.vocab_size)
    greedy_flags = resolve_batch_greedy_flags(
        getattr(batch, "sampling_info", None), bs, reqs=reqs
    )
    max_sampled_bs = int(os.getenv("SGLANG_SM89_MAX_SAMPLED_GRAMMAR_BS", "0"))
    if max_sampled_bs > 0 and bs > max_sampled_bs and (
        any(not g for g in greedy_flags) or selector_sample is not None
    ):
        return original_fn(cls, verify_ids_2d)
    allow_forced = os.getenv("SGLANG_SM89_JUMP_FORWARD", "1").strip() != "0"

    if verify_ids_2d.is_cuda:
        if _STEP_TLS.h2d_event is not None and not _STEP_TLS.h2d_event.query():
            _STEP_TLS.h2d_event.synchronize()
        draft_cpu = _borrow_pinned_buf(
            "draft_pin", (bs, int(verify_ids_2d.shape[1])), verify_ids_2d.dtype, pin=True
        )
        draft_cpu.copy_(verify_ids_2d[:bs], non_blocking=True)

        cand_cpu: Optional[torch.Tensor] = None
        if cand_gpu is not None:
            cand_slice = cand_gpu[:bs]
            cand_cpu = _borrow_pinned_buf(
                "cand_pin", tuple(int(s) for s in cand_slice.shape), cand_slice.dtype, pin=True
            )
            cand_cpu.copy_(cand_slice, non_blocking=True)

        q_cpu: Optional[torch.Tensor] = None
        if use_sampled:
            q_slice = q_gpu[:bs]
            q_cpu = _borrow_pinned_buf(
                "q_pin", tuple(int(s) for s in q_slice.shape), q_slice.dtype, pin=True
            )
            q_cpu.copy_(q_slice, non_blocking=True)

        torch.cuda.current_stream(verify_ids_2d.device).synchronize()
    else:
        draft_cpu = verify_ids_2d[:bs].detach().to("cpu").clone()
        cand_cpu = cand_gpu[:bs].detach().to("cpu").clone() if cand_gpu is not None else None
        q_cpu = q_gpu[:bs].detach().to("cpu").clone() if use_sampled else None

    draft_changed, cand_changed, q_changed = apply_coupled_conditional_resampling(
        reqs=reqs[:bs],
        draft_cpu=draft_cpu,
        cand_cpu=cand_cpu,
        q_cpu=q_cpu,
        greedy_flags=greedy_flags,
        vocab_size=vocab_size,
        allow_forced_single=allow_forced,
    )

    uploaded = False
    if draft_changed:
        verify_ids_2d[:bs, 1:].copy_(draft_cpu[:, 1:], non_blocking=verify_ids_2d.is_cuda)
        uploaded = verify_ids_2d.is_cuda

    if q_changed and sel_q_target is not None and q_cpu is not None:
        sel_q_target[:bs].copy_(q_cpu, non_blocking=sel_q_target.is_cuda)
        uploaded = uploaded or sel_q_target.is_cuda
        if cand_changed and sel_cand_target is not None and cand_cpu is not None:
            sel_cand_target[:bs].copy_(cand_cpu, non_blocking=sel_cand_target.is_cuda)

    if uploaded:
        ev = _STEP_TLS.h2d_event
        if ev is None:
            ev = torch.cuda.Event()
            _STEP_TLS.h2d_event = ev
        ev.record(torch.cuda.current_stream(verify_ids_2d.device))

    return original_fn(cls, verify_ids_2d)


def register_hooks(
    HookRegistry=None, HookType=None, *, install_wrapper_on_step: bool = True
) -> None:
    if HookRegistry is None or HookType is None:
        from sglang.srt.plugins.hook_registry import HookRegistry as _HR
        from sglang.srt.plugins.hook_registry import HookType as _HT

        HookRegistry = _HR
        HookType = _HT

    specs = []
    if install_wrapper_on_step:
        specs.append(
            (
                "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.forward_batch_generation",
                around_dflash_forward_batch_generation,
                HookType.AROUND,
            )
        )
    specs.extend(
        (
            (
                "sglang.srt.models.dflash.DFlash2DraftModel.compute_candidates",
                after_compute_candidates,
                HookType.AFTER,
            ),
            (
                "sglang.srt.model_executor.model_runner.ModelRunner.forward",
                after_draft_runner_forward,
                HookType.AFTER,
            ),
            (
                "sglang.srt.speculative.dflash_worker_v2.GrammarTree.from_linear_chain",
                around_grammar_tree_from_linear_chain,
                HookType.AROUND,
            ),
        )
    )
    for target_path, handler, hook_type in specs:
        HookRegistry.register(target_path, handler, hook_type)
