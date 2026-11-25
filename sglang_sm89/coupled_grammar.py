"""Eager/CUDA-Graph-isolated Coupled Conditional Resampling for SGLang DFlash2 + XGrammar."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class _CoupledStepTLS(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[Tuple[Any, Any, Optional[Callable[[], None]]]] = []
        self.eager_cand_ids: Optional[torch.Tensor] = None
        self.draft_ran_graph: Optional[bool] = None


_STEP_TLS = _CoupledStepTLS()
_SCRATCH_TLS = threading.local()


def _borrow_bitmask_row(vocab_words: int) -> torch.Tensor:
    row = getattr(_SCRATCH_TLS, "mask_row", None)
    if row is None or row.shape != (1, vocab_words):
        row = torch.full((1, vocab_words), -1, dtype=torch.int32)
        _SCRATCH_TLS.mask_row = row
    return row


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


def resolve_batch_greedy_flags(sampling_info: Any, bs: int) -> list[bool]:
    """Resolve per-request greedy status (temp < 1e-5 or top_k == 1) for a batch."""
    if sampling_info is None or getattr(sampling_info, "is_all_greedy", True):
        return [True] * bs

    temps = getattr(sampling_info, "temperatures", None)
    top_ks = getattr(sampling_info, "top_ks", None)
    temp_vals = (
        temps.view(-1)[:bs].detach().cpu().tolist()
        if temps is not None
        else [1.0] * bs
    )
    top_k_vals = (
        top_ks.view(-1)[:bs].detach().cpu().tolist()
        if top_ks is not None
        else [0] * bs
    )
    return [
        (float(temp_vals[i]) < 1e-5) or (int(top_k_vals[i]) == 1)
        for i in range(bs)
    ]


def apply_coupled_conditional_resampling(
    *,
    reqs: list[Any],
    draft_cpu: torch.Tensor,
    cand_cpu: Optional[torch.Tensor],
    q_cpu: Optional[torch.Tensor],
    greedy_flags: Optional[list[bool]],
    vocab_size: int,
    allow_forced_single: bool = True,
) -> Tuple[bool, bool]:
    """Refine speculative draft tokens 1..block_size-1 using coupled conditional resampling.

    Preserves exact constrained draft distributions on sampled requests (TV = 0), ensures
    zero duplicate candidate IDs, and rolls back stateful grammars cleanly.
    """
    bs, block_size = int(draft_cpu.shape[0]), int(draft_cpu.shape[1])
    vocab_words = (vocab_size + 31) // 32
    mask_buf = _borrow_bitmask_row(vocab_words)
    mask_np = mask_buf[0].numpy()

    draft_changed = False
    q_changed = False

    for b in range(bs):
        grammar = getattr(reqs[b], "grammar", None)
        if grammar is None or grammar.is_terminated():
            continue

        is_greedy = True if (greedy_flags is None or b >= len(greedy_flags)) else greedy_flags[b]
        accepted = 0
        try:
            for pos in range(1, block_size):
                if grammar.is_terminated():
                    break
                mask_buf.fill_(-1)
                grammar.fill_vocab_mask(mask_buf, 0)
                _clear_padding_bits(mask_np, vocab_size)

                slot = pos - 1
                cur_tok = int(draft_cpu[b, pos].item())
                cand_row = cand_cpu[b, slot] if cand_cpu is not None else None
                q_row = q_cpu[b, slot] if q_cpu is not None else None

                forced_tok = (
                    find_single_legal_token(mask_np, vocab_size)
                    if allow_forced_single
                    else -1
                )
                if forced_tok >= 0:
                    if forced_tok != cur_tok:
                        draft_cpu[b, pos] = forced_tok
                        draft_changed = True
                    if cand_row is not None or q_row is not None:
                        _put_unique_at_front(cand_row, q_row, forced_tok)
                        if q_row is not None:
                            q_changed = True
                    grammar.accept_token(forced_tok)
                    accepted += 1
                    continue

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
                            break
                        draft_cpu[b, pos] = picked
                        draft_changed = True

                    if cand_row is not None or q_row is not None:
                        _put_unique_at_front(cand_row, q_row, picked)
                        if q_row is not None:
                            q_changed = True
                    grammar.accept_token(picked)
                    accepted += 1
                    continue

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
                    q_changed = True

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
                        draft_cpu[b, pos] = picked
                        draft_changed = True
                else:
                    fallback_tok = find_lowest_legal_token(mask_np, vocab_size)
                    if fallback_tok < 0:
                        break
                    _put_unique_at_front(cand_row, q_row, fallback_tok)
                    draft_cpu[b, pos] = fallback_tok
                    draft_changed = True
                    q_changed = True
                    picked = fallback_tok

                grammar.accept_token(picked)
                accepted += 1
        finally:
            if accepted > 0:
                grammar.rollback(accepted)

    return draft_changed, q_changed


def _ensure_draft_runner_forward_hook(worker: Any) -> None:
    """Attach an idempotent instance forward observer for worker.draft_model_runner if unhooked."""
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
    """Capture eager DFlash2DraftModel.compute_candidates output IDs inside active steps."""
    if _STEP_TLS.stack and result is not None:
        cand = result[0] if isinstance(result, (tuple, list)) else result
        if isinstance(cand, torch.Tensor):
            _STEP_TLS.eager_cand_ids = cand
    return result


def after_draft_runner_forward(result, runner_self, *args, **kwargs):
    """Record whether worker.draft_model_runner executed via CUDA Graph for this step."""
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
    """Track step context and isolate eager vs CUDA Graph sampling buffers."""
    batch = kwargs["batch"] if "batch" in kwargs else args[0]
    grammar_barrier = kwargs.get("grammar_barrier", None)
    _ensure_draft_runner_forward_hook(worker)

    prev_eager = _STEP_TLS.eager_cand_ids
    prev_graph = _STEP_TLS.draft_ran_graph
    _STEP_TLS.eager_cand_ids = None
    _STEP_TLS.draft_ran_graph = None
    _STEP_TLS.stack.append((worker, batch, grammar_barrier))
    try:
        return original_fn(worker, *args, **kwargs)
    finally:
        _STEP_TLS.stack.pop()
        _STEP_TLS.eager_cand_ids = prev_eager
        _STEP_TLS.draft_ran_graph = prev_graph


def around_grammar_tree_from_linear_chain(original_fn, cls, verify_ids_2d: torch.Tensor):
    """Refine drafted verify_ids_2d (and live proposal distributions) before target verify."""
    if not _STEP_TLS.stack:
        return original_fn(cls, verify_ids_2d)

    worker, batch, grammar_barrier = _STEP_TLS.stack[-1]
    if not getattr(batch, "has_grammar", False):
        return original_fn(cls, verify_ids_2d)

    if grammar_barrier is not None:
        grammar_barrier()

    reqs = getattr(batch, "reqs", None)
    bs = len(reqs) if reqs is not None else 0
    num_draft_pos = int(verify_ids_2d.shape[1]) - 1
    if bs <= 0 or num_draft_pos <= 0:
        return original_fn(cls, verify_ids_2d)

    if not any(
        (g := getattr(r, "grammar", None)) is not None and not g.is_terminated()
        for r in reqs[:bs]
    ):
        return original_fn(cls, verify_ids_2d)

    sampler = getattr(worker, "_draft_sampler", None)
    selector_sample = getattr(worker, "_selector_sample", None)

    cand_gpu: Optional[torch.Tensor] = None
    q_gpu: Optional[torch.Tensor] = None
    sel_cand_target: Optional[torch.Tensor] = None
    sel_q_target: Optional[torch.Tensor] = None

    if selector_sample is not None:
        # Rule A: sampled & mixed batches always read/write worker._selector_sample exclusively.
        sel_cand_target, sel_q_target = selector_sample[0], selector_sample[1]
        cand_gpu = sel_cand_target
        q_gpu = sel_q_target
    else:
        # Rule B: all-greedy batches never touch stale sampler buffers on eager steps.
        if _STEP_TLS.eager_cand_ids is not None:
            cand_gpu = _reshape_eager_candidates(_STEP_TLS.eager_cand_ids, bs, num_draft_pos)
        elif (
            _STEP_TLS.draft_ran_graph is True
            and sampler is not None
            and getattr(sampler, "candidate_out", None) is not None
        ):
            cand_gpu = sampler.candidate_out[:bs]

    draft_cpu = verify_ids_2d[:bs].detach().to("cpu", non_blocking=False).clone()
    cand_cpu = (
        cand_gpu[:bs].detach().to("cpu", non_blocking=False).clone()
        if cand_gpu is not None
        else None
    )
    use_sampled = selector_sample is not None and q_gpu is not None
    q_cpu = (
        q_gpu[:bs].detach().to("cpu", non_blocking=False).clone()
        if use_sampled
        else None
    )

    vocab_size = int(worker.target_worker.model_runner.model_config.vocab_size)
    greedy_flags = resolve_batch_greedy_flags(getattr(batch, "sampling_info", None), bs)
    allow_forced = os.getenv("SGLANG_SM89_JUMP_FORWARD", "1").strip() != "0"

    draft_changed, q_changed = apply_coupled_conditional_resampling(
        reqs=reqs[:bs],
        draft_cpu=draft_cpu,
        cand_cpu=cand_cpu,
        q_cpu=q_cpu,
        greedy_flags=greedy_flags,
        vocab_size=vocab_size,
        allow_forced_single=allow_forced,
    )

    if draft_changed:
        verify_ids_2d[:bs, 1:].copy_(draft_cpu[:, 1:].to(verify_ids_2d.device))

    if q_changed and sel_cand_target is not None and sel_q_target is not None:
        if cand_cpu is not None:
            sel_cand_target[:bs].copy_(cand_cpu.to(sel_cand_target.device))
        if q_cpu is not None:
            sel_q_target[:bs].copy_(q_cpu.to(sel_q_target.device))

    return original_fn(cls, verify_ids_2d)


def register_hooks(HookRegistry=None, HookType=None, *, install_wrapper_on_step: bool = True) -> None:
    if HookRegistry is None or HookType is None:
        from sglang.srt.plugins.hook_registry import HookRegistry as _HR
        from sglang.srt.plugins.hook_registry import HookType as _HT

        HookRegistry = _HR
        HookType = _HT

    if install_wrapper_on_step:
        HookRegistry.register(
            "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.forward_batch_generation",
            around_dflash_forward_batch_generation,
            HookType.AROUND,
        )
    HookRegistry.register(
        "sglang.srt.models.dflash.DFlash2DraftModel.compute_candidates",
        after_compute_candidates,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.forward",
        after_draft_runner_forward,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dflash_worker_v2.GrammarTree.from_linear_chain",
        around_grammar_tree_from_linear_chain,
        HookType.AROUND,
    )
