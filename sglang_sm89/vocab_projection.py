"""Clean-room Ada Lovelace (sm_89) sparse vocab projection and deferred verify dispatch."""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Optional, Tuple

import numpy as np
import torch

from .coupled_grammar import _STEP_TLS

logger = logging.getLogger(__name__)

_EXT_LOAD_ATTEMPTED = False
_EXT_LOADED = False


class _DeferredVerifyTLS(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.pending: Optional[dict[str, Any]] = None


_VERIFY_TLS = _DeferredVerifyTLS()


def ensure_extension_loaded() -> bool:
    """Import compiled sglang_sm89._C CUDA extension once and verify torch.ops bindings."""
    global _EXT_LOAD_ATTEMPTED, _EXT_LOADED
    if _EXT_LOADED:
        return True
    if _EXT_LOAD_ATTEMPTED:
        return _EXT_LOADED
    _EXT_LOAD_ATTEMPTED = True
    try:
        import sglang_sm89._C  # noqa: F401

        op_ns = getattr(torch.ops, "sglang_sm89", None)
        _EXT_LOADED = (
            op_ns is not None
            and hasattr(op_ns, "sparse_vocab_proj_bf16")
            and hasattr(op_ns, "sparse_vocab_proj_fp8")
            and hasattr(op_ns, "select_sparse_vocab_proj")
        )
    except Exception:
        _EXT_LOADED = False
    return _EXT_LOADED


def _select_sparse_numpy(
    vocab_mask_cpu: torch.Tensor,
    vocab_size: int,
    k_chunk: int = 8,
    max_eff_row_ratio: float = 0.55,
) -> bool:
    """NumPy fallback matching C++ 64-row uint64 tile roofline cost model."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if k_chunk <= 0:
        raise ValueError("k_chunk must be positive")

    mask_c = vocab_mask_cpu.contiguous()
    m_rows = 1 if mask_c.dim() == 1 else int(mask_c.shape[0])
    if m_rows == 0:
        return True
    mask_2d = mask_c.view(1, -1) if mask_c.dim() == 1 else mask_c
    mask_words = int(mask_2d.shape[1])
    n_words = (vocab_size + 31) // 32
    if mask_words < n_words:
        raise ValueError(f"vocab_mask_cpu words ({mask_words}) < ceil(vocab_size/32) ({n_words})")

    n_tiles = (vocab_size + 63) // 64
    even_words = n_tiles * 2
    u32_in = mask_2d.numpy().view(np.uint32)
    if mask_words < even_words:
        padded = np.zeros((m_rows, even_words), dtype=np.uint32)
        padded[:, :mask_words] = u32_in
    else:
        padded = np.ascontiguousarray(u32_in[:, :even_words])
    tiles_u64 = padded.view(np.uint64)

    rem = vocab_size & 63
    if rem != 0:
        if mask_words >= even_words:
            tiles_u64 = tiles_u64.copy()
        tiles_u64[:, -1] &= np.uint64((1 << rem) - 1)

    tile_union = np.bitwise_or.reduce(tiles_u64, axis=0)
    active_m = np.count_nonzero(tiles_u64, axis=0)
    nz_tiles = np.flatnonzero(active_m)
    if nz_tiles.size == 0:
        return True

    budget = float(vocab_size) * float(max_eff_row_ratio)
    eff_rows = 0
    for tile_idx in nz_tiles.tolist():
        pop_u = int(tile_union[tile_idx]).bit_count()
        passes = int(math.ceil(int(active_m[tile_idx]) / k_chunk))
        eff_rows += pop_u * passes
        if float(eff_rows) > budget:
            return False
    return True


def select_sparse_vocab_proj(
    vocab_mask_cpu: torch.Tensor,
    vocab_size: int,
    k_chunk: int = 8,
    max_eff_row_ratio: float = 0.55,
    *,
    force_numpy: bool = False,
) -> bool:
    """Return True when sparse kTileV=64 vocab projection beats dense GEMM under roofline cost."""
    if not force_numpy and ensure_extension_loaded():
        return bool(
            torch.ops.sglang_sm89.select_sparse_vocab_proj(
                vocab_mask_cpu,
                int(vocab_size),
                int(k_chunk),
                float(max_eff_row_ratio),
            )
        )
    return _select_sparse_numpy(
        vocab_mask_cpu,
        int(vocab_size),
        k_chunk=int(k_chunk),
        max_eff_row_ratio=float(max_eff_row_ratio),
    )


def sparse_vocab_bf16_lm_head(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    vocab_mask: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute BF16 hidden @ weight.T for active bitmask rows only (inactive -> -inf)."""
    if not ensure_extension_loaded():
        raise RuntimeError("sglang_sm89._C extension is required for sparse_vocab_bf16_lm_head")
    if out is None:
        out = torch.empty((hidden.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=hidden.device)
    torch.ops.sglang_sm89.sparse_vocab_proj_bf16(
        hidden.contiguous(),
        weight.contiguous(),
        vocab_mask.contiguous(),
        out,
    )
    return out


def sparse_vocab_fp8_lm_head(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    vocab_mask: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute row-scaled E4M3 FP8 hidden @ weight.T for active bitmask rows only (inactive -> -inf)."""
    if not ensure_extension_loaded():
        raise RuntimeError("sglang_sm89._C extension is required for sparse_vocab_fp8_lm_head")
    if out is None:
        out = torch.empty((hidden.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=hidden.device)
    torch.ops.sglang_sm89.sparse_vocab_proj_fp8(
        hidden.contiguous(),
        weight.contiguous(),
        weight_scale.contiguous(),
        vocab_mask.contiguous(),
        out,
    )
    return out


def _quantize_fp8_lm_head(
    weight: torch.Tensor, *, chunk_rows: int = 8192
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, k = int(weight.shape[0]), int(weight.shape[1])
    device = weight.device
    w_fp8 = torch.empty((n, k), device=device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((n,), device=device, dtype=torch.float32)

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        blk = weight[start:end].float()
        sc = (blk.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
        q = (blk / sc.unsqueeze(1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        w_fp8[start:end].copy_(q)
        scales[start:end].copy_(sc)

    return w_fp8.contiguous(), scales.contiguous(), w_fp8.T, scales.view(n, 1).contiguous()


def _ensure_verify_scratch(
    lp_self: Any, *, cap_rows: int, n_vocab: int, hidden_dim: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_buf = getattr(lp_self, "_sm89_verify_logits_buf", None)
    if (
        logits_buf is None
        or logits_buf.shape[0] < cap_rows
        or logits_buf.shape[1] != n_vocab
        or logits_buf.device != device
    ):
        logits_buf = torch.empty((max(cap_rows, 64), n_vocab), dtype=torch.bfloat16, device=device)
        lp_self._sm89_verify_logits_buf = logits_buf

    hidden_buf = getattr(lp_self, "_sm89_verify_hidden_buf", None)
    if (
        hidden_buf is None
        or hidden_buf.shape[0] < cap_rows
        or hidden_buf.shape[1] != hidden_dim
        or hidden_buf.device != device
    ):
        hidden_buf = torch.empty((max(cap_rows, 64), hidden_dim), dtype=torch.bfloat16, device=device)
        lp_self._sm89_verify_hidden_buf = hidden_buf

    return logits_buf, hidden_buf


def _get_fp8_verify_bundle(lp_self: Any, weight: torch.Tensor):
    cached = getattr(lp_self, "_sm89_fp8_verify_bundle", None)
    if cached is not None:
        return cached
    bundle = _quantize_fp8_lm_head(weight)
    lp_self._sm89_fp8_verify_bundle = bundle
    return bundle


def around_target_logits_processor_forward(
    original_fn,
    lp_self,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    lm_head,
    logits_metadata,
    aux_hidden_states=None,
    hidden_states_before_norm=None,
):
    """Defer TARGET_VERIFY lm_head projection until grammar bitmask sparsity is known."""
    from sglang.srt.layers import layernorm_sp
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    meta = (
        LogitsMetadata.from_forward_batch(logits_metadata)
        if isinstance(logits_metadata, ForwardBatch)
        else logits_metadata
    )

    if (
        not meta.forward_mode.is_target_verify()
        or meta.extend_return_logprob
        or getattr(lp_self, "use_fp32_lm_head", False)
        or getattr(lp_self, "use_fp32_lm_head_output", False)
        or getattr(lp_self, "final_logit_softcapping", None) is not None
        or getattr(lm_head, "weight", None) is None
        or lm_head.weight.dtype != torch.bfloat16
        or getattr(lp_self, "do_tensor_parallel_all_gather", False)
        or getattr(lp_self, "logit_scale", None) is not None
    ):
        return original_fn(
            lp_self,
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            aux_hidden_states=aux_hidden_states,
            hidden_states_before_norm=hidden_states_before_norm,
        )

    hidden_states, hidden_states_before_norm = layernorm_sp.maybe_exit_gather(
        hidden_states=hidden_states,
        hidden_states_before_norm=hidden_states_before_norm,
        input_ids=input_ids,
        forward_mode=meta.forward_mode,
    )
    if getattr(lp_self, "expected_aux_hidden_states", None) is not None:
        lp_self._check_aux_hidden_states(aux_hidden_states, hidden_states, meta)

    (
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        _input_logprob_indices,
        _token_to_seq_idx,
    ) = lp_self._get_pruned_states(
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        meta,
    )

    hidden_states_to_store = lp_self._get_hidden_states_to_store(
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        meta,
    )
    last_hidden_states_to_store = lp_self._get_last_hidden_states_to_store(
        hidden_states, aux_hidden_states, meta
    )

    m = int(pruned_states.shape[0])
    n_vocab = int(lm_head.weight.shape[0])
    hidden_dim = int(pruned_states.shape[1])
    logits_buf = getattr(lp_self, "_sm89_verify_logits_buf", None)
    hidden_buf = getattr(lp_self, "_sm89_verify_hidden_buf", None)
    if (
        logits_buf is None
        or logits_buf.shape[0] < m
        or logits_buf.shape[1] != n_vocab
        or hidden_buf is None
        or hidden_buf.shape[0] < m
        or hidden_buf.shape[1] != hidden_dim
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("verify logits/hidden scratch buffers too small during CUDA graph capture")
        logits_buf, hidden_buf = _ensure_verify_scratch(
            lp_self,
            cap_rows=m,
            n_vocab=n_vocab,
            hidden_dim=hidden_dim,
            device=pruned_states.device,
        )

    if os.getenv("SGLANG_SM89_FP8_VERIFY_HEAD", "0").strip() == "1":
        if not torch.cuda.is_current_stream_capturing():
            _get_fp8_verify_bundle(lp_self, lm_head.weight)

    hidden_buf[:m].copy_(pruned_states)
    lp_self._sm89_verify_deferred = True
    return LogitsProcessorOutput(
        next_token_logits=logits_buf[:m],
        hidden_states=hidden_states_to_store,
        last_hidden_states=last_hidden_states_to_store,
        mm_input_embeds=meta.mm_input_embeds,
    )


def after_tp_worker_forward_batch_generation(result, tp_worker, *args, **kwargs):
    """Capture deferred verify handles right after target forward (eager or CUDA graph)."""
    if not kwargs.get("is_verify", False) or not _STEP_TLS.stack:
        return result

    dflash_worker = _STEP_TLS.stack[-1][0]
    lp = getattr(dflash_worker, "_target_logits_processor", None)
    if lp is None or not getattr(lp, "_sm89_verify_deferred", False):
        return result

    logits_output = getattr(result, "logits_output", None)
    hidden_buf = getattr(lp, "_sm89_verify_hidden_buf", None)
    if logits_output is None or hidden_buf is None or logits_output.next_token_logits is None:
        return result

    lm_head = getattr(tp_worker.model_runner.model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return result

    m = int(logits_output.next_token_logits.shape[0])
    _VERIFY_TLS.pending = {
        "lp": lp,
        "hidden": hidden_buf[:m],
        "weight": weight,
        "out": logits_output.next_token_logits,
        "vocab_mask_gpu": None,
        "use_sparse": False,
        "done": False,
    }
    return result


def around_build_grammar_vocab_mask(original_fn, *args, **kwargs):
    """Run C++ 64-row roofline check on CPU bitmask before async H2D mask transfer."""
    from sglang.srt.speculative import spec_utils as su

    pending = _VERIFY_TLS.pending
    if pending is None:
        return original_fn(*args, **kwargs)

    reqs = kwargs["reqs"]
    tree = kwargs["tree"]
    sampling_info = kwargs["sampling_info"]
    device = kwargs["device"]
    barrier = kwargs.get("barrier")

    if barrier is not None:
        barrier()
    vocab_mask_cpu, grammar = su.generate_token_bitmask(
        reqs,
        *tree.resolve(),
        sampling_info.vocab_size,
    )
    if vocab_mask_cpu is None:
        return None

    n_weight = int(pending["weight"].shape[0])
    max_ratio = float(os.getenv("SGLANG_SM89_MAX_EFF_ROW_RATIO", "0.55"))
    use_sparse = (
        int(vocab_mask_cpu.shape[1]) * 32 >= n_weight
        and select_sparse_vocab_proj(
            vocab_mask_cpu,
            n_weight,
            k_chunk=8,
            max_eff_row_ratio=max_ratio,
        )
    )

    vocab_mask_gpu = vocab_mask_cpu.to(device, non_blocking=True)
    sampling_info.grammar_mask = None
    pending["vocab_mask_gpu"] = vocab_mask_gpu
    pending["use_sparse"] = use_sparse
    return su.GrammarMask(grammar, vocab_mask_gpu)


def materialize_deferred_target_logits() -> None:
    """Populate deferred next_token_logits using sparse kTileV=64 projection or dense GEMM."""
    pending = _VERIFY_TLS.pending
    if pending is None or pending["done"]:
        return
    pending["done"] = True
    _VERIFY_TLS.pending = None

    lp = pending["lp"]
    hidden = pending["hidden"]
    weight = pending["weight"]
    out = pending["out"]
    vocab_mask_gpu = pending["vocab_mask_gpu"]
    use_sparse = bool(pending["use_sparse"])
    use_fp8 = os.getenv("SGLANG_SM89_FP8_VERIFY_HEAD", "0").strip() == "1"

    if use_fp8:
        w_fp8, w_scale, w_fp8_t, w_scale_col = _get_fp8_verify_bundle(lp, weight)
        if vocab_mask_gpu is not None and use_sparse and ensure_extension_loaded():
            sparse_vocab_fp8_lm_head(hidden, w_fp8, w_scale, vocab_mask_gpu, out=out)
            return
        from sglang.kernels.ops.gemm import fp8_scaled_mm
        from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

        h2d = hidden if hidden.is_contiguous() else hidden.contiguous()
        xq, xs = sglang_per_token_quant_fp8(h2d)
        logits = fp8_scaled_mm(xq, w_fp8_t, xs, w_scale_col, out.dtype, None)
        out.copy_(logits)
        return

    if vocab_mask_gpu is not None and use_sparse and ensure_extension_loaded():
        sparse_vocab_bf16_lm_head(hidden, weight, vocab_mask_gpu, out=out)
    else:
        torch.mm(hidden, weight.T, out=out)


def before_apply_dflash_verify_logits_adjustments(*args, **kwargs) -> None:
    materialize_deferred_target_logits()


def before_dflash_accept_block(*args, **kwargs) -> None:
    materialize_deferred_target_logits()


def clear_target_verify_tls() -> None:
    _VERIFY_TLS.pending = None


def register_hooks(HookRegistry=None, HookType=None) -> None:
    if HookRegistry is None or HookType is None:
        from sglang.srt.plugins.hook_registry import HookRegistry as _HR
        from sglang.srt.plugins.hook_registry import HookType as _HT

        HookRegistry = _HR
        HookType = _HT

    HookRegistry.register(
        "sglang.srt.layers.logits_processor.LogitsProcessor.forward",
        around_target_logits_processor_forward,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation",
        after_tp_worker_forward_batch_generation,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.speculative.spec_utils.build_grammar_vocab_mask",
        around_build_grammar_vocab_mask,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dflash_worker_v2.apply_dflash_verify_logits_adjustments",
        before_apply_dflash_verify_logits_adjustments,
        HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._accept_block",
        before_dflash_accept_block,
        HookType.BEFORE,
    )
