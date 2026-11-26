"""Clean-room Ada Lovelace (sm_89) sparse vocab projection via LogitsProcessor._get_logits."""

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


class _VerifyTicketTLS(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.pending: Optional[dict[str, Any]] = None


_VERIFY_TLS = _VerifyTicketTLS()


def ensure_extension_loaded() -> bool:
    """Import compiled sglang_sm89._C extension once and verify torch.ops bindings."""
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


def _prepare_sparse_out(hidden: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor]) -> torch.Tensor:
    if not ensure_extension_loaded():
        raise RuntimeError("sglang_sm89._C CUDA extension must be built for sparse_vocab_proj")
    if out is not None:
        return out
    return torch.empty((hidden.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=hidden.device)


def sparse_vocab_bf16_lm_head(
    hidden: torch.Tensor, weight: torch.Tensor, vocab_mask: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute BF16 hidden @ weight.T for active bitmask rows only (inactive -> -inf)."""
    target = _prepare_sparse_out(hidden, weight, out)
    torch.ops.sglang_sm89.sparse_vocab_proj_bf16(
        hidden.contiguous(), weight.contiguous(), vocab_mask.contiguous(), target
    )
    return target


def sparse_vocab_fp8_lm_head(
    hidden: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, vocab_mask: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute row-scaled E4M3 FP8 hidden @ weight.T for active bitmask rows only (inactive -> -inf)."""
    target = _prepare_sparse_out(hidden, weight, out)
    torch.ops.sglang_sm89.sparse_vocab_proj_fp8(
        hidden.contiguous(), weight.contiguous(), weight_scale.contiguous(), vocab_mask.contiguous(), target
    )
    return target


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


def _is_cuda_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _borrow_arena_slice(
    lp_self: Any,
    attr: str,
    rows: int,
    cols: int,
    device: torch.device,
) -> torch.Tensor:
    arena = getattr(lp_self, attr, None)
    if (
        arena is None
        or arena.shape[0] < rows
        or arena.shape[1] != cols
        or arena.device != device
    ):
        if _is_cuda_capturing(device):
            raise RuntimeError(f"Missing pre-allocated scratch arena {attr} during CUDA graph capture")
        arena = torch.empty((max(rows, 64), cols), dtype=torch.bfloat16, device=device)
        setattr(lp_self, attr, arena)
    return arena[:rows]


def _get_fp8_verify_bundle(lp_self: Any, weight: torch.Tensor):
    cached = getattr(lp_self, "_sm89_fp8_verify_bundle", None)
    if cached is not None:
        return cached
    bundle = _quantize_fp8_lm_head(weight)
    lp_self._sm89_fp8_verify_bundle = bundle
    return bundle


def around_target_get_logits(
    original_fn,
    lp_self,
    hidden_states: torch.Tensor,
    lm_head,
    logits_metadata,
    embedding_bias: Optional[torch.Tensor] = None,
    use_logits_buffer: bool = True,
) -> torch.Tensor:
    """Defer TARGET_VERIFY projection inside LogitsProcessor._get_logits until mask builds."""
    fmode = getattr(logits_metadata, "forward_mode", None)
    if fmode is None or not fmode.is_target_verify():
        return original_fn(
            lp_self,
            hidden_states,
            lm_head,
            logits_metadata,
            embedding_bias=embedding_bias,
            use_logits_buffer=use_logits_buffer,
        )

    weight = getattr(lm_head, "weight", None)
    can_defer = (
        not getattr(logits_metadata, "extend_return_logprob", False)
        and embedding_bias is None
        and not getattr(lp_self, "use_fp32_lm_head", False)
        and not getattr(lp_self, "use_fp32_lm_head_output", False)
        and getattr(lp_self, "final_logit_softcapping", None) is None
        and getattr(lp_self, "logit_scale", None) is None
        and not getattr(lp_self, "do_tensor_parallel_all_gather", False)
        and int(getattr(lp_self, "tp_size", 1)) == 1
        and weight is not None
        and weight.dtype == torch.bfloat16
    )
    if not can_defer:
        lp_self._sm89_eager_verify_bypassed = True
        return original_fn(
            lp_self,
            hidden_states,
            lm_head,
            logits_metadata,
            embedding_bias=embedding_bias,
            use_logits_buffer=use_logits_buffer,
        )

    rows, hidden_dim = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    n_vocab = int(weight.shape[0])
    device = hidden_states.device

    hidden_slot = _borrow_arena_slice(lp_self, "_sm89_verify_hidden_arena", rows, hidden_dim, device)
    ext_buf = (
        getattr(logits_metadata, "next_token_logits_buffer", None)
        if use_logits_buffer
        else None
    )
    if (
        ext_buf is not None
        and ext_buf.shape[0] >= rows
        and ext_buf.shape[1] == n_vocab
        and ext_buf.dtype == torch.bfloat16
        and ext_buf.device == device
    ):
        logits_slot = ext_buf[:rows]
    else:
        logits_slot = _borrow_arena_slice(lp_self, "_sm89_verify_logits_arena", rows, n_vocab, device)

    if os.getenv("SGLANG_SM89_FP8_VERIFY_HEAD", "0").strip() == "1":
        if not _is_cuda_capturing(device):
            _get_fp8_verify_bundle(lp_self, weight)

    hidden_slot.copy_(hidden_states)
    lp_self._sm89_verify_armed = True
    lp_self._sm89_eager_verify_bypassed = False
    return logits_slot


def after_tp_worker_forward_batch_generation(result, tp_worker, *args, **kwargs):
    """Store deferred TARGET_VERIFY work ticket once target forward completes."""
    if not (_STEP_TLS.stack and kwargs.get("is_verify", False)):
        return result

    active_worker = _STEP_TLS.stack[-1][0]
    lp = getattr(active_worker, "_target_logits_processor", None)
    if lp is None or not getattr(lp, "_sm89_verify_armed", False):
        return result
    if getattr(lp, "_sm89_eager_verify_bypassed", False):
        lp._sm89_eager_verify_bypassed = False
        return result

    logits_output = getattr(result, "logits_output", None)
    hidden_arena = getattr(lp, "_sm89_verify_hidden_arena", None)
    out_tensor = getattr(logits_output, "next_token_logits", None)
    if hidden_arena is None or out_tensor is None:
        return result

    lm_head = getattr(tp_worker.model_runner.model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return result

    m_rows = int(out_tensor.shape[0])
    _VERIFY_TLS.pending = {
        "lp": lp,
        "hidden": hidden_arena[:m_rows],
        "weight": weight,
        "out": out_tensor,
        "vocab_mask_gpu": None,
        "use_sparse": False,
        "done": False,
    }
    return result


def around_build_grammar_vocab_mask(original_fn, *args, **kwargs):
    """Run C++ 64-row roofline check on CPU bitmask before non-blocking H2D mask upload."""
    from sglang.srt.speculative import spec_utils as su

    ticket = _VERIFY_TLS.pending
    if ticket is None:
        return original_fn(*args, **kwargs)

    barrier = kwargs.get("barrier")
    if barrier is not None:
        barrier()

    sampling_info = kwargs["sampling_info"]
    vocab_mask_cpu, grammar = su.generate_token_bitmask(
        kwargs["reqs"],
        *kwargs["tree"].resolve(),
        sampling_info.vocab_size,
    )
    if vocab_mask_cpu is None:
        return None

    n_weight = int(ticket["weight"].shape[0])
    ratio_limit = float(os.getenv("SGLANG_SM89_MAX_EFF_ROW_RATIO", "0.55"))
    use_sparse = int(vocab_mask_cpu.shape[1]) * 32 >= n_weight and select_sparse_vocab_proj(
        vocab_mask_cpu,
        n_weight,
        k_chunk=8,
        max_eff_row_ratio=ratio_limit,
    )

    vocab_mask_gpu = vocab_mask_cpu.to(kwargs["device"], non_blocking=True)
    sampling_info.grammar_mask = None
    ticket["vocab_mask_gpu"] = vocab_mask_gpu
    ticket["use_sparse"] = use_sparse
    return su.GrammarMask(grammar, vocab_mask_gpu)


def materialize_deferred_target_logits() -> None:
    """Execute deferred sparse kTileV=64 projection or dense fallback into next_token_logits."""
    ticket = _VERIFY_TLS.pending
    if ticket is None or ticket["done"]:
        return
    ticket["done"] = True
    _VERIFY_TLS.pending = None

    lp = ticket["lp"]
    hidden = ticket["hidden"]
    weight = ticket["weight"]
    out = ticket["out"]
    vocab_mask_gpu = ticket["vocab_mask_gpu"]
    use_sparse = bool(ticket["use_sparse"])
    use_fp8 = os.getenv("SGLANG_SM89_FP8_VERIFY_HEAD", "0").strip() == "1"

    if use_fp8:
        w_fp8, w_scale, w_fp8_t, w_scale_col = _get_fp8_verify_bundle(lp, weight)
        if vocab_mask_gpu is not None and use_sparse and ensure_extension_loaded():
            sparse_vocab_fp8_lm_head(hidden, w_fp8, w_scale, vocab_mask_gpu, out=out)
            return
        from sglang.kernels.ops.gemm import fp8_scaled_mm
        from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

        h_cont = hidden if hidden.is_contiguous() else hidden.contiguous()
        xq, xs = sglang_per_token_quant_fp8(h_cont)
        logits = fp8_scaled_mm(xq, w_fp8_t, xs, w_scale_col, out.dtype, None)
        out.copy_(logits)
        return

    if vocab_mask_gpu is not None and use_sparse and ensure_extension_loaded():
        sparse_vocab_bf16_lm_head(hidden, weight, vocab_mask_gpu, out=out)
    else:
        torch.mm(hidden, weight.T, out=out)


def clear_target_verify_tls() -> None:
    _VERIFY_TLS.pending = None


def before_apply_dflash_verify_logits_adjustments(*_args, **_kwargs) -> None:
    materialize_deferred_target_logits()


def before_dflash_accept_block(*_args, **_kwargs) -> None:
    materialize_deferred_target_logits()


def register_hooks(HookRegistry=None, HookType=None) -> None:
    if HookRegistry is None or HookType is None:
        from sglang.srt.plugins.hook_registry import HookRegistry as _HR
        from sglang.srt.plugins.hook_registry import HookType as _HT

        HookRegistry = _HR
        HookType = _HT

    hook_specs = (
        (
            "sglang.srt.layers.logits_processor.LogitsProcessor._get_logits",
            around_target_get_logits,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation",
            after_tp_worker_forward_batch_generation,
            HookType.AFTER,
        ),
        (
            "sglang.srt.speculative.spec_utils.build_grammar_vocab_mask",
            around_build_grammar_vocab_mask,
            HookType.AROUND,
        ),
        (
            "sglang.srt.speculative.dflash_worker_v2.apply_dflash_verify_logits_adjustments",
            before_apply_dflash_verify_logits_adjustments,
            HookType.BEFORE,
        ),
        (
            "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._accept_block",
            before_dflash_accept_block,
            HookType.BEFORE,
        ),
    )
    for target_path, fn, htype in hook_specs:
        HookRegistry.register(target_path, fn, htype)
