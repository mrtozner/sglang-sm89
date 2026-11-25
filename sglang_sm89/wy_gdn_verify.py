"""Native sm_89 CUDA WY GDN ReplaySSM verify kernel & SGLang 0.5.23 hooks."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from contextvars import ContextVar
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)

_EXT_READY: Optional[bool] = None
_IN_REPLAYSSM_VERIFY: ContextVar[bool] = ContextVar(
    "sglang_sm89_in_replayssm_verify", default=False
)
_PATCHED_GDN_BACKEND_GLOBALS = False
_SIG_CACHE: dict[Any, inspect.Signature] = {}

_GDN_SPEC_MODULES = (
    "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode",
    "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode",
    "sglang.srt.layers.attention.linear.gdn_replayssm_spec_decode",
)


def _max_bf16_tokens() -> int:
    # Default 32: routes c=1..4 (<=32 tokens) through wy_gdn_sm89 (1.04x-1.65x faster)
    # and routes BF16 c=8 (>32 tokens) through Triton to guarantee >=1.00x everywhere.
    raw = os.getenv("SGLANG_SM89_WY_GDN_MAX_BF16_TOKENS", "32").strip()
    try:
        return int(raw)
    except ValueError:
        return 32


def _force_wy_cuda(force_cuda: bool) -> bool:
    if force_cuda:
        return True
    return os.getenv("SGLANG_SM89_WY_GDN_FORCE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _wy_extension_ready() -> bool:
    global _EXT_READY
    if _EXT_READY is not None:
        return _EXT_READY
    try:
        from . import _C  # noqa: F401

        ns = getattr(torch.ops, "sglang_sm89", None)
        _EXT_READY = (
            ns is not None
            and getattr(ns, "wy_gdn_replayssm_verify", None) is not None
            and getattr(ns, "wy_gdn_replayssm_flush_fold", None) is not None
        )
    except Exception as exc:
        logger.debug("[sglang_sm89] _C extension unavailable for wy_gdn_verify: %s", exc)
        _EXT_READY = False
    return _EXT_READY


def _can_use_wy_gdn_fast_path(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    rawv_cache: Optional[torch.Tensor],
    rawk_cache: Optional[torch.Tensor],
    beta_cache: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    out: Optional[torch.Tensor],
    max_cache_len: int,
    max_spec_len: int,
    launch_mode: str,
    use_qk_l2norm_in_kernel: bool,
    null_block_id: int,
    force_cuda: bool = False,
    has_fallback: bool = False,
) -> bool:
    """Check strict specialization invariants without triggering host-device syncs."""
    if launch_mode not in ("verify", "both", "flush") or not use_qk_l2norm_in_kernel or null_block_id != -1:
        return False
    if (
        not _force_wy_cuda(force_cuda)
        and has_fallback
        and checkpoint_state.dtype == torch.bfloat16
        and int(q.shape[0]) > _max_bf16_tokens()
    ):
        return False
    if q.device.type != "cuda":
        return False
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return False
    if q.shape[2] != 128 or v.shape[2] != 128 or k.shape != q.shape:
        return False
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        return False
    B = int(query_start_loc.shape[0]) - 1
    H = int(q.shape[1])
    HV = int(v.shape[1])
    if B <= 0 or H <= 0 or HV < H or (HV % H) != 0:
        return False
    if not (1 <= int(max_spec_len) <= 16):
        return False
    if not (1 <= int(max_cache_len) <= 64) or (int(max_cache_len) & (int(max_cache_len) - 1)) != 0:
        return False

    supported_act = (torch.bfloat16, torch.float16, torch.float32)
    act_dtype = q.dtype
    if act_dtype not in supported_act:
        return False
    if (
        k.dtype != act_dtype
        or v.dtype != act_dtype
        or a.dtype not in supported_act
        or b.dtype not in supported_act
        or d_cache.dtype != act_dtype
        or k_cache.dtype != act_dtype
    ):
        return False
    if out is not None and (out.dtype != act_dtype or out.device != q.device):
        return False

    state_dtype = checkpoint_state.dtype
    if act_dtype == torch.bfloat16 and state_dtype not in (torch.bfloat16, torch.float32):
        return False
    if act_dtype == torch.float16 and state_dtype not in (torch.float16, torch.float32):
        return False
    if act_dtype == torch.float32 and state_dtype != torch.float32:
        return False

    if A_log.dtype != torch.float32 or g_cache.dtype != torch.float32:
        return False
    if dt_bias.dtype not in supported_act:
        return False
    if rawv_cache is not None and rawv_cache.dtype != d_cache.dtype:
        return False
    if rawk_cache is not None and rawk_cache.dtype != k_cache.dtype:
        return False
    if beta_cache is not None and beta_cache.dtype != torch.float32:
        return False

    idx_dtypes = (torch.int32, torch.int64)
    if (
        query_start_loc.dtype not in idx_dtypes
        or ssm_state_indices.dtype not in idx_dtypes
        or replay_indices.dtype not in idx_dtypes
        or write_pos.dtype not in idx_dtypes
        or cache_base.dtype not in idx_dtypes
    ):
        return False
    if is_flush.dtype not in (torch.int8, torch.uint8, torch.bool):
        return False

    # Contiguous inner dimensions required by coalesced 128-wide loads/stores.
    if q.stride(2) != 1 or q.stride(1) != 128:
        return False
    if k.stride(2) != 1 or k.stride(1) != 128:
        return False
    if v.stride(2) != 1 or v.stride(1) != 128:
        return False
    if out is not None and (out.stride(2) != 1 or out.stride(1) != 128):
        return False
    if a.stride(1) != 1 or b.stride(1) != 1:
        return False

    return _wy_extension_ready()


def _resolve_upstream_gdn_replayssm():
    for mod_path in _GDN_SPEC_MODULES:
        try:
            mod = __import__(mod_path, fromlist=["gdn_replayssm_spec_decode"])
            fn = getattr(mod, "gdn_replayssm_spec_decode", None)
            if fn is not None:
                return mod, fn
        except ImportError:
            continue
    return None, None


def wy_gdn_replayssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    rawv_cache: Optional[torch.Tensor] = None,
    rawk_cache: Optional[torch.Tensor] = None,
    beta_cache: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    replay_indices: Optional[torch.Tensor] = None,
    write_pos: Optional[torch.Tensor] = None,
    cache_base: Optional[torch.Tensor] = None,
    is_flush: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    max_cache_len: int = 64,
    max_spec_len: int = 16,
    launch_mode: str = "verify",
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = -1,
    softplus_threshold: float = 20.0,
    fallback_fn: Optional[Callable[..., Any]] = None,
    force_cuda: bool = False,
    **extra_kwargs: Any,
) -> torch.Tensor:
    """Accelerated sm_89 CUDA ReplaySSM verify with automatic upstream fallback."""
    upstream_fn = None if fallback_fn is not None else _resolve_upstream_gdn_replayssm()[1]
    has_fallback = (fallback_fn is not None) or (upstream_fn is not None)
    if (
        query_start_loc is None
        or ssm_state_indices is None
        or replay_indices is None
        or write_pos is None
        or cache_base is None
        or is_flush is None
        or not _can_use_wy_gdn_fast_path(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            checkpoint_state=checkpoint_state,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            rawv_cache=rawv_cache,
            rawk_cache=rawk_cache,
            beta_cache=beta_cache,
            query_start_loc=query_start_loc,
            ssm_state_indices=ssm_state_indices,
            replay_indices=replay_indices,
            write_pos=write_pos,
            cache_base=cache_base,
            is_flush=is_flush,
            out=out,
            max_cache_len=int(max_cache_len),
            max_spec_len=int(max_spec_len),
            launch_mode=str(launch_mode),
            use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
            null_block_id=int(null_block_id),
            force_cuda=bool(force_cuda),
            has_fallback=has_fallback,
        )
    ):
        if fallback_fn is not None:
            return fallback_fn()
        if upstream_fn is not None:
            upstream_kwargs: dict[str, Any] = dict(
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                checkpoint_state=checkpoint_state,
                d_cache=d_cache,
                k_cache=k_cache,
                g_cache=g_cache,
                rawv_cache=rawv_cache,
                rawk_cache=rawk_cache,
                beta_cache=beta_cache,
                query_start_loc=query_start_loc,
                ssm_state_indices=ssm_state_indices,
                replay_indices=replay_indices,
                write_pos=write_pos,
                cache_base=cache_base,
                is_flush=is_flush,
                out=out,
                scale=scale,
                max_cache_len=max_cache_len,
                max_spec_len=max_spec_len,
                launch_mode=launch_mode,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                null_block_id=null_block_id,
                **extra_kwargs,
            )
            sig = _SIG_CACHE.get(upstream_fn)
            if sig is None:
                try:
                    sig = inspect.signature(upstream_fn)
                    _SIG_CACHE[upstream_fn] = sig
                except Exception:
                    sig = None
            if sig is not None and "softplus_threshold" in sig.parameters:
                upstream_kwargs["softplus_threshold"] = softplus_threshold
            return upstream_fn(**upstream_kwargs)
        raise RuntimeError(
            "wy_gdn_replayssm_verify fast path unsupported for given inputs and no fallback available"
        )

    if out is None:
        out = torch.empty(
            (int(v.shape[0]), int(v.shape[1]), int(v.shape[2])),
            dtype=v.dtype,
            device=v.device,
        )

    eff_scale = float(scale) if scale is not None else (float(q.shape[-1]) ** -0.5)
    eff_max_cache = int(max_cache_len)
    eff_max_spec = int(max_spec_len)
    eff_sp_thresh = float(softplus_threshold)
    store_exact_raw = (
        rawv_cache is not None and rawk_cache is not None and beta_cache is not None
    )

    if launch_mode in ("both", "flush"):
        torch.ops.sglang_sm89.wy_gdn_replayssm_flush_fold(
            checkpoint_state,
            d_cache,
            k_cache,
            g_cache,
            rawv_cache,
            rawk_cache,
            ssm_state_indices,
            replay_indices,
            write_pos,
            cache_base,
            is_flush,
            eff_max_cache,
            bool(store_exact_raw),
            beta_cache if store_exact_raw else None,
        )

    if launch_mode in ("both", "verify"):
        # Single kernel launch on launch_mode="verify" (matches SGLang GDNAttnBackend contract)
        torch.ops.sglang_sm89.wy_gdn_replayssm_verify(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            checkpoint_state,
            d_cache,
            k_cache,
            g_cache,
            rawv_cache,
            rawk_cache,
            beta_cache,
            query_start_loc,
            ssm_state_indices,
            replay_indices,
            write_pos,
            cache_base,
            is_flush,
            out,
            eff_scale,
            eff_max_cache,
            eff_max_spec,
            False,
            eff_sp_thresh,
        )

    if launch_mode in ("both", "flush"):
        torch.ops.sglang_sm89.wy_gdn_replayssm_verify(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            checkpoint_state,
            d_cache,
            k_cache,
            g_cache,
            rawv_cache,
            rawk_cache,
            beta_cache,
            query_start_loc,
            ssm_state_indices,
            replay_indices,
            write_pos,
            cache_base,
            is_flush,
            out,
            eff_scale,
            eff_max_cache,
            eff_max_spec,
            True,
            eff_sp_thresh,
        )
    return out


def _bind_gdn_spec_args(original_fn, args, kwargs) -> Optional[dict[str, Any]]:
    if not args:
        return dict(kwargs)
    try:
        sig = _SIG_CACHE.get(original_fn)
        if sig is None:
            sig = inspect.signature(original_fn)
            _SIG_CACHE[original_fn] = sig
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        return None


def around_gdn_replayssm_spec_decode(original_fn, *args, **kwargs):
    """Intercept gdn_replayssm_spec_decode(..., launch_mode='verify') for sm_89 CUDA."""
    bound_kwargs = _bind_gdn_spec_args(original_fn, args, kwargs)
    if bound_kwargs is None:
        return original_fn(*args, **kwargs)

    launch_mode = bound_kwargs.get("launch_mode")
    if launch_mode is None and _IN_REPLAYSSM_VERIFY.get():
        launch_mode = "verify"
        bound_kwargs["launch_mode"] = "verify"
    if launch_mode != "verify":
        return original_fn(*args, **kwargs)

    return wy_gdn_replayssm_verify(
        **bound_kwargs,
        fallback_fn=lambda: original_fn(*args, **kwargs),
    )


def around_replayssm_target_verify(original_fn, backend, *args, **kwargs):
    """Hook GDNLinearAttnBackend._replayssm_target_verify to route verify through sm_89 WY CUDA."""
    global _PATCHED_GDN_BACKEND_GLOBALS
    if not _PATCHED_GDN_BACKEND_GLOBALS:
        _PATCHED_GDN_BACKEND_GLOBALS = True
        mod = sys.modules.get(getattr(original_fn, "__module__", ""))
        if mod is not None:
            orig_spec = getattr(mod, "gdn_replayssm_spec_decode", None)
            if orig_spec is not None and not getattr(
                orig_spec, "_sglang_sm89_wrapped", False
            ):
                def _wrapped_gdn_spec(*sp_args, **sp_kwargs):
                    return around_gdn_replayssm_spec_decode(
                        orig_spec, *sp_args, **sp_kwargs
                    )

                _wrapped_gdn_spec._sglang_sm89_wrapped = True
                setattr(mod, "gdn_replayssm_spec_decode", _wrapped_gdn_spec)

    token = _IN_REPLAYSSM_VERIFY.set(True)
    try:
        return original_fn(backend, *args, **kwargs)
    finally:
        _IN_REPLAYSSM_VERIFY.reset(token)


def register_hooks(HookRegistry, HookType) -> None:
    """Register SGLang 0.5.23 GDN ReplaySSM verify hooks."""
    backend_targets: list[str] = []
    try:
        gdn_backend_mod = importlib.import_module(
            "sglang.srt.layers.attention.linear.gdn_backend"
        )
        for cls_name in ("GDNAttnBackend", "GDNLinearAttnBackend"):
            cls = getattr(gdn_backend_mod, cls_name, None)
            if cls is not None and hasattr(cls, "_replayssm_target_verify"):
                backend_targets.append(
                    f"sglang.srt.layers.attention.linear.gdn_backend.{cls_name}._replayssm_target_verify"
                )
    except ImportError:
        pass

    if not backend_targets:
        backend_targets = [
            "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend._replayssm_target_verify",
            "sglang.srt.layers.attention.linear.gdn_backend.GDNLinearAttnBackend._replayssm_target_verify",
        ]

    for target in backend_targets:
        try:
            HookRegistry.register(
                target,
                around_replayssm_target_verify,
                HookType.AROUND,
            )
        except Exception:
            logger.debug("[sglang_sm89] Failed registering %s", target, exc_info=True)

    spec_targets: list[str] = []
    for mod_path in _GDN_SPEC_MODULES:
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, "gdn_replayssm_spec_decode"):
                spec_targets.append(f"{mod_path}.gdn_replayssm_spec_decode")
        except ImportError:
            continue

    if not spec_targets:
        spec_targets = [f"{m}.gdn_replayssm_spec_decode" for m in _GDN_SPEC_MODULES]

    for target in spec_targets:
        try:
            HookRegistry.register(
                target,
                around_gdn_replayssm_spec_decode,
                HookType.AROUND,
            )
        except Exception:
            logger.debug("[sglang_sm89] Failed registering %s", target, exc_info=True)
