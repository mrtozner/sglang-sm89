"""Unified SGLang HookRegistry entry point for sglang-sm89 v0.3.0."""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Callable

logger = logging.getLogger(__name__)

_REGISTERED = False


def env_flag(*names: str, default: str = "1") -> bool:
    """Resolve the first defined environment variable in ``names`` as a boolean flag."""
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw.strip().lower() not in ("0", "false", "no", "off", "")
    return default.strip().lower() not in ("0", "false", "no", "off", "")


def register() -> None:
    """Register all enabled sm_89 SGLang runtime hooks idempotently."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    if not env_flag("SGLANG_SM89_ENABLE", default="1"):
        logger.info("[sglang_sm89] Disabled via SGLANG_SM89_ENABLE=0")
        return

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    fp8_configs_on = env_flag("SGLANG_SM89_FP8_CONFIGS", default="1")
    dflash_fp8_on = env_flag("SGLANG_SM89_DFLASH_FP8", default="1")
    replayssm_on = env_flag("SGLANG_SM89_REPLAYSSM", default="1")
    coupled_grammar_on = env_flag(
        "SGLANG_SM89_COUPLED_GRAMMAR", "SGLANG_SM89_IN_DRAFT_GRAMMAR", default="1"
    )
    sparse_proj_on = env_flag(
        "SGLANG_SM89_SPARSE_PROJ", "SGLANG_SM89_ACTIVE_ROW_LMHEAD", default="1"
    )
    wy_gdn_on = env_flag("SGLANG_SM89_WY_GDN", default="1")
    sam_graft_on = env_flag("SGLANG_SM89_SAM_GRAFT", default="0")

    if fp8_configs_on:
        from .autotune_fp8 import ensure_sm89_fp8_configs_installed

        ensure_sm89_fp8_configs_installed()

    if dflash_fp8_on:
        from .draft_fp8 import after_dflash_worker_init, around_compute_candidates

        for target_path, cb, hook_kind in (
            (
                "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.__init__",
                partial(after_dflash_worker_init, quantize_draft=True, quantize_head=True),
                HookType.AFTER,
            ),
            (
                "sglang.srt.models.dflash.DFlash2DraftModel.compute_candidates",
                around_compute_candidates,
                HookType.AROUND,
            ),
        ):
            HookRegistry.register(target_path, cb, hook_kind)

    if replayssm_on:
        from .gdn_pool_bridge import register_hooks as register_gdn_bridge_hooks

        register_gdn_bridge_hooks(HookRegistry, HookType)

    ext_loaded = False
    try:
        from .vocab_projection import ensure_extension_loaded

        ext_loaded = bool(ensure_extension_loaded())
    except ImportError:
        pass
    except Exception:
        logger.debug("[sglang_sm89] Native extension _C load deferred", exc_info=True)

    if coupled_grammar_on or sparse_proj_on:
        from .coupled_grammar import around_dflash_forward_batch_generation
        from .vocab_projection import clear_target_verify_tls

        def _dflash_step_guard(original_fn: Callable, worker, *args, **kwargs):
            try:
                return around_dflash_forward_batch_generation(
                    original_fn, worker, *args, **kwargs
                )
            finally:
                clear_target_verify_tls()

        HookRegistry.register(
            "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.forward_batch_generation",
            _dflash_step_guard,
            HookType.AROUND,
        )

    if coupled_grammar_on:
        from .coupled_grammar import register_hooks as register_grammar_hooks

        register_grammar_hooks(HookRegistry, HookType, install_wrapper_on_step=False)

    if sparse_proj_on:
        from .vocab_projection import register_hooks as register_vocab_hooks

        register_vocab_hooks(HookRegistry, HookType)

    if wy_gdn_on:
        from .wy_gdn_verify import register_hooks as register_wy_gdn_hooks

        register_wy_gdn_hooks(HookRegistry, HookType)

    if sam_graft_on:
        try:
            from . import sam_drafter

            reg_fn = getattr(sam_drafter, "register_hooks", None)
            if callable(reg_fn):
                reg_fn(HookRegistry, HookType)
        except ImportError:
            pass

    logger.info(
        "[sglang_sm89] Registered v0.3.0 hooks (fp8_configs=%s dflash_fp8=%s replayssm=%s "
        "coupled_grammar=%s sparse_proj=%s wy_gdn=%s sam_graft=%s ext_loaded=%s)",
        fp8_configs_on,
        dflash_fp8_on,
        replayssm_on,
        coupled_grammar_on,
        sparse_proj_on,
        wy_gdn_on,
        sam_graft_on,
        ext_loaded,
    )
