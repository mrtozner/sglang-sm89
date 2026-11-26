"""Clean-room GDN ReplaySSM bridge for DFlash2/DSpark on SGLang 0.5.23."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def around_build_hybrid_req_pool(original_fn: Any, configurator: Any, *args: Any, **kwargs: Any):
    """Enable ReplaySSM pool allocation for DFLASH/DSPARK GDN models via sentinel guard.

    SGLang's KVCacheConfigurator._build_hybrid_req_pool checks only
    ``self.hybrid_kda_config is None`` and ``self.hybrid_kda_config is not None``
    when gating speculative ReplaySSM pools. Temporarily attaching a sentinel
    object inside ``try ... finally`` lets upstream build the exact pool without
    duplicating constructor arguments.
    """
    from sglang.srt.mem_cache.kv_cache_configurator import get_exec, get_spec

    spec_cfg = get_spec()
    algo = (spec_cfg.speculative_algorithm or "").upper()
    should_bridge = (
        get_exec().mamba.enable_linear_replayssm_spec
        and algo in ("DFLASH", "DSPARK")
        and getattr(configurator, "hybrid_kda_config", None) is None
        and getattr(configurator, "hybrid_gdn_config", None) is not None
    )
    if not should_bridge:
        return original_fn(configurator, *args, **kwargs)

    configurator.hybrid_kda_config = object()
    try:
        return original_fn(configurator, *args, **kwargs)
    finally:
        configurator.hybrid_kda_config = None


def _resolve_mamba_batch_indices(
    linear_backend: Any,
    req_pool: Any,
    request_number: int,
    req_pool_indices: Optional[torch.Tensor],
    mamba_track_indices: Optional[torch.Tensor],
):
    from sglang.srt.speculative.spec_utils import pp_spec_stable_rows_enabled

    if pp_spec_stable_rows_enabled() and req_pool_indices is not None:
        state_indices_tensor = linear_backend._translate_mamba_indices(
            req_pool.get_mamba_indices(req_pool_indices[:request_number])
        )
    else:
        state_indices_tensor = linear_backend.forward_metadata.mamba_cache_indices[
            :request_number
        ]

    track_indices_t = (
        linear_backend._translate_mamba_indices(mamba_track_indices)
        if mamba_track_indices is not None
        else None
    )
    replay_indices = (
        req_pool_indices[:request_number] if req_pool_indices is not None else None
    )
    return state_indices_tensor, track_indices_t, replay_indices


def around_update_mamba_state_after_mtp_verify(
    original_fn: Any,
    backend: Any,
    last_correct_step_indices: torch.Tensor,
    mamba_track_indices: Optional[torch.Tensor],
    mamba_steps_to_track: Optional[torch.Tensor],
    model: Any,
    req_pool_indices: Optional[torch.Tensor] = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Route HybridLinearAttnBackend GDN MTP state updates through ReplaySSM kernels."""
    linear_backend = getattr(backend, "linear_attn_backend", None)
    req_pool = getattr(linear_backend, "req_to_token_pool", None)
    mamba_pool = getattr(req_pool, "mamba_pool", None) if req_pool is not None else None

    is_kda = bool(getattr(mamba_pool, "replayssm_is_kda", False)) if mamba_pool is not None else False
    has_fold = bool(getattr(mamba_pool, "replayssm_spec_fold", False)) if mamba_pool is not None else False
    has_base = (getattr(mamba_pool, "replayssm_cache_base", None) is not None) if mamba_pool is not None else False
    is_gdn_replayssm = (mamba_pool is not None) and (not is_kda) and (has_fold or has_base)
    if not is_gdn_replayssm:
        return original_fn(
            backend,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
            model,
            req_pool_indices,
            *args,
            **kwargs,
        )

    spec_state = req_pool.get_speculative_mamba2_params_all_layers()
    request_number = int(last_correct_step_indices.shape[0])
    state_indices_tensor, track_indices_t, replay_indices = _resolve_mamba_batch_indices(
        linear_backend, req_pool, request_number, req_pool_indices, mamba_track_indices
    )
    accept_lens = last_correct_step_indices + 1

    if getattr(mamba_pool, "replayssm_spec_fold", False):
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold import (
            commit_gdn_replayssm_fold_after_verify,
        )

        commit_gdn_replayssm_fold_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_indices_tensor,
            accept_lens=accept_lens,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=track_indices_t,
            mamba_steps_to_track=mamba_steps_to_track,
            src_indices_raw=None,
            null_block_id=-1,
        )
        return

    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        commit_gdn_replayssm_circular,
        commit_gdn_replayssm_spec,
    )
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import get_spec

    max_spec_len = int(
        getattr(req_pool, "speculative_num_draft_tokens", None)
        or getattr(get_spec(), "speculative_num_draft_tokens", None)
        or spec_state.intermediate_conv_window[0].shape[1]
    )
    max_cache_len = int(
        getattr(mamba_pool, "linear_replayssm_cache_len", None)
        or spec_state.replayssm_d.shape[-2]
    )

    commit_gdn_replayssm_spec(
        mamba_pool.replayssm_spec_write_pos,
        mamba_pool.replayssm_cache_base,
        mamba_pool.replayssm_is_flush,
        accept_lens,
        replay_indices,
        max_cache_len,
        max_spec_len,
        fold_every_commit=(spec_state.temporal.dtype != torch.float32),
        cache_buf_len=None,
        null_block_id=-1,
    )

    commit_gdn_replayssm_circular(
        checkpoint_state=spec_state.temporal,
        d_cache=spec_state.replayssm_d,
        k_cache=spec_state.replayssm_k,
        g_cache=spec_state.replayssm_g,
        d_residual_cache=getattr(spec_state, "replayssm_rawv", None),
        k_residual_cache=getattr(spec_state, "replayssm_rawk", None),
        state_batch_indices=state_indices_tensor,
        replay_indices=replay_indices,
        write_pos=mamba_pool.replayssm_spec_write_pos,
        cache_base=mamba_pool.replayssm_cache_base,
        is_flush=mamba_pool.replayssm_is_flush,
        accept_lens=accept_lens,
        mamba_track_indices=track_indices_t,
        mamba_steps_to_track=mamba_steps_to_track,
        null_block_id=-1,
    )

    fused_conv_window_scatter_with_mask(
        spec_state.conv[0],
        spec_state.intermediate_conv_window[0],
        state_indices_tensor,
        last_correct_step_indices,
    )
    if track_indices_t is not None and mamba_steps_to_track is not None:
        fused_conv_window_scatter_with_mask(
            spec_state.conv[0],
            spec_state.intermediate_conv_window[0],
            track_indices_t,
            mamba_steps_to_track,
        )


def register_hooks(HookRegistry: Any, HookType: Any) -> None:
    """Register clean-room GDN ReplaySSM bridge hooks into SGLang HookRegistry."""
    targets = (
        (
            "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_hybrid_req_pool",
            around_build_hybrid_req_pool,
        ),
        (
            "sglang.srt.layers.attention.hybrid_linear_attn_backend.HybridLinearAttnBackend.update_mamba_state_after_mtp_verify",
            around_update_mamba_state_after_mtp_verify,
        ),
    )
    for target_name, hook_fn in targets:
        HookRegistry.register(target_name, hook_fn, HookType.AROUND)
