"""Clean-room rowwise FP8 linear and draft-only LM head quantization for DFlash2."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

_FP8_E4M3_MAX = 448.0


def quantize_weight_rowwise_fp8(
    weight_2d: torch.Tensor, *, stripe_rows: int = 4096
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize [out_dim, in_dim] weights into column-major FP8 (in_dim, out_dim) + scales."""
    if weight_2d.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {tuple(weight_2d.shape)}")
    rows, cols = int(weight_2d.shape[0]), int(weight_2d.shape[1])
    device = weight_2d.device
    q_mat = torch.empty((rows, cols), device=device, dtype=torch.float8_e4m3fn)
    row_scales = torch.empty((rows, 1), device=device, dtype=torch.float32)

    for row0 in range(0, rows, stripe_rows):
        row1 = min(row0 + stripe_rows, rows)
        slab = weight_2d[row0:row1].to(torch.float32)
        scale = (slab.abs().amax(dim=1, keepdim=True) / _FP8_E4M3_MAX).clamp_min(1e-12)
        q_slab = (slab / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        q_mat[row0:row1].copy_(q_slab)
        row_scales[row0:row1].copy_(scale)

    return q_mat.T, row_scales.contiguous()


class QuantizedLinear8(nn.Module):
    """Rowwise E4M3 linear wrapper backed by SGLang's fp8_scaled_mm."""

    def __init__(
        self,
        weight_col_major: torch.Tensor,
        weight_scales: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        tuple_output: bool,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_col_major", weight_col_major)
        self.register_buffer("weight_scales", weight_scales)
        if bias is not None:
            self.register_buffer("bias", bias.detach())
        else:
            self.bias = None
        self.tuple_output = bool(tuple_output)
        self.in_dim = int(weight_col_major.shape[0])
        self.out_dim = int(weight_col_major.shape[1])
        self.in_features = self.input_size = self.in_dim
        self.out_features = self.output_size = self.out_dim

    @classmethod
    def quantize_from(cls, linear_mod: nn.Module, *, tuple_output: bool) -> "QuantizedLinear8":
        w_t, w_s = quantize_weight_rowwise_fp8(linear_mod.weight.detach())
        bias = getattr(linear_mod, "bias", None)
        return cls(w_t, w_s, bias, tuple_output=tuple_output)

    def forward(self, x: torch.Tensor):
        from sglang.kernels.ops.gemm import fp8_scaled_mm
        from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_dim)
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()
        x_q, x_scale = sglang_per_token_quant_fp8(x_flat)
        y = fp8_scaled_mm(
            x_q,
            self.weight_col_major,
            x_scale,
            self.weight_scales,
            x.dtype,
            self.bias,
        )
        if len(leading_shape) != 1:
            y = y.view(*leading_shape, self.out_dim)
        return (y, None) if self.tuple_output else y


class DraftHeadFP8:
    """Draft-only rowwise FP8 LM head payload attached to a draft model instance."""

    def __init__(self, weight_col_major: torch.Tensor, scales: torch.Tensor, num_org: int) -> None:
        self.weight_col_major = weight_col_major
        self.scales = scales
        self.num_org = int(num_org)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.gemm import fp8_scaled_mm
        from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

        h_contig = hidden if hidden.is_contiguous() else hidden.contiguous()
        h_q, h_s = sglang_per_token_quant_fp8(h_contig)
        return fp8_scaled_mm(
            h_q,
            self.weight_col_major,
            h_s,
            self.scales,
            h_contig.dtype,
            None,
        )


def _eligible_bf16_linear(mod: Any) -> bool:
    if mod is None or isinstance(mod, QuantizedLinear8):
        return False
    w = getattr(mod, "weight", None)
    if w is None or w.ndim != 2 or w.dtype not in (torch.bfloat16, torch.float16):
        return False
    return (int(w.shape[0]) % 16 == 0) and (int(w.shape[1]) % 16 == 0)


def quantize_draft_backbone(draft_model: nn.Module) -> Tuple[int, int]:
    """Replace eligible draft backbone linears with QuantizedLinear8 after fused-KV setup."""
    count = 0
    saved_bytes = 0

    def _replace(holder: nn.Module, field: str, tuple_output: bool) -> None:
        nonlocal count, saved_bytes
        mod = getattr(holder, field, None)
        if not _eligible_bf16_linear(mod):
            return
        w = mod.weight
        bytes_before = w.numel() * w.element_size()
        setattr(holder, field, QuantizedLinear8.quantize_from(mod, tuple_output=tuple_output))
        del mod, w
        count += 1
        saved_bytes += bytes_before // 2

    if hasattr(draft_model, "fc"):
        _replace(draft_model, "fc", tuple_output=False)

    for block in getattr(draft_model, "layers", ()):
        attn = getattr(block, "self_attn", None)
        if attn is not None:
            _replace(attn, "qkv_proj", tuple_output=True)
            _replace(attn, "o_proj", tuple_output=True)
        mlp = getattr(block, "mlp", None)
        if mlp is not None:
            _replace(mlp, "gate_up_proj", tuple_output=True)
            _replace(mlp, "down_proj", tuple_output=True)

    return count, saved_bytes


def attach_draft_head_fp8(draft_model: nn.Module, target_lm_head: Any) -> int:
    """Build and attach a draft-only DraftHeadFP8 copy of target_lm_head."""
    w = getattr(target_lm_head, "weight", None)
    if w is None or w.ndim != 2 or w.dtype not in (torch.bfloat16, torch.float16):
        return 0
    if int(getattr(target_lm_head, "tp_size", 1)) != 1:
        return 0
    shard = getattr(target_lm_head, "shard_indices", None)
    if shard is not None:
        if int(getattr(shard, "num_added_elements", 0)) != 0:
            return 0
        num_org = int(getattr(target_lm_head, "org_vocab_size", shard.num_org_elements))
    else:
        num_org = int(getattr(target_lm_head, "org_vocab_size", w.shape[0]))
    if num_org <= 0 or (num_org % 16) != 0 or (int(w.shape[1]) % 16) != 0:
        return 0

    w_t, scales = quantize_weight_rowwise_fp8(w[:num_org])
    draft_model._sm89_draft_head_fp8 = DraftHeadFP8(w_t, scales, num_org)
    return num_org


def _unwrap_target_lm_head(target_worker: Any) -> Any:
    model = getattr(getattr(target_worker, "model_runner", None), "model", None)
    head = getattr(model, "lm_head", None) if model is not None else None
    if head is None:
        return None
    try:
        from sglang.srt.lora.utils import unwrap_lora_layer

        return unwrap_lora_layer(head)
    except Exception:
        return head


def after_dflash_worker_init(
    _result: Any,
    worker: Any,
    *args: Any,
    quantize_draft: bool = True,
    quantize_head: bool = True,
    **kwargs: Any,
) -> None:
    """Hook AFTER DFlashWorkerV2.__init__ (runs after _init_fused_kv_helper)."""
    draft_model = getattr(worker, "draft_model", None)
    if draft_model is None:
        return

    dirty = False
    if quantize_draft:
        n_replaced, freed = quantize_draft_backbone(draft_model)
        if n_replaced > 0:
            dirty = True
            logger.info(
                "[sglang_sm89] Draft FP8 backbone enabled (%d layers, -%.1f MiB, fused_kv=%s)",
                n_replaced,
                freed / (1024 * 1024),
                getattr(worker, "_use_fused_kv_materialize", None),
            )

    if quantize_head:
        tw = getattr(worker, "target_worker", None) or getattr(worker, "_target_worker", None)
        head = _unwrap_target_lm_head(tw) if tw is not None else None
        if head is not None:
            num_org = attach_draft_head_fp8(draft_model, head)
            if num_org > 0:
                dirty = True
                logger.info("[sglang_sm89] Draft FP8 head attached (vocab=%d)", num_org)

    if dirty and torch.cuda.is_available():
        torch.cuda.empty_cache()


def around_compute_candidates(original_fn: Any, draft_model: Any, hidden: torch.Tensor):
    """Route DFlash2DraftModel.compute_candidates through DraftHeadFP8 when attached."""
    head_fp8: Optional[DraftHeadFP8] = getattr(draft_model, "_sm89_draft_head_fp8", None)
    if head_fp8 is None:
        return original_fn(draft_model, hidden)

    from sglang.srt.models.dflash import _radix_topk

    logits = head_fp8.project(hidden)
    vals, ids = _radix_topk(logits, draft_model.candidate_selector.top_k)
    return ids.long(), draft_model._transform_unary_logits(vals)
