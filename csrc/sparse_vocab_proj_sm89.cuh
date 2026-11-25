#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

namespace sglang_sm89 {

void sparse_vocab_proj_bf16(
    const at::Tensor& hidden,
    const at::Tensor& weight,
    const at::Tensor& vocab_mask,
    at::Tensor& out);

void sparse_vocab_proj_fp8(
    const at::Tensor& hidden,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& vocab_mask,
    at::Tensor& out);

bool select_sparse_vocab_proj_cpu(
    const at::Tensor& vocab_mask_cpu,
    int64_t vocab_size,
    int64_t k_chunk,
    double max_eff_row_ratio);

}  // namespace sglang_sm89
