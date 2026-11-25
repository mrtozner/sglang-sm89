#pragma once

#include <torch/types.h>
#include <cstdint>

namespace sglang_sm89 {

void wy_gdn_replayssm_verify_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const at::Tensor& checkpoint_state,
    at::Tensor& d_cache,
    at::Tensor& k_cache,
    at::Tensor& g_cache,
    const c10::optional<at::Tensor>& rawv_cache,
    const c10::optional<at::Tensor>& rawk_cache,
    const c10::optional<at::Tensor>& beta_cache,
    const at::Tensor& query_start_loc,
    const at::Tensor& ssm_state_indices,
    const at::Tensor& replay_indices,
    const at::Tensor& write_pos,
    const at::Tensor& cache_base,
    const at::Tensor& is_flush,
    at::Tensor& out,
    double scale,
    int64_t max_cache_len,
    int64_t max_spec_len,
    bool is_flush_pass,
    double softplus_threshold);

void wy_gdn_replayssm_flush_fold_cuda(
    at::Tensor& checkpoint_state,
    const at::Tensor& d_cache,
    const at::Tensor& k_cache,
    const at::Tensor& g_cache,
    const c10::optional<at::Tensor>& rawv_cache,
    const c10::optional<at::Tensor>& rawk_cache,
    const at::Tensor& ssm_state_indices,
    const at::Tensor& replay_indices,
    at::Tensor& write_pos,
    const at::Tensor& cache_base,
    const at::Tensor& is_flush,
    int64_t max_cache_len,
    bool store_exact_raw,
    const c10::optional<at::Tensor>& beta_cache = c10::nullopt);

}  // namespace sglang_sm89
