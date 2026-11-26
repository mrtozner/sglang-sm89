#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <tuple>

namespace sglang_sm89 {

std::tuple<int64_t, bool, bool> coupled_refine_step_cpu(
    at::Tensor& mask_row_cpu,
    at::Tensor& draft_row_cpu,
    const c10::optional<at::Tensor>& cand_row_cpu,
    const c10::optional<at::Tensor>& q_row_cpu,
    int64_t pos,
    int64_t vocab_size,
    bool is_greedy,
    bool allow_forced_single,
    double u_draw,
    bool reset_mask = false);

}  // namespace sglang_sm89
