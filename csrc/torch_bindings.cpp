#include <torch/extension.h>
#include <torch/library.h>

#include "coupled_grammar_cpu.cuh"
#include "sparse_vocab_proj_sm89.cuh"
#include "wy_gdn_sm89.cuh"

TORCH_LIBRARY(sglang_sm89, m) {
    m.def(
        "sparse_vocab_proj_bf16("
        "Tensor hidden, Tensor weight, Tensor vocab_mask, Tensor(a!) out) -> ()");
    m.def(
        "sparse_vocab_proj_fp8("
        "Tensor hidden, Tensor weight, Tensor weight_scale, Tensor vocab_mask, Tensor(a!) out) -> ()");
    m.def(
        "select_sparse_vocab_proj("
        "Tensor vocab_mask_cpu, int vocab_size, int k_chunk=8, float max_eff_row_ratio=0.55) -> bool");
    m.def(
        "wy_gdn_replayssm_verify("
        "Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, "
        "Tensor A_log, Tensor dt_bias, Tensor checkpoint_state, "
        "Tensor(a!) d_cache, Tensor(b!) k_cache, Tensor(c!) g_cache, "
        "Tensor(d!)? rawv_cache, Tensor(e!)? rawk_cache, Tensor(f!)? beta_cache, "
        "Tensor query_start_loc, Tensor ssm_state_indices, Tensor replay_indices, "
        "Tensor write_pos, Tensor cache_base, Tensor is_flush, Tensor(g!) out, "
        "float scale, int max_cache_len, int max_spec_len, "
        "bool is_flush_pass=False, float softplus_threshold=20.0) -> ()");
    m.def(
        "wy_gdn_replayssm_flush_fold("
        "Tensor(a!) checkpoint_state, "
        "Tensor d_cache, Tensor k_cache, Tensor g_cache, "
        "Tensor? rawv_cache, Tensor? rawk_cache, "
        "Tensor ssm_state_indices, Tensor replay_indices, "
        "Tensor(b!) write_pos, Tensor cache_base, Tensor is_flush, "
        "int max_cache_len, bool store_exact_raw=False, Tensor? beta_cache=None) -> ()");
}

TORCH_LIBRARY_IMPL(sglang_sm89, CUDA, m) {
    m.impl("sparse_vocab_proj_bf16", &sglang_sm89::sparse_vocab_proj_bf16);
    m.impl("sparse_vocab_proj_fp8", &sglang_sm89::sparse_vocab_proj_fp8);
    m.impl("wy_gdn_replayssm_verify", &sglang_sm89::wy_gdn_replayssm_verify_cuda);
    m.impl("wy_gdn_replayssm_flush_fold", &sglang_sm89::wy_gdn_replayssm_flush_fold_cuda);
}

TORCH_LIBRARY_IMPL(sglang_sm89, CPU, m) {
    m.impl("select_sparse_vocab_proj", &sglang_sm89::select_sparse_vocab_proj_cpu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Ada Lovelace (sm_89) SGLang CUDA and CPU acceleration kernels";
    m.def(
        "select_sparse_vocab_proj",
        &sglang_sm89::select_sparse_vocab_proj_cpu,
        py::arg("vocab_mask_cpu"),
        py::arg("vocab_size"),
        py::arg("k_chunk") = 8,
        py::arg("max_eff_row_ratio") = 0.55);
    m.def(
        "coupled_refine_step_cpu",
        &sglang_sm89::coupled_refine_step_cpu,
        py::arg("mask_row_cpu"),
        py::arg("draft_row_cpu"),
        py::arg("cand_row_cpu"),
        py::arg("q_row_cpu"),
        py::arg("pos"),
        py::arg("vocab_size"),
        py::arg("is_greedy"),
        py::arg("allow_forced_single"),
        py::arg("u_draw"),
        py::arg("reset_mask") = false);
}
