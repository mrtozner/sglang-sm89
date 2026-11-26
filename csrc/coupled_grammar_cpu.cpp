#include "coupled_grammar_cpu.cuh"

#include <c10/util/Exception.h>

#include <cstdint>
#include <cstring>
#include <tuple>

namespace sglang_sm89 {
namespace {

inline bool token_bit_is_set(const uint32_t* words, int64_t tok, int64_t vocab_size) {
    if (tok < 0 || tok >= vocab_size) return false;
    return ((words[tok >> 5] >> (tok & 31)) & 1u) != 0u;
}

// Fast 64-bit word scan across mask_row_cpu.
// When detect_unique is true, stops immediately as soon as popcount reaches >= 2.
// Always outputs first_tok (lowest legal token ID, or -1 if bitmask is empty).
inline void scan_mask_64(
    const uint32_t* words,
    int64_t vocab_size,
    bool detect_unique,
    int64_t& out_unique,
    int64_t& out_first) {
    out_unique = -1;
    out_first = -1;

    const int64_t n_words = (vocab_size + 31) / 32;
    const int64_t n_pairs = n_words / 2;
    int64_t found_cnt = 0;

    for (int64_t p = 0; p < n_pairs; ++p) {
        uint64_t v = static_cast<uint64_t>(words[p * 2]) |
                     (static_cast<uint64_t>(words[p * 2 + 1]) << 32);
        if ((p * 2 + 2) == n_words) {
            const int rem64 = static_cast<int>(vocab_size & 63);
            if (rem64 != 0) {
                v &= (1ULL << rem64) - 1ULL;
            }
        }
        if (v == 0ULL) continue;

        if (out_first < 0) {
            out_first = (p << 6) + __builtin_ctzll(v);
        }
        if (!detect_unique) {
            return;
        }
        found_cnt += __builtin_popcountll(v);
        if (found_cnt > 1) {
            return;
        }
    }

    if ((n_words & 1) != 0) {
        uint32_t w = words[n_words - 1];
        const int rem32 = static_cast<int>(vocab_size & 31);
        if (rem32 != 0) {
            w &= (1u << rem32) - 1u;
        }
        if (w != 0u) {
            if (out_first < 0) {
                out_first = ((n_words - 1) << 5) + __builtin_ctz(w);
            }
            if (!detect_unique) {
                return;
            }
            found_cnt += __builtin_popcount(w);
            if (found_cnt > 1) {
                return;
            }
        }
    }

    if (detect_unique && found_cnt == 1) {
        out_unique = out_first;
    }
}

inline void place_unique_front(
    int64_t* cand_ptr,
    float* q_ptr,
    int64_t k_width,
    int64_t tok) {
    if (cand_ptr != nullptr && k_width > 0) {
        int64_t found_at = -1;
        for (int64_t i = 0; i < k_width; ++i) {
            if (cand_ptr[i] == tok) {
                found_at = i;
                break;
            }
        }
        if (found_at > 0) {
            cand_ptr[found_at] = cand_ptr[0];
            cand_ptr[0] = tok;
        } else if (found_at < 0) {
            cand_ptr[0] = tok;
        }
    }
    if (q_ptr != nullptr && k_width > 0) {
        std::memset(q_ptr, 0, static_cast<size_t>(k_width) * sizeof(float));
        q_ptr[0] = 1.0f;
    }
}

}  // namespace

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
    bool reset_mask) {
    TORCH_CHECK(mask_row_cpu.device().is_cpu() && mask_row_cpu.scalar_type() == at::kInt &&
                    mask_row_cpu.is_contiguous(),
                "mask_row_cpu must be contiguous CPU int32");
    TORCH_CHECK(draft_row_cpu.device().is_cpu() && draft_row_cpu.scalar_type() == at::kLong &&
                    draft_row_cpu.is_contiguous() && draft_row_cpu.dim() == 1,
                "draft_row_cpu must be contiguous 1D CPU int64");
    TORCH_CHECK(vocab_size > 0 && pos >= 0 && pos < draft_row_cpu.size(0),
                "Invalid vocab_size or pos");

    const int64_t n_words = (vocab_size + 31) / 32;
    const int64_t mask_words = (mask_row_cpu.dim() == 1) ? mask_row_cpu.size(0) : mask_row_cpu.size(1);
    TORCH_CHECK(mask_words >= n_words, "mask_row_cpu smaller than ceil_div(vocab_size, 32)");

    int32_t* mask_i32 = mask_row_cpu.data_ptr<int32_t>();
    const uint32_t* words_u32 = reinterpret_cast<const uint32_t*>(mask_i32);
    int64_t* draft_ptr = draft_row_cpu.data_ptr<int64_t>();
    const int64_t cur_tok = draft_ptr[pos];

    int64_t* cand_ptr = nullptr;
    int64_t k_width = 0;
    if (cand_row_cpu.has_value()) {
        const at::Tensor& c = *cand_row_cpu;
        TORCH_CHECK(c.device().is_cpu() && c.scalar_type() == at::kLong && c.is_contiguous() && c.dim() == 1,
                    "cand_row_cpu must be contiguous 1D CPU int64");
        cand_ptr = c.data_ptr<int64_t>();
        k_width = c.size(0);
    }

    float* q_ptr = nullptr;
    if (q_row_cpu.has_value()) {
        const at::Tensor& q = *q_row_cpu;
        TORCH_CHECK(q.device().is_cpu() && q.scalar_type() == at::kFloat && q.is_contiguous() && q.dim() == 1,
                    "q_row_cpu must be contiguous 1D CPU float32");
        if (cand_ptr != nullptr) {
            TORCH_CHECK(q.size(0) == k_width, "q_row_cpu length must match cand_row_cpu");
        }
        q_ptr = q.data_ptr<float>();
    }

    auto finish = [&](int64_t picked, bool d_chg, bool q_chg) -> std::tuple<int64_t, bool, bool> {
        if (reset_mask) {
            std::memset(mask_i32, 0xFF, static_cast<size_t>(n_words) * sizeof(int32_t));
        }
        return {picked, d_chg, q_chg};
    };

    int64_t unique_tok = -1;
    int64_t first_tok = -1;
    if (allow_forced_single) {
        scan_mask_64(words_u32, vocab_size, true, unique_tok, first_tok);
        if (first_tok < 0) {
            return finish(-1, false, false);
        }
        if (unique_tok >= 0) {
            bool d_chg = false;
            if (unique_tok != cur_tok) {
                draft_ptr[pos] = unique_tok;
                d_chg = true;
            }
            const bool q_chg = (q_ptr != nullptr);
            if (cand_ptr != nullptr || q_ptr != nullptr) {
                place_unique_front(cand_ptr, q_ptr, k_width, unique_tok);
            }
            return finish(unique_tok, d_chg, q_chg);
        }
    }

    if (is_greedy || cand_ptr == nullptr || q_ptr == nullptr) {
        int64_t picked = -1;
        bool d_chg = false;
        if (token_bit_is_set(words_u32, cur_tok, vocab_size)) {
            picked = cur_tok;
        } else {
            if (cand_ptr != nullptr) {
                for (int64_t i = 0; i < k_width; ++i) {
                    if (token_bit_is_set(words_u32, cand_ptr[i], vocab_size)) {
                        picked = cand_ptr[i];
                        break;
                    }
                }
            }
            if (picked < 0) {
                if (first_tok < 0) {
                    int64_t dummy_u = -1;
                    scan_mask_64(words_u32, vocab_size, false, dummy_u, first_tok);
                }
                picked = first_tok;
            }
            if (picked < 0) {
                return finish(-1, false, false);
            }
            draft_ptr[pos] = picked;
            d_chg = true;
        }
        const bool q_chg = (q_ptr != nullptr);
        if (cand_ptr != nullptr || q_ptr != nullptr) {
            place_unique_front(cand_ptr, q_ptr, k_width, picked);
        }
        return finish(picked, d_chg, q_chg);
    }

    double z_mass = 0.0;
    int64_t match_idx = -1;
    int64_t last_legal_idx = -1;
    const bool cur_legal = token_bit_is_set(words_u32, cur_tok, vocab_size);

    for (int64_t i = 0; i < k_width; ++i) {
        const int64_t tok = cand_ptr[i];
        if (tok == cur_tok && match_idx < 0) {
            match_idx = i;
        }
        const float qv = q_ptr[i];
        if (qv > 0.0f && token_bit_is_set(words_u32, tok, vocab_size)) {
            z_mass += static_cast<double>(qv);
            last_legal_idx = i;
        } else {
            q_ptr[i] = 0.0f;
        }
    }

    if (z_mass > 1e-12) {
        const double inv_z = 1.0 / z_mass;
        for (int64_t i = 0; i < k_width; ++i) {
            if (q_ptr[i] > 0.0f) {
                q_ptr[i] = static_cast<float>(static_cast<double>(q_ptr[i]) * inv_z);
            }
        }

        if (cur_legal && match_idx >= 0 && q_ptr[match_idx] > 0.0f) {
            return finish(cur_tok, false, true);
        }

        double cdf = 0.0;
        int64_t sel = last_legal_idx;
        for (int64_t i = 0; i < k_width; ++i) {
            const double pk = static_cast<double>(q_ptr[i]);
            if (pk <= 0.0) continue;
            cdf += pk;
            if (u_draw < cdf) {
                sel = i;
                break;
            }
        }
        const int64_t picked = cand_ptr[sel];
        draft_ptr[pos] = picked;
        return finish(picked, true, true);
    }

    if (first_tok < 0) {
        int64_t dummy_u = -1;
        scan_mask_64(words_u32, vocab_size, false, dummy_u, first_tok);
    }
    if (first_tok < 0) {
        return finish(-1, false, false);
    }
    place_unique_front(cand_ptr, q_ptr, k_width, first_tok);
    draft_ptr[pos] = first_tok;
    return finish(first_tok, true, true);
}

}  // namespace sglang_sm89
