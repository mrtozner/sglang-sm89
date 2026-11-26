#include "sparse_vocab_proj_sm89.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace sglang_sm89 {
namespace {

constexpr int kTileV = 64;
constexpr int kThreadsPerBlock = 128;
constexpr int kSmemRows = 64;
constexpr int kChunkM = 8;

__device__ __forceinline__ __nv_bfloat16 dev_bf16_neg_inf() {
    return __ushort_as_bfloat16(static_cast<uint16_t>(0xFF80u));
}

__device__ __forceinline__ float2 bf16_pair_to_float2(uint32_t raw_bits) {
    __nv_bfloat162 b2;
    static_assert(sizeof(b2) == sizeof(raw_bits), "bfloat162 must be 4 bytes");
    __builtin_memcpy(&b2, &raw_bits, sizeof(raw_bits));
    return __bfloat1622float2(b2);
}

__device__ __forceinline__ __half2 e4m3_pair_to_half2(uint16_t raw_pair) {
#if __CUDA_ARCH__ >= 890
    uint32_t out_u32;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out_u32) : "h"(raw_pair));
    __half2 out;
    static_assert(sizeof(out) == sizeof(out_u32), "half2 must be 4 bytes");
    __builtin_memcpy(&out, &out_u32, sizeof(out_u32));
    return out;
#else
    auto cvt_one = [](uint8_t byte) -> float {
        const int sign = (byte >> 7) & 1;
        const int exp = (byte >> 3) & 0xF;
        const int man = byte & 0x7;
        float mag;
        if (exp == 0) {
            mag = ldexpf(static_cast<float>(man) * 0.125f, -6);
        } else if (exp == 15 && man == 7) {
            mag = 0.0f;
        } else {
            mag = ldexpf(1.0f + static_cast<float>(man) * 0.125f, exp - 7);
        }
        return sign ? -mag : mag;
    };
    return __floats2half2_rn(
        cvt_one(static_cast<uint8_t>(raw_pair & 0xFFu)),
        cvt_one(static_cast<uint8_t>((raw_pair >> 8) & 0xFFu)));
#endif
}

__device__ __forceinline__ uint64_t read_tile_mask64(
    const int32_t* __restrict__ vocab_mask,
    int m,
    int mask_words,
    int word_base,
    uint64_t tile_tail) {
    const int32_t* row_ptr = vocab_mask + static_cast<size_t>(m) * mask_words + word_base;
    uint32_t lo = 0u;
    uint32_t hi = 0u;
    if ((word_base + 2 <= mask_words) &&
        ((reinterpret_cast<uintptr_t>(row_ptr) & 7u) == 0u)) {
        const uint2 pair = __ldg(reinterpret_cast<const uint2*>(row_ptr));
        lo = pair.x;
        hi = pair.y;
    } else {
        if (word_base < mask_words) lo = static_cast<uint32_t>(__ldg(row_ptr));
        if (word_base + 1 < mask_words) hi = static_cast<uint32_t>(__ldg(row_ptr + 1));
    }
    return (static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32)) & tile_tail;
}

__device__ __forceinline__ void accum_slot(
    int slot,
    float delta,
    float& a0,
    float& a1,
    float& a2,
    float& a3,
    float& a4,
    float& a5,
    float& a6,
    float& a7) {
    if (slot < 4) {
        if (slot < 2) {
            if (slot == 0) a0 += delta;
            else a1 += delta;
        } else {
            if (slot == 2) a2 += delta;
            else a3 += delta;
        }
    } else {
        if (slot < 6) {
            if (slot == 4) a4 += delta;
            else a5 += delta;
        } else {
            if (slot == 6) a6 += delta;
            else a7 += delta;
        }
    }
}

__device__ __forceinline__ float select_slot(
    int slot,
    float a0,
    float a1,
    float a2,
    float a3,
    float a4,
    float a5,
    float a6,
    float a7) {
    if (slot < 4) {
        if (slot < 2) return (slot == 0) ? a0 : a1;
        return (slot == 2) ? a2 : a3;
    }
    if (slot < 6) return (slot == 4) ? a4 : a5;
    return (slot == 6) ? a6 : a7;
}

template <bool USE_FP8>
__device__ __forceinline__ void eval_warp_row_group(
    const __nv_bfloat16* __restrict__ hidden,
    const void* __restrict__ weight,
    float row_scale,
    __nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ m_offsets,
    int m_base,
    int m0,
    int m_cnt,
    int row,
    int N,
    int K,
    int lane) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    if constexpr (!USE_FP8) {
        const __nv_bfloat16* w_row =
            reinterpret_cast<const __nv_bfloat16*>(weight) + static_cast<size_t>(row) * K;
        for (int k = lane * 8; k < K; k += 256) {
            const uint4 wv = __ldg(reinterpret_cast<const uint4*>(w_row + k));
            const float2 wf0 = bf16_pair_to_float2(wv.x);
            const float2 wf1 = bf16_pair_to_float2(wv.y);
            const float2 wf2 = bf16_pair_to_float2(wv.z);
            const float2 wf3 = bf16_pair_to_float2(wv.w);

            #pragma unroll 1
            for (int j = 0; j < m_cnt; ++j) {
                const int m_idx = m_base + static_cast<int>(m_offsets[m0 + j]);
                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                    hidden + static_cast<size_t>(m_idx) * K + k));

                float2 xf = bf16_pair_to_float2(xv.x);
                float dot = wf0.x * xf.x + wf0.y * xf.y;
                xf = bf16_pair_to_float2(xv.y);
                dot = fmaf(wf1.x, xf.x, fmaf(wf1.y, xf.y, dot));
                xf = bf16_pair_to_float2(xv.z);
                dot = fmaf(wf2.x, xf.x, fmaf(wf2.y, xf.y, dot));
                xf = bf16_pair_to_float2(xv.w);
                dot = fmaf(wf3.x, xf.x, fmaf(wf3.y, xf.y, dot));
                accum_slot(j, dot, a0, a1, a2, a3, a4, a5, a6, a7);
            }
        }
    } else {
        const size_t row_off = static_cast<size_t>(row) * K;
        const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(weight) + row_off;
        for (int k = lane * 16; k < K; k += 512) {
            const uint4 wv = __ldg(reinterpret_cast<const uint4*>(w_bytes + k));
            const __half2 wh0 = e4m3_pair_to_half2(static_cast<uint16_t>(wv.x & 0xFFFFu));
            const __half2 wh1 = e4m3_pair_to_half2(static_cast<uint16_t>(wv.x >> 16));
            const __half2 wh2 = e4m3_pair_to_half2(static_cast<uint16_t>(wv.y & 0xFFFFu));
            const __half2 wh3 = e4m3_pair_to_half2(static_cast<uint16_t>(wv.y >> 16));
            const __half2 wh4 = e4m3_pair_to_half2(static_cast<uint16_t>(wv.z & 0xFFFFu));
            const __half2 wh5 = e4m3_pair_to_half2(static_cast<uint16_t>(wv.z >> 16));
            const __half2 wh6 = e4m3_pair_to_half2(static_cast<uint16_t>(wv.w & 0xFFFFu));
            const __half2 wh7 = e4m3_pair_to_half2(static_cast<uint16_t>(wv.w >> 16));

            #pragma unroll 1
            for (int j = 0; j < m_cnt; ++j) {
                const int m_idx = m_base + static_cast<int>(m_offsets[m0 + j]);
                const __nv_bfloat16* x_ptr = hidden + static_cast<size_t>(m_idx) * K + k;
                float dot = 0.0f;
                {
                    const uint4 xv_lo = __ldg(reinterpret_cast<const uint4*>(x_ptr));
                    float2 wf = __half22float2(wh0);
                    float2 xf = bf16_pair_to_float2(xv_lo.x);
                    dot = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, dot));
                    wf = __half22float2(wh1);
                    xf = bf16_pair_to_float2(xv_lo.y);
                    dot = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, dot));
                    wf = __half22float2(wh2);
                    xf = bf16_pair_to_float2(xv_lo.z);
                    dot = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, dot));
                    wf = __half22float2(wh3);
                    xf = bf16_pair_to_float2(xv_lo.w);
                    dot = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, dot));
                }
                {
                    const uint4 xv_hi = __ldg(reinterpret_cast<const uint4*>(x_ptr + 8));
                    float2 wf = __half22float2(wh4);
                    float2 xf = bf16_pair_to_float2(xv_hi.x);
                    dot = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, dot));
                    wf = __half22float2(wh5);
                    xf = bf16_pair_to_float2(xv_hi.y);
                    dot = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, dot));
                    wf = __half22float2(wh6);
                    xf = bf16_pair_to_float2(xv_hi.z);
                    dot = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, dot));
                    wf = __half22float2(wh7);
                    xf = bf16_pair_to_float2(xv_hi.w);
                    dot = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, dot));
                }
                accum_slot(j, dot, a0, a1, a2, a3, a4, a5, a6, a7);
            }
        }
    }

    #pragma unroll 1
    for (int j = 0; j < m_cnt; ++j) {
        float sum = select_slot(j, a0, a1, a2, a3, a4, a5, a6, a7);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, offset);
        }
        if (lane == 0) {
            const int m_idx = m_base + static_cast<int>(m_offsets[m0 + j]);
            out[static_cast<size_t>(m_idx) * N + row] =
                __float2bfloat16(sum * row_scale);
        }
    }
}

template <bool USE_FP8>
__global__ void __launch_bounds__(kThreadsPerBlock, 4)
sparse_vocab_proj_tile64_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const void* __restrict__ weight,
    const float* __restrict__ weight_scale,
    const int32_t* __restrict__ vocab_mask,
    __nv_bfloat16* __restrict__ out,
    int M, int N, int K, int mask_words) {
    const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int tile_id = blockIdx.x, row_base = tile_id * kTileV;
    if (row_base >= N) return;

    const int rem_rows = N - row_base;
    const int tile_rows = (rem_rows < kTileV) ? rem_rows : kTileV;
    const uint64_t tile_tail = (rem_rows >= kTileV)
        ? ~0ULL
        : ((rem_rows <= 0) ? 0ULL : ((1ULL << rem_rows) - 1ULL));
    const int word_base = tile_id * 2;

    __shared__ uint64_t s_tile_masks[kSmemRows];
    __shared__ uint64_t s_tile_union;
    __shared__ uint8_t s_active_v[kTileV];
    __shared__ int s_num_active_v;
    __shared__ uint8_t s_warp_m[4][kSmemRows];

    uint64_t local_union = 0ULL;
    for (int m = tid; m < M; m += kThreadsPerBlock) {
        const uint64_t mw = read_tile_mask64(vocab_mask, m, mask_words, word_base, tile_tail);
        if (m < kSmemRows) {
            s_tile_masks[m] = mw;
        }
        local_union |= mw;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_union |= __shfl_xor_sync(0xFFFFFFFFu, local_union, offset);
    }

    if (tid == 0) {
        s_tile_union = 0ULL;
    }
    __syncthreads();

    if (lane == 0 && local_union != 0ULL) {
        atomicOr(reinterpret_cast<unsigned long long*>(&s_tile_union),
                 static_cast<unsigned long long>(local_union));
    }
    __syncthreads();

    const uint64_t u_mask = s_tile_union;
    const __nv_bfloat16 neg_inf = dev_bf16_neg_inf();

    if (u_mask == 0ULL) {
        if (tile_rows == kTileV) {
            const int total_cells = M * kTileV;
            for (int cell = tid; cell < total_cells; cell += kThreadsPerBlock) {
                const int m = cell >> 6;
                const int r = cell & 63;
                out[static_cast<size_t>(m) * N + (row_base + r)] = neg_inf;
            }
        } else {
            const int total_cells = M * tile_rows;
            for (int cell = tid; cell < total_cells; cell += kThreadsPerBlock) {
                const int m = cell / tile_rows;
                const int r = cell - m * tile_rows;
                out[static_cast<size_t>(m) * N + (row_base + r)] = neg_inf;
            }
        }
        return;
    }

    const bool row_valid = (tid < tile_rows) && (((u_mask >> tid) & 1ULL) != 0ULL);
    if (tid < tile_rows && !row_valid) {
        const int row = row_base + tid;
        for (int m = 0; m < M; ++m) {
            out[static_cast<size_t>(m) * N + row] = neg_inf;
        }
    }

    if (warp_id < 2) {
        const uint32_t b = __ballot_sync(0xFFFFFFFFu, row_valid);
        const int w0_cnt = __popc(static_cast<uint32_t>(u_mask & 0xFFFFFFFFULL));
        const uint32_t lane_pfx = (lane == 0) ? 0u : ((1u << lane) - 1u);
        const int pos = (warp_id == 0 ? 0 : w0_cnt) + __popc(b & lane_pfx);
        if (row_valid) {
            s_active_v[pos] = static_cast<uint8_t>(tid);
        }
    }
    if (tid == 0) {
        s_num_active_v = __popcll(u_mask);
    }
    __syncthreads();

    const int n_active = s_num_active_v;
    if (n_active == 0) return;

    const int m_step = (M <= kSmemRows) ? kSmemRows : 32;
    for (int a = warp_id; a < n_active; a += 4) {
        const int r_local = static_cast<int>(s_active_v[a]);
        const int row = row_base + r_local;
        const uint64_t r_bit = 1ULL << r_local;
        const float row_scale = USE_FP8 ? __ldg(weight_scale + row) : 1.0f;

        for (int m_base = 0; m_base < M; m_base += m_step) {
            int n_m = 0;
            if (M <= kSmemRows) {
                const int m_lo = lane;
                const int m_hi = 32 + lane;
                bool ok_lo = false;
                bool ok_hi = false;
                if (m_lo < M) {
                    ok_lo = (s_tile_masks[m_lo] & r_bit) != 0ULL;
                    if (!ok_lo) out[static_cast<size_t>(m_lo) * N + row] = neg_inf;
                }
                if (m_hi < M) {
                    ok_hi = (s_tile_masks[m_hi] & r_bit) != 0ULL;
                    if (!ok_hi) out[static_cast<size_t>(m_hi) * N + row] = neg_inf;
                }
                const uint32_t b_lo = __ballot_sync(0xFFFFFFFFu, ok_lo);
                const uint32_t b_hi = __ballot_sync(0xFFFFFFFFu, ok_hi);
                n_m = __popc(b_lo) + __popc(b_hi);
                if (n_m == 0) continue;

                const uint32_t lane_pfx = (lane == 0) ? 0u : ((1u << lane) - 1u);
                const int hi_off = __popc(b_lo);
                if (ok_lo) s_warp_m[warp_id][__popc(b_lo & lane_pfx)] = static_cast<uint8_t>(m_lo);
                if (ok_hi) s_warp_m[warp_id][hi_off + __popc(b_hi & lane_pfx)] = static_cast<uint8_t>(m_hi);
            } else {
                const int m = m_base + lane;
                bool ok = false;
                if (m < M) {
                    const uint64_t mw = (m < kSmemRows)
                        ? s_tile_masks[m]
                        : read_tile_mask64(vocab_mask, m, mask_words, word_base, tile_tail);
                    ok = (mw & r_bit) != 0ULL;
                    if (!ok) out[static_cast<size_t>(m) * N + row] = neg_inf;
                }
                const uint32_t b = __ballot_sync(0xFFFFFFFFu, ok);
                n_m = __popc(b);
                if (n_m == 0) continue;

                const uint32_t lane_pfx = (lane == 0) ? 0u : ((1u << lane) - 1u);
                if (ok) s_warp_m[warp_id][__popc(b & lane_pfx)] = static_cast<uint8_t>(lane);
            }
            __syncwarp();

            for (int m0 = 0; m0 < n_m; m0 += kChunkM) {
                const int m_cnt = (m0 + kChunkM <= n_m) ? kChunkM : (n_m - m0);
                eval_warp_row_group<USE_FP8>(
                    hidden, weight, row_scale, out,
                    s_warp_m[warp_id], m_base, m0, m_cnt, row, N, K, lane);
            }
        }
    }
}

void validate_proj_inputs(
    const at::Tensor& hidden, const at::Tensor& weight,
    const at::Tensor& vocab_mask, const at::Tensor& out, int k_align) {
    TORCH_CHECK(hidden.dim() == 2 && weight.dim() == 2 && vocab_mask.dim() == 2 && out.dim() == 2,
                "All projection operands must be rank-2 tensors");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
                "hidden and out must be bfloat16");
    TORCH_CHECK(vocab_mask.scalar_type() == at::kInt, "vocab_mask must be int32");
    TORCH_CHECK(hidden.is_cuda() && weight.is_cuda() && vocab_mask.is_cuda() && out.is_cuda(),
                "Projection operands must reside on CUDA");
    TORCH_CHECK(hidden.is_contiguous() && weight.is_contiguous() && vocab_mask.is_contiguous() && out.is_contiguous(),
                "Projection operands must be contiguous");

    const int64_t k_dim = hidden.size(1);
    const int64_t n_vocab = weight.size(0);
    const int64_t m_rows = hidden.size(0);
    TORCH_CHECK(weight.size(1) == k_dim && (k_dim % k_align) == 0,
                "Invalid K dimension or alignment");
    TORCH_CHECK(vocab_mask.size(0) == m_rows && vocab_mask.size(1) >= (n_vocab + 31) / 32,
                "vocab_mask shape incompatible with [M, ceil_div(N, 32)]");
    TORCH_CHECK(out.size(0) == m_rows && out.size(1) == n_vocab, "out shape mismatch");
}

template <bool USE_FP8>
void launch_sparse_vocab_proj(
    const at::Tensor& hidden,
    const at::Tensor& weight,
    const float* scale_ptr,
    const at::Tensor& vocab_mask,
    at::Tensor& out) {
    const int m_rows = static_cast<int>(hidden.size(0));
    const int n_vocab = static_cast<int>(weight.size(0));
    const int k_dim = static_cast<int>(hidden.size(1));
    const int mask_words = static_cast<int>(vocab_mask.size(1));
    if (m_rows <= 0 || n_vocab <= 0) return;

    const at::cuda::OptionalCUDAGuard guard(hidden.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 grid((n_vocab + kTileV - 1) / kTileV, 1, 1);
    const dim3 block(kThreadsPerBlock, 1, 1);
    sparse_vocab_proj_tile64_kernel<USE_FP8><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr()),
        weight.data_ptr(),
        scale_ptr,
        vocab_mask.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        m_rows,
        n_vocab,
        k_dim,
        mask_words);
}

}  // namespace

void sparse_vocab_proj_bf16(
    const at::Tensor& hidden,
    const at::Tensor& weight,
    const at::Tensor& vocab_mask,
    at::Tensor& out) {
    validate_proj_inputs(hidden, weight, vocab_mask, out, 8);
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bfloat16");
    launch_sparse_vocab_proj<false>(hidden, weight, nullptr, vocab_mask, out);
}

void sparse_vocab_proj_fp8(
    const at::Tensor& hidden, const at::Tensor& weight,
    const at::Tensor& weight_scale, const at::Tensor& vocab_mask, at::Tensor& out) {
    validate_proj_inputs(hidden, weight, vocab_mask, out, 16);
    TORCH_CHECK(weight.scalar_type() == at::kFloat8_e4m3fn, "weight must be float8_e4m3fn");
    TORCH_CHECK(weight_scale.is_cuda() && weight_scale.is_contiguous() && weight_scale.scalar_type() == at::kFloat,
                "weight_scale must be contiguous CUDA float32");
    TORCH_CHECK(weight_scale.numel() == weight.size(0), "weight_scale length must equal N");
    launch_sparse_vocab_proj<true>(hidden, weight, weight_scale.data_ptr<float>(), vocab_mask, out);
}

bool select_sparse_vocab_proj_cpu(
    const at::Tensor& vocab_mask_cpu,
    int64_t vocab_size,
    int64_t k_chunk,
    double max_eff_row_ratio) {
    TORCH_CHECK(vocab_mask_cpu.device().is_cpu(), "vocab_mask_cpu must be on CPU");
    TORCH_CHECK(vocab_mask_cpu.scalar_type() == at::kInt, "vocab_mask_cpu must be int32");
    TORCH_CHECK(vocab_size > 0, "vocab_size must be positive");
    TORCH_CHECK(k_chunk > 0, "k_chunk must be positive");

    const at::Tensor mask_c = vocab_mask_cpu.contiguous();
    const int64_t M = (mask_c.dim() == 1) ? 1 : mask_c.size(0);
    const int64_t mask_words = (mask_c.dim() == 1) ? mask_c.size(0) : mask_c.size(1);
    const int64_t n_words = (vocab_size + 31) / 32;
    TORCH_CHECK(mask_words >= n_words, "vocab_mask_cpu width smaller than ceil_div(vocab_size, 32)");
    if (M == 0) return true;

    const int64_t n_tiles = (vocab_size + kTileV - 1) / kTileV;
    const double budget = static_cast<double>(vocab_size) * max_eff_row_ratio;
    const uint32_t* base = reinterpret_cast<const uint32_t*>(mask_c.data_ptr<int32_t>());
    int64_t eff_rows = 0;

    for (int64_t tile = 0; tile < n_tiles; ++tile) {
        const int64_t rem = vocab_size - tile * kTileV;
        const uint64_t tile_tail = (rem >= kTileV)
            ? ~0ULL
            : ((rem <= 0) ? 0ULL : ((1ULL << rem) - 1ULL));
        const int64_t w0 = tile * 2;
        uint64_t u64 = 0ULL;
        int64_t active_m = 0;

        if (w0 + 2 <= mask_words) {
            for (int64_t m = 0; m < M; ++m) {
                const uint32_t* rp = base + m * mask_words + w0;
                const uint64_t val =
                    (static_cast<uint64_t>(rp[0]) | (static_cast<uint64_t>(rp[1]) << 32)) & tile_tail;
                if (val != 0ULL) {
                    active_m += 1;
                    u64 |= val;
                }
            }
        } else {
            for (int64_t m = 0; m < M; ++m) {
                const uint32_t* rp = base + m * mask_words + w0;
                const uint32_t lo = (w0 < mask_words) ? rp[0] : 0u;
                const uint32_t hi = (w0 + 1 < mask_words) ? rp[1] : 0u;
                const uint64_t val =
                    (static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32)) & tile_tail;
                if (val != 0ULL) {
                    active_m += 1;
                    u64 |= val;
                }
            }
        }

        if (active_m > 0) {
            const int pop_u = __builtin_popcountll(u64);
            const int64_t passes = (active_m + k_chunk - 1) / k_chunk;
            eff_rows += static_cast<int64_t>(pop_u) * passes;
            if (static_cast<double>(eff_rows) > budget) {
                return false;
            }
        }
    }

    return true;
}

}  // namespace sglang_sm89
