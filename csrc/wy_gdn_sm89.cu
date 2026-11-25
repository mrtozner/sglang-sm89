#include "wy_gdn_sm89.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace sglang_sm89 {

namespace {

constexpr int kHeadDim = 128;
constexpr int kBV = 64;
constexpr int kMaxSpecLen = 16;
constexpr int kMaxCacheLen = 64;
constexpr int kBlockThreads = 128;

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__device__ __forceinline__ int64_t read_idx(
    const void* __restrict__ ptr,
    int64_t idx,
    int64_t stride,
    bool is_i64) {
    if (is_i64) {
        return reinterpret_cast<const int64_t*>(ptr)[idx * stride];
    }
    return static_cast<int64_t>(reinterpret_cast<const int32_t*>(ptr)[idx * stride]);
}

inline int scalar_kind(at::ScalarType t) {
    if (t == at::kFloat) return 0;
    if (t == at::kBFloat16) return 1;
    return 2;
}

__device__ __forceinline__ float read_scalar_elem(
    const void* __restrict__ ptr,
    int64_t idx,
    int kind) {
    if (kind == 0) {
        return reinterpret_cast<const float*>(ptr)[idx];
    }
    if (kind == 1) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
    }
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

__device__ __forceinline__ uint32_t pack_bf16_pair(float x0, float x1) {
    __nv_bfloat162 b2 = __floats2bfloat162_rn(x0, x1);
    return *reinterpret_cast<const uint32_t*>(&b2);
}

__device__ __forceinline__ uint32_t f32_to_tf32(float x) {
    uint32_t res = 0u;
#if __CUDA_ARCH__ >= 800
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(res) : "f"(x));
#else
    res = __float_as_uint(x) & 0xFFFFE000u;
#endif
    return res;
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float d[4],
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)d; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

__device__ __forceinline__ void mma_m16n8k8_tf32(
    float d[4],
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)d; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

template <typename StateT>
__device__ __forceinline__ void h0_mma_pass(
    const StateT* __restrict__ h0_head,
    const float smem_q[kMaxSpecLen][kHeadDim],
    const float smem_k[kMaxSpecLen][kHeadDim],
    int v_row_base,
    int spec_len,
    int groupID,
    int tid_in_group,
    float d_q0[4],
    float d_q1[4],
    float d_k0[4],
    float d_k1[4]) {
    if constexpr (std::is_same_v<StateT, __nv_bfloat16>) {
        // BF16 h0 -> native mma.sync.m16n8k16.f32.bf16.bf16.f32
        const int k_base0 = tid_in_group * 2;
        for (int kk = 0; kk < 8; ++kk) {
            const int k_off = kk * 16 + k_base0;
            const __nv_bfloat16* p0 = reinterpret_cast<const __nv_bfloat16*>(h0_head)
                + static_cast<size_t>(v_row_base + groupID) * kHeadDim + k_off;
            const __nv_bfloat16* p1 = reinterpret_cast<const __nv_bfloat16*>(h0_head)
                + static_cast<size_t>(v_row_base + groupID + 8) * kHeadDim + k_off;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(p0);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(p1);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(p0 + 8);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(p1 + 8);

            const uint32_t bq0 = (groupID < spec_len) ? pack_bf16_pair(smem_q[groupID][k_off], smem_q[groupID][k_off + 1]) : 0u;
            const uint32_t bq1 = (groupID < spec_len) ? pack_bf16_pair(smem_q[groupID][k_off + 8], smem_q[groupID][k_off + 9]) : 0u;
            const uint32_t bk0 = (groupID < spec_len) ? pack_bf16_pair(smem_k[groupID][k_off], smem_k[groupID][k_off + 1]) : 0u;
            const uint32_t bk1 = (groupID < spec_len) ? pack_bf16_pair(smem_k[groupID][k_off + 8], smem_k[groupID][k_off + 9]) : 0u;
            mma_m16n8k16_bf16(d_q0, a0, a1, a2, a3, bq0, bq1);
            mma_m16n8k16_bf16(d_k0, a0, a1, a2, a3, bk0, bk1);

            if (spec_len > 8) {
                const int t_hi = 8 + groupID;
                const uint32_t bq2 = (t_hi < spec_len) ? pack_bf16_pair(smem_q[t_hi][k_off], smem_q[t_hi][k_off + 1]) : 0u;
                const uint32_t bq3 = (t_hi < spec_len) ? pack_bf16_pair(smem_q[t_hi][k_off + 8], smem_q[t_hi][k_off + 9]) : 0u;
                const uint32_t bk2 = (t_hi < spec_len) ? pack_bf16_pair(smem_k[t_hi][k_off], smem_k[t_hi][k_off + 1]) : 0u;
                const uint32_t bk3 = (t_hi < spec_len) ? pack_bf16_pair(smem_k[t_hi][k_off + 8], smem_k[t_hi][k_off + 9]) : 0u;
                mma_m16n8k16_bf16(d_q1, a0, a1, a2, a3, bq2, bq3);
                mma_m16n8k16_bf16(d_k1, a0, a1, a2, a3, bk2, bk3);
            }
        }
    } else {
        // FP32 or FP16 h0 -> mma.sync.m16n8k8.f32.tf32.tf32.f32
        for (int kk = 0; kk < 16; ++kk) {
            const int k_off = kk * 8 + tid_in_group;
            const StateT* p0 = h0_head + static_cast<size_t>(v_row_base + groupID) * kHeadDim + k_off;
            const StateT* p1 = h0_head + static_cast<size_t>(v_row_base + groupID + 8) * kHeadDim + k_off;
            const uint32_t a0 = f32_to_tf32(to_float(p0[0]));
            const uint32_t a1 = f32_to_tf32(to_float(p1[0]));
            const uint32_t a2 = f32_to_tf32(to_float(p0[4]));
            const uint32_t a3 = f32_to_tf32(to_float(p1[4]));

            const uint32_t bq0 = (groupID < spec_len) ? f32_to_tf32(smem_q[groupID][k_off]) : 0u;
            const uint32_t bq1 = (groupID < spec_len) ? f32_to_tf32(smem_q[groupID][k_off + 4]) : 0u;
            const uint32_t bk0 = (groupID < spec_len) ? f32_to_tf32(smem_k[groupID][k_off]) : 0u;
            const uint32_t bk1 = (groupID < spec_len) ? f32_to_tf32(smem_k[groupID][k_off + 4]) : 0u;
            mma_m16n8k8_tf32(d_q0, a0, a1, a2, a3, bq0, bq1);
            mma_m16n8k8_tf32(d_k0, a0, a1, a2, a3, bk0, bk1);

            if (spec_len > 8) {
                const int t_hi = 8 + groupID;
                const uint32_t bq2 = (t_hi < spec_len) ? f32_to_tf32(smem_q[t_hi][k_off]) : 0u;
                const uint32_t bq3 = (t_hi < spec_len) ? f32_to_tf32(smem_q[t_hi][k_off + 4]) : 0u;
                const uint32_t bk2 = (t_hi < spec_len) ? f32_to_tf32(smem_k[t_hi][k_off]) : 0u;
                const uint32_t bk3 = (t_hi < spec_len) ? f32_to_tf32(smem_k[t_hi][k_off + 4]) : 0u;
                mma_m16n8k8_tf32(d_q1, a0, a1, a2, a3, bq2, bq3);
                mma_m16n8k8_tf32(d_k1, a0, a1, a2, a3, bk2, bk3);
            }
        }
    }
}

__device__ __forceinline__ void store_mma_frag_to_smem(
    float smem[kMaxSpecLen][kBV + 1],
    const float d0[4],
    const float d1[4],
    int warp,
    int groupID,
    int tid_in_group,
    int spec_len) {
    const int r0 = warp * 16 + groupID;
    const int r1 = r0 + 8;
    const int t0 = tid_in_group * 2;
    const int t1 = t0 + 1;
    if (t0 < spec_len) {
        smem[t0][r0] = d0[0];
        smem[t0][r1] = d0[2];
    }
    if (t1 < spec_len) {
        smem[t1][r0] = d0[1];
        smem[t1][r1] = d0[3];
    }
    if (spec_len > 8) {
        const int t2 = 8 + t0;
        const int t3 = 8 + t1;
        if (t2 < spec_len) {
            smem[t2][r0] = d1[0];
            smem[t2][r1] = d1[2];
        }
        if (t3 < spec_len) {
            smem[t3][r0] = d1[1];
            smem[t3][r1] = d1[3];
        }
    }
}

template <typename ActT, typename StateT, typename CacheT, bool IS_FLUSH>
__global__ void __launch_bounds__(kBlockThreads, 2)
wy_gdn_replayssm_verify_kernel(
    const ActT* __restrict__ q,
    const ActT* __restrict__ k,
    const ActT* __restrict__ v,
    const void* __restrict__ a,
    const void* __restrict__ b,
    const float* __restrict__ A_log,
    const void* __restrict__ dt_bias,
    const StateT* __restrict__ h0,
    CacheT* __restrict__ d_cache,
    CacheT* __restrict__ k_cache,
    float* __restrict__ g_cache,
    CacheT* __restrict__ rawv_cache,
    CacheT* __restrict__ rawk_cache,
    float* __restrict__ beta_cache,
    const void* __restrict__ query_start_loc,
    const void* __restrict__ ssm_state_indices,
    const void* __restrict__ replay_indices,
    const void* __restrict__ write_pos,
    const void* __restrict__ cache_base,
    const int8_t* __restrict__ is_flush,
    ActT* __restrict__ out,
    int64_t stride_q_t,
    int64_t stride_k_t,
    int64_t stride_v_t,
    int64_t stride_a_t,
    int64_t stride_b_t,
    int64_t stride_o_t,
    int64_t stride_state_slot,
    int64_t stride_d_slot,
    int64_t stride_k_slot,
    int64_t stride_g_slot,
    int64_t stride_rawv_slot,
    int64_t stride_rawk_slot,
    int64_t stride_beta_slot,
    int64_t stride_qsl,
    int64_t stride_indices,
    int64_t stride_replay_indices,
    int64_t stride_wp,
    int64_t stride_cb,
    int64_t stride_flush,
    int v_per_h,
    int max_cache_len,
    float scale,
    float softplus_threshold,
    int a_kind,
    int b_kind,
    int dt_bias_kind,
    bool store_exact_raw,
    bool store_d_residual,
    bool store_k_residual,
    bool qsl_i64,
    bool state_idx_i64,
    bool replay_idx_i64,
    bool wp_i64,
    bool cb_i64) {
    const int i_v = static_cast<int>(blockIdx.x);
    const int v_base = i_v * kBV;
    const int i_n = static_cast<int>(blockIdx.y);
    const int i_hv = static_cast<int>(blockIdx.z);
    const int i_h = i_hv / v_per_h;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;

    const int64_t bos = read_idx(query_start_loc, i_n, stride_qsl, qsl_i64);
    const int64_t eos = read_idx(query_start_loc, i_n + 1, stride_qsl, qsl_i64);
    int spec_len = static_cast<int>(eos - bos);
    if (spec_len <= 0) return;
    if (spec_len > kMaxSpecLen) spec_len = kMaxSpecLen;

    const int64_t state_idx = read_idx(ssm_state_indices, i_n, stride_indices, state_idx_i64);
    const int64_t replay_idx = read_idx(replay_indices, i_n, stride_replay_indices, replay_idx_i64);

    if constexpr (IS_FLUSH) {
        if (state_idx <= -1 || replay_idx < 0) return;
        if (is_flush[replay_idx * stride_flush] == 0) return;
    } else {
        if (state_idx <= -1) {
            if (tid < kBV) {
                const ActT zero_val = from_float<ActT>(0.0f);
                const int v_col = v_base + tid;
                for (int t = 0; t < spec_len; ++t) {
                    const size_t o_off = static_cast<size_t>(bos + t) * stride_o_t
                                       + static_cast<size_t>(i_hv) * kHeadDim + v_col;
                    out[o_off] = zero_val;
                }
            }
            return;
        }
        if (replay_idx < 0 || is_flush[replay_idx * stride_flush] != 0) return;
    }

    int b_write_pos = 0;
    if constexpr (!IS_FLUSH) {
        b_write_pos = static_cast<int>(read_idx(write_pos, replay_idx, stride_wp, wp_i64));
        if (b_write_pos < 0) b_write_pos = 0;
        if (b_write_pos > max_cache_len) b_write_pos = max_cache_len;
    }
    const int b_cache_base = static_cast<int>(read_idx(cache_base, replay_idx, stride_cb, cb_i64));
    const int cache_mask = max_cache_len - 1;

    __shared__ float smem_q[kMaxSpecLen][kHeadDim];
    __shared__ float smem_k[kMaxSpecLen][kHeadDim];
    __shared__ float s_hw_q[kMaxSpecLen][kBV + 1];
    __shared__ float s_hw_k[kMaxSpecLen][kBV + 1];
    __shared__ float s_kk[kMaxSpecLen][kMaxSpecLen];
    __shared__ float s_kq[kMaxSpecLen][kMaxSpecLen];
    __shared__ float s_A[kMaxSpecLen][kMaxSpecLen];
    __shared__ float s_F[kMaxSpecLen][kMaxSpecLen];
    __shared__ float s_k_hist[8][kHeadDim];
    __shared__ float s_d_hist[8][kBV + 1];
    __shared__ float s_qnorm[kMaxSpecLen][4];
    __shared__ float s_knorm[kMaxSpecLen][4];
    __shared__ float s_g[kMaxSpecLen];
    __shared__ float s_G[kMaxSpecLen];
    __shared__ float s_expG[kMaxSpecLen];
    __shared__ float s_beta[kMaxSpecLen];
    __shared__ float s_replay_decay[kMaxCacheLen];
    __shared__ float s_total_decay;

    // 1. Scalar gates (g_s, beta_s, G_s, expG_s) and g_cache / beta_cache writes
    if (tid < spec_len) {
        const float A_log_val = A_log[i_hv];
        const float dt_bias_val = read_scalar_elem(dt_bias, i_hv, dt_bias_kind);
        const float a_val = read_scalar_elem(a, static_cast<int64_t>(bos + tid) * stride_a_t + i_hv, a_kind);
        const float b_val = read_scalar_elem(b, static_cast<int64_t>(bos + tid) * stride_b_t + i_hv, b_kind);
        const float x = a_val + dt_bias_val;
        const float sp_x = (x <= softplus_threshold) ? log1pf(expf(x)) : x;
        const float g_val = -expf(A_log_val) * sp_x;
        const float beta_val = 1.0f / (1.0f + expf(-b_val));
        s_g[tid] = g_val;
        s_beta[tid] = beta_val;

        if (i_v == 0 && (b_write_pos + tid < max_cache_len)) {
            const int phys_spec = (b_cache_base + b_write_pos + tid) & cache_mask;
            g_cache[static_cast<size_t>(replay_idx) * stride_g_slot + static_cast<size_t>(i_hv) * max_cache_len + phys_spec] = g_val;
            if (beta_cache != nullptr) {
                beta_cache[static_cast<size_t>(replay_idx) * stride_beta_slot + static_cast<size_t>(i_hv) * max_cache_len + phys_spec] = beta_val;
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < spec_len; ++t) {
            acc += s_g[t];
            s_G[t] = acc;
            s_expG[t] = expf(acc);
        }
    }

    // 2. Load q, k once into shared memory and accumulate L2 norms
    for (int t = 0; t < spec_len; ++t) {
        const float q_raw = to_float(q[static_cast<size_t>(bos + t) * stride_q_t + static_cast<size_t>(i_h) * kHeadDim + tid]);
        const float k_raw = to_float(k[static_cast<size_t>(bos + t) * stride_k_t + static_cast<size_t>(i_h) * kHeadDim + tid]);
        smem_q[t][tid] = q_raw;
        smem_k[t][tid] = k_raw;
        const float q_sq = warp_sum(q_raw * q_raw);
        const float k_sq = warp_sum(k_raw * k_raw);
        if (lane == 0) {
            s_qnorm[t][warp] = q_sq;
            s_knorm[t][warp] = k_sq;
        }
    }
    __syncthreads();

    const bool write_k = (i_v == 0) && (i_hv == i_h * v_per_h);
    for (int t = 0; t < spec_len; ++t) {
        const float q_sum = s_qnorm[t][0] + s_qnorm[t][1] + s_qnorm[t][2] + s_qnorm[t][3];
        const float k_sum = s_knorm[t][0] + s_knorm[t][1] + s_knorm[t][2] + s_knorm[t][3];
        const float q_rnorm = (1.0f / sqrtf(q_sum + 1e-6f)) * scale;
        const float k_rnorm = 1.0f / sqrtf(k_sum + 1e-6f);

        const float q_precise = smem_q[t][tid] * q_rnorm;
        const float k_raw = smem_k[t][tid];
        const float k_precise = k_raw * k_rnorm;
        smem_q[t][tid] = q_precise;
        smem_k[t][tid] = k_precise;

        if (write_k && (b_write_pos + t < max_cache_len)) {
            const int phys_spec = (b_cache_base + b_write_pos + t) & cache_mask;
            const size_t k_off = static_cast<size_t>(replay_idx) * stride_k_slot
                               + (static_cast<size_t>(i_h) * max_cache_len + phys_spec) * kHeadDim + tid;
            const CacheT k_store = from_float<CacheT>(k_precise);
            k_cache[k_off] = k_store;

            if (rawk_cache != nullptr && (store_exact_raw || store_k_residual)) {
                const size_t rk_off = static_cast<size_t>(replay_idx) * stride_rawk_slot
                                    + (static_cast<size_t>(i_h) * max_cache_len + phys_spec) * kHeadDim + tid;
                if (store_exact_raw) {
                    rawk_cache[rk_off] = from_float<CacheT>(k_raw);
                } else {
                    rawk_cache[rk_off] = from_float<CacheT>(k_precise - to_float(k_store));
                }
            }
        }
    }
    __syncthreads();

    // 3. Tensor Core pass over h0[state_idx, i_hv, v_base + 0..63, 0..127]
    const StateT* h0_head = h0 + static_cast<size_t>(state_idx) * stride_state_slot
                          + static_cast<size_t>(i_hv) * (kHeadDim * kHeadDim);
    float d_q0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d_q1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d_k0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d_k1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    h0_mma_pass<StateT>(
        h0_head, smem_q, smem_k, v_base + warp * 16, spec_len,
        groupID, tid_in_group, d_q0, d_q1, d_k0, d_k1);

    // 4. Fold circular ring history into d_q0..d_k1 when !IS_FLUSH && b_write_pos > 0
    if constexpr (!IS_FLUSH) {
        if (b_write_pos > 0) {
            if (tid < b_write_pos) {
                const int phys_c = (b_cache_base + tid) & cache_mask;
                s_replay_decay[tid] = g_cache[
                    static_cast<size_t>(replay_idx) * stride_g_slot
                    + static_cast<size_t>(i_hv) * max_cache_len + phys_c];
            }
            __syncthreads();

            if (tid == 0) {
                float g_tot = 0.0f;
                for (int c = 0; c < b_write_pos; ++c) {
                    g_tot += s_replay_decay[c];
                    s_replay_decay[c] = g_tot;
                }
                s_total_decay = expf(g_tot);
                for (int c = 0; c < b_write_pos; ++c) {
                    s_replay_decay[c] = expf(g_tot - s_replay_decay[c]);
                }
            }
            __syncthreads();

            const float total_decay = s_total_decay;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                d_q0[i] *= total_decay;
                d_q1[i] *= total_decay;
                d_k0[i] *= total_decay;
                d_k1[i] *= total_decay;
            }

            for (int c_blk = 0; c_blk < b_write_pos; c_blk += 8) {
                const int cnt = (c_blk + 8 <= b_write_pos) ? 8 : (b_write_pos - c_blk);
                for (int u = 0; u < cnt; ++u) {
                    const int c = c_blk + u;
                    const int phys_c = (b_cache_base + c) & cache_mask;
                    const size_t k_off = static_cast<size_t>(replay_idx) * stride_k_slot
                                       + (static_cast<size_t>(i_h) * max_cache_len + phys_c) * kHeadDim + tid;
                    float kh = to_float(k_cache[k_off]);
                    if (store_k_residual && rawk_cache != nullptr) {
                        const size_t rk_off = static_cast<size_t>(replay_idx) * stride_rawk_slot
                                            + (static_cast<size_t>(i_h) * max_cache_len + phys_c) * kHeadDim + tid;
                        kh += to_float(rawk_cache[rk_off]);
                    }
                    s_k_hist[u][tid] = kh;

                    if (tid < kBV) {
                        const size_t d_off = static_cast<size_t>(replay_idx) * stride_d_slot
                                           + (static_cast<size_t>(i_hv) * max_cache_len + phys_c) * kHeadDim + (v_base + tid);
                        float dh = to_float(d_cache[d_off]);
                        if (store_d_residual && rawv_cache != nullptr) {
                            const size_t rv_off = static_cast<size_t>(replay_idx) * stride_rawv_slot
                                                + (static_cast<size_t>(i_hv) * max_cache_len + phys_c) * kHeadDim + (v_base + tid);
                            dh += to_float(rawv_cache[rv_off]);
                        }
                        s_d_hist[u][tid] = s_replay_decay[c] * dh;
                    }
                }
                for (int u = cnt; u < 8; ++u) {
                    s_k_hist[u][tid] = 0.0f;
                    if (tid < kBV) s_d_hist[u][tid] = 0.0f;
                }
                __syncthreads();

                // Compute Q @ K_hist^T (warp 0) and K @ K_hist^T (warp 1) -> reuse s_A (Pq) and s_F (Pk)
                if (warp < 2) {
                    const bool is_q = (warp == 0);
                    const int t_row0 = groupID;
                    const int t_row1 = groupID + 8;
                    float p_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    const float (*src_mat)[kHeadDim] = is_q ? smem_q : smem_k;
                    for (int kk = 0; kk < 16; ++kk) {
                        const int k_off = kk * 8 + tid_in_group;
                        const uint32_t a0 = (t_row0 < spec_len) ? f32_to_tf32(src_mat[t_row0][k_off]) : 0u;
                        const uint32_t a1 = (t_row1 < spec_len) ? f32_to_tf32(src_mat[t_row1][k_off]) : 0u;
                        const uint32_t a2 = (t_row0 < spec_len) ? f32_to_tf32(src_mat[t_row0][k_off + 4]) : 0u;
                        const uint32_t a3 = (t_row1 < spec_len) ? f32_to_tf32(src_mat[t_row1][k_off + 4]) : 0u;

                        const uint32_t b0 = f32_to_tf32(s_k_hist[groupID][k_off]);
                        const uint32_t b1 = f32_to_tf32(s_k_hist[groupID][k_off + 4]);
                        mma_m16n8k8_tf32(p_acc, a0, a1, a2, a3, b0, b1);
                    }
                    const int u0 = tid_in_group * 2;
                    const int u1 = u0 + 1;
                    float (*dst_mat)[kMaxSpecLen] = is_q ? s_A : s_F;
                    if (t_row0 < spec_len) {
                        dst_mat[t_row0][u0] = p_acc[0];
                        dst_mat[t_row0][u1] = p_acc[1];
                    }
                    if (t_row1 < spec_len) {
                        dst_mat[t_row1][u0] = p_acc[2];
                        dst_mat[t_row1][u1] = p_acc[3];
                    }
                }
                __syncthreads();

                // Accumulate d_scaled^T[64, 8] @ [Pq^T, Pk^T][8, 16] directly into d_q0..d_k1
                {
                    const int r0 = warp * 16 + groupID;
                    const int r1 = r0 + 8;
                    const uint32_t a0 = f32_to_tf32(s_d_hist[tid_in_group][r0]);
                    const uint32_t a1 = f32_to_tf32(s_d_hist[tid_in_group][r1]);
                    const uint32_t a2 = f32_to_tf32(s_d_hist[tid_in_group + 4][r0]);
                    const uint32_t a3 = f32_to_tf32(s_d_hist[tid_in_group + 4][r1]);

                    const uint32_t bq0 = (groupID < spec_len) ? f32_to_tf32(s_A[groupID][tid_in_group]) : 0u;
                    const uint32_t bq1 = (groupID < spec_len) ? f32_to_tf32(s_A[groupID][tid_in_group + 4]) : 0u;
                    const uint32_t bk0 = (groupID < spec_len) ? f32_to_tf32(s_F[groupID][tid_in_group]) : 0u;
                    const uint32_t bk1 = (groupID < spec_len) ? f32_to_tf32(s_F[groupID][tid_in_group + 4]) : 0u;
                    mma_m16n8k8_tf32(d_q0, a0, a1, a2, a3, bq0, bq1);
                    mma_m16n8k8_tf32(d_k0, a0, a1, a2, a3, bk0, bk1);

                    if (spec_len > 8) {
                        const int t_hi = 8 + groupID;
                        const uint32_t bq2 = (t_hi < spec_len) ? f32_to_tf32(s_A[t_hi][tid_in_group]) : 0u;
                        const uint32_t bq3 = (t_hi < spec_len) ? f32_to_tf32(s_A[t_hi][tid_in_group + 4]) : 0u;
                        const uint32_t bk2 = (t_hi < spec_len) ? f32_to_tf32(s_F[t_hi][tid_in_group]) : 0u;
                        const uint32_t bk3 = (t_hi < spec_len) ? f32_to_tf32(s_F[t_hi][tid_in_group + 4]) : 0u;
                        mma_m16n8k8_tf32(d_q1, a0, a1, a2, a3, bq2, bq3);
                        mma_m16n8k8_tf32(d_k1, a0, a1, a2, a3, bk2, bk3);
                    }
                }
                if (c_blk + 8 < b_write_pos) __syncthreads();
            }
        }
    }

    store_mma_frag_to_smem(s_hw_q, d_q0, d_q1, warp, groupID, tid_in_group, spec_len);
    store_mma_frag_to_smem(s_hw_k, d_k0, d_k1, warp, groupID, tid_in_group, spec_len);

    // 5. Compute kk_mat = K @ K^T and kq_mat = K @ Q^T via Tensor Cores (4 warps in parallel)
    {
        const bool is_kk = (warp >= 2);
        const int n_base = ((warp & 1) == 0) ? 0 : 8;
        if (n_base < spec_len) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int kk = 0; kk < 16; ++kk) {
                const int k_off = kk * 8 + tid_in_group;
                const uint32_t a0 = (groupID < spec_len) ? f32_to_tf32(smem_k[groupID][k_off]) : 0u;
                const uint32_t a1 = (groupID + 8 < spec_len) ? f32_to_tf32(smem_k[groupID + 8][k_off]) : 0u;
                const uint32_t a2 = (groupID < spec_len) ? f32_to_tf32(smem_k[groupID][k_off + 4]) : 0u;
                const uint32_t a3 = (groupID + 8 < spec_len) ? f32_to_tf32(smem_k[groupID + 8][k_off + 4]) : 0u;

                const int col_idx = n_base + groupID;
                uint32_t b0 = 0u, b1 = 0u;
                if (col_idx < spec_len) {
                    if (is_kk) {
                        b0 = f32_to_tf32(smem_k[col_idx][k_off]);
                        b1 = f32_to_tf32(smem_k[col_idx][k_off + 4]);
                    } else {
                        b0 = f32_to_tf32(smem_q[col_idx][k_off]);
                        b1 = f32_to_tf32(smem_q[col_idx][k_off + 4]);
                    }
                }
                mma_m16n8k8_tf32(acc, a0, a1, a2, a3, b0, b1);
            }

            const int r0 = groupID;
            const int r1 = groupID + 8;
            const int c0 = n_base + tid_in_group * 2;
            const int c1 = c0 + 1;
            float (*dst)[kMaxSpecLen] = is_kk ? s_kk : s_kq;
            if (r0 < spec_len && c0 < spec_len) dst[r0][c0] = acc[0];
            if (r0 < spec_len && c1 < spec_len) dst[r0][c1] = acc[1];
            if (r1 < spec_len && c0 < spec_len) dst[r1][c0] = acc[2];
            if (r1 < spec_len && c1 < spec_len) dst[r1][c1] = acc[3];
        }
    }
    __syncthreads();

    // 6. Build strictly lower A_mat[i, j] and causal F_mat[j, i]
    const int n_sq = spec_len * spec_len;
    for (int idx = tid; idx < n_sq; idx += kBlockThreads) {
        const int i = idx / spec_len;
        const int j = idx - i * spec_len;
        s_A[i][j] = (i > j) ? (s_beta[i] * expf(s_G[i] - s_G[j]) * s_kk[i][j]) : 0.0f;
        s_F[j][i] = (j <= i) ? (expf(s_G[i] - s_G[j]) * s_kq[j][i]) : 0.0f;
    }
    __syncthreads();

    // 7. Direct forward substitution (I + A_mat) @ D_T = R^T for v = v_base + 0..63
    if (tid < kBV) {
        const int v_col = v_base + tid;
        float D_spec[kMaxSpecLen];
        for (int t = 0; t < spec_len; ++t) {
            const float v_val = to_float(v[static_cast<size_t>(bos + t) * stride_v_t + static_cast<size_t>(i_hv) * kHeadDim + v_col]);
            float d_val = s_beta[t] * (v_val - s_expG[t] * s_hw_k[t][tid]);
            for (int j = 0; j < t; ++j) {
                d_val = fmaf(-s_A[t][j], D_spec[j], d_val);
            }
            D_spec[t] = d_val;

            if (b_write_pos + t < max_cache_len) {
                const int phys_spec = (b_cache_base + b_write_pos + t) & cache_mask;
                const size_t d_off = static_cast<size_t>(replay_idx) * stride_d_slot
                                   + (static_cast<size_t>(i_hv) * max_cache_len + phys_spec) * kHeadDim + v_col;
                const CacheT d_hi = from_float<CacheT>(d_val);
                d_cache[d_off] = d_hi;

                if (rawv_cache != nullptr && (store_exact_raw || store_d_residual)) {
                    const size_t rv_off = static_cast<size_t>(replay_idx) * stride_rawv_slot
                                        + (static_cast<size_t>(i_hv) * max_cache_len + phys_spec) * kHeadDim + v_col;
                    if (store_exact_raw) {
                        rawv_cache[rv_off] = from_float<CacheT>(v_val);
                    } else {
                        rawv_cache[rv_off] = from_float<CacheT>(d_val - to_float(d_hi));
                    }
                }
            }
        }

        for (int t = 0; t < spec_len; ++t) {
            float out_val = s_expG[t] * s_hw_q[t][tid];
            for (int j = 0; j <= t; ++j) {
                out_val = fmaf(D_spec[j], s_F[j][t], out_val);
            }
            const size_t o_off = static_cast<size_t>(bos + t) * stride_o_t
                               + static_cast<size_t>(i_hv) * kHeadDim + v_col;
            out[o_off] = from_float<ActT>(out_val);
        }
    }
}

template <typename StateT, typename CacheT>
__global__ void __launch_bounds__(kBlockThreads, 2)
wy_gdn_replayssm_flush_fold_kernel(
    StateT* __restrict__ h0,
    const CacheT* __restrict__ d_cache,
    const CacheT* __restrict__ k_cache,
    const float* __restrict__ g_cache,
    const CacheT* __restrict__ rawv_cache,
    const CacheT* __restrict__ rawk_cache,
    const float* __restrict__ beta_cache,
    const void* __restrict__ ssm_state_indices,
    const void* __restrict__ replay_indices,
    const void* __restrict__ write_pos,
    const void* __restrict__ cache_base,
    const int8_t* __restrict__ is_flush,
    int64_t stride_state_slot,
    int64_t stride_d_slot,
    int64_t stride_k_slot,
    int64_t stride_g_slot,
    int64_t stride_rawv_slot,
    int64_t stride_rawk_slot,
    int64_t stride_beta_slot,
    int64_t stride_indices,
    int64_t stride_replay_indices,
    int64_t stride_wp,
    int64_t stride_cb,
    int64_t stride_flush,
    int v_per_h,
    int max_cache_len,
    bool store_exact_raw,
    bool store_d_residual,
    bool store_k_residual,
    bool state_idx_i64,
    bool replay_idx_i64,
    bool wp_i64,
    bool cb_i64) {
    const int i_v = static_cast<int>(blockIdx.x);
    const int v_base = i_v * kBV;
    const int i_n = static_cast<int>(blockIdx.y);
    const int i_hv = static_cast<int>(blockIdx.z);
    const int i_h = i_hv / v_per_h;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int64_t state_idx = read_idx(ssm_state_indices, i_n, stride_indices, state_idx_i64);
    if (state_idx <= -1) return;
    const int64_t replay_idx = read_idx(replay_indices, i_n, stride_replay_indices, replay_idx_i64);
    if (replay_idx < 0 || is_flush[replay_idx * stride_flush] == 0) return;

    int b_write_pos = static_cast<int>(read_idx(write_pos, replay_idx, stride_wp, wp_i64));
    if (b_write_pos <= 0) return;
    if (b_write_pos > max_cache_len) b_write_pos = max_cache_len;
    const int b_cache_base = static_cast<int>(read_idx(cache_base, replay_idx, stride_cb, cb_i64));
    const int cache_mask = max_cache_len - 1;

    StateT* h0_head = h0 + static_cast<size_t>(state_idx) * stride_state_slot
                    + static_cast<size_t>(i_hv) * (kHeadDim * kHeadDim);

    if (store_exact_raw) {
        // Exact-raw fold matching SGLang _launch_gdn_exact_fold (write_pos left intact)
        __shared__ float s_k_norm[kHeadDim];
        __shared__ float s_knorm_sum[4];

        for (int c = 0; c < b_write_pos; ++c) {
            const int phys_c = (b_cache_base + c) & cache_mask;
            const float g_c = g_cache[
                static_cast<size_t>(replay_idx) * stride_g_slot
                + static_cast<size_t>(i_hv) * max_cache_len + phys_c];
            const float beta_c = (beta_cache != nullptr)
                ? beta_cache[
                    static_cast<size_t>(replay_idx) * stride_beta_slot
                    + static_cast<size_t>(i_hv) * max_cache_len + phys_c]
                : 0.0f;
            const float dec = expf(g_c);

            float k_raw = (rawk_cache != nullptr)
                ? to_float(rawk_cache[
                    static_cast<size_t>(replay_idx) * stride_rawk_slot
                    + (static_cast<size_t>(i_h) * max_cache_len + phys_c) * kHeadDim + tid])
                : 0.0f;
            const float k_sq = warp_sum(k_raw * k_raw);
            if (lane == 0) s_knorm_sum[warp] = k_sq;
            __syncthreads();
            const float k_tot = s_knorm_sum[0] + s_knorm_sum[1] + s_knorm_sum[2] + s_knorm_sum[3];
            s_k_norm[tid] = k_raw * (1.0f / sqrtf(k_tot + 1e-6f));
            __syncthreads();

            for (int r = 0; r < 16; ++r) {
                const int v_row = v_base + warp * 16 + r;
                StateT* row_ptr = h0_head + static_cast<size_t>(v_row) * kHeadDim;
                float h0_0 = dec * to_float(row_ptr[0 * 32 + lane]);
                float h0_1 = dec * to_float(row_ptr[1 * 32 + lane]);
                float h0_2 = dec * to_float(row_ptr[2 * 32 + lane]);
                float h0_3 = dec * to_float(row_ptr[3 * 32 + lane]);

                const float v_hat = warp_sum(
                    h0_0 * s_k_norm[0 * 32 + lane] +
                    h0_1 * s_k_norm[1 * 32 + lane] +
                    h0_2 * s_k_norm[2 * 32 + lane] +
                    h0_3 * s_k_norm[3 * 32 + lane]);

                float delta = 0.0f;
                if (lane == 0) {
                    const float v_raw = (rawv_cache != nullptr)
                        ? to_float(rawv_cache[
                            static_cast<size_t>(replay_idx) * stride_rawv_slot
                            + (static_cast<size_t>(i_hv) * max_cache_len + phys_c) * kHeadDim + v_row])
                        : 0.0f;
                    delta = beta_c * (v_raw - v_hat);
                }
                delta = __shfl_sync(0xFFFFFFFFu, delta, 0);
                row_ptr[0 * 32 + lane] = from_float<StateT>(fmaf(delta, s_k_norm[0 * 32 + lane], h0_0));
                row_ptr[1 * 32 + lane] = from_float<StateT>(fmaf(delta, s_k_norm[1 * 32 + lane], h0_1));
                row_ptr[2 * 32 + lane] = from_float<StateT>(fmaf(delta, s_k_norm[2 * 32 + lane], h0_2));
                row_ptr[3 * 32 + lane] = from_float<StateT>(fmaf(delta, s_k_norm[3 * 32 + lane], h0_3));
            }
            __syncthreads();
        }
        return;
    }

    // Residual / projected-D fold: h0 = exp(sum g) * h0 + sum_c (decay[c] * d_hist[c]) @ k_hist[c]^T
    __shared__ float s_d[kMaxCacheLen][kBV + 1];
    __shared__ float s_replay_decay[kMaxCacheLen];
    __shared__ float s_total_decay;

    if (tid < b_write_pos) {
        const int phys_c = (b_cache_base + tid) & cache_mask;
        s_replay_decay[tid] = g_cache[
            static_cast<size_t>(replay_idx) * stride_g_slot
            + static_cast<size_t>(i_hv) * max_cache_len + phys_c];
    }
    __syncthreads();

    if (tid == 0) {
        float g_tot = 0.0f;
        for (int c = 0; c < b_write_pos; ++c) {
            g_tot += s_replay_decay[c];
            s_replay_decay[c] = g_tot;
        }
        s_total_decay = expf(g_tot);
        for (int c = 0; c < b_write_pos; ++c) {
            s_replay_decay[c] = expf(g_tot - s_replay_decay[c]);
        }
    }
    __syncthreads();

    if (tid < kBV) {
        const int v_col = v_base + tid;
        for (int c = 0; c < b_write_pos; ++c) {
            const int phys_c = (b_cache_base + c) & cache_mask;
            const size_t d_off = static_cast<size_t>(replay_idx) * stride_d_slot
                               + (static_cast<size_t>(i_hv) * max_cache_len + phys_c) * kHeadDim + v_col;
            float dh = to_float(d_cache[d_off]);
            if (store_d_residual && rawv_cache != nullptr) {
                const size_t rv_off = static_cast<size_t>(replay_idx) * stride_rawv_slot
                                    + (static_cast<size_t>(i_hv) * max_cache_len + phys_c) * kHeadDim + v_col;
                dh += to_float(rawv_cache[rv_off]);
            }
            s_d[c][tid] = s_replay_decay[c] * dh;
        }
    }
    __syncthreads();

    const float total_decay = s_total_decay;
    const CacheT* k_slot_base = k_cache + static_cast<size_t>(replay_idx) * stride_k_slot
                              + static_cast<size_t>(i_h) * max_cache_len * kHeadDim;
    const CacheT* rk_slot_base = (store_k_residual && rawk_cache != nullptr)
                               ? (rawk_cache + static_cast<size_t>(replay_idx) * stride_rawk_slot
                                  + static_cast<size_t>(i_h) * max_cache_len * kHeadDim)
                               : nullptr;

    for (int r = 0; r < 16; ++r) {
        const int v_local = warp * 16 + r;
        const int v_row = v_base + v_local;
        StateT* row_ptr = h0_head + static_cast<size_t>(v_row) * kHeadDim;
        float h0_0 = total_decay * to_float(row_ptr[0 * 32 + lane]);
        float h0_1 = total_decay * to_float(row_ptr[1 * 32 + lane]);
        float h0_2 = total_decay * to_float(row_ptr[2 * 32 + lane]);
        float h0_3 = total_decay * to_float(row_ptr[3 * 32 + lane]);

        for (int c = 0; c < b_write_pos; ++c) {
            const float d_val = s_d[c][v_local];
            const int phys_c = (b_cache_base + c) & cache_mask;
            const size_t c_off = static_cast<size_t>(phys_c) * kHeadDim;
            float k0 = to_float(__ldg(k_slot_base + c_off + 0 * 32 + lane));
            float k1 = to_float(__ldg(k_slot_base + c_off + 1 * 32 + lane));
            float k2 = to_float(__ldg(k_slot_base + c_off + 2 * 32 + lane));
            float k3 = to_float(__ldg(k_slot_base + c_off + 3 * 32 + lane));
            if (rk_slot_base != nullptr) {
                k0 += to_float(__ldg(rk_slot_base + c_off + 0 * 32 + lane));
                k1 += to_float(__ldg(rk_slot_base + c_off + 1 * 32 + lane));
                k2 += to_float(__ldg(rk_slot_base + c_off + 2 * 32 + lane));
                k3 += to_float(__ldg(rk_slot_base + c_off + 3 * 32 + lane));
            }
            h0_0 = fmaf(d_val, k0, h0_0);
            h0_1 = fmaf(d_val, k1, h0_1);
            h0_2 = fmaf(d_val, k2, h0_2);
            h0_3 = fmaf(d_val, k3, h0_3);
        }

        row_ptr[0 * 32 + lane] = from_float<StateT>(h0_0);
        row_ptr[1 * 32 + lane] = from_float<StateT>(h0_1);
        row_ptr[2 * 32 + lane] = from_float<StateT>(h0_2);
        row_ptr[3 * 32 + lane] = from_float<StateT>(h0_3);
    }
}

bool is_index_type(at::ScalarType t) {
    return t == at::kInt || t == at::kLong;
}

bool is_float_act_type(at::ScalarType t) {
    return t == at::kBFloat16 || t == at::kHalf || t == at::kFloat;
}

bool is_state_type(at::ScalarType t) {
    return t == at::kBFloat16 || t == at::kHalf || t == at::kFloat;
}

template <typename ActT, typename StateT, typename CacheT, bool IS_FLUSH>
void dispatch_verify_kernel(
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
    int B,
    int H,
    int HV,
    int max_cache_len,
    float scale,
    float softplus_threshold,
    cudaStream_t stream) {
    const int a_kind = scalar_kind(a.scalar_type());
    const int b_kind = scalar_kind(b.scalar_type());
    const int dt_bias_kind = scalar_kind(dt_bias.scalar_type());
    const bool has_beta = beta_cache.has_value() && beta_cache->defined();
    const bool has_rawv = rawv_cache.has_value() && rawv_cache->defined();
    const bool has_rawk = rawk_cache.has_value() && rawk_cache->defined();
    const bool store_exact_raw = has_rawv && has_rawk && has_beta;
    const bool store_d_residual = !store_exact_raw && has_rawv;
    const bool store_k_residual = !store_exact_raw && has_rawk;

    CacheT* rawv_ptr = has_rawv ? reinterpret_cast<CacheT*>(const_cast<void*>(rawv_cache->data_ptr())) : nullptr;
    CacheT* rawk_ptr = has_rawk ? reinterpret_cast<CacheT*>(const_cast<void*>(rawk_cache->data_ptr())) : nullptr;
    float* beta_ptr = has_beta ? const_cast<float*>(beta_cache->data_ptr<float>()) : nullptr;

    dim3 grid(2, static_cast<unsigned>(B), static_cast<unsigned>(HV));
    dim3 block(kBlockThreads, 1, 1);

    wy_gdn_replayssm_verify_kernel<ActT, StateT, CacheT, IS_FLUSH><<<grid, block, 0, stream>>>(
        reinterpret_cast<const ActT*>(q.data_ptr()),
        reinterpret_cast<const ActT*>(k.data_ptr()),
        reinterpret_cast<const ActT*>(v.data_ptr()),
        a.data_ptr(),
        b.data_ptr(),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr(),
        reinterpret_cast<const StateT*>(checkpoint_state.data_ptr()),
        reinterpret_cast<CacheT*>(d_cache.data_ptr()),
        reinterpret_cast<CacheT*>(k_cache.data_ptr()),
        g_cache.data_ptr<float>(),
        rawv_ptr,
        rawk_ptr,
        beta_ptr,
        query_start_loc.data_ptr(),
        ssm_state_indices.data_ptr(),
        replay_indices.data_ptr(),
        write_pos.data_ptr(),
        cache_base.data_ptr(),
        reinterpret_cast<const int8_t*>(is_flush.data_ptr()),
        reinterpret_cast<ActT*>(out.data_ptr()),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        a.stride(0),
        b.stride(0),
        out.stride(0),
        checkpoint_state.stride(0),
        d_cache.stride(0),
        k_cache.stride(0),
        g_cache.stride(0),
        has_rawv ? rawv_cache->stride(0) : 0,
        has_rawk ? rawk_cache->stride(0) : 0,
        has_beta ? beta_cache->stride(0) : 0,
        query_start_loc.stride(0),
        ssm_state_indices.stride(0),
        replay_indices.stride(0),
        write_pos.stride(0),
        cache_base.stride(0),
        is_flush.stride(0),
        HV / H,
        max_cache_len,
        scale,
        softplus_threshold,
        a_kind,
        b_kind,
        dt_bias_kind,
        store_exact_raw,
        store_d_residual,
        store_k_residual,
        query_start_loc.scalar_type() == at::kLong,
        ssm_state_indices.scalar_type() == at::kLong,
        replay_indices.scalar_type() == at::kLong,
        write_pos.scalar_type() == at::kLong,
        cache_base.scalar_type() == at::kLong);
}

template <typename ActT, typename StateT, typename CacheT>
void dispatch_verify_flush(
    bool is_flush_pass,
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
    int B,
    int H,
    int HV,
    int max_cache_len,
    float scale,
    float softplus_threshold,
    cudaStream_t stream) {
    if (is_flush_pass) {
        dispatch_verify_kernel<ActT, StateT, CacheT, true>(
            q, k, v, a, b, A_log, dt_bias, checkpoint_state,
            d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
            query_start_loc, ssm_state_indices, replay_indices,
            write_pos, cache_base, is_flush, out,
            B, H, HV, max_cache_len, scale, softplus_threshold, stream);
    } else {
        dispatch_verify_kernel<ActT, StateT, CacheT, false>(
            q, k, v, a, b, A_log, dt_bias, checkpoint_state,
            d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
            query_start_loc, ssm_state_indices, replay_indices,
            write_pos, cache_base, is_flush, out,
            B, H, HV, max_cache_len, scale, softplus_threshold, stream);
    }
}

template <typename StateT, typename CacheT>
void launch_flush_fold_typed(
    at::Tensor& checkpoint_state,
    const at::Tensor& d_cache,
    const at::Tensor& k_cache,
    const at::Tensor& g_cache,
    const c10::optional<at::Tensor>& rawv_cache,
    const c10::optional<at::Tensor>& rawk_cache,
    const c10::optional<at::Tensor>& beta_cache,
    const at::Tensor& ssm_state_indices,
    const at::Tensor& replay_indices,
    const at::Tensor& write_pos,
    const at::Tensor& cache_base,
    const at::Tensor& is_flush,
    int B,
    int H,
    int HV,
    int max_cache_len,
    bool store_exact_raw,
    cudaStream_t stream) {
    const bool has_rawv = rawv_cache.has_value() && rawv_cache->defined();
    const bool has_rawk = rawk_cache.has_value() && rawk_cache->defined();
    const bool has_beta = beta_cache.has_value() && beta_cache->defined();
    const bool eff_exact_raw = store_exact_raw && has_rawv && has_rawk && has_beta;
    const bool store_d_residual = !eff_exact_raw && has_rawv;
    const bool store_k_residual = !eff_exact_raw && has_rawk;

    const CacheT* rawv_ptr = has_rawv ? reinterpret_cast<const CacheT*>(rawv_cache->data_ptr()) : nullptr;
    const CacheT* rawk_ptr = has_rawk ? reinterpret_cast<const CacheT*>(rawk_cache->data_ptr()) : nullptr;
    const float* beta_ptr = has_beta ? beta_cache->data_ptr<float>() : nullptr;

    dim3 grid(2, static_cast<unsigned>(B), static_cast<unsigned>(HV));
    dim3 block(kBlockThreads, 1, 1);

    wy_gdn_replayssm_flush_fold_kernel<StateT, CacheT><<<grid, block, 0, stream>>>(
        reinterpret_cast<StateT*>(checkpoint_state.data_ptr()),
        reinterpret_cast<const CacheT*>(d_cache.data_ptr()),
        reinterpret_cast<const CacheT*>(k_cache.data_ptr()),
        g_cache.data_ptr<float>(),
        rawv_ptr,
        rawk_ptr,
        beta_ptr,
        ssm_state_indices.data_ptr(),
        replay_indices.data_ptr(),
        write_pos.data_ptr(),
        cache_base.data_ptr(),
        reinterpret_cast<const int8_t*>(is_flush.data_ptr()),
        checkpoint_state.stride(0),
        d_cache.stride(0),
        k_cache.stride(0),
        g_cache.stride(0),
        has_rawv ? rawv_cache->stride(0) : 0,
        has_rawk ? rawk_cache->stride(0) : 0,
        has_beta ? beta_cache->stride(0) : 0,
        ssm_state_indices.stride(0),
        replay_indices.stride(0),
        write_pos.stride(0),
        cache_base.stride(0),
        is_flush.stride(0),
        HV / H,
        max_cache_len,
        eff_exact_raw,
        store_d_residual,
        store_k_residual,
        ssm_state_indices.scalar_type() == at::kLong,
        replay_indices.scalar_type() == at::kLong,
        write_pos.scalar_type() == at::kLong,
        cache_base.scalar_type() == at::kLong);
}

void check_3d_inner_contig(const at::Tensor& t, int64_t d1, int64_t d2, const char* name) {
    TORCH_CHECK(t.dim() == 3 && t.size(1) == d1 && t.size(2) == d2, name, " shape mismatch");
    TORCH_CHECK(t.stride(2) == 1 && t.stride(1) == d2, name, " inner dimensions must be contiguous");
}

void check_4d_inner_contig(const at::Tensor& t, int64_t d1, int64_t d2, int64_t d3, const char* name) {
    TORCH_CHECK(t.dim() == 4 && t.size(1) == d1 && t.size(2) == d2 && t.size(3) == d3, name, " shape mismatch");
    TORCH_CHECK(t.stride(3) == 1 && t.stride(2) == d3 && t.stride(1) == d2 * d3, name, " inner dimensions must be contiguous");
}

}  // namespace

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
    double softplus_threshold) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && a.is_cuda() && b.is_cuda(), "q,k,v,a,b must be on CUDA");
    TORCH_CHECK(A_log.is_cuda() && dt_bias.is_cuda() && checkpoint_state.is_cuda() && out.is_cuda(), "state/out tensors must be on CUDA");
    TORCH_CHECK(d_cache.is_cuda() && k_cache.is_cuda() && g_cache.is_cuda(), "ring caches must be on CUDA");
    TORCH_CHECK(query_start_loc.is_cuda() && ssm_state_indices.is_cuda() && replay_indices.is_cuda(), "indices must be on CUDA");
    TORCH_CHECK(write_pos.is_cuda() && cache_base.is_cuda() && is_flush.is_cuda(), "replay metadata must be on CUDA");

    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && out.dim() == 3, "q,k,v,out must be 3D");
    const int64_t total_tokens = q.size(0);
    const int64_t H = q.size(1);
    const int64_t K = q.size(2);
    const int64_t HV = v.size(1);
    const int64_t V = v.size(2);

    TORCH_CHECK(K == kHeadDim && V == kHeadDim, "wy_gdn_sm89 requires K==128 and V==128");
    TORCH_CHECK(H > 0 && HV >= H && (HV % H == 0), "HV must be a positive multiple of H");
    TORCH_CHECK(max_spec_len >= 1 && max_spec_len <= kMaxSpecLen, "max_spec_len must be in [1, 16]");
    TORCH_CHECK(max_cache_len >= 1 && max_cache_len <= kMaxCacheLen && ((max_cache_len & (max_cache_len - 1)) == 0),
                "max_cache_len must be a power of 2 in [1, 64]");

    TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) >= 2, "query_start_loc must have length B+1 >= 2");
    const int64_t B = query_start_loc.size(0) - 1;
    TORCH_CHECK(ssm_state_indices.dim() == 1 && ssm_state_indices.size(0) >= B, "ssm_state_indices size mismatch");
    TORCH_CHECK(replay_indices.dim() == 1 && replay_indices.size(0) >= B, "replay_indices size mismatch");

    TORCH_CHECK(k.size(0) == total_tokens && v.size(0) == total_tokens && out.size(0) == total_tokens, "token dim mismatch");
    TORCH_CHECK(a.dim() == 2 && a.size(0) == total_tokens && a.size(1) == HV && a.stride(1) == 1, "a shape/stride mismatch");
    TORCH_CHECK(b.dim() == 2 && b.size(0) == total_tokens && b.size(1) == HV && b.stride(1) == 1, "b shape/stride mismatch");
    check_3d_inner_contig(q, H, K, "q");
    check_3d_inner_contig(k, H, K, "k");
    check_3d_inner_contig(v, HV, V, "v");
    check_3d_inner_contig(out, HV, V, "out");

    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == HV && A_log.is_contiguous() && A_log.scalar_type() == at::kFloat,
                "A_log must be contiguous float32 [HV]");
    TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == HV && dt_bias.is_contiguous() && is_float_act_type(dt_bias.scalar_type()),
                "dt_bias must be contiguous float32/bfloat16/float16 [HV]");

    check_4d_inner_contig(checkpoint_state, HV, V, K, "checkpoint_state");
    const int64_t num_replay_slots = d_cache.size(0);
    check_4d_inner_contig(d_cache, HV, max_cache_len, V, "d_cache");
    check_4d_inner_contig(k_cache, H, max_cache_len, K, "k_cache");
    TORCH_CHECK(k_cache.size(0) == num_replay_slots, "k_cache slot count mismatch");
    TORCH_CHECK(g_cache.dim() == 3 && g_cache.size(0) == num_replay_slots && g_cache.size(1) == HV && g_cache.size(2) == max_cache_len,
                "g_cache shape mismatch");
    TORCH_CHECK(g_cache.stride(2) == 1 && g_cache.stride(1) == max_cache_len && g_cache.scalar_type() == at::kFloat,
                "g_cache must be float32 with contiguous inner dims");

    if (rawv_cache.has_value() && rawv_cache->defined()) {
        TORCH_CHECK(rawv_cache->is_cuda() && rawv_cache->size(0) == num_replay_slots, "rawv_cache device/slot mismatch");
        check_4d_inner_contig(*rawv_cache, HV, max_cache_len, V, "rawv_cache");
        TORCH_CHECK(rawv_cache->scalar_type() == d_cache.scalar_type(), "rawv_cache dtype must match d_cache");
    }
    if (rawk_cache.has_value() && rawk_cache->defined()) {
        TORCH_CHECK(rawk_cache->is_cuda() && rawk_cache->size(0) == num_replay_slots, "rawk_cache device/slot mismatch");
        check_4d_inner_contig(*rawk_cache, H, max_cache_len, K, "rawk_cache");
        TORCH_CHECK(rawk_cache->scalar_type() == k_cache.scalar_type(), "rawk_cache dtype must match k_cache");
    }
    if (beta_cache.has_value() && beta_cache->defined()) {
        TORCH_CHECK(beta_cache->is_cuda() && beta_cache->dim() == 3 && beta_cache->size(0) == num_replay_slots &&
                    beta_cache->size(1) == HV && beta_cache->size(2) == max_cache_len, "beta_cache shape mismatch");
        TORCH_CHECK(beta_cache->stride(2) == 1 && beta_cache->stride(1) == max_cache_len && beta_cache->scalar_type() == at::kFloat,
                    "beta_cache must be float32 with contiguous inner dims");
    }

    TORCH_CHECK(is_index_type(query_start_loc.scalar_type()) && is_index_type(ssm_state_indices.scalar_type()) &&
                is_index_type(replay_indices.scalar_type()) && is_index_type(write_pos.scalar_type()) &&
                is_index_type(cache_base.scalar_type()), "index tensors must be int32 or int64");
    TORCH_CHECK(is_flush.scalar_type() == at::kChar || is_flush.scalar_type() == at::kByte || is_flush.scalar_type() == at::kBool,
                "is_flush must be int8/uint8/bool");
    TORCH_CHECK(write_pos.dim() == 1 && write_pos.size(0) == num_replay_slots, "write_pos shape mismatch");
    TORCH_CHECK(cache_base.dim() == 1 && cache_base.size(0) == num_replay_slots, "cache_base shape mismatch");
    TORCH_CHECK(is_flush.dim() == 1 && is_flush.size(0) == num_replay_slots, "is_flush shape mismatch");

    const at::ScalarType act_t = q.scalar_type();
    TORCH_CHECK(is_float_act_type(act_t), "q must be bfloat16, float16, or float32");
    TORCH_CHECK(k.scalar_type() == act_t && v.scalar_type() == act_t && out.scalar_type() == act_t,
                "q,k,v,out must share scalar type");
    TORCH_CHECK(is_float_act_type(a.scalar_type()) && is_float_act_type(b.scalar_type()),
                "a,b must be bfloat16, float16, or float32");
    const at::ScalarType state_t = checkpoint_state.scalar_type();
    TORCH_CHECK(is_state_type(state_t), "checkpoint_state must be bfloat16, float16, or float32");
    const at::ScalarType cache_t = d_cache.scalar_type();
    TORCH_CHECK(k_cache.scalar_type() == cache_t && cache_t == act_t,
                "d_cache and k_cache must match activation scalar type");

    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int B_i = static_cast<int>(B);
    const int H_i = static_cast<int>(H);
    const int HV_i = static_cast<int>(HV);
    const int max_c_i = static_cast<int>(max_cache_len);
    const float scale_f = static_cast<float>(scale);
    const float sp_thresh_f = static_cast<float>(softplus_threshold);

    if (act_t == at::kBFloat16) {
        if (state_t == at::kBFloat16) {
            dispatch_verify_flush<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
                is_flush_pass, q, k, v, a, b, A_log, dt_bias, checkpoint_state,
                d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
                query_start_loc, ssm_state_indices, replay_indices,
                write_pos, cache_base, is_flush, out,
                B_i, H_i, HV_i, max_c_i, scale_f, sp_thresh_f, stream);
        } else if (state_t == at::kFloat) {
            dispatch_verify_flush<__nv_bfloat16, float, __nv_bfloat16>(
                is_flush_pass, q, k, v, a, b, A_log, dt_bias, checkpoint_state,
                d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
                query_start_loc, ssm_state_indices, replay_indices,
                write_pos, cache_base, is_flush, out,
                B_i, H_i, HV_i, max_c_i, scale_f, sp_thresh_f, stream);
        } else {
            TORCH_CHECK(false, "Unsupported checkpoint_state dtype for bfloat16 activations");
        }
    } else if (act_t == at::kHalf) {
        if (state_t == at::kHalf) {
            dispatch_verify_flush<__half, __half, __half>(
                is_flush_pass, q, k, v, a, b, A_log, dt_bias, checkpoint_state,
                d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
                query_start_loc, ssm_state_indices, replay_indices,
                write_pos, cache_base, is_flush, out,
                B_i, H_i, HV_i, max_c_i, scale_f, sp_thresh_f, stream);
        } else if (state_t == at::kFloat) {
            dispatch_verify_flush<__half, float, __half>(
                is_flush_pass, q, k, v, a, b, A_log, dt_bias, checkpoint_state,
                d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
                query_start_loc, ssm_state_indices, replay_indices,
                write_pos, cache_base, is_flush, out,
                B_i, H_i, HV_i, max_c_i, scale_f, sp_thresh_f, stream);
        } else {
            TORCH_CHECK(false, "Unsupported checkpoint_state dtype for float16 activations");
        }
    } else {
        TORCH_CHECK(state_t == at::kFloat, "float32 activations require float32 checkpoint_state");
        dispatch_verify_flush<float, float, float>(
            is_flush_pass, q, k, v, a, b, A_log, dt_bias, checkpoint_state,
            d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
            query_start_loc, ssm_state_indices, replay_indices,
            write_pos, cache_base, is_flush, out,
            B_i, H_i, HV_i, max_c_i, scale_f, sp_thresh_f, stream);
    }
}

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
    const c10::optional<at::Tensor>& beta_cache) {
    TORCH_CHECK(checkpoint_state.is_cuda() && d_cache.is_cuda() && k_cache.is_cuda() && g_cache.is_cuda(),
                "flush_fold state/caches must be on CUDA");
    TORCH_CHECK(ssm_state_indices.is_cuda() && replay_indices.is_cuda() &&
                write_pos.is_cuda() && cache_base.is_cuda() && is_flush.is_cuda(),
                "flush_fold indices/metadata must be on CUDA");

    const int64_t HV = checkpoint_state.size(1);
    const int64_t H = k_cache.size(1);
    const int64_t B = ssm_state_indices.size(0);
    if (B <= 0) return;

    TORCH_CHECK(H > 0 && HV >= H && (HV % H == 0), "HV must be a positive multiple of H");
    TORCH_CHECK(max_cache_len >= 1 && max_cache_len <= kMaxCacheLen && ((max_cache_len & (max_cache_len - 1)) == 0),
                "max_cache_len must be a power of 2 in [1, 64]");
    check_4d_inner_contig(checkpoint_state, HV, kHeadDim, kHeadDim, "checkpoint_state");
    check_4d_inner_contig(d_cache, HV, max_cache_len, kHeadDim, "d_cache");
    check_4d_inner_contig(k_cache, H, max_cache_len, kHeadDim, "k_cache");
    TORCH_CHECK(g_cache.dim() == 3 && g_cache.size(1) == HV && g_cache.size(2) == max_cache_len &&
                g_cache.stride(2) == 1 && g_cache.stride(1) == max_cache_len && g_cache.scalar_type() == at::kFloat,
                "g_cache mismatch");
    TORCH_CHECK(d_cache.scalar_type() == k_cache.scalar_type(), "d_cache and k_cache dtype mismatch");

    if (rawv_cache.has_value() && rawv_cache->defined()) {
        check_4d_inner_contig(*rawv_cache, HV, max_cache_len, kHeadDim, "rawv_cache");
        TORCH_CHECK(rawv_cache->scalar_type() == d_cache.scalar_type(), "rawv_cache dtype mismatch");
    }
    if (rawk_cache.has_value() && rawk_cache->defined()) {
        check_4d_inner_contig(*rawk_cache, H, max_cache_len, kHeadDim, "rawk_cache");
        TORCH_CHECK(rawk_cache->scalar_type() == k_cache.scalar_type(), "rawk_cache dtype mismatch");
    }
    if (beta_cache.has_value() && beta_cache->defined()) {
        TORCH_CHECK(beta_cache->dim() == 3 && beta_cache->size(1) == HV && beta_cache->size(2) == max_cache_len &&
                    beta_cache->stride(2) == 1 && beta_cache->stride(1) == max_cache_len &&
                    beta_cache->scalar_type() == at::kFloat, "beta_cache mismatch");
    }

    const at::ScalarType state_t = checkpoint_state.scalar_type();
    const at::ScalarType cache_t = d_cache.scalar_type();
    TORCH_CHECK(is_state_type(state_t) && is_float_act_type(cache_t), "Unsupported dtype for flush_fold");

    const at::cuda::OptionalCUDAGuard device_guard(checkpoint_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int B_i = static_cast<int>(B);
    const int H_i = static_cast<int>(H);
    const int HV_i = static_cast<int>(HV);
    const int max_c_i = static_cast<int>(max_cache_len);

    if (cache_t == at::kBFloat16) {
        if (state_t == at::kBFloat16) {
            launch_flush_fold_typed<__nv_bfloat16, __nv_bfloat16>(
                checkpoint_state, d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
                ssm_state_indices, replay_indices, write_pos, cache_base, is_flush,
                B_i, H_i, HV_i, max_c_i, store_exact_raw, stream);
        } else if (state_t == at::kFloat) {
            launch_flush_fold_typed<float, __nv_bfloat16>(
                checkpoint_state, d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
                ssm_state_indices, replay_indices, write_pos, cache_base, is_flush,
                B_i, H_i, HV_i, max_c_i, store_exact_raw, stream);
        } else {
            TORCH_CHECK(false, "Unsupported checkpoint_state dtype for bfloat16 cache");
        }
    } else if (cache_t == at::kHalf) {
        if (state_t == at::kHalf) {
            launch_flush_fold_typed<__half, __half>(
                checkpoint_state, d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
                ssm_state_indices, replay_indices, write_pos, cache_base, is_flush,
                B_i, H_i, HV_i, max_c_i, store_exact_raw, stream);
        } else if (state_t == at::kFloat) {
            launch_flush_fold_typed<float, __half>(
                checkpoint_state, d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
                ssm_state_indices, replay_indices, write_pos, cache_base, is_flush,
                B_i, H_i, HV_i, max_c_i, store_exact_raw, stream);
        } else {
            TORCH_CHECK(false, "Unsupported checkpoint_state dtype for float16 cache");
        }
    } else {
        TORCH_CHECK(state_t == at::kFloat, "float32 cache requires float32 checkpoint_state");
        launch_flush_fold_typed<float, float>(
            checkpoint_state, d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
            ssm_state_indices, replay_indices, write_pos, cache_base, is_flush,
            B_i, H_i, HV_i, max_c_i, store_exact_raw, stream);
    }
}

}  // namespace sglang_sm89
