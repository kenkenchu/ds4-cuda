#define DS4_CUDA_RUNTIME_TYPES
#include <cuda_runtime.h>

#include "ds4_cuda.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <new>
#include <cmath>
#include <cstdint>
#include <float.h>

// ============================================================
// Quantized weight block types (CUDA-compatible)
// ============================================================

#define QK_K 256

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;

// ============================================================
// IQ2_XXS lookup tables (constant memory, read-only on device)
// ============================================================

__constant__ uint8_t d_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

__constant__ uint8_t d_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

__constant__ uint64_t d_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

struct ds4_cuda_tensor {
    void *base;
    uint64_t offset;
    uint64_t bytes;
    int owner;
};

static int g_cuda_device = -1;
static char g_cuda_name[256];
static int g_cuda_cc_major = 0;
static int g_cuda_cc_minor = 0;
static uint64_t g_cuda_total = 0;
static uint64_t g_cuda_tensor_live_bytes = 0;
static uint64_t g_cuda_tensor_peak_bytes = 0;
static cudaStream_t g_cuda_prefill_stream = nullptr;
static cudaStream_t g_cuda_decode_stream = nullptr;

static void ds4_cuda_reset_state(void) {
    g_cuda_device = -1;
    g_cuda_name[0] = '\0';
    g_cuda_cc_major = 0;
    g_cuda_cc_minor = 0;
    g_cuda_total = 0;
    g_cuda_tensor_live_bytes = 0;
    g_cuda_tensor_peak_bytes = 0;
    g_cuda_prefill_stream = nullptr;
    g_cuda_decode_stream = nullptr;
}

static void ds4_cuda_report_error(const char *file, int line, const char *expr, cudaError_t err) {
    fprintf(stderr,
            "ds4: CUDA error %s:%d: %s failed: %s (%d)\n",
            file,
            line,
            expr ? expr : "<unknown>",
            cudaGetErrorString(err),
            (int)err);
}

#define DS4_CUDA_CHECK(expr)                                                   \
    do {                                                                       \
        cudaError_t _ds4_cuda_err = (expr);                                    \
        if (_ds4_cuda_err != cudaSuccess) {                                    \
            ds4_cuda_report_error(__FILE__, __LINE__, #expr, _ds4_cuda_err);   \
            goto fail;                                                         \
        }                                                                      \
    } while (0)

static inline ds4_cuda_tensor *ds4_cuda_tensor_new(void) {
    return new (std::nothrow) ds4_cuda_tensor{};
}

static inline const ds4_cuda_tensor *ds4_cuda_tensor_const_obj(const ds4_cuda_tensor *tensor) {
    return tensor;
}

static inline ds4_cuda_tensor *ds4_cuda_tensor_obj(ds4_cuda_tensor *tensor) {
    return tensor;
}

static bool ds4_cuda_tensor_range_ok(const ds4_cuda_tensor *tensor, uint64_t offset, uint64_t bytes) {
    if (!tensor) return false;
    if (offset > tensor->bytes) return false;
    if (bytes > tensor->bytes - offset) return false;
    return true;
}

static void ds4_cuda_tensor_account_alloc(uint64_t bytes) {
    g_cuda_tensor_live_bytes += bytes;
    if (g_cuda_tensor_live_bytes > g_cuda_tensor_peak_bytes) {
        g_cuda_tensor_peak_bytes = g_cuda_tensor_live_bytes;
    }
}

static void ds4_cuda_tensor_account_free(uint64_t bytes) {
    if (bytes <= g_cuda_tensor_live_bytes) {
        g_cuda_tensor_live_bytes -= bytes;
    } else {
        g_cuda_tensor_live_bytes = 0;
    }
}

static int ds4_cuda_copy_host_range_to_device(const void *src, uint64_t bytes, void **dst) {
    if (!src || !dst || bytes == 0) return 0;
    if (bytes > (uint64_t)SIZE_MAX) return 0;
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaMalloc", err);
        return 0;
    }
    err = cudaMemcpy(ptr, src, (size_t)bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaMemcpyHostToDevice", err);
        cudaFree(ptr);
        return 0;
    }
    *dst = ptr;
    return 1;
}

static inline float ds4_cuda_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline float ds4_cuda_rms_scale(const float *x, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double v = (double)x[i];
        ss += v * v;
    }
    return 1.0f / sqrtf((float)(ss / (double)n) + eps);
}

static __host__ __device__ inline float ds4_cuda_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

#define DS4_CUDA_LAUNCH(kernel, grid, block, shared, stream, ...)              \
    do {                                                                       \
        kernel<<<(grid), (block), (shared), (stream)>>>(__VA_ARGS__);          \
        cudaError_t _ds4_cuda_launch_err = cudaPeekAtLastError();              \
        if (_ds4_cuda_launch_err != cudaSuccess) {                              \
            ds4_cuda_report_error(__FILE__, __LINE__, #kernel,                 \
                                  _ds4_cuda_launch_err);                       \
            return 0;                                                           \
        }                                                                       \
    } while (0)

__global__ static void ds4_cuda_kernel_add(float *out, const float *a, const float *b, uint32_t n) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = a[i] + b[i];
}

static inline __host__ __device__ float ds4_cuda_softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ static void ds4_cuda_kernel_router_softplus_sqrt_rows(float *out,
                                                                  const float *logits,
                                                                  uint32_t n_tok) {
    const uint32_t token = (uint32_t)blockIdx.y;
    const uint32_t expert = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= n_tok || expert >= 256u) return;
    const float v = logits[(uint64_t)token * 256u + expert];
    out[(uint64_t)token * 256u + expert] = sqrtf(ds4_cuda_softplus_stable(v));
}

__global__ static void ds4_cuda_kernel_router_add_bias_rows(float *scores,
                                                            const float *bias,
                                                            uint32_t n_tok) {
    const uint32_t token = (uint32_t)blockIdx.y;
    const uint32_t expert = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= n_tok || expert >= 256u) return;
    scores[(uint64_t)token * 256u + expert] += bias[expert];
}

__global__ static void ds4_cuda_kernel_router_topk_rows(int32_t *selected,
                                                        const float *scores,
                                                        uint32_t n_tok) {
    const uint32_t token = (uint32_t)blockIdx.x;
    if (token >= n_tok || threadIdx.x != 0) return;

    const float *row = scores + (uint64_t)token * 256u;
    float best_val[6];
    int32_t best_idx[6];
    for (int i = 0; i < 6; i++) {
        best_val[i] = -FLT_MAX;
        best_idx[i] = -1;
    }

    for (int expert = 0; expert < 256; expert++) {
        const float v = row[expert];
        for (int slot = 0; slot < 6; slot++) {
            if (best_idx[slot] < 0 || v > best_val[slot]) {
                for (int shift = 5; shift > slot; shift--) {
                    best_val[shift] = best_val[shift - 1];
                    best_idx[shift] = best_idx[shift - 1];
                }
                best_val[slot] = v;
                best_idx[slot] = expert;
                break;
            }
        }
    }

    int32_t *dst = selected + (uint64_t)token * 6u;
    for (int i = 0; i < 6; i++) dst[i] = best_idx[i];
}

__global__ static void ds4_cuda_kernel_router_hash_selected_rows(int32_t *selected,
                                                                 const int32_t *tokens,
                                                                 const int32_t *hash_table,
                                                                 uint32_t hash_rows,
                                                                 uint32_t n_tok,
                                                                 bool use_token_buffer,
                                                                 int32_t single_token) {
    const uint32_t token = (uint32_t)blockIdx.x;
    if (token >= n_tok || threadIdx.x != 0) return;

    const int32_t tok = use_token_buffer ? tokens[token] : single_token;
    if (tok < 0 || (uint32_t)tok >= hash_rows) return;

    const int32_t *row = hash_table + (uint64_t)tok * 6u;
    int32_t *dst = selected + (uint64_t)token * 6u;
    for (int i = 0; i < 6; i++) dst[i] = row[i];
}

__global__ static void ds4_cuda_kernel_router_gather_weights_rows(float *weights,
                                                                  const float *probs,
                                                                  const int32_t *selected,
                                                                  uint32_t n_tok) {
    const uint32_t token = (uint32_t)blockIdx.x;
    if (token >= n_tok || threadIdx.x != 0) return;

    const float *row = probs + (uint64_t)token * 256u;
    const int32_t *sel = selected + (uint64_t)token * 6u;
    float *dst = weights + (uint64_t)token * 6u;
    float sum = 0.0f;
    for (int i = 0; i < 6; i++) {
        const int32_t idx = sel[i];
        dst[i] = (idx >= 0 && idx < 256) ? row[idx] : 0.0f;
        sum += dst[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int i = 0; i < 6; i++) dst[i] = dst[i] / sum * 1.5f;
}

__global__ static void ds4_cuda_kernel_repeat_hc(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_embd * n_hc;
    if (idx >= total) return;
    uint32_t embd = (uint32_t)(idx % n_embd);
    out[idx] = row[embd];
}

__global__ static void ds4_cuda_kernel_rms_norm_plain(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    extern __shared__ double ssum[];
    uint32_t row = blockIdx.x;
    if (row >= rows) return;

    const float *xin = x + (uint64_t)row * n;
    float *yout = out + (uint64_t)row * n;

    double local = 0.0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        double v = (double)xin[i];
        local += v * v;
    }
    ssum[threadIdx.x] = local;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    double scale = 0.0;
    if (threadIdx.x == 0) {
        scale = 1.0 / sqrt(ssum[0] / (double)n + (double)eps);
        ssum[0] = scale;
    }
    __syncthreads();

    const float yscale = (float)ssum[0];
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        yout[i] = xin[i] * yscale;
    }
}

__global__ static void ds4_cuda_kernel_quantize_q8_0_batch(const float *x, int8_t *xq,
                                                           float *xscale, uint32_t n_tok,
                                                           uint32_t in_dim, uint32_t blocks) {
    extern __shared__ float shmem[];
    const uint32_t token = (uint32_t)blockIdx.y;
    const uint32_t block = (uint32_t)blockIdx.x;
    if (token >= n_tok || block >= blocks) return;

    const uint64_t base = (uint64_t)token * in_dim + (uint64_t)block * 32u;
    const uint32_t bn = in_dim > (uint64_t)block * 32u ?
        (uint32_t)(((uint64_t)in_dim - (uint64_t)block * 32u) < 32u ? ((uint64_t)in_dim - (uint64_t)block * 32u) : 32u) : 0u;
    const float *xin = x + base;

    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < bn; i += blockDim.x) {
        const float ax = fabsf(xin[i]);
        if (ax > local) local = ax;
    }
    shmem[threadIdx.x] = local;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] = fmaxf(shmem[threadIdx.x], shmem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    __shared__ float scale;
    if (threadIdx.x == 0) {
        scale = shmem[0] / 127.0f;
        xscale[(uint64_t)token * blocks + block] = scale;
    }
    __syncthreads();

    const float inv = scale != 0.0f ? 1.0f / scale : 0.0f;
    int8_t *xqo = xq + ((uint64_t)token * blocks + block) * 32u;
    for (uint32_t i = threadIdx.x; i < 32u; i += blockDim.x) {
        int8_t v = 0;
        if (i < bn) {
            int qv = (int)lrintf(xin[i] * inv);
            if (qv > 127) qv = 127;
            if (qv < -128) qv = -128;
            v = (int8_t)qv;
        }
        xqo[i] = v;
    }
}

__global__ static void ds4_cuda_kernel_matmul_q8_0_batch(float *out, const uint8_t *weights,
                                                         const int8_t *xq, const float *xscale,
                                                         uint32_t out_dim, uint32_t blocks,
                                                         uint32_t n_tok) {
    __shared__ int32_t dot_sh[32];
    const uint32_t row = (uint32_t)blockIdx.x;
    const uint32_t token = (uint32_t)blockIdx.y;
    if (row >= out_dim || token >= n_tok) return;

    const uint8_t *row_ptr = weights + (uint64_t)row * blocks * 34u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = row_ptr + (uint64_t)b * 34u;
        const uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        const float wscale = ds4_cuda_half_to_float(scale_bits);
        const int8_t *wq = (const int8_t *)(blk + 2);
        const int8_t *xqb = xq + ((uint64_t)token * blocks + b) * 32u;
        int32_t local = 0;
        for (uint32_t i = threadIdx.x; i < 32u; i += blockDim.x) {
            local += (int32_t)wq[i] * (int32_t)xqb[i];
        }
        dot_sh[threadIdx.x] = local;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) dot_sh[threadIdx.x] += dot_sh[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            acc += wscale * xscale[(uint64_t)token * blocks + b] * (float)dot_sh[0];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[(uint64_t)token * out_dim + row] = acc;
    }
}

__global__ static void ds4_cuda_kernel_matmul_f32_batch(float *out, const float *weights,
                                                         const float *x, uint32_t in_dim,
                                                         uint32_t out_dim, uint32_t n_tok) {
    __shared__ double dot_sh[32];
    const uint32_t row = (uint32_t)blockIdx.x;
    const uint32_t token = (uint32_t)blockIdx.y;
    if (row >= out_dim || token >= n_tok) return;

    const float *row_ptr = weights + (uint64_t)row * in_dim;
    double local = 0.0;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        local += (double)row_ptr[i] * (double)x[(uint64_t)token * in_dim + i];
    }
    dot_sh[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) dot_sh[threadIdx.x] += dot_sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[(uint64_t)token * out_dim + row] = (float)dot_sh[0];
    }
}

__global__ static void ds4_cuda_kernel_matmul_f16_batch(float *out, const uint16_t *weights,
                                                         const float *x, uint32_t in_dim,
                                                         uint32_t out_dim, uint32_t n_tok) {
    __shared__ double dot_sh[32];
    const uint32_t row = (uint32_t)blockIdx.x;
    const uint32_t token = (uint32_t)blockIdx.y;
    if (row >= out_dim || token >= n_tok) return;

    const uint16_t *row_ptr = weights + (uint64_t)row * in_dim;
    double local = 0.0;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        local += (double)ds4_cuda_half_to_float(row_ptr[i]) * (double)x[(uint64_t)token * in_dim + i];
    }
    dot_sh[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) dot_sh[threadIdx.x] += dot_sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[(uint64_t)token * out_dim + row] = (float)dot_sh[0];
    }
}

__global__ static void ds4_cuda_kernel_embed_token_hc(float *out_hc, const uint16_t *weights,
                                                      uint32_t token, uint32_t n_vocab,
                                                      uint32_t n_embd, uint32_t n_hc) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_embd * n_hc;
    if (idx >= total) return;
    if (token >= n_vocab) return;
    uint32_t embd = (uint32_t)(idx % n_embd);
    const uint16_t *row = weights + (uint64_t)token * n_embd;
    out_hc[idx] = ds4_cuda_half_to_float(row[embd]);
}

__global__ static void ds4_cuda_kernel_embed_tokens_hc(float *out_hc, const int32_t *tokens,
                                                       const uint16_t *weights, uint32_t n_vocab,
                                                       uint32_t n_tokens, uint32_t n_embd,
                                                       uint32_t n_hc) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_tokens * n_embd * n_hc;
    if (idx >= total) return;
    uint32_t embd = (uint32_t)(idx % n_embd);
    uint64_t token_idx = idx / ((uint64_t)n_embd * n_hc);
    int32_t token = tokens[token_idx];
    if (token < 0 || (uint32_t)token >= n_vocab) return;
    const uint16_t *row = weights + (uint64_t)token * n_embd;
    out_hc[idx] = ds4_cuda_half_to_float(row[embd]);
}

__global__ static void ds4_cuda_kernel_rms_norm_weight_rows(float *out, const float *x,
                                                            const float *weight, uint32_t n,
                                                            uint32_t rows, float eps) {
    extern __shared__ double ssum[];
    uint32_t row = blockIdx.x;
    if (row >= rows) return;

    const float *xin = x + (uint64_t)row * n;
    float *yout = out + (uint64_t)row * n;

    double local = 0.0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        double v = (double)xin[i];
        local += v * v;
    }
    ssum[threadIdx.x] = local;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        __syncthreads();
    }

    __shared__ float scale;
    if (threadIdx.x == 0) {
        scale = 1.0f / sqrtf((float)(ssum[0] / (double)n) + eps);
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        yout[i] = xin[i] * scale * weight[i];
    }
}

__global__ static void ds4_cuda_kernel_head_rms_norm(float *x, uint32_t n_tok, uint32_t n_head,
                                                     uint32_t head_dim, float eps) {
    extern __shared__ double ssum[];
    uint32_t row = blockIdx.x;
    uint32_t rows = n_tok * n_head;
    if (row >= rows) return;

    float *xin = x + (uint64_t)row * head_dim;
    double local = 0.0;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        double v = (double)xin[i];
        local += v * v;
    }
    ssum[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        __syncthreads();
    }
    __shared__ float scale;
    if (threadIdx.x == 0) {
        scale = 1.0f / sqrtf((float)(ssum[0] / (double)head_dim) + eps);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        xin[i] *= scale;
    }
}

__global__ static void ds4_cuda_kernel_hc_post(float *out_hc, const float *block_out,
                                                const float *residual_hc, const float *post,
                                                const float *comb, uint32_t n_embd,
                                                uint32_t n_hc) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_embd * n_hc;
    if (idx >= total) return;
    uint32_t dst = (uint32_t)(idx / n_embd);
    uint32_t d = (uint32_t)(idx % n_embd);
    float acc = block_out[d] * post[dst];
    for (uint32_t src = 0; src < n_hc; src++) {
        acc += comb[dst + (uint64_t)src * n_hc] * residual_hc[(uint64_t)src * n_embd + d];
    }
    out_hc[idx] = acc;
}

__global__ static void ds4_cuda_kernel_rope_tail(float *x, uint32_t n_head, uint32_t head_dim,
                                                 uint32_t n_rot, uint32_t pos,
                                                 uint64_t n_ctx_orig, float freq_base,
                                                 float freq_scale, float ext_factor,
                                                 float attn_factor, float beta_fast,
                                                 float beta_slow, bool inverse) {
    (void)n_ctx_orig;
    (void)freq_base;
    (void)freq_scale;
    (void)ext_factor;
    (void)attn_factor;
    (void)beta_fast;
    (void)beta_slow;
    const uint32_t head = (uint32_t)blockIdx.y;
    const uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= n_head || i >= n_rot || (i & 1u) != 0) return;

    const uint32_t n_nope = head_dim - n_rot;
    float *tail = x + (uint64_t)head * head_dim + n_nope;
    const float theta = (float)pos * powf(freq_base, -2.0f / (float)n_rot) * freq_scale;
    const float c = cosf(theta);
    const float s = (inverse ? -1.0f : 1.0f) * sinf(theta);
    const float x0 = tail[i + 0];
    const float x1 = tail[i + 1];
    tail[i + 0] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__global__ static void ds4_cuda_kernel_hc_split_sinkhorn(float *out, const float *mix,
                                                         const float *scale, const float *base,
                                                         int n_hc, int iters, float eps) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }
    for (int i = 0; i < n_hc; i++) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[16 * 16];
    for (int dst = 0; dst < n_hc; dst++) {
        float row_max = -FLT_MAX;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }
    for (int src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
        const float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }
    for (int iter = 1; iter < iters; iter++) {
        for (int dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }
        for (int src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }
    for (int i = 0; i < n_hc * n_hc; i++) out[2 * n_hc + i] = c[i];
}

__global__ static void ds4_cuda_kernel_hc_weighted_sum(float *out, const float *x,
                                                       const float *weights, uint32_t n_embd,
                                                       uint32_t n_hc) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_embd) return;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)h * n_embd + idx] * weights[h];
    }
    out[idx] = acc;
}

// ============================================================
// IQ2_XXS dequantization helpers (device-side)
// ============================================================

// Dequantize one 16-element sub-block of an IQ2_XXS block.
// xb: pointer to the block_iq2_xxs
// il: sub-block index [0..15], produces 16 floats in vals[0..15]
__device__ static void dequantize_iq2_xxs_sub16(const block_iq2_xxs *xb, int il, float *vals) {
    const float d = ds4_cuda_half_to_float(xb->d);
    const int ib32 = il / 2;
    const int il_mod = il % 2;

    const uint16_t *q2 = xb->qs + (uint64_t)4 * ib32;
    const uint32_t aux32_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux32_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);

    const uint8_t *aux8 = (const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;

    const uint8_t *grid0 = (const uint8_t *)(d_iq2xxs_grid + aux8[2 * il_mod + 0]);
    const uint8_t signs0 = d_ksigns_iq2xs[(aux32_s >> (14 * il_mod)) & 127];
    for (int i = 0; i < 8; i++) {
        vals[i] = dl * (float)grid0[i] * ((signs0 & d_kmask_iq2xs[i]) ? -1.0f : 1.0f);
    }

    const uint8_t *grid1 = (const uint8_t *)(d_iq2xxs_grid + aux8[2 * il_mod + 1]);
    const uint8_t signs1 = d_ksigns_iq2xs[(aux32_s >> (14 * il_mod + 7)) & 127];
    for (int i = 0; i < 8; i++) {
        vals[8 + i] = dl * (float)grid1[i] * ((signs1 & d_kmask_iq2xs[i]) ? -1.0f : 1.0f);
    }
}

// ============================================================
// IQ2_XXS single-expert matvec kernel
// ============================================================

// Each thread block handles one output row.
// Grid: (out_dim), Block: 32
__global__ static void ds4_cuda_kernel_moe_iq2_xxs_matvec(
        float *out,
        const block_iq2_xxs *weights,   // one expert's full weight matrix [out_dim, blocks_per_row]
        const float *x,                  // input vector [in_dim]
        uint32_t in_dim,                 // must be a multiple of QK_K
        uint32_t out_dim) {
    __shared__ float dot_sh[32];
    const uint32_t row = (uint32_t)blockIdx.x;
    if (row >= out_dim) return;

    const uint32_t blocks_per_row = in_dim / QK_K;
    const block_iq2_xxs *row_w = weights + (uint64_t)row * blocks_per_row;

    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks_per_row; b++) {
        const block_iq2_xxs *blk = row_w + b;
        float local = 0.0f;
        for (int il = (int)threadIdx.x; il < 16; il += (int)blockDim.x) {
            float wvals[16];
            dequantize_iq2_xxs_sub16(blk, il, wvals);
            for (int i = 0; i < 16; i++) {
                const uint32_t xi = b * QK_K + (uint32_t)il * 16u + (uint32_t)i;
                local += wvals[i] * x[xi];
            }
        }
        dot_sh[threadIdx.x] = local;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) dot_sh[threadIdx.x] += dot_sh[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) acc += dot_sh[0];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = acc;
}

// ============================================================
// Q2_K dequantization helper
// ============================================================

__device__ static void dequantize_q2_K_sub16(const block_q2_K *xb, int il, float *vals) {
    const float d = ds4_cuda_half_to_float(xb->d);
    const float min = ds4_cuda_half_to_float(xb->dmin);
    const uint8_t *q = (const uint8_t *)xb->qs;
    const uint8_t sc = xb->scales[il];

    q = q + 32u * (uint32_t)(il / 8) + 16u * (uint32_t)(il & 1);
    const int sub_il = (il / 2) % 4;
    float coef;
    uint8_t mask;
    if (sub_il > 1) {
        if (sub_il > 2) { coef = 1.0f / 64.0f; mask = 192; }
        else            { coef = 1.0f / 16.0f; mask = 48;  }
    } else {
        if (sub_il > 0) { coef = 1.0f / 4.0f;  mask = 12;  }
        else            { coef = 1.0f;          mask = 3;   }
    }
    const float dl = d * (float)(sc & 0xF) * coef;
    const float ml = min * (float)(sc >> 4);

    for (int i = 0; i < 16; i++) {
        vals[i] = dl * (float)(q[i] & mask) - ml;
    }
}

// ============================================================
// Q2_K sum6 down matvec kernel (MoE decode)
// ============================================================

// For a single token, computes down projection for all 6 selected experts
// and sums the results. Grid: (out_dim), Block: 32
__global__ static void ds4_cuda_kernel_moe_q2_k_sum6(
        float *out,
        const block_q2_K *const *down_weights,  // array of 6 expert weight pointers
        const float *mid,                       // mid activation [6, in_dim]
        const float *route_weights,             // route weights [6]
        uint32_t in_dim,                        // must be a multiple of QK_K
        uint32_t out_dim,
        uint32_t n_expert) {
    __shared__ float dot_sh[32];
    const uint32_t row = (uint32_t)blockIdx.x;
    if (row >= out_dim) return;

    const uint32_t blocks_per_row = in_dim / QK_K;

    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) {
        const float rw = route_weights[e];
        const block_q2_K *row_w = down_weights[e] + (uint64_t)row * blocks_per_row;
        const float *x = mid + (uint64_t)e * in_dim;

        float local = 0.0f;
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            const block_q2_K *blk = row_w + b;
            for (int il = (int)threadIdx.x; il < 16; il += (int)blockDim.x) {
                float wvals[16];
                dequantize_q2_K_sub16(blk, il, wvals);
                for (int i = 0; i < 16; i++) {
                    const uint32_t xi = b * QK_K + (uint32_t)il * 16u + (uint32_t)i;
                    local += wvals[i] * x[xi];
                }
            }
        }
        dot_sh[threadIdx.x] = local;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) dot_sh[threadIdx.x] += dot_sh[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) acc += dot_sh[0] * rw;
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = acc;
}

// ============================================================
// Weighted SwiGLU activation kernel
// ============================================================

// For each expert: mid[i] = silu(clamp(gate, -clamp, clamp)) * clamp(up, -clamp, clamp) * weight
// Grid: (n_expert * out_dim + 255) / 256, Block: 256
__global__ static void ds4_cuda_kernel_moe_swiglu_weighted(
        float *mid,
        const float *gate,
        const float *up,
        const float *weights,
        uint32_t out_dim,    // elements per expert
        uint32_t n_expert,
        float clamp_val) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_expert * out_dim;
    if (idx >= total) return;

    float g = gate[idx];
    float u = up[idx];
    const uint32_t expert = (uint32_t)(idx / out_dim);

    if (clamp_val > 0.0f) {
        if (g < -clamp_val) g = -clamp_val;
        if (g >  clamp_val) g =  clamp_val;
        if (u < -clamp_val) u = -clamp_val;
        if (u >  clamp_val) u =  clamp_val;
    }

    const float s = 1.0f / (1.0f + expf(-g));
    const float w = weights[expert];
    mid[idx] = s * g * u * w;
}

// ============================================================
// Shared expert Q8_0 fused gate+up+swiglu kernel (decode)
// ============================================================

// Fuses Q8_0 gate matvec + Q8_0 up matvec + SwiGLU into one dispatch.
// For single-token decode. Grid: (out_dim), Block: 32
__global__ static void ds4_cuda_kernel_shared_gate_up_swiglu_q8_0(
        float *gate_out,
        float *up_out,
        float *mid,
        const uint8_t *gate_weights,   // Q8_0 layout [out_dim, blocks * 34]
        const uint8_t *up_weights,     // Q8_0 layout [out_dim, blocks * 34]
        const int8_t *xq,              // quantized input [blocks * 32]
        const float *xscale,           // input scale [blocks]
        uint32_t in_dim,
        uint32_t out_dim) {
    __shared__ int32_t dot_sh[32];
    const uint32_t row = (uint32_t)blockIdx.x;
    if (row >= out_dim) return;

    const uint32_t blocks = in_dim / 32u;

    // Gate matvec
    {
        const uint8_t *row_ptr = gate_weights + (uint64_t)row * blocks * 34u;
        float gate_acc = 0.0f;
        for (uint32_t b = 0; b < blocks; b++) {
            const uint8_t *blk = row_ptr + (uint64_t)b * 34u;
            const uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            const float wscale = ds4_cuda_half_to_float(scale_bits);
            const int8_t *wq = (const int8_t *)(blk + 2);
            const int8_t *xqb = xq + (uint64_t)b * 32u;
            int32_t local = 0;
            for (uint32_t i = threadIdx.x; i < 32u; i += blockDim.x) {
                local += (int32_t)wq[i] * (int32_t)xqb[i];
            }
            dot_sh[threadIdx.x] = local;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) dot_sh[threadIdx.x] += dot_sh[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0) gate_acc += wscale * xscale[b] * (float)dot_sh[0];
            __syncthreads();
        }
        if (threadIdx.x == 0) gate_out[row] = gate_acc;
    }

    // Up matvec
    {
        const uint8_t *row_ptr = up_weights + (uint64_t)row * blocks * 34u;
        float up_acc = 0.0f;
        for (uint32_t b = 0; b < blocks; b++) {
            const uint8_t *blk = row_ptr + (uint64_t)b * 34u;
            const uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            const float wscale = ds4_cuda_half_to_float(scale_bits);
            const int8_t *wq = (const int8_t *)(blk + 2);
            const int8_t *xqb = xq + (uint64_t)b * 32u;
            int32_t local = 0;
            for (uint32_t i = threadIdx.x; i < 32u; i += blockDim.x) {
                local += (int32_t)wq[i] * (int32_t)xqb[i];
            }
            dot_sh[threadIdx.x] = local;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) dot_sh[threadIdx.x] += dot_sh[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0) up_acc += wscale * xscale[b] * (float)dot_sh[0];
            __syncthreads();
        }
        if (threadIdx.x == 0) up_out[row] = up_acc;
    }

    // SwiGLU: mid = silu(gate) * up
    if (threadIdx.x == 0) {
        const float g = gate_out[row];
        const float u = up_out[row];
        const float s = 1.0f / (1.0f + expf(-g));
        mid[row] = s * g * u;
    }
}

// ============================================================
// Fused shared expert Q8_0 down + HC expand kernel (decode)
// ============================================================

// Performs Q8_0 down-projection of shared_mid, adds routed_out,
// then HC-expands via post/comb/split coefficients into out_hc.
// Grid: (out_dim), Block: 32
__global__ static void ds4_cuda_kernel_shared_down_hc_expand_q8_0(
        float *out_hc,
        float *shared_out,
        const uint8_t *down_weights,   // Q8_0 layout [out_dim, blocks * 34]
        const int8_t *xq,              // quantized shared_mid [blocks * 32]
        const float *xscale,           // scale per block
        const float *routed_out,       // [out_dim]
        const float *residual_hc,      // [n_hc * n_embd]
        const float *post,             // [n_hc]
        const float *comb,             // [n_hc * n_hc]
        uint32_t in_dim,               // shared_dim (FF_EXP = 2048)
        uint32_t out_dim,              // n_embd = 4096
        uint32_t n_hc) {               // 4
    __shared__ int32_t dot_sh[32];
    const uint32_t row = (uint32_t)blockIdx.x;
    if (row >= out_dim) return;

    const uint32_t blocks = in_dim / 32u;
    const uint8_t *row_ptr = down_weights + (uint64_t)row * blocks * 34u;
    float so = 0.0f;

    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = row_ptr + (uint64_t)b * 34u;
        const uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        const float wscale = ds4_cuda_half_to_float(scale_bits);
        const int8_t *wq = (const int8_t *)(blk + 2);
        const int8_t *xqb = xq + (uint64_t)b * 32u;
        int32_t local = 0;
        for (uint32_t i = threadIdx.x; i < 32u; i += blockDim.x) {
            local += (int32_t)wq[i] * (int32_t)xqb[i];
        }
        dot_sh[threadIdx.x] = local;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) dot_sh[threadIdx.x] += dot_sh[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) so += wscale * xscale[b] * (float)dot_sh[0];
        __syncthreads();
    }

    if (threadIdx.x == 0) shared_out[row] = so;

    // HC expand: for each HC stream h, compute the output
    // out_hc[h * n_embd + row] = (shared_out[row] + routed_out[row]) * post[h] + sum_src(comb[h + s * n_hc] * residual_hc[s * n_embd + row])
    __syncthreads();
    if (threadIdx.x == 0) {
        const float base = so + routed_out[row];
        for (uint32_t h = 0; h < n_hc; h++) {
            float acc = base * post[h];
            for (uint32_t s = 0; s < n_hc; s++) {
                acc += comb[h + (uint64_t)s * n_hc] * residual_hc[(uint64_t)s * out_dim + row];
            }
            out_hc[(uint64_t)h * out_dim + row] = acc;
        }
    }
}

int ds4_cuda_register_host_memory(const void *ptr, uint64_t bytes) {
    if (!ptr || bytes == 0) return 1;
    cudaError_t err = cudaHostRegister((void *)ptr, (size_t)bytes, cudaHostRegisterPortable);
    if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaHostRegister", err);
        return 0;
    }
    return 1;
}

void ds4_cuda_unregister_host_memory(const void *ptr) {
    if (!ptr) return;
    cudaError_t err = cudaHostUnregister((void *)ptr);
    if (err != cudaSuccess && err != cudaErrorHostMemoryNotRegistered) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaHostUnregister", err);
    }
}

int ds4_cuda_init(bool quality) {
    (void)quality;
    if (g_cuda_device >= 0) return 1;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        fprintf(stderr, "ds4: CUDA backend unavailable: %s\n", cudaGetErrorString(err));
        ds4_cuda_reset_state();
        return 0;
    }

    int best = -1;
    int best_score = -1;
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) continue;
        int score = prop.major * 100 + prop.minor;
        if (prop.unifiedAddressing) score += 10000;
        if (prop.major == 12 && prop.minor == 1) score += 100000;
        if (score > best_score) {
            best = i;
            best_score = score;
        }
    }

    if (best < 0) {
        fprintf(stderr, "ds4: CUDA backend unavailable: no usable device found\n");
        ds4_cuda_reset_state();
        return 0;
    }

    err = cudaSetDevice(best);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA set device failed: %s\n", cudaGetErrorString(err));
        ds4_cuda_reset_state();
        return 0;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, best);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA device query failed: %s\n", cudaGetErrorString(err));
        ds4_cuda_reset_state();
        return 0;
    }

    DS4_CUDA_CHECK(cudaFree(0));

    DS4_CUDA_CHECK(cudaStreamCreateWithFlags(&g_cuda_prefill_stream, cudaStreamNonBlocking));
    DS4_CUDA_CHECK(cudaStreamCreateWithFlags(&g_cuda_decode_stream, cudaStreamNonBlocking));

    g_cuda_device = best;
    snprintf(g_cuda_name, sizeof(g_cuda_name), "%s", prop.name);
    g_cuda_cc_major = prop.major;
    g_cuda_cc_minor = prop.minor;
    g_cuda_total = (uint64_t)prop.totalGlobalMem;

    if (!prop.unifiedAddressing) {
        fprintf(stderr, "ds4: CUDA warning: selected device does not report unified addressing\n");
    }
    if (prop.major < 12 || (prop.major == 12 && prop.minor < 1)) {
        fprintf(stderr,
                "ds4: CUDA warning: selected device is sm_%d%d, optimized target is GB10/Blackwell sm_121\n",
                prop.major,
                prop.minor);
    }

    (void)cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1u << 20);
    fprintf(stderr,
            "ds4: CUDA backend initialized on device %d: %s (sm_%d%d, %.2f GiB global memory)\n",
            g_cuda_device,
            g_cuda_name,
            g_cuda_cc_major,
            g_cuda_cc_minor,
            (double)g_cuda_total / (1024.0 * 1024.0 * 1024.0));
    return 1;

fail:
    if (g_cuda_prefill_stream) {
        cudaStreamDestroy(g_cuda_prefill_stream);
    }
    if (g_cuda_decode_stream) {
        cudaStreamDestroy(g_cuda_decode_stream);
    }
    ds4_cuda_reset_state();
    return 0;
}

void ds4_cuda_cleanup(void) {
    if (g_cuda_device >= 0) {
        (void)cudaDeviceSynchronize();
    }
    if (g_cuda_prefill_stream) {
        (void)cudaStreamDestroy(g_cuda_prefill_stream);
    }
    if (g_cuda_decode_stream) {
        (void)cudaStreamDestroy(g_cuda_decode_stream);
    }
    ds4_cuda_reset_state();
}

bool ds4_cuda_available(void) {
    return g_cuda_device >= 0;
}

const char *ds4_cuda_device_name(void) {
    return g_cuda_name[0] ? g_cuda_name : "unknown";
}

uint64_t ds4_cuda_total_memory(void) {
    return g_cuda_total;
}

uint64_t ds4_cuda_free_memory(void) {
    size_t free_b = 0;
    size_t total_b = 0;
    if (g_cuda_device < 0) return 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return (uint64_t)free_b;
}

int ds4_cuda_compute_capability_major(void) {
    return g_cuda_cc_major;
}

int ds4_cuda_compute_capability_minor(void) {
    return g_cuda_cc_minor;
}

void ds4_cuda_print_memory_report(const char *label) {
    size_t free_b = 0;
    size_t total_b = 0;
    if (g_cuda_device < 0) return;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return;
    fprintf(stderr,
            "ds4: CUDA memory %s: free %.2f GiB / total %.2f GiB, tensors live %.2f MiB peak %.2f MiB\n",
            label ? label : "",
            (double)free_b / (1024.0 * 1024.0 * 1024.0),
            (double)total_b / (1024.0 * 1024.0 * 1024.0),
            (double)g_cuda_tensor_live_bytes / (1024.0 * 1024.0),
            (double)g_cuda_tensor_peak_bytes / (1024.0 * 1024.0));
}

cudaStream_t ds4_cuda_prefill_stream(void) {
    return g_cuda_prefill_stream;
}

cudaStream_t ds4_cuda_decode_stream(void) {
    return g_cuda_decode_stream;
}

int ds4_cuda_synchronize(void) {
    if (g_cuda_device < 0) return 0;
    return cudaDeviceSynchronize() == cudaSuccess;
}

int ds4_cuda_stream_synchronize(cudaStream_t stream) {
    if (g_cuda_device < 0) return 0;
    if (!stream) return 1;
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

ds4_cuda_tensor *ds4_cuda_tensor_alloc(uint64_t bytes) {
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return NULL;
    if (bytes == 0 || bytes > (uint64_t)SIZE_MAX) return NULL;

    ds4_cuda_tensor *tensor = ds4_cuda_tensor_new();
    if (!tensor) return NULL;

    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaMalloc", err);
        delete tensor;
        return NULL;
    }

    tensor->base = ptr;
    tensor->offset = 0;
    tensor->bytes = bytes;
    tensor->owner = 1;
    ds4_cuda_tensor_account_alloc(bytes);
    return tensor;
}

ds4_cuda_tensor *ds4_cuda_tensor_view(const ds4_cuda_tensor *base, uint64_t offset, uint64_t bytes) {
    const ds4_cuda_tensor *obj = ds4_cuda_tensor_const_obj(base);
    if (!obj) return NULL;
    if (!ds4_cuda_tensor_range_ok(obj, offset, bytes)) return NULL;
    if (obj->offset > UINT64_MAX - offset) return NULL;

    ds4_cuda_tensor *view = ds4_cuda_tensor_new();
    if (!view) return NULL;
    view->base = obj->base;
    view->offset = obj->offset + offset;
    view->bytes = bytes;
    view->owner = 0;
    return view;
}

void ds4_cuda_tensor_free(ds4_cuda_tensor *tensor) {
    if (!tensor) return;
    ds4_cuda_tensor *obj = ds4_cuda_tensor_obj(tensor);
    if (obj->owner && obj->base) {
        (void)cudaFree(obj->base);
        ds4_cuda_tensor_account_free(obj->bytes);
    }
    obj->base = NULL;
    obj->offset = 0;
    obj->bytes = 0;
    obj->owner = 0;
    delete obj;
}

uint64_t ds4_cuda_tensor_bytes(const ds4_cuda_tensor *tensor) {
    if (!tensor) return 0;
    return ds4_cuda_tensor_const_obj(tensor)->bytes;
}

void *ds4_cuda_tensor_device_ptr(ds4_cuda_tensor *tensor) {
    if (!tensor) return NULL;
    ds4_cuda_tensor *obj = ds4_cuda_tensor_obj(tensor);
    return static_cast<uint8_t *>(obj->base) + obj->offset;
}

const void *ds4_cuda_tensor_device_ptr_const(const ds4_cuda_tensor *tensor) {
    if (!tensor) return NULL;
    const ds4_cuda_tensor *obj = ds4_cuda_tensor_const_obj(tensor);
    return static_cast<const uint8_t *>(obj->base) + obj->offset;
}

int ds4_cuda_tensor_write(ds4_cuda_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    if (bytes > (uint64_t)SIZE_MAX) return 0;
    ds4_cuda_tensor *obj = ds4_cuda_tensor_obj(tensor);
    if (!ds4_cuda_tensor_range_ok(obj, offset, bytes)) return 0;
    if (bytes == 0) return 1;

    cudaError_t err = cudaMemcpy(static_cast<uint8_t *>(obj->base) + obj->offset + offset,
                                 data,
                                 (size_t)bytes,
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaMemcpyHostToDevice", err);
        return 0;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

int ds4_cuda_tensor_read(const ds4_cuda_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    if (bytes > (uint64_t)SIZE_MAX) return 0;
    const ds4_cuda_tensor *obj = ds4_cuda_tensor_const_obj(tensor);
    if (!ds4_cuda_tensor_range_ok(obj, offset, bytes)) return 0;
    if (bytes == 0) return 1;

    cudaError_t err = cudaMemcpy(data,
                                 static_cast<const uint8_t *>(obj->base) + obj->offset + offset,
                                 (size_t)bytes,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaMemcpyDeviceToHost", err);
        return 0;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

int ds4_cuda_tensor_copy(ds4_cuda_tensor *dst, uint64_t dst_offset,
                         const ds4_cuda_tensor *src, uint64_t src_offset,
                         uint64_t bytes) {
    if (!dst || !src) return 0;
    if (bytes > (uint64_t)SIZE_MAX) return 0;
    ds4_cuda_tensor *d = ds4_cuda_tensor_obj(dst);
    const ds4_cuda_tensor *s = ds4_cuda_tensor_const_obj(src);
    if (!ds4_cuda_tensor_range_ok(d, dst_offset, bytes)) return 0;
    if (!ds4_cuda_tensor_range_ok(s, src_offset, bytes)) return 0;
    if (bytes == 0) return 1;

    cudaError_t err = cudaMemcpy(static_cast<uint8_t *>(d->base) + d->offset + dst_offset,
                                 static_cast<const uint8_t *>(s->base) + s->offset + src_offset,
                                 (size_t)bytes,
                                 cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaMemcpyDeviceToDevice", err);
        return 0;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

int ds4_cuda_tensor_memset(ds4_cuda_tensor *tensor, uint64_t offset, int value, uint64_t bytes) {
    if (!tensor) return 0;
    if (bytes > (uint64_t)SIZE_MAX) return 0;
    ds4_cuda_tensor *obj = ds4_cuda_tensor_obj(tensor);
    if (!ds4_cuda_tensor_range_ok(obj, offset, bytes)) return 0;
    if (bytes == 0) return 1;

    cudaError_t err = cudaMemset(static_cast<uint8_t *>(obj->base) + obj->offset + offset,
                                value,
                                (size_t)bytes);
    if (err != cudaSuccess) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaMemset", err);
        return 0;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

int ds4_cuda_repeat_hc_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *row,
                              uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;
    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t out_bytes = row_bytes * n_hc;
    if (ds4_cuda_tensor_bytes(row) < row_bytes || ds4_cuda_tensor_bytes(out) < out_bytes) return 0;

    cudaStream_t stream = ds4_cuda_prefill_stream();
    if (!stream) stream = 0;
    const uint64_t total = (uint64_t)n_embd * n_hc;
    const uint32_t block = 256u;
    const uint32_t grid = (uint32_t)((total + block - 1u) / block);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_repeat_hc,
                    grid,
                    block,
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(row)),
                    n_embd,
                    n_hc);
    return ds4_cuda_stream_synchronize(stream);
}

int ds4_cuda_add_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *a,
                        const ds4_cuda_tensor *b, uint32_t n) {
    if (!out || !a || !b || n == 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (ds4_cuda_tensor_bytes(out) < bytes || ds4_cuda_tensor_bytes(a) < bytes || ds4_cuda_tensor_bytes(b) < bytes) {
        return 0;
    }

    cudaStream_t stream = ds4_cuda_decode_stream();
    if (!stream) stream = 0;
    const uint32_t block = 256u;
    const uint32_t grid = (n + block - 1u) / block;
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_add,
                    grid,
                    block,
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(a)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(b)),
                    n);
    return ds4_cuda_stream_synchronize(stream);
}

static int ds4_cuda_router_select_impl(ds4_cuda_tensor *selected, ds4_cuda_tensor *weights,
                                       ds4_cuda_tensor *probs, const void *model_map,
                                       uint64_t model_size, uint64_t bias_offset,
                                       uint64_t hash_offset, uint32_t hash_rows,
                                       uint32_t n_expert_groups, uint32_t n_group_used,
                                       bool has_bias, bool hash_mode,
                                       const ds4_cuda_tensor *logits,
                                       const ds4_cuda_tensor *tokens,
                                       uint32_t n_tokens,
                                       int32_t single_token,
                                       bool use_token_buffer) {
    if (!selected || !weights || !probs || !logits || !model_map || n_tokens == 0) return 0;
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: CUDA router group gating is not part of this DeepSeek V4 Flash path\n");
        return 0;
    }
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;

    const uint64_t logits_bytes = (uint64_t)n_tokens * 256u * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_tokens * 6u * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)n_tokens * 6u * sizeof(float);
    if (ds4_cuda_tensor_bytes(logits) < logits_bytes ||
        ds4_cuda_tensor_bytes(probs) < logits_bytes ||
        ds4_cuda_tensor_bytes(selected) < selected_bytes ||
        ds4_cuda_tensor_bytes(weights) < weights_bytes) {
        fprintf(stderr, "ds4: CUDA router select received undersized buffers\n");
        return 0;
    }
    if (use_token_buffer && (!tokens || ds4_cuda_tensor_bytes(tokens) < (uint64_t)n_tokens * sizeof(int32_t))) {
        fprintf(stderr, "ds4: CUDA router select received undersized token buffer\n");
        return 0;
    }

    cudaStream_t stream = ds4_cuda_prefill_stream();
    ds4_cuda_tensor *scores = NULL;
    if (has_bias && !hash_mode) {
        scores = ds4_cuda_tensor_alloc(logits_bytes);
        if (!scores) return 0;
    }

    const uint32_t block = 256u;
    const dim3 grid((uint32_t)((256u + block - 1u) / block), n_tokens, 1u);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_router_softplus_sqrt_rows,
                    grid,
                    block,
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(probs)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(logits)),
                    n_tokens);

    void *bias_dev = NULL;
    void *hash_dev = NULL;
    if (has_bias && !hash_mode) {
        const uint64_t bias_bytes = 256u * sizeof(float);
        if (bias_offset > model_size || bias_bytes > model_size - bias_offset ||
            !ds4_cuda_copy_host_range_to_device((const uint8_t *)model_map + bias_offset,
                                                bias_bytes, &bias_dev)) {
            ds4_cuda_tensor_free(scores);
            return 0;
        }
        if (cudaMemcpyAsync(ds4_cuda_tensor_device_ptr(scores),
                            ds4_cuda_tensor_device_ptr_const(probs),
                            logits_bytes,
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            ds4_cuda_report_error(__FILE__, __LINE__, "cudaMemcpyAsync", cudaGetLastError());
            cudaFree(bias_dev);
            ds4_cuda_tensor_free(scores);
            return 0;
        }
        DS4_CUDA_LAUNCH(ds4_cuda_kernel_router_add_bias_rows,
                        dim3((uint32_t)((256u + block - 1u) / block), n_tokens, 1u),
                        block,
                        0,
                        stream,
                        static_cast<float *>(ds4_cuda_tensor_device_ptr(scores)),
                        static_cast<const float *>(bias_dev),
                        n_tokens);
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset ||
            !ds4_cuda_copy_host_range_to_device((const uint8_t *)model_map + hash_offset,
                                                hash_bytes, &hash_dev)) {
            cudaFree(bias_dev);
            ds4_cuda_tensor_free(scores);
            return 0;
        }
        DS4_CUDA_LAUNCH(ds4_cuda_kernel_router_hash_selected_rows,
                        dim3(n_tokens, 1u, 1u),
                        dim3(1u, 1u, 1u),
                        0,
                        stream,
                        static_cast<int32_t *>(ds4_cuda_tensor_device_ptr(selected)),
                        use_token_buffer ? static_cast<const int32_t *>(ds4_cuda_tensor_device_ptr_const(tokens)) : NULL,
                        static_cast<const int32_t *>(hash_dev),
                        hash_rows,
                        n_tokens,
                        use_token_buffer,
                        single_token);
    } else {
        DS4_CUDA_LAUNCH(ds4_cuda_kernel_router_topk_rows,
                        dim3(n_tokens, 1u, 1u),
                        dim3(1u, 1u, 1u),
                        0,
                        stream,
                        static_cast<int32_t *>(ds4_cuda_tensor_device_ptr(selected)),
                        static_cast<const float *>(has_bias ? ds4_cuda_tensor_device_ptr_const(scores)
                                                            : ds4_cuda_tensor_device_ptr_const(probs)),
                        n_tokens);
    }

    DS4_CUDA_LAUNCH(ds4_cuda_kernel_router_gather_weights_rows,
                    dim3(n_tokens, 1u, 1u),
                    dim3(1u, 1u, 1u),
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(weights)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(probs)),
                    static_cast<const int32_t *>(ds4_cuda_tensor_device_ptr_const(selected)),
                    n_tokens);

    const int ok = ds4_cuda_stream_synchronize(stream);
    cudaFree(bias_dev);
    cudaFree(hash_dev);
    ds4_cuda_tensor_free(scores);
    return ok;
}

int ds4_cuda_router_select_tensor(ds4_cuda_tensor *selected, ds4_cuda_tensor *weights,
                                  ds4_cuda_tensor *probs, const void *model_map,
                                  uint64_t model_size, uint64_t bias_offset,
                                  uint64_t hash_offset, uint32_t hash_rows,
                                  uint32_t token, uint32_t n_expert_groups,
                                  uint32_t n_group_used, bool has_bias,
                                  bool hash_mode, const ds4_cuda_tensor *logits) {
    return ds4_cuda_router_select_impl(selected, weights, probs, model_map, model_size,
                                       bias_offset, hash_offset, hash_rows,
                                       n_expert_groups, n_group_used, has_bias,
                                       hash_mode, logits, NULL, 1u, (int32_t)token, false);
}

int ds4_cuda_router_select_batch_tensor(ds4_cuda_tensor *selected, ds4_cuda_tensor *weights,
                                        ds4_cuda_tensor *probs, const void *model_map,
                                        uint64_t model_size, uint64_t bias_offset,
                                        uint64_t hash_offset, uint32_t hash_rows,
                                        uint32_t n_expert_groups, uint32_t n_group_used,
                                        bool has_bias, bool hash_mode,
                                        const ds4_cuda_tensor *logits,
                                        const ds4_cuda_tensor *tokens,
                                        uint32_t n_tokens) {
    return ds4_cuda_router_select_impl(selected, weights, probs, model_map, model_size,
                                       bias_offset, hash_offset, hash_rows,
                                       n_expert_groups, n_group_used, has_bias,
                                       hash_mode, logits, tokens, n_tokens, 0, true);
}

int ds4_cuda_rms_norm_plain_rows_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *x,
                                        uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || n == 0 || rows == 0 || (n & 3u) != 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;
    const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
    if (ds4_cuda_tensor_bytes(out) < bytes || ds4_cuda_tensor_bytes(x) < bytes) return 0;

    cudaStream_t stream = ds4_cuda_prefill_stream();
    if (!stream) stream = 0;
    const uint32_t block = 256u;
    const size_t shared = (size_t)block * sizeof(double);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_rms_norm_plain,
                    rows,
                    block,
                    shared,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(x)),
                    n,
                    rows,
                    eps);
    return ds4_cuda_stream_synchronize(stream);
}

int ds4_cuda_rms_norm_plain_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *x,
                                   uint32_t n, float eps) {
    return ds4_cuda_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_cuda_embed_token_hc_tensor(
        ds4_cuda_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc) {
    if (!out_hc || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0 || n_hc == 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;
    const uint64_t out_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (ds4_cuda_tensor_bytes(out_hc) < out_bytes || weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return 0;
    }

    cudaStream_t stream = ds4_cuda_prefill_stream();
    const uint16_t *weights_host = (const uint16_t *)((const uint8_t *)model_map + weight_offset);
    void *weights_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(weights_host, weight_bytes, &weights_dev)) return 0;
    dim3 grid((uint32_t)((out_bytes + 255u) / 256u), 1, 1);
    dim3 block(256, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_embed_token_hc,
                    grid,
                    block,
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out_hc)),
                    static_cast<const uint16_t *>(weights_dev),
                    token,
                    n_vocab,
                    n_embd,
                    n_hc);
    const int ok = ds4_cuda_stream_synchronize(stream);
    cudaFree(weights_dev);
    return ok;
}

int ds4_cuda_embed_tokens_hc_tensor(
        ds4_cuda_tensor       *out_hc,
        const ds4_cuda_tensor *tokens,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint32_t               n_vocab,
        uint32_t               n_tokens,
        uint32_t               n_embd,
        uint32_t               n_hc) {
    if (!out_hc || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) {
        fprintf(stderr, "ds4: CUDA batch embedding invalid arguments\n");
        return 0;
    }
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;
    const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * n_hc * sizeof(float);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (ds4_cuda_tensor_bytes(out_hc) < out_bytes || ds4_cuda_tensor_bytes(tokens) < token_bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        fprintf(stderr, "ds4: CUDA batch embedding undersized buffers or weight range\n");
        return 0;
    }

    cudaStream_t stream = ds4_cuda_prefill_stream();
    const uint16_t *weights_host = (const uint16_t *)((const uint8_t *)model_map + weight_offset);
    void *weights_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(weights_host, weight_bytes, &weights_dev)) return 0;
    dim3 grid((uint32_t)((out_bytes + 255u) / 256u), 1, 1);
    dim3 block(256, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_embed_tokens_hc,
                    grid,
                    block,
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out_hc)),
                    static_cast<const int32_t *>(ds4_cuda_tensor_device_ptr_const(tokens)),
                    static_cast<const uint16_t *>(weights_dev),
                    n_vocab,
                    n_tokens,
                    n_embd,
                    n_hc);
    const int ok = ds4_cuda_stream_synchronize(stream);
    cudaFree(weights_dev);
    return ok;
}

int ds4_cuda_rms_norm_weight_rows_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint32_t               n,
        uint32_t               rows,
        float                  eps) {
    if (!out || !x || !model_map || n == 0 || rows == 0 || (n & 3u) != 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint64_t bytes = row_bytes * rows;
    if (ds4_cuda_tensor_bytes(out) < bytes || ds4_cuda_tensor_bytes(x) < bytes ||
        weight_offset > model_size || row_bytes > model_size - weight_offset) {
        return 0;
    }

    cudaStream_t stream = ds4_cuda_prefill_stream();
    const float *weight_host = (const float *)((const uint8_t *)model_map + weight_offset);
    void *weight_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(weight_host, row_bytes, &weight_dev)) return 0;
    dim3 grid(rows, 1, 1);
    dim3 block(256, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_rms_norm_weight_rows,
                    grid,
                    block,
                    (size_t)block.x * sizeof(double),
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(x)),
                    static_cast<const float *>(weight_dev),
                    n,
                    rows,
                    eps);
    const int ok = ds4_cuda_stream_synchronize(stream);
    cudaFree(weight_dev);
    return ok;
}

int ds4_cuda_rms_norm_weight_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint32_t               n,
        float                  eps) {
    return ds4_cuda_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, 1, eps);
}

int ds4_cuda_head_rms_norm_tensor(
        ds4_cuda_tensor *x,
        uint32_t         n_tok,
        uint32_t         n_head,
        uint32_t         head_dim,
        float            eps) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || (head_dim & 3u) != 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;
    const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
    if (ds4_cuda_tensor_bytes(x) < bytes) return 0;

    cudaStream_t stream = ds4_cuda_prefill_stream();
    dim3 grid(n_tok * n_head, 1, 1);
    dim3 block(256, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_head_rms_norm,
                    grid,
                    block,
                    (size_t)block.x * sizeof(double),
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(x)),
                    n_tok,
                    n_head,
                    head_dim,
                    eps);
    return ds4_cuda_stream_synchronize(stream);
}

static int ds4_cuda_matmul_f16_impl(ds4_cuda_tensor *out,
                                    const void *model_map,
                                    uint64_t model_size,
                                    uint64_t weight_offset,
                                    uint64_t in_dim,
                                    uint64_t out_dim,
                                    const ds4_cuda_tensor *x,
                                    uint64_t n_tok) {
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;

    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    const uint64_t row_bytes = in_dim * sizeof(uint16_t);
    const uint64_t weight_bytes = row_bytes * out_dim;
    if (ds4_cuda_tensor_bytes(x) < x_bytes || ds4_cuda_tensor_bytes(out) < out_bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return 0;
    }
    cudaStream_t stream = ds4_cuda_prefill_stream();
    const uint16_t *weights_host = (const uint16_t *)((const uint8_t *)model_map + weight_offset);
    void *weights_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(weights_host, weight_bytes, &weights_dev)) return 0;
    dim3 grid((uint32_t)out_dim, (uint32_t)n_tok, 1);
    dim3 block(32, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_matmul_f16_batch,
                    grid,
                    block,
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out)),
                    static_cast<const uint16_t *>(weights_dev),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(x)),
                    (uint32_t)in_dim,
                    (uint32_t)out_dim,
                    (uint32_t)n_tok);
    const int ok = ds4_cuda_stream_synchronize(stream);
    cudaFree(weights_dev);
    return ok;
}

int ds4_cuda_matmul_f16_tensor(ds4_cuda_tensor *out, const void *model_map,
                               uint64_t model_size, uint64_t weight_offset,
                               uint64_t in_dim, uint64_t out_dim,
                               const ds4_cuda_tensor *x, uint64_t n_tok) {
    return ds4_cuda_matmul_f16_impl(out, model_map, model_size, weight_offset,
                                    in_dim, out_dim, x, n_tok);
}

int ds4_cuda_matmul_f16_pair_tensor(ds4_cuda_tensor *out_a, ds4_cuda_tensor *out_b,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t weight_a_offset, uint64_t weight_b_offset,
                                    uint64_t in_dim, uint64_t out_dim,
                                    const ds4_cuda_tensor *x, uint64_t n_tok) {
    if (n_tok != 1) return 0;
    return ds4_cuda_matmul_f16_impl(out_a, model_map, model_size, weight_a_offset,
                                    in_dim, out_dim, x, n_tok) &&
           ds4_cuda_matmul_f16_impl(out_b, model_map, model_size, weight_b_offset,
                                    in_dim, out_dim, x, n_tok);
}

int ds4_cuda_matmul_f32_tensor(ds4_cuda_tensor *out, const void *model_map,
                               uint64_t model_size, uint64_t weight_offset,
                               uint64_t in_dim, uint64_t out_dim,
                               const ds4_cuda_tensor *x, uint64_t n_tok) {
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;

    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    const uint64_t weight_bytes = in_dim * out_dim * sizeof(float);
    if (ds4_cuda_tensor_bytes(x) < x_bytes || ds4_cuda_tensor_bytes(out) < out_bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return 0;
    }
    cudaStream_t stream = ds4_cuda_prefill_stream();
    const float *weights_host = (const float *)((const uint8_t *)model_map + weight_offset);
    void *weights_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(weights_host, weight_bytes, &weights_dev)) return 0;
    dim3 grid((uint32_t)out_dim, (uint32_t)n_tok, 1);
    dim3 block(32, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_matmul_f32_batch,
                    grid,
                    block,
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out)),
                    static_cast<const float *>(weights_dev),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(x)),
                    (uint32_t)in_dim,
                    (uint32_t)out_dim,
                    (uint32_t)n_tok);
    const int ok = ds4_cuda_stream_synchronize(stream);
    cudaFree(weights_dev);
    return ok;
}

static int ds4_cuda_matmul_q8_0_impl(ds4_cuda_tensor *out,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_offset,
                                     uint64_t in_dim,
                                     uint64_t out_dim,
                                     const ds4_cuda_tensor *x,
                                     uint64_t n_tok) {
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;

    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (ds4_cuda_tensor_bytes(x) < x_bytes || ds4_cuda_tensor_bytes(out) < out_bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return 0;
    }

    const uint8_t *weights_host = (const uint8_t *)model_map + weight_offset;
    void *weights_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(weights_host, weight_bytes, &weights_dev)) return 0;

    const uint64_t xq_bytes = n_tok * blocks * 32u * sizeof(int8_t);
    const uint64_t xscale_bytes = n_tok * blocks * sizeof(float);
    int8_t *xq = NULL;
    float *xscale = NULL;
    cudaError_t err = cudaMalloc(&xq, (size_t)xq_bytes);
    if (err != cudaSuccess) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaMalloc", err);
        cudaFree(weights_dev);
        return 0;
    }
    err = cudaMalloc(&xscale, (size_t)xscale_bytes);
    if (err != cudaSuccess) {
        ds4_cuda_report_error(__FILE__, __LINE__, "cudaMalloc", err);
        cudaFree(xq);
        cudaFree(weights_dev);
        return 0;
    }

    cudaStream_t stream = ds4_cuda_prefill_stream();
    dim3 qgrid((uint32_t)blocks, (uint32_t)n_tok, 1);
    dim3 qblock(32, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_quantize_q8_0_batch,
                    qgrid,
                    qblock,
                    (size_t)qblock.x * sizeof(float),
                    stream,
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(x)),
                    xq,
                    xscale,
                    (uint32_t)n_tok,
                    (uint32_t)in_dim,
                    (uint32_t)blocks);

    dim3 grid((uint32_t)out_dim, (uint32_t)n_tok, 1);
    dim3 block(32, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_matmul_q8_0_batch,
                    grid,
                    block,
                    (size_t)block.x * sizeof(int32_t),
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out)),
                    static_cast<const uint8_t *>(weights_dev),
                    xq,
                    xscale,
                    (uint32_t)out_dim,
                    (uint32_t)blocks,
                    (uint32_t)n_tok);

    const int ok = ds4_cuda_stream_synchronize(stream);
    cudaFree(xscale);
    cudaFree(xq);
    cudaFree(weights_dev);
    return ok;
}

int ds4_cuda_matmul_q8_0_tensor(ds4_cuda_tensor *out, const void *model_map,
                                uint64_t model_size, uint64_t weight_offset,
                                uint64_t in_dim, uint64_t out_dim,
                                const ds4_cuda_tensor *x, uint64_t n_tok) {
    return ds4_cuda_matmul_q8_0_impl(out, model_map, model_size, weight_offset,
                                     in_dim, out_dim, x, n_tok);
}

int ds4_cuda_matmul_q8_0_pair_tensor(ds4_cuda_tensor *out_a, ds4_cuda_tensor *out_b,
                                     const void *model_map, uint64_t model_size,
                                     uint64_t weight_a_offset, uint64_t weight_b_offset,
                                     uint64_t in_dim, uint64_t out_dim,
                                     const ds4_cuda_tensor *x, uint64_t n_tok) {
    return ds4_cuda_matmul_q8_0_impl(out_a, model_map, model_size, weight_a_offset,
                                     in_dim, out_dim, x, n_tok) &&
           ds4_cuda_matmul_q8_0_impl(out_b, model_map, model_size, weight_b_offset,
                                     in_dim, out_dim, x, n_tok);
}

int ds4_cuda_hc_post_tensor(ds4_cuda_tensor *out_hc, const ds4_cuda_tensor *block_out,
                            const ds4_cuda_tensor *residual_hc, const ds4_cuda_tensor *post,
                            const ds4_cuda_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc == 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;
    const uint64_t out_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
    const uint64_t post_bytes = n_hc * sizeof(float);
    const uint64_t comb_bytes = (uint64_t)n_hc * n_hc * sizeof(float);
    if (ds4_cuda_tensor_bytes(out_hc) < out_bytes ||
        ds4_cuda_tensor_bytes(block_out) < (uint64_t)n_embd * sizeof(float) ||
        ds4_cuda_tensor_bytes(residual_hc) < out_bytes ||
        ds4_cuda_tensor_bytes(post) < post_bytes ||
        ds4_cuda_tensor_bytes(comb) < comb_bytes) {
        return 0;
    }

    cudaStream_t stream = ds4_cuda_prefill_stream();
    dim3 grid((uint32_t)((out_bytes + 255u) / 256u), 1, 1);
    dim3 block(256, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_hc_post,
                    grid,
                    block,
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out_hc)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(block_out)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(residual_hc)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(post)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(comb)),
                    n_embd,
                    n_hc);
    return ds4_cuda_stream_synchronize(stream);
}

int ds4_cuda_rope_tail_inplace_tensor(ds4_cuda_tensor *x, uint32_t n_head,
                                      uint32_t head_dim, uint32_t n_rot,
                                      uint32_t pos, uint32_t n_ctx_orig,
                                      float freq_base, float freq_scale,
                                      float ext_factor, float attn_factor,
                                      float beta_fast, float beta_slow,
                                      bool inverse) {
    if (!x || n_head == 0 || head_dim == 0 || n_rot == 0 || n_rot > head_dim) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;
    const uint64_t bytes = (uint64_t)n_head * head_dim * sizeof(float);
    if (ds4_cuda_tensor_bytes(x) < bytes) return 0;

    cudaStream_t stream = ds4_cuda_prefill_stream();
    dim3 grid((n_rot + 31u) / 32u, n_head, 1);
    dim3 block(32, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_rope_tail,
                    grid,
                    block,
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(x)),
                    n_head,
                    head_dim,
                    n_rot,
                    pos,
                    (uint64_t)n_ctx_orig,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    beta_fast,
                    beta_slow,
                    inverse);
    return ds4_cuda_stream_synchronize(stream);
}

int ds4_cuda_hc_pre_from_state_tensor(ds4_cuda_tensor *out_hc,
                                      ds4_cuda_tensor *post,
                                      ds4_cuda_tensor *comb,
                                      const ds4_cuda_tensor *residual_hc,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t fn_offset,
                                      uint64_t scale_offset,
                                      uint64_t base_offset,
                                      uint32_t n_embd,
                                      uint32_t n_hc,
                                      int sinkhorn_iters,
                                      float eps) {
    if (!out_hc || !post || !comb || !residual_hc || !model_map || n_embd == 0 || n_hc == 0) {
        return 0;
    }
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;
    const uint64_t hc_dim = (uint64_t)n_embd * n_hc;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    const uint64_t mix_dim = 2u * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_dim * sizeof(float);
    const uint64_t fn_bytes = hc_dim * mix_dim * sizeof(uint16_t);
    const uint64_t scale_bytes = 3u * sizeof(float);
    const uint64_t base_bytes = mix_dim * sizeof(float);
    if (ds4_cuda_tensor_bytes(out_hc) < hc_bytes ||
        ds4_cuda_tensor_bytes(post) < (uint64_t)n_hc * sizeof(float) ||
        ds4_cuda_tensor_bytes(comb) < (uint64_t)n_hc * n_hc * sizeof(float) ||
        ds4_cuda_tensor_bytes(residual_hc) < hc_bytes ||
        fn_offset > model_size || scale_offset > model_size || base_offset > model_size ||
        fn_bytes > model_size - fn_offset || scale_bytes > model_size - scale_offset ||
        base_bytes > model_size - base_offset) {
        return 0;
    }

    cudaStream_t stream = ds4_cuda_prefill_stream();
    ds4_cuda_tensor *flat = ds4_cuda_tensor_alloc(hc_bytes);
    ds4_cuda_tensor *mix = ds4_cuda_tensor_alloc(mix_bytes);
    ds4_cuda_tensor *split = ds4_cuda_tensor_alloc((uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float));
    if (!flat || !mix || !split) {
        ds4_cuda_tensor_free(flat);
        ds4_cuda_tensor_free(mix);
        ds4_cuda_tensor_free(split);
        return 0;
    }

    if (!ds4_cuda_rms_norm_plain_tensor(flat, residual_hc, n_embd * n_hc, eps)) {
        ds4_cuda_tensor_free(flat);
        ds4_cuda_tensor_free(mix);
        ds4_cuda_tensor_free(split);
        return 0;
    }

    const float *scale_host = (const float *)((const uint8_t *)model_map + scale_offset);
    void *scale_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(scale_host, scale_bytes, &scale_dev)) {
        ds4_cuda_tensor_free(flat);
        ds4_cuda_tensor_free(mix);
        ds4_cuda_tensor_free(split);
        return 0;
    }
    const float *base_host = (const float *)((const uint8_t *)model_map + base_offset);
    void *base_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(base_host, base_bytes, &base_dev)) {
        cudaFree(scale_dev);
        ds4_cuda_tensor_free(flat);
        ds4_cuda_tensor_free(mix);
        ds4_cuda_tensor_free(split);
        return 0;
    }

    if (!ds4_cuda_matmul_f16_tensor(mix, model_map, model_size, fn_offset, hc_dim, mix_dim, flat, 1)) {
        cudaFree(scale_dev);
        cudaFree(base_dev);
        ds4_cuda_tensor_free(flat);
        ds4_cuda_tensor_free(mix);
        ds4_cuda_tensor_free(split);
        return 0;
    }

    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_hc_split_sinkhorn,
                    grid,
                    block,
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(split)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(mix)),
                    static_cast<const float *>(scale_dev),
                    static_cast<const float *>(base_dev),
                    (int)n_hc,
                    sinkhorn_iters,
                    eps);

    DS4_CUDA_LAUNCH(ds4_cuda_kernel_hc_weighted_sum,
                    dim3((uint32_t)((n_embd + 255u) / 256u), 1, 1),
                    dim3(256, 1, 1),
                    0,
                    stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out_hc)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(residual_hc)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(split)),
                    n_embd,
                    n_hc);

    const size_t post_bytes = (size_t)n_hc * sizeof(float);
    const size_t comb_bytes = (size_t)n_hc * n_hc * sizeof(float);
    if (cudaMemcpyAsync(ds4_cuda_tensor_device_ptr(post), ds4_cuda_tensor_device_ptr_const(split),
                        post_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(ds4_cuda_tensor_device_ptr(comb), (const uint8_t *)ds4_cuda_tensor_device_ptr_const(split) + 2u * n_hc * sizeof(float),
                        comb_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        cudaFree(scale_dev);
        cudaFree(base_dev);
        ds4_cuda_tensor_free(flat);
        ds4_cuda_tensor_free(mix);
        ds4_cuda_tensor_free(split);
        return 0;
    }

    const int ok = ds4_cuda_stream_synchronize(stream);
    cudaFree(scale_dev);
    cudaFree(base_dev);
    ds4_cuda_tensor_free(flat);
    ds4_cuda_tensor_free(mix);
    ds4_cuda_tensor_free(split);
    return ok;
}

// ============================================================
// Routed MoE (single-token decode) wrapper
// ============================================================

int ds4_cuda_routed_moe_one_tensor(
        ds4_cuda_tensor       *out,
        ds4_cuda_tensor       *gate,
        ds4_cuda_tensor       *up,
        ds4_cuda_tensor       *mid,
        ds4_cuda_tensor       *experts,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               gate_offset,
        uint64_t               up_offset,
        uint64_t               down_offset,
        uint32_t               gate_type,
        uint32_t               down_type,
        uint64_t               gate_expert_bytes,
        uint64_t               gate_row_bytes,
        uint64_t               down_expert_bytes,
        uint64_t               down_row_bytes,
        uint32_t               expert_in_dim,
        uint32_t               expert_mid_dim,
        uint32_t               out_dim,
        const ds4_cuda_tensor *selected,
        const ds4_cuda_tensor *weights,
        uint32_t               n_expert,
        float                  clamp,
        const ds4_cuda_tensor *x) {
    if (!out || !gate || !up || !mid || !experts || !model_map || !selected || !weights || !x) return 0;
    if (n_expert < 1 || n_expert > 6) return 0;
    if (gate_type != 16) { // DS4_TENSOR_IQ2_XXS
        fprintf(stderr, "ds4: CUDA routed MoE only supports IQ2_XXS gate type, got %u\n", gate_type);
        return 0;
    }
    if (down_type != 10) { // DS4_TENSOR_Q2_K
        fprintf(stderr, "ds4: CUDA routed MoE only supports Q2_K down type, got %u\n", down_type);
        return 0;
    }
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;

    const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
    const uint64_t gate_bytes = (uint64_t)expert_mid_dim * sizeof(float);
    const uint64_t up_bytes = gate_bytes;
    const uint64_t mid_bytes = (uint64_t)n_expert * expert_mid_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t sel_bytes = (uint64_t)n_expert * sizeof(int32_t);
    const uint64_t wgt_bytes = (uint64_t)n_expert * sizeof(float);
    if (ds4_cuda_tensor_bytes(x) < x_bytes ||
        ds4_cuda_tensor_bytes(gate) < gate_bytes * n_expert ||
        ds4_cuda_tensor_bytes(up) < up_bytes * n_expert ||
        ds4_cuda_tensor_bytes(mid) < mid_bytes ||
        ds4_cuda_tensor_bytes(out) < out_bytes ||
        ds4_cuda_tensor_bytes(selected) < sel_bytes ||
        ds4_cuda_tensor_bytes(weights) < wgt_bytes) {
        fprintf(stderr, "ds4: CUDA routed MoE received undersized buffers\n");
        return 0;
    }

    cudaStream_t stream = ds4_cuda_prefill_stream();

    int32_t host_sel[6];
    float host_wgt[6];
    if (cudaMemcpy(host_sel, ds4_cuda_tensor_device_ptr_const(selected), sel_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    if (cudaMemcpy(host_wgt, ds4_cuda_tensor_device_ptr_const(weights), wgt_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;

    // Get input x
    ds4_cuda_tensor *x_dev = ds4_cuda_tensor_alloc(x_bytes);
    if (!x_dev) return 0;
    if (cudaMemcpyAsync(ds4_cuda_tensor_device_ptr(x_dev),
                        ds4_cuda_tensor_device_ptr_const(x),
                        x_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        ds4_cuda_tensor_free(x_dev);
        return 0;
    }

    // Stage gate weights for all selected experts
    void *gate_w_dev[6] = {NULL};
    void *up_w_dev[6] = {NULL};
    void *down_w_dev[6] = {NULL};
    const block_q2_K *down_ptrs[6];
    bool stag_ok = true;
    for (uint32_t e = 0; e < n_expert && stag_ok; e++) {
        const uint64_t goff = gate_offset + (uint64_t)host_sel[e] * gate_expert_bytes;
        stag_ok = (goff <= model_size && gate_row_bytes * expert_mid_dim <= model_size - goff);
        if (stag_ok) stag_ok = ds4_cuda_copy_host_range_to_device(
                (const uint8_t *)model_map + goff,
                gate_row_bytes * expert_mid_dim, &gate_w_dev[e]);

        const uint64_t uoff = up_offset + (uint64_t)host_sel[e] * gate_expert_bytes;
        if (stag_ok) stag_ok = (uoff <= model_size && gate_row_bytes * expert_mid_dim <= model_size - uoff);
        if (stag_ok) stag_ok = ds4_cuda_copy_host_range_to_device(
                (const uint8_t *)model_map + uoff,
                gate_row_bytes * expert_mid_dim, &up_w_dev[e]);

        const uint64_t doff = down_offset + (uint64_t)host_sel[e] * down_expert_bytes;
        if (stag_ok) stag_ok = (doff <= model_size && down_row_bytes * out_dim <= model_size - doff);
        if (stag_ok) stag_ok = ds4_cuda_copy_host_range_to_device(
                (const uint8_t *)model_map + doff,
                down_row_bytes * out_dim, &down_w_dev[e]);
    }
    if (!stag_ok) {
        ds4_cuda_tensor_free(x_dev);
        for (uint32_t e = 0; e < n_expert; e++) {
            if (gate_w_dev[e]) cudaFree(gate_w_dev[e]);
            if (up_w_dev[e]) cudaFree(up_w_dev[e]);
            if (down_w_dev[e]) cudaFree(down_w_dev[e]);
        }
        return 0;
    }

    // Launch IQ2_XXS gate matvec for each expert
    for (uint32_t e = 0; e < n_expert; e++) {
        dim3 grid(expert_mid_dim, 1, 1);
        dim3 block(32, 1, 1);
        DS4_CUDA_LAUNCH(ds4_cuda_kernel_moe_iq2_xxs_matvec,
                        grid, block, (size_t)block.x * sizeof(float), stream,
                        static_cast<float *>(ds4_cuda_tensor_device_ptr(gate)) + (uint64_t)e * expert_mid_dim,
                        static_cast<const block_iq2_xxs *>(gate_w_dev[e]),
                        static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(x_dev)),
                        expert_in_dim, expert_mid_dim);
    }

    // Launch IQ2_XXS up matvec for each expert
    for (uint32_t e = 0; e < n_expert; e++) {
        dim3 grid(expert_mid_dim, 1, 1);
        dim3 block(32, 1, 1);
        DS4_CUDA_LAUNCH(ds4_cuda_kernel_moe_iq2_xxs_matvec,
                        grid, block, (size_t)block.x * sizeof(float), stream,
                        static_cast<float *>(ds4_cuda_tensor_device_ptr(up)) + (uint64_t)e * expert_mid_dim,
                        static_cast<const block_iq2_xxs *>(up_w_dev[e]),
                        static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(x_dev)),
                        expert_in_dim, expert_mid_dim);
    }

    // Setup down weight pointers array for sum6 kernel
    ds4_cuda_tensor *down_ptrs_dev = ds4_cuda_tensor_alloc((uint64_t)n_expert * sizeof(const block_q2_K *));
    if (!down_ptrs_dev) {
        ds4_cuda_tensor_free(x_dev);
        for (uint32_t e = 0; e < n_expert; e++) {
            if (gate_w_dev[e]) cudaFree(gate_w_dev[e]);
            if (up_w_dev[e]) cudaFree(up_w_dev[e]);
            if (down_w_dev[e]) cudaFree(down_w_dev[e]);
        }
        return 0;
    }
    for (uint32_t e = 0; e < n_expert; e++) down_ptrs[e] = (const block_q2_K *)down_w_dev[e];
    if (cudaMemcpyAsync(ds4_cuda_tensor_device_ptr(down_ptrs_dev), down_ptrs,
                        (uint64_t)n_expert * sizeof(const block_q2_K *),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        ds4_cuda_tensor_free(down_ptrs_dev);
        ds4_cuda_tensor_free(x_dev);
        for (uint32_t e = 0; e < n_expert; e++) {
            if (gate_w_dev[e]) cudaFree(gate_w_dev[e]);
            if (up_w_dev[e]) cudaFree(up_w_dev[e]);
            if (down_w_dev[e]) cudaFree(down_w_dev[e]);
        }
        return 0;
    }

    // Stage route weights for sum6 kernel
    ds4_cuda_tensor *rw_dev = ds4_cuda_tensor_alloc(wgt_bytes);
    if (!rw_dev) {
        ds4_cuda_tensor_free(down_ptrs_dev);
        ds4_cuda_tensor_free(x_dev);
        for (uint32_t e = 0; e < n_expert; e++) {
            if (gate_w_dev[e]) cudaFree(gate_w_dev[e]);
            if (up_w_dev[e]) cudaFree(up_w_dev[e]);
            if (down_w_dev[e]) cudaFree(down_w_dev[e]);
        }
        return 0;
    }
    if (cudaMemcpyAsync(ds4_cuda_tensor_device_ptr(rw_dev), host_wgt, wgt_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        ds4_cuda_tensor_free(down_ptrs_dev);
        ds4_cuda_tensor_free(rw_dev);
        ds4_cuda_tensor_free(x_dev);
        for (uint32_t e = 0; e < n_expert; e++) {
            if (gate_w_dev[e]) cudaFree(gate_w_dev[e]);
            if (up_w_dev[e]) cudaFree(up_w_dev[e]);
            if (down_w_dev[e]) cudaFree(down_w_dev[e]);
        }
        return 0;
    }

    // Launch SwiGLU weighted kernel
    {
        dim3 swigrid(((uint64_t)n_expert * expert_mid_dim + 255u) / 256u, 1, 1);
        dim3 swiblock(256, 1, 1);
        DS4_CUDA_LAUNCH(ds4_cuda_kernel_moe_swiglu_weighted,
                        swigrid, swiblock, 0, stream,
                        static_cast<float *>(ds4_cuda_tensor_device_ptr(mid)),
                        static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(gate)),
                        static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(up)),
                        static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(rw_dev)),
                        expert_mid_dim, n_expert, clamp);
    }

    // Launch Q2_K sum6 down matvec
    {
        dim3 grid(out_dim, 1, 1);
        dim3 block(32, 1, 1);
        DS4_CUDA_LAUNCH(ds4_cuda_kernel_moe_q2_k_sum6,
                        grid, block, (size_t)block.x * sizeof(float), stream,
                        static_cast<float *>(ds4_cuda_tensor_device_ptr(out)),
                        (const block_q2_K *const *)ds4_cuda_tensor_device_ptr_const(down_ptrs_dev),
                        static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(mid)),
                        static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(rw_dev)),
                        expert_mid_dim, out_dim, n_expert);
    }

    ds4_cuda_tensor_free(rw_dev);
    ds4_cuda_tensor_free(down_ptrs_dev);

    const int ok = ds4_cuda_stream_synchronize(stream);
    ds4_cuda_tensor_free(x_dev);
    for (uint32_t e = 0; e < n_expert; e++) {
        cudaFree(gate_w_dev[e]);
        cudaFree(up_w_dev[e]);
        cudaFree(down_w_dev[e]);
    }
    return ok;
}

// ============================================================
// Shared expert fused gate+up+swiglu (decode)
// ============================================================

int ds4_cuda_shared_gate_up_swiglu_q8_0_tensor(
        ds4_cuda_tensor       *gate,
        ds4_cuda_tensor       *up,
        ds4_cuda_tensor       *mid,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               gate_offset,
        uint64_t               up_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *x) {
    if (!gate || !up || !mid || !model_map || !x || in_dim == 0 || out_dim == 0) return 0;
    if ((in_dim & 31u) != 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;

    const uint64_t x_bytes = in_dim * sizeof(float);
    const uint64_t out_bytes = out_dim * sizeof(float);
    const uint64_t blocks = in_dim / 32u;
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (ds4_cuda_tensor_bytes(x) < x_bytes ||
        ds4_cuda_tensor_bytes(gate) < out_bytes ||
        ds4_cuda_tensor_bytes(up) < out_bytes ||
        ds4_cuda_tensor_bytes(mid) < out_bytes) {
        return 0;
    }
    if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
        up_offset > model_size || weight_bytes > model_size - up_offset) {
        return 0;
    }

    cudaStream_t stream = ds4_cuda_prefill_stream();

    const uint64_t xq_bytes = blocks * 32u * sizeof(int8_t);
    const uint64_t xscale_bytes = blocks * sizeof(float);

    const uint8_t *gate_host = (const uint8_t *)model_map + gate_offset;
    void *gate_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(gate_host, weight_bytes, &gate_dev)) return 0;

    const uint8_t *up_host = (const uint8_t *)model_map + up_offset;
    void *up_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(up_host, weight_bytes, &up_dev)) { cudaFree(gate_dev); return 0; }

    int8_t *xq = NULL;
    float *xscale = NULL;
    if (cudaMalloc(&xq, (size_t)xq_bytes) != cudaSuccess ||
        cudaMalloc(&xscale, (size_t)xscale_bytes) != cudaSuccess) {
        cudaFree(gate_dev); cudaFree(up_dev);
        if (xq) cudaFree(xq);
        return 0;
    }

    dim3 qgrid((uint32_t)blocks, 1u, 1);
    dim3 qblock(32, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_quantize_q8_0_batch,
                    qgrid, qblock, (size_t)qblock.x * sizeof(float), stream,
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(x)),
                    xq, xscale, 1u, (uint32_t)in_dim, (uint32_t)blocks);

    dim3 grid((uint32_t)out_dim, 1, 1);
    dim3 block(32, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_shared_gate_up_swiglu_q8_0,
                    grid, block, (size_t)block.x * sizeof(int32_t), stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(gate)),
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(up)),
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(mid)),
                    static_cast<const uint8_t *>(gate_dev),
                    static_cast<const uint8_t *>(up_dev),
                    xq, xscale,
                    (uint32_t)in_dim, (uint32_t)out_dim);

    const int ok = ds4_cuda_stream_synchronize(stream);
    cudaFree(gate_dev);
    cudaFree(up_dev);
    cudaFree(xq);
    cudaFree(xscale);
    return ok;
}

// ============================================================
// Shared expert fused down + HC expand (decode)
// ============================================================

int ds4_cuda_shared_down_hc_expand_q8_0_tensor(
        ds4_cuda_tensor       *out_hc,
        ds4_cuda_tensor       *shared_out,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *shared_mid,
        const ds4_cuda_tensor *routed_out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *split,
        uint32_t               n_embd,
        uint32_t               n_hc) {
    if (!out_hc || !shared_out || !model_map || !shared_mid || !routed_out ||
        !residual_hc || !split || n_embd == 0 || n_hc == 0 || n_hc != 4) return 0;
    if ((in_dim & 31u) != 0) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;

    const uint64_t blocks = in_dim / 32u;
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    const uint64_t mid_bytes = in_dim * sizeof(float);
    const uint64_t embd_bytes = out_dim * sizeof(float);
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t mix_hc = 2u * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t split_bytes = mix_hc * sizeof(float);

    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    if (ds4_cuda_tensor_bytes(shared_mid) < mid_bytes ||
        ds4_cuda_tensor_bytes(shared_out) < embd_bytes ||
        ds4_cuda_tensor_bytes(routed_out) < embd_bytes ||
        ds4_cuda_tensor_bytes(residual_hc) < hc_bytes ||
        ds4_cuda_tensor_bytes(split) < split_bytes ||
        ds4_cuda_tensor_bytes(out_hc) < hc_bytes) {
        return 0;
    }

    cudaStream_t stream = ds4_cuda_prefill_stream();

    const uint8_t *w_host = (const uint8_t *)model_map + weight_offset;
    void *w_dev = NULL;
    if (!ds4_cuda_copy_host_range_to_device(w_host, weight_bytes, &w_dev)) return 0;

    const uint64_t xq_bytes = blocks * 32u * sizeof(int8_t);
    const uint64_t xscale_bytes = blocks * sizeof(float);
    int8_t *xq = NULL;
    float *xscale = NULL;
    if (cudaMalloc(&xq, (size_t)xq_bytes) != cudaSuccess ||
        cudaMalloc(&xscale, (size_t)xscale_bytes) != cudaSuccess) {
        cudaFree(w_dev);
        if (xq) cudaFree(xq);
        return 0;
    }

    dim3 qgrid((uint32_t)blocks, 1u, 1);
    dim3 qblock(32, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_quantize_q8_0_batch,
                    qgrid, qblock, (size_t)qblock.x * sizeof(float), stream,
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(shared_mid)),
                    xq, xscale, 1u, (uint32_t)in_dim, (uint32_t)blocks);

    // Fused down + HC expand
    dim3 grid((uint32_t)out_dim, 1, 1);
    dim3 block(32, 1, 1);
    DS4_CUDA_LAUNCH(ds4_cuda_kernel_shared_down_hc_expand_q8_0,
                    grid, block, (size_t)block.x * sizeof(int32_t), stream,
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(out_hc)),
                    static_cast<float *>(ds4_cuda_tensor_device_ptr(shared_out)),
                    static_cast<const uint8_t *>(w_dev),
                    xq, xscale,
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(routed_out)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(residual_hc)),
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(split)) + n_hc,
                    static_cast<const float *>(ds4_cuda_tensor_device_ptr_const(split)) + (uint64_t)2u * n_hc,
                    (uint32_t)in_dim, (uint32_t)out_dim, n_hc);

    const int ok = ds4_cuda_stream_synchronize(stream);
    cudaFree(w_dev);
    cudaFree(xq);
    cudaFree(xscale);
    return ok;
}
