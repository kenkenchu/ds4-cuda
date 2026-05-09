#define DS4_CUDA_RUNTIME_TYPES
#include <cuda_runtime.h>

#include "ds4_cuda.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <new>
#include <cmath>

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

    const uint8_t *weights = (const uint8_t *)model_map + weight_offset;
    const uint16_t *row = (const uint16_t *)weights + (uint64_t)token * n_embd;
    float *stage = (float *)malloc((size_t)out_bytes);
    if (!stage) return 0;
    for (uint32_t h = 0; h < n_hc; h++) {
        float *dst = stage + (uint64_t)h * n_embd;
        for (uint32_t i = 0; i < n_embd; i++) {
            dst[i] = ds4_cuda_f16_to_f32(row[i]);
        }
    }
    int ok = ds4_cuda_tensor_write(out_hc, 0, stage, out_bytes);
    free(stage);
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

    int32_t *token_host = (int32_t *)malloc((size_t)token_bytes);
    float *stage = (float *)malloc((size_t)out_bytes);
    if (!token_host || !stage) {
        fprintf(stderr, "ds4: CUDA batch embedding host allocation failed\n");
        free(token_host);
        free(stage);
        return 0;
    }
    if (!ds4_cuda_tensor_read(tokens, 0, token_host, token_bytes)) {
        fprintf(stderr, "ds4: CUDA batch embedding failed to read token tensor\n");
        free(token_host);
        free(stage);
        return 0;
    }

    const uint16_t *weights = (const uint16_t *)((const uint8_t *)model_map + weight_offset);
    for (uint32_t t = 0; t < n_tokens; t++) {
        int32_t token = token_host[t];
        if (token < 0 || (uint32_t)token >= n_vocab) {
            fprintf(stderr, "ds4: CUDA batch embedding token %d outside vocabulary\n", (int)token);
            free(token_host);
            free(stage);
            return 0;
        }
        const uint16_t *row = weights + (uint64_t)token * n_embd;
        float *dst_token = stage + (uint64_t)t * n_embd * n_hc;
        for (uint32_t h = 0; h < n_hc; h++) {
            float *dst = dst_token + (uint64_t)h * n_embd;
            for (uint32_t i = 0; i < n_embd; i++) {
                dst[i] = ds4_cuda_f16_to_f32(row[i]);
            }
        }
    }

    int ok = ds4_cuda_tensor_write(out_hc, 0, stage, out_bytes);
    if (!ok) {
        fprintf(stderr, "ds4: CUDA batch embedding failed to write output tensor\n");
    }
    free(token_host);
    free(stage);
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

    float *x_host = (float *)malloc((size_t)bytes);
    float *out_host = (float *)malloc((size_t)bytes);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        return 0;
    }
    if (!ds4_cuda_tensor_read(x, 0, x_host, bytes)) {
        free(x_host);
        free(out_host);
        return 0;
    }

    const float *weight = (const float *)((const uint8_t *)model_map + weight_offset);
    for (uint32_t r = 0; r < rows; r++) {
        const float *xin = x_host + (uint64_t)r * n;
        float *yout = out_host + (uint64_t)r * n;
        float scale = ds4_cuda_rms_scale(xin, n, eps);
        for (uint32_t i = 0; i < n; i++) {
            yout[i] = xin[i] * scale * weight[i];
        }
    }

    int ok = ds4_cuda_tensor_write(out, 0, out_host, bytes);
    free(x_host);
    free(out_host);
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

    float *host = (float *)malloc((size_t)bytes);
    if (!host) return 0;
    if (!ds4_cuda_tensor_read(x, 0, host, bytes)) {
        free(host);
        return 0;
    }

    const uint32_t rows = n_tok * n_head;
    for (uint32_t r = 0; r < rows; r++) {
        float *row = host + (uint64_t)r * head_dim;
        float scale = ds4_cuda_rms_scale(row, head_dim, eps);
        for (uint32_t i = 0; i < head_dim; i++) {
            row[i] *= scale;
        }
    }

    int ok = ds4_cuda_tensor_write(x, 0, host, bytes);
    free(host);
    return ok;
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
    if (in_dim > (uint64_t)SIZE_MAX / sizeof(float) || out_dim > (uint64_t)SIZE_MAX / sizeof(float)) return 0;
    if (g_cuda_device < 0 && !ds4_cuda_init(false)) return 0;

    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    const uint64_t row_bytes = in_dim * sizeof(uint16_t);
    const uint64_t weight_bytes = row_bytes * out_dim;
    if (ds4_cuda_tensor_bytes(x) < x_bytes || ds4_cuda_tensor_bytes(out) < out_bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return 0;
    }

    float *x_host = (float *)malloc((size_t)x_bytes);
    float *out_host = (float *)malloc((size_t)out_bytes);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        return 0;
    }
    if (!ds4_cuda_tensor_read(x, 0, x_host, x_bytes)) {
        free(x_host);
        free(out_host);
        return 0;
    }

    const uint16_t *weights = (const uint16_t *)((const uint8_t *)model_map + weight_offset);
    for (uint64_t t = 0; t < n_tok; t++) {
        const float *xin = x_host + t * in_dim;
        float *yout = out_host + t * out_dim;
        for (uint64_t o = 0; o < out_dim; o++) {
            const uint16_t *row = weights + o * in_dim;
            double acc = 0.0;
            for (uint64_t i = 0; i < in_dim; i++) {
                acc += (double)ds4_cuda_f16_to_f32(row[i]) * (double)xin[i];
            }
            yout[o] = (float)acc;
        }
    }

    int ok = ds4_cuda_tensor_write(out, 0, out_host, out_bytes);
    free(x_host);
    free(out_host);
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

    float *x_host = (float *)malloc((size_t)x_bytes);
    float *out_host = (float *)malloc((size_t)out_bytes);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        return 0;
    }
    if (!ds4_cuda_tensor_read(x, 0, x_host, x_bytes)) {
        free(x_host);
        free(out_host);
        return 0;
    }

    const float *weights = (const float *)((const uint8_t *)model_map + weight_offset);
    for (uint64_t t = 0; t < n_tok; t++) {
        const float *xin = x_host + t * in_dim;
        float *yout = out_host + t * out_dim;
        for (uint64_t o = 0; o < out_dim; o++) {
            const float *row = weights + o * in_dim;
            double acc = 0.0;
            for (uint64_t i = 0; i < in_dim; i++) {
                acc += (double)row[i] * (double)xin[i];
            }
            yout[o] = (float)acc;
        }
    }

    int ok = ds4_cuda_tensor_write(out, 0, out_host, out_bytes);
    free(x_host);
    free(out_host);
    return ok;
}
