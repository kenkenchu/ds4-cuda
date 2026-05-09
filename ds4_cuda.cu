#include "ds4_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

static int g_cuda_device = -1;
static char g_cuda_name[256];
static int g_cuda_cc_major = 0;
static int g_cuda_cc_minor = 0;
static uint64_t g_cuda_total = 0;

static void ds4_cuda_reset_state(void) {
    g_cuda_device = -1;
    g_cuda_name[0] = '\0';
    g_cuda_cc_major = 0;
    g_cuda_cc_minor = 0;
    g_cuda_total = 0;
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

    int best = 0;
    int best_score = -1;
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) continue;
        int score = prop.major * 100 + prop.minor;
        if (prop.unifiedAddressing) score += 10000;
        if (score > best_score) {
            best = i;
            best_score = score;
        }
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

    g_cuda_device = best;
    snprintf(g_cuda_name, sizeof(g_cuda_name), "%s", prop.name);
    g_cuda_cc_major = prop.major;
    g_cuda_cc_minor = prop.minor;
    g_cuda_total = (uint64_t)prop.totalGlobalMem;

    if (!prop.unifiedAddressing) {
        fprintf(stderr, "ds4: CUDA warning: selected device does not report unified addressing\n");
    }
    if (prop.major < 12) {
        fprintf(stderr,
                "ds4: CUDA warning: selected device is sm_%d%d, optimized target is GB10/Blackwell sm_121\n",
                prop.major,
                prop.minor);
    }

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1u << 20);
    fprintf(stderr,
            "ds4: CUDA backend initialized on device %d: %s (sm_%d%d, %.2f GiB global memory)\n",
            g_cuda_device,
            g_cuda_name,
            g_cuda_cc_major,
            g_cuda_cc_minor,
            (double)g_cuda_total / (1024.0 * 1024.0 * 1024.0));
    return 1;
}

void ds4_cuda_cleanup(void) {
    if (g_cuda_device >= 0) {
        cudaDeviceSynchronize();
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
            "ds4: CUDA memory %s: free %.2f GiB / total %.2f GiB\n",
            label ? label : "",
            (double)free_b / (1024.0 * 1024.0 * 1024.0),
            (double)total_b / (1024.0 * 1024.0 * 1024.0));
}
