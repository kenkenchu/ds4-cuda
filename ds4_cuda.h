#ifndef DS4_CUDA_H
#define DS4_CUDA_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int ds4_cuda_init(bool quality);
void ds4_cuda_cleanup(void);
bool ds4_cuda_available(void);
const char *ds4_cuda_device_name(void);
uint64_t ds4_cuda_total_memory(void);
uint64_t ds4_cuda_free_memory(void);
int ds4_cuda_compute_capability_major(void);
int ds4_cuda_compute_capability_minor(void);
void ds4_cuda_print_memory_report(const char *label);

#ifdef __cplusplus
}
#endif

#endif
