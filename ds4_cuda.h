#ifndef DS4_CUDA_H
#define DS4_CUDA_H

#include <stdbool.h>
#include <stdint.h>

#ifndef DS4_CUDA_RUNTIME_TYPES
typedef struct CUstream_st *cudaStream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_cuda_tensor ds4_cuda_tensor;

int ds4_cuda_init(bool quality);
void ds4_cuda_cleanup(void);
bool ds4_cuda_available(void);
const char *ds4_cuda_device_name(void);
uint64_t ds4_cuda_total_memory(void);
uint64_t ds4_cuda_free_memory(void);
int ds4_cuda_compute_capability_major(void);
int ds4_cuda_compute_capability_minor(void);
void ds4_cuda_print_memory_report(const char *label);
cudaStream_t ds4_cuda_prefill_stream(void);
cudaStream_t ds4_cuda_decode_stream(void);
int ds4_cuda_synchronize(void);
int ds4_cuda_stream_synchronize(cudaStream_t stream);
int ds4_cuda_register_host_memory(const void *ptr, uint64_t bytes);
void ds4_cuda_unregister_host_memory(const void *ptr);

ds4_cuda_tensor *ds4_cuda_tensor_alloc(uint64_t bytes);
ds4_cuda_tensor *ds4_cuda_tensor_view(const ds4_cuda_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_cuda_tensor_free(ds4_cuda_tensor *tensor);
uint64_t ds4_cuda_tensor_bytes(const ds4_cuda_tensor *tensor);
void *ds4_cuda_tensor_device_ptr(ds4_cuda_tensor *tensor);
const void *ds4_cuda_tensor_device_ptr_const(const ds4_cuda_tensor *tensor);
int ds4_cuda_tensor_write(ds4_cuda_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_cuda_tensor_read(const ds4_cuda_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_cuda_tensor_copy(ds4_cuda_tensor *dst, uint64_t dst_offset,
                         const ds4_cuda_tensor *src, uint64_t src_offset,
                         uint64_t bytes);
int ds4_cuda_tensor_memset(ds4_cuda_tensor *tensor, uint64_t offset, int value, uint64_t bytes);
int ds4_cuda_repeat_hc_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *row,
                              uint32_t n_embd, uint32_t n_hc);
int ds4_cuda_add_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *a,
                        const ds4_cuda_tensor *b, uint32_t n);
int ds4_cuda_router_select_tensor(ds4_cuda_tensor *selected, ds4_cuda_tensor *weights,
                                  ds4_cuda_tensor *probs, const void *model_map,
                                  uint64_t model_size, uint64_t bias_offset,
                                  uint64_t hash_offset, uint32_t hash_rows,
                                  uint32_t token, uint32_t n_expert_groups,
                                  uint32_t n_group_used, bool has_bias,
                                  bool hash_mode, const ds4_cuda_tensor *logits);
int ds4_cuda_router_select_batch_tensor(ds4_cuda_tensor *selected, ds4_cuda_tensor *weights,
                                        ds4_cuda_tensor *probs, const void *model_map,
                                        uint64_t model_size, uint64_t bias_offset,
                                        uint64_t hash_offset, uint32_t hash_rows,
                                        uint32_t n_expert_groups, uint32_t n_group_used,
                                        bool has_bias, bool hash_mode,
                                        const ds4_cuda_tensor *logits,
                                        const ds4_cuda_tensor *tokens,
                                        uint32_t n_tokens);
int ds4_cuda_rms_norm_plain_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *x,
                                   uint32_t n, float eps);
int ds4_cuda_rms_norm_plain_rows_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *x,
                                        uint32_t n, uint32_t rows, float eps);
int ds4_cuda_embed_token_hc_tensor(ds4_cuda_tensor *out_hc, const void *model_map,
                                   uint64_t model_size, uint64_t weight_offset,
                                   uint32_t n_vocab, uint32_t token,
                                   uint32_t n_embd, uint32_t n_hc);
int ds4_cuda_embed_tokens_hc_tensor(ds4_cuda_tensor *out_hc, const ds4_cuda_tensor *tokens,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t weight_offset, uint32_t n_vocab,
                                    uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc);
int ds4_cuda_rms_norm_weight_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *x,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t weight_offset, uint32_t n, float eps);
int ds4_cuda_rms_norm_weight_rows_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *x,
                                         const void *model_map, uint64_t model_size,
                                         uint64_t weight_offset, uint32_t n,
                                         uint32_t rows, float eps);
int ds4_cuda_head_rms_norm_tensor(ds4_cuda_tensor *x, uint32_t n_tok,
                                  uint32_t n_head, uint32_t head_dim, float eps);
int ds4_cuda_matmul_f16_tensor(ds4_cuda_tensor *out, const void *model_map,
                               uint64_t model_size, uint64_t weight_offset,
                               uint64_t in_dim, uint64_t out_dim,
                               const ds4_cuda_tensor *x, uint64_t n_tok);
int ds4_cuda_matmul_f16_pair_tensor(ds4_cuda_tensor *out_a, ds4_cuda_tensor *out_b,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t weight_a_offset, uint64_t weight_b_offset,
                                    uint64_t in_dim, uint64_t out_dim,
                                    const ds4_cuda_tensor *x, uint64_t n_tok);
int ds4_cuda_matmul_f32_tensor(ds4_cuda_tensor *out, const void *model_map,
                               uint64_t model_size, uint64_t weight_offset,
                               uint64_t in_dim, uint64_t out_dim,
                               const ds4_cuda_tensor *x, uint64_t n_tok);
int ds4_cuda_matmul_q8_0_tensor(ds4_cuda_tensor *out, const void *model_map,
                                uint64_t model_size, uint64_t weight_offset,
                                uint64_t in_dim, uint64_t out_dim,
                                const ds4_cuda_tensor *x, uint64_t n_tok);
int ds4_cuda_matmul_q8_0_pair_tensor(ds4_cuda_tensor *out_a, ds4_cuda_tensor *out_b,
                                     const void *model_map, uint64_t model_size,
                                     uint64_t weight_a_offset, uint64_t weight_b_offset,
                                     uint64_t in_dim, uint64_t out_dim,
                                     const ds4_cuda_tensor *x, uint64_t n_tok);
int ds4_cuda_hc_post_tensor(ds4_cuda_tensor *out_hc, const ds4_cuda_tensor *block_out,
                            const ds4_cuda_tensor *residual_hc, const ds4_cuda_tensor *post,
                            const ds4_cuda_tensor *comb, uint32_t n_embd, uint32_t n_hc);
int ds4_cuda_rope_tail_inplace_tensor(ds4_cuda_tensor *x, uint32_t n_head,
                                      uint32_t head_dim, uint32_t n_rot,
                                      uint32_t pos, uint32_t n_ctx_orig,
                                      float freq_base, float freq_scale,
                                      float ext_factor, float attn_factor,
                                      float beta_fast, float beta_slow,
                                      bool inverse);
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
                                      float eps);

#ifdef __cplusplus
}
#endif

#endif
