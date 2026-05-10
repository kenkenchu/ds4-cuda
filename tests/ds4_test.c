#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#ifndef DS4_NO_CUDA
#include "../ds4_cuda.h"
#endif
#include <math.h>

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float test_f16_to_float(uint16_t h) {
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

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float test_softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void test_router_topk_desc(const float *score, int n, int k, int *idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (idx[j] < 0 || score[i] > score[idx[j]]) {
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = i;
                break;
            }
        }
    }
}

static void test_router_expected(const float *logits, const float *bias, const int32_t *hash_row,
                                 bool hash_mode, int *selected, float *weights) {
    float probs[256];
    float scores[256];
    for (int i = 0; i < 256; i++) {
        probs[i] = sqrtf(test_softplus_stable(logits[i]));
        scores[i] = probs[i] + (bias ? bias[i] : 0.0f);
    }

    if (hash_mode) {
        for (int i = 0; i < 6; i++) selected[i] = hash_row[i];
    } else {
        test_router_topk_desc(scores, 256, 6, selected);
    }

    float sum = 0.0f;
    for (int i = 0; i < 6; i++) {
        weights[i] = (selected[i] >= 0 && selected[i] < 256) ? probs[selected[i]] : 0.0f;
        sum += weights[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int i = 0; i < 6; i++) {
        weights[i] = weights[i] / sum * 1.5f;
    }
}

#ifndef DS4_NO_METAL
#include "../ds4_metal.h"

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    ds4_engine_options opt = {
        .model_path = test_model_path(),
        .backend = DS4_BACKEND_METAL,
        .quality = quality,
    };
    TEST_ASSERT(ds4_engine_open(slot, &opt) == 0);
    return *slot;
}

static void test_close_engines(void) {
    ds4_engine_close(test_engine_fast);
    ds4_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    ds4_engine_close(*slot);
    *slot = NULL;
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several DS4 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    ds4_metal_tensor *x = ds4_metal_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_metal_tensor *out = ds4_metal_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_metal_tensor_free(x);
        ds4_metal_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_metal_tensor_free(x);
        ds4_metal_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(ds4_metal_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_metal_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_metal_set_quality(false);
    TEST_ASSERT(ds4_metal_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_metal_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    ds4_metal_tensor_free(x);
    ds4_metal_tensor_free(out);
    free(weights_raw);
}

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

static int test_count_substr(const char *s, const char *needle) {
    int count = 0;
    size_t n = strlen(needle);
    const char *p = s;
    while ((p = strstr(p, needle)) != NULL) {
        count++;
        p += n;
    }
    return count;
}

static int test_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = test_hex_digit(hex[0]);
        int lo = test_hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool test_token_bytes_equal(ds4_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = ds4_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "ds4-test: long-context prefill %d/%d\n", current, total);
    }
}

static void test_long_security_continuation(void) {
    const char *prompt_path = getenv("DS4_TEST_LONG_PROMPT");
    if (!prompt_path || !prompt_path[0]) {
        prompt_path = "tests/long_context_security_prompt.txt";
    }
    char *prompt_text = test_read_file(prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        free(prompt_text);
        return;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    TEST_ASSERT(prompt.len > 30000);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 100000) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        free(prompt_text);
        return;
    }

    char err[160];
    ds4_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    ds4_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 700; generated++) {
        int token = ds4_session_sample(session, 0.8f, 40, 0.95f, 0.05f, &rng);
        if (token == ds4_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    const char *text = out.ptr ? out.ptr : "";
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    TEST_ASSERT(strstr(text, "</think>") != NULL);
    TEST_ASSERT(test_count_substr(text, "</think>") == 1);
    TEST_ASSERT(test_count_substr(text, "The most critical security issue") == 1);
    TEST_ASSERT(strstr(text, "arbitrary file") != NULL);

    buf_free(&out);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    free(prompt_text);
}

#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static bool test_read_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= TEST_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before vector case");
    }
    return false;
}

static bool test_fill_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;

        if (!strncmp(p, "step ", 5)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                TEST_ASSERT(!"bad vector step line");
                return false;
            }
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(ntop >= 0 && ntop <= TEST_VEC_MAX_TOP);
            vc->steps[step_index].ntop = ntop;
            TEST_ASSERT(test_hex_to_bytes(hex,
                                          vc->steps[step_index].selected,
                                          TEST_VEC_MAX_TOKEN_BYTES,
                                          &vc->steps[step_index].selected_len));
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                TEST_ASSERT(!"bad vector top line");
                return false;
            }
            test_vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            TEST_ASSERT(test_hex_to_bytes(hex, top->bytes,
                                          TEST_VEC_MAX_TOKEN_BYTES, &top->len));
            continue;
        }

        TEST_ASSERT(!"unexpected vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated vector case");
    return false;
}

static void test_logprob_vector_case(ds4_engine *engine, const test_vec_case *vc) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);

    ds4_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int nscore = ds4_session_top_logprobs(session, scores, 20);
        int token = ds4_session_argmax(session);
        if (!test_token_bytes_equal(engine, token, step->selected, step->selected_len)) {
            fprintf(stderr, "ds4-test: vector %s step %d selected token mismatch\n",
                    vc->id, i);
            TEST_ASSERT(false);
        }

        for (int t = 0; t < step->ntop; t++) {
            bool found = false;
            float local_lp = 0.0f;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id < 0) continue;
                if (test_token_bytes_equal(engine, scores[j].id,
                                           step->top[t].bytes,
                                           step->top[t].len)) {
                    found = true;
                    local_lp = scores[j].logprob;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "ds4-test: vector %s step %d official top token missing locally\n",
                        vc->id, i);
                TEST_ASSERT(false);
            } else if (fabsf(local_lp - step->top[t].logprob) > 4.0f) {
                fprintf(stderr,
                        "ds4-test: vector %s step %d logprob delta too high: local=%g official=%g\n",
                        vc->id, i, local_lp, step->top[t].logprob);
                TEST_ASSERT(false);
            }
        }

        if (i + 1 < vc->nsteps) {
            TEST_ASSERT(ds4_session_eval(session, token, err, sizeof(err)) == 0);
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static void test_official_logprob_vectors(void) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        fclose(fp);
        return;
    }

    test_vec_case vc;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        fprintf(stderr, "ds4-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
    }
    fclose(fp);
}

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_tool_call_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    for (int i = 0; i < r.max_tokens; i++) {
        int token = ds4_session_sample(session, r.temperature, r.top_k,
                                       r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr ? text.ptr : "", &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message(text.ptr ? text.ptr : "",
                                          &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

#endif

#ifndef DS4_NO_CUDA
static void test_cuda_tensor_layer(void) {
    if (!ds4_cuda_init(false)) {
        fprintf(stderr, "ds4-test: CUDA unavailable, skipping tensor layer test\n");
        return;
    }

    TEST_ASSERT(ds4_cuda_available());
    TEST_ASSERT(ds4_cuda_prefill_stream() != NULL);
    TEST_ASSERT(ds4_cuda_decode_stream() != NULL);
    TEST_ASSERT(ds4_cuda_stream_synchronize(ds4_cuda_prefill_stream()) != 0);
    TEST_ASSERT(ds4_cuda_stream_synchronize(ds4_cuda_decode_stream()) != 0);

    ds4_cuda_tensor *a = ds4_cuda_tensor_alloc(64);
    ds4_cuda_tensor *b = ds4_cuda_tensor_alloc(64);
    TEST_ASSERT(a != NULL);
    TEST_ASSERT(b != NULL);
    if (!a || !b) {
        ds4_cuda_tensor_free(a);
        ds4_cuda_tensor_free(b);
        ds4_cuda_cleanup();
        return;
    }

    uint8_t host_a[64];
    uint8_t host_b[64];
    uint8_t host_view[16];
    for (uint32_t i = 0; i < 64; i++) {
        host_a[i] = (uint8_t)(i ^ 0x5a);
        host_b[i] = 0;
    }

    TEST_ASSERT(ds4_cuda_tensor_write(a, 0, host_a, sizeof(host_a)) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(a, 0, host_b, sizeof(host_b)) != 0);
    TEST_ASSERT(memcmp(host_a, host_b, sizeof(host_a)) == 0);

    ds4_cuda_tensor *view = ds4_cuda_tensor_view(a, 16, 16);
    TEST_ASSERT(view != NULL);
    if (!view) {
        ds4_cuda_tensor_free(a);
        ds4_cuda_tensor_free(b);
        ds4_cuda_cleanup();
        return;
    }
    memset(host_view, 0, sizeof(host_view));
    TEST_ASSERT(ds4_cuda_tensor_read(view, 0, host_view, sizeof(host_view)) != 0);
    TEST_ASSERT(memcmp(host_a + 16, host_view, sizeof(host_view)) == 0);

    TEST_ASSERT(ds4_cuda_tensor_copy(b, 8, a, 0, 32) != 0);
    memset(host_b, 0, sizeof(host_b));
    TEST_ASSERT(ds4_cuda_tensor_read(b, 0, host_b, sizeof(host_b)) != 0);
    for (uint32_t i = 0; i < 8; i++) {
        TEST_ASSERT(host_b[i] == 0);
    }
    TEST_ASSERT(memcmp(host_a, host_b + 8, 32) == 0);
    for (uint32_t i = 40; i < 64; i++) {
        TEST_ASSERT(host_b[i] == 0);
    }

    TEST_ASSERT(ds4_cuda_tensor_memset(b, 0, 0x3c, 64) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(b, 0, host_b, sizeof(host_b)) != 0);
    for (uint32_t i = 0; i < 64; i++) {
        TEST_ASSERT(host_b[i] == 0x3c);
    }

    TEST_ASSERT(ds4_cuda_tensor_bytes(a) == 64);
    TEST_ASSERT(ds4_cuda_tensor_bytes(view) == 16);
    TEST_ASSERT(ds4_cuda_tensor_device_ptr(a) != NULL);
    TEST_ASSERT(ds4_cuda_tensor_device_ptr_const(view) != NULL);

    TEST_ASSERT(ds4_cuda_tensor_write(a, 60, host_a, 4) != 0);
    TEST_ASSERT(ds4_cuda_tensor_write(a, 61, host_a, 4) == 0);
    TEST_ASSERT(ds4_cuda_tensor_read(a, 61, host_b, 4) == 0);
    TEST_ASSERT(ds4_cuda_tensor_copy(b, 61, a, 0, 4) == 0);
    TEST_ASSERT(ds4_cuda_tensor_memset(a, 61, 0, 4) == 0);

    ds4_cuda_tensor_free(view);
    ds4_cuda_tensor_free(a);
    ds4_cuda_tensor_free(b);
    ds4_cuda_cleanup();
}

static void test_cuda_row_primitives(void) {
    if (!ds4_cuda_init(false)) {
        fprintf(stderr, "ds4-test: CUDA unavailable, skipping row primitive test\n");
        return;
    }

    const uint32_t n = 16;
    const uint32_t rows = 3;
    const uint32_t n_hc = 4;
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint64_t rows_bytes = (uint64_t)n * rows * sizeof(float);

    ds4_cuda_tensor *a = ds4_cuda_tensor_alloc(row_bytes);
    ds4_cuda_tensor *b = ds4_cuda_tensor_alloc(row_bytes);
    ds4_cuda_tensor *out = ds4_cuda_tensor_alloc(row_bytes);
    ds4_cuda_tensor *repeat_out = ds4_cuda_tensor_alloc(row_bytes * n_hc);
    ds4_cuda_tensor *rms_x = ds4_cuda_tensor_alloc(rows_bytes);
    ds4_cuda_tensor *rms_out = ds4_cuda_tensor_alloc(rows_bytes);
    TEST_ASSERT(a && b && out && repeat_out && rms_x && rms_out);
    if (!a || !b || !out || !repeat_out || !rms_x || !rms_out) {
        ds4_cuda_tensor_free(a);
        ds4_cuda_tensor_free(b);
        ds4_cuda_tensor_free(out);
        ds4_cuda_tensor_free(repeat_out);
        ds4_cuda_tensor_free(rms_x);
        ds4_cuda_tensor_free(rms_out);
        ds4_cuda_cleanup();
        return;
    }

    float host_a[n];
    float host_b[n];
    float host_out[n];
    float host_repeat[n * n_hc];
    float host_rms_x[n * rows];
    float host_rms_out[n * rows];

    for (uint32_t i = 0; i < n; i++) {
        host_a[i] = (float)((int)(i % 7u) - 3) * 0.5f;
        host_b[i] = (float)((int)(i % 5u) - 2) * 0.25f;
    }
    for (uint32_t i = 0; i < n * rows; i++) {
        host_rms_x[i] = (float)((int)(i % 11u) - 5) * 0.125f;
    }

    TEST_ASSERT(ds4_cuda_tensor_write(a, 0, host_a, row_bytes) != 0);
    TEST_ASSERT(ds4_cuda_tensor_write(b, 0, host_b, row_bytes) != 0);
    TEST_ASSERT(ds4_cuda_add_tensor(out, a, b, n) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(out, 0, host_out, row_bytes) != 0);
    for (uint32_t i = 0; i < n; i++) {
        TEST_ASSERT(host_out[i] == host_a[i] + host_b[i]);
    }

    TEST_ASSERT(ds4_cuda_repeat_hc_tensor(repeat_out, a, n, n_hc) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(repeat_out, 0, host_repeat, row_bytes * n_hc) != 0);
    for (uint32_t r = 0; r < n_hc; r++) {
        TEST_ASSERT(memcmp(host_a, host_repeat + (uint64_t)r * n, row_bytes) == 0);
    }

    TEST_ASSERT(ds4_cuda_tensor_write(rms_x, 0, host_rms_x, rows_bytes) != 0);
    TEST_ASSERT(ds4_cuda_rms_norm_plain_rows_tensor(rms_out, rms_x, n, rows, 1.0e-6f) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(rms_out, 0, host_rms_out, rows_bytes) != 0);
    for (uint32_t r = 0; r < rows; r++) {
        double ss = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            double v = host_rms_x[(uint64_t)r * n + i];
            ss += v * v;
        }
        float scale = 1.0f / sqrtf((float)(ss / (double)n) + 1.0e-6f);
        for (uint32_t i = 0; i < n; i++) {
            float want = host_rms_x[(uint64_t)r * n + i] * scale;
            float got = host_rms_out[(uint64_t)r * n + i];
            TEST_ASSERT(fabsf(got - want) < 1.0e-5f);
        }
    }

    ds4_cuda_tensor_free(a);
    ds4_cuda_tensor_free(b);
    ds4_cuda_tensor_free(out);
    ds4_cuda_tensor_free(repeat_out);
    ds4_cuda_tensor_free(rms_x);
    ds4_cuda_tensor_free(rms_out);
    ds4_cuda_cleanup();
}

static void test_cuda_weight_primitives(void) {
    if (!ds4_cuda_init(false)) {
        fprintf(stderr, "ds4-test: CUDA unavailable, skipping weight primitive test\n");
        return;
    }

    const uint32_t n_vocab = 8;
    const uint32_t n_embd = 16;
    const uint32_t n_hc = 4;
    const uint32_t rows = 2;
    const uint32_t head_dim = 8;
    const uint32_t n_head = 2;
    const uint32_t n_tok = 2;
    const uint32_t mm_in_dim = 6;
    const uint32_t mm_out_dim = 5;
    const uint64_t embd_weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    const uint64_t rms_weight_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t embd_out_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
    const uint64_t batch_embd_out_bytes = (uint64_t)n_tok * n_embd * n_hc * sizeof(float);
    const uint64_t rms_in_bytes = (uint64_t)n_embd * rows * sizeof(float);
    const uint64_t head_bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
    const uint64_t mm_weight_bytes = (uint64_t)mm_in_dim * mm_out_dim * sizeof(uint16_t);
    const uint64_t mm_x_bytes = (uint64_t)mm_in_dim * sizeof(float);
    const uint64_t mm_out_bytes = (uint64_t)mm_out_dim * sizeof(float);

    uint16_t *embd_weights = malloc((size_t)embd_weight_bytes);
    float *rms_weights = malloc((size_t)rms_weight_bytes);
    uint16_t *mm_weights = malloc((size_t)mm_weight_bytes);
    TEST_ASSERT(embd_weights != NULL);
    TEST_ASSERT(rms_weights != NULL);
    TEST_ASSERT(mm_weights != NULL);
    if (!embd_weights || !rms_weights || !mm_weights) {
        free(embd_weights);
        free(rms_weights);
        free(mm_weights);
        ds4_cuda_cleanup();
        return;
    }

    for (uint32_t t = 0; t < n_vocab; t++) {
        for (uint32_t i = 0; i < n_embd; i++) {
            float v = (float)((int)((t * 7u + i * 3u) % 19u) - 9) / 16.0f;
            embd_weights[(uint64_t)t * n_embd + i] = test_float_to_f16(v);
        }
    }
    for (uint32_t i = 0; i < n_embd; i++) {
        rms_weights[i] = 0.5f + (float)i * 0.03125f;
    }
    for (uint32_t o = 0; o < mm_out_dim; o++) {
        for (uint32_t i = 0; i < mm_in_dim; i++) {
            float v = (float)((int)((o * 5u + i * 11u) % 17u) - 8) / 32.0f;
            mm_weights[(uint64_t)o * mm_in_dim + i] = test_float_to_f16(v);
        }
    }

    ds4_cuda_tensor *embd_out = ds4_cuda_tensor_alloc(embd_out_bytes);
    ds4_cuda_tensor *batch_embd_out = ds4_cuda_tensor_alloc(batch_embd_out_bytes);
    ds4_cuda_tensor *tokens = ds4_cuda_tensor_alloc((uint64_t)n_tok * sizeof(int32_t));
    ds4_cuda_tensor *rms_in = ds4_cuda_tensor_alloc(rms_in_bytes);
    ds4_cuda_tensor *rms_out = ds4_cuda_tensor_alloc(rms_in_bytes);
    ds4_cuda_tensor *head = ds4_cuda_tensor_alloc(head_bytes);
    ds4_cuda_tensor *mm_x = ds4_cuda_tensor_alloc(mm_x_bytes);
    ds4_cuda_tensor *mm_out = ds4_cuda_tensor_alloc(mm_out_bytes);
    TEST_ASSERT(embd_out && batch_embd_out && tokens && rms_in && rms_out && head && mm_x && mm_out);
    if (!embd_out || !batch_embd_out || !tokens || !rms_in || !rms_out || !head || !mm_x || !mm_out) {
        ds4_cuda_tensor_free(embd_out);
        ds4_cuda_tensor_free(batch_embd_out);
        ds4_cuda_tensor_free(tokens);
        ds4_cuda_tensor_free(rms_in);
        ds4_cuda_tensor_free(rms_out);
        ds4_cuda_tensor_free(head);
        ds4_cuda_tensor_free(mm_x);
        ds4_cuda_tensor_free(mm_out);
        free(embd_weights);
        free(rms_weights);
        free(mm_weights);
        ds4_cuda_cleanup();
        return;
    }

    int32_t token_host[2] = { 3, 6 };
    float rms_host_in[rows * n_embd];
    float head_orig[head_bytes / sizeof(float)];
    float head_host[head_bytes / sizeof(float)];
    float mm_x_host[mm_in_dim];
    float mm_out_host[mm_out_dim];
    for (uint32_t i = 0; i < rows * n_embd; i++) {
        rms_host_in[i] = (float)((int)(i % 13u) - 6) * 0.125f;
    }
    for (uint32_t i = 0; i < head_bytes / sizeof(float); i++) {
        head_orig[i] = (float)((int)(i % 9u) - 4) * 0.2f;
        head_host[i] = head_orig[i];
    }
    for (uint32_t i = 0; i < mm_in_dim; i++) {
        mm_x_host[i] = (float)((int)(i % 5u) - 2) * 0.3f;
    }

    TEST_ASSERT(ds4_cuda_tensor_write(tokens, 0, token_host, sizeof(token_host)) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(tokens, 0, token_host, sizeof(token_host)) != 0);
    TEST_ASSERT(ds4_cuda_tensor_write(rms_in, 0, rms_host_in, rms_in_bytes) != 0);
    TEST_ASSERT(ds4_cuda_tensor_write(head, 0, head_host, head_bytes) != 0);
    TEST_ASSERT(ds4_cuda_tensor_write(mm_x, 0, mm_x_host, mm_x_bytes) != 0);

    TEST_ASSERT(ds4_cuda_embed_token_hc_tensor(embd_out, embd_weights, embd_weight_bytes, 0,
                                               n_vocab, (uint32_t)token_host[0], n_embd, n_hc) != 0);
    float embd_host[embd_out_bytes / sizeof(float)];
    float batch_embd_host[batch_embd_out_bytes / sizeof(float)];
    TEST_ASSERT(ds4_cuda_tensor_read(embd_out, 0, embd_host, embd_out_bytes) != 0);
    for (uint32_t h = 0; h < n_hc; h++) {
        for (uint32_t i = 0; i < n_embd; i++) {
            float want = test_f16_to_float(embd_weights[(uint64_t)token_host[0] * n_embd + i]);
            TEST_ASSERT(fabsf(embd_host[(uint64_t)h * n_embd + i] - want) < 1.0e-5f);
        }
    }

    TEST_ASSERT(ds4_cuda_embed_tokens_hc_tensor(batch_embd_out, tokens, embd_weights, embd_weight_bytes, 0,
                                                n_vocab, n_tok, n_embd, n_hc) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(batch_embd_out, 0, batch_embd_host, batch_embd_out_bytes) != 0);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t h = 0; h < n_hc; h++) {
            for (uint32_t i = 0; i < n_embd; i++) {
                float want = test_f16_to_float(embd_weights[(uint64_t)token_host[t] * n_embd + i]);
                float got = batch_embd_host[((uint64_t)t * n_hc + h) * n_embd + i];
                TEST_ASSERT(fabsf(got - want) < 1.0e-5f);
            }
        }
    }

    TEST_ASSERT(ds4_cuda_rms_norm_weight_rows_tensor(rms_out, rms_in, rms_weights, rms_weight_bytes, 0,
                                                     n_embd, rows, 1.0e-6f) != 0);
    float rms_host_out[rows * n_embd];
    TEST_ASSERT(ds4_cuda_tensor_read(rms_out, 0, rms_host_out, rms_in_bytes) != 0);
    for (uint32_t r = 0; r < rows; r++) {
        double ss = 0.0;
        for (uint32_t i = 0; i < n_embd; i++) {
            double v = rms_host_in[(uint64_t)r * n_embd + i];
            ss += v * v;
        }
        float scale = 1.0f / sqrtf((float)(ss / (double)n_embd) + 1.0e-6f);
        for (uint32_t i = 0; i < n_embd; i++) {
            float want = rms_host_in[(uint64_t)r * n_embd + i] * scale * rms_weights[i];
            TEST_ASSERT(fabsf(rms_host_out[(uint64_t)r * n_embd + i] - want) < 1.0e-5f);
        }
    }

    TEST_ASSERT(ds4_cuda_head_rms_norm_tensor(head, n_tok, n_head, head_dim, 1.0e-6f) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(head, 0, head_host, head_bytes) != 0);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            const float *row = head_host + ((uint64_t)t * n_head + h) * head_dim;
            const float *orig = head_orig + ((uint64_t)t * n_head + h) * head_dim;
            double ss = 0.0;
            for (uint32_t i = 0; i < head_dim; i++) {
                double v = orig[i];
                ss += v * v;
            }
            float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + 1.0e-6f);
            for (uint32_t i = 0; i < head_dim; i++) {
                TEST_ASSERT(fabsf(row[i] - orig[i] * scale) < 1.0e-5f);
            }
        }
    }

    TEST_ASSERT(ds4_cuda_matmul_f16_tensor(mm_out, mm_weights, mm_weight_bytes, 0,
                                           mm_in_dim, mm_out_dim, mm_x, 1) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(mm_out, 0, mm_out_host, mm_out_bytes) != 0);
    for (uint32_t o = 0; o < mm_out_dim; o++) {
        double acc = 0.0;
        for (uint32_t i = 0; i < mm_in_dim; i++) {
            float w = test_f16_to_float(mm_weights[(uint64_t)o * mm_in_dim + i]);
            acc += (double)w * (double)mm_x_host[i];
        }
        TEST_ASSERT(fabsf(mm_out_host[o] - (float)acc) < 1.0e-5f);
    }

    ds4_cuda_tensor_free(embd_out);
    ds4_cuda_tensor_free(batch_embd_out);
    ds4_cuda_tensor_free(tokens);
    ds4_cuda_tensor_free(rms_in);
    ds4_cuda_tensor_free(rms_out);
    ds4_cuda_tensor_free(head);
    ds4_cuda_tensor_free(mm_x);
    ds4_cuda_tensor_free(mm_out);
    free(embd_weights);
    free(rms_weights);
    free(mm_weights);
    ds4_cuda_cleanup();
}

static void test_cuda_router_select(void) {
    if (!ds4_cuda_init(false)) {
        fprintf(stderr, "ds4-test: CUDA unavailable, skipping router select test\n");
        return;
    }

    const uint32_t n_tok = 2;
    const uint32_t logits_bytes = (uint64_t)n_tok * 256u * sizeof(float);
    const uint32_t selected_bytes = (uint64_t)n_tok * 6u * sizeof(int32_t);
    const uint32_t weights_bytes = (uint64_t)n_tok * 6u * sizeof(float);
    const uint32_t probs_bytes = logits_bytes;

    float *logits = malloc(logits_bytes);
    float *bias = malloc(256u * sizeof(float));
    int32_t *hash_table = malloc(3u * 6u * sizeof(int32_t));
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(bias != NULL);
    TEST_ASSERT(hash_table != NULL);
    if (!logits || !bias || !hash_table) {
        free(logits);
        free(bias);
        free(hash_table);
        ds4_cuda_cleanup();
        return;
    }

    for (uint32_t i = 0; i < 256u; i++) {
        bias[i] = (float)((int)(i % 9u) - 4) * 0.03125f;
        for (uint32_t t = 0; t < n_tok; t++) {
            logits[(uint64_t)t * 256u + i] = (float)((int)((t * 11u + i * 7u) % 23u) - 11) * 0.125f;
        }
    }
    for (uint32_t r = 0; r < 3u; r++) {
        for (uint32_t j = 0; j < 6u; j++) {
            hash_table[(uint64_t)r * 6u + j] = (int32_t)((r * 37u + j * 19u) % 256u);
        }
    }

    ds4_cuda_tensor *logits_dev = ds4_cuda_tensor_alloc(logits_bytes);
    ds4_cuda_tensor *probs_dev = ds4_cuda_tensor_alloc(probs_bytes);
    ds4_cuda_tensor *selected_dev = ds4_cuda_tensor_alloc(selected_bytes);
    ds4_cuda_tensor *weights_dev = ds4_cuda_tensor_alloc(weights_bytes);
    ds4_cuda_tensor *tokens_dev = ds4_cuda_tensor_alloc((uint64_t)n_tok * sizeof(int32_t));
    TEST_ASSERT(logits_dev && probs_dev && selected_dev && weights_dev && tokens_dev);
    if (!logits_dev || !probs_dev || !selected_dev || !weights_dev || !tokens_dev) {
        ds4_cuda_tensor_free(logits_dev);
        ds4_cuda_tensor_free(probs_dev);
        ds4_cuda_tensor_free(selected_dev);
        ds4_cuda_tensor_free(weights_dev);
        ds4_cuda_tensor_free(tokens_dev);
        free(logits);
        free(bias);
        free(hash_table);
        ds4_cuda_cleanup();
        return;
    }

    int32_t token_ids[2] = { 2, 1 };
    TEST_ASSERT(ds4_cuda_tensor_write(logits_dev, 0, logits, logits_bytes) != 0);
    TEST_ASSERT(ds4_cuda_tensor_write(tokens_dev, 0, token_ids, sizeof(token_ids)) != 0);

    TEST_ASSERT(ds4_cuda_router_select_batch_tensor(selected_dev, weights_dev, probs_dev,
                                                    bias, 256u * sizeof(float),
                                                    0, 0, 0, 1, 0, true, false,
                                                    logits_dev, tokens_dev, n_tok) != 0);

    float probs_host[2 * 256];
    int32_t selected_host[2 * 6];
    float weights_host[2 * 6];
    TEST_ASSERT(ds4_cuda_tensor_read(probs_dev, 0, probs_host, probs_bytes) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(selected_dev, 0, selected_host, selected_bytes) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(weights_dev, 0, weights_host, weights_bytes) != 0);

    for (uint32_t t = 0; t < n_tok; t++) {
        int want_selected[6];
        float want_weights[6];
        test_router_expected(logits + (uint64_t)t * 256u, bias, NULL, false,
                             want_selected, want_weights);
        for (int i = 0; i < 6; i++) {
            TEST_ASSERT(selected_host[(uint64_t)t * 6u + i] == want_selected[i]);
            TEST_ASSERT(fabsf(weights_host[(uint64_t)t * 6u + i] - want_weights[i]) < 1.0e-5f);
        }
        for (uint32_t i = 0; i < 256u; i++) {
            float want = sqrtf(test_softplus_stable(logits[(uint64_t)t * 256u + i]));
            TEST_ASSERT(fabsf(probs_host[(uint64_t)t * 256u + i] - want) < 1.0e-5f);
        }
    }

    TEST_ASSERT(ds4_cuda_router_select_tensor(selected_dev, weights_dev, probs_dev,
                                              hash_table, 3u * 6u * sizeof(int32_t),
                                              0, 0, 3u, 2, 1, 0, false, true,
                                              logits_dev) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(selected_dev, 0, selected_host, selected_bytes) != 0);
    TEST_ASSERT(ds4_cuda_tensor_read(weights_dev, 0, weights_host, weights_bytes) != 0);

    int want_selected[6];
    float want_weights[6];
    test_router_expected(logits, NULL, hash_table + 2u * 6u, true, want_selected, want_weights);
    for (int i = 0; i < 6; i++) {
        TEST_ASSERT(selected_host[i] == want_selected[i]);
        TEST_ASSERT(fabsf(weights_host[i] - want_weights[i]) < 1.0e-5f);
    }

    ds4_cuda_tensor_free(logits_dev);
    ds4_cuda_tensor_free(probs_dev);
    ds4_cuda_tensor_free(selected_dev);
    ds4_cuda_tensor_free(weights_dev);
    ds4_cuda_tensor_free(tokens_dev);
    free(logits);
    free(bias);
    free(hash_table);
    ds4_cuda_cleanup();
}
#endif

static void test_server_unit_group(void) {
    ds4_server_unit_tests_run();
}

typedef void (*test_fn)(void);

typedef struct {
    const char *flag;
    const char *name;
    const char *desc;
    test_fn fn;
} ds4_test_entry;

static const ds4_test_entry test_entries[] = {
#ifndef DS4_NO_CUDA
    {"--cuda-tensors", "cuda-tensors", "CUDA tensor allocation and copy boundaries", test_cuda_tensor_layer},
    {"--cuda-primitives", "cuda-primitives", "CUDA row primitive parity checks", test_cuda_row_primitives},
    {"--cuda-weights", "cuda-weights", "CUDA weight-backed primitive parity checks", test_cuda_weight_primitives},
    {"--cuda-router", "cuda-router", "CUDA router selection parity checks", test_cuda_router_select},
#endif
#ifndef DS4_NO_METAL
    {"--long-context", "long-context", "long Metal continuation regression", test_long_security_continuation},
    {"--tool-call-quality", "tool-call-quality", "model emits valid DSML tool calls", test_tool_call_quality},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison", test_official_logprob_vectors},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_f16_matvec_fast_nr0_4},
#endif
    {"--server", "server", "server parser/rendering/cache unit tests", test_server_unit_group},
};

static void test_print_help(const char *prog) {
    printf("Usage: %s [--all | TEST...]\n\n", prog);
    puts("Tests:");
    puts("  --all");
    puts("      Run every test. This is the default, ordered from slower to faster.");
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        printf("  %-20s %s\n", test_entries[i].flag, test_entries[i].desc);
    }
    puts("  --list");
    puts("      Print test names only.");
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context regression prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Simple official-vector fixture.");
}

static const ds4_test_entry *test_find_entry(const char *arg) {
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

static void test_run_entry(const ds4_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    fprintf(stderr, "%s: ", entry->name);
    ds4_log(stderr,
            test_failures == before ? DS4_LOG_OK : DS4_LOG_ERROR,
            "%s",
            test_failures == before ? "OK" : "ERR");
    fputc('\n', stderr);
}

int main(int argc, char **argv) {
    bool run_all = argc == 1;
    bool selected[sizeof(test_entries) / sizeof(test_entries[0])] = {0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) {
            run_all = true;
        } else if (!strcmp(argv[i], "--list")) {
            for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
                puts(test_entries[j].flag);
            }
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            test_print_help(argv[0]);
            return 0;
        } else {
            const ds4_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "ds4-test: unknown test switch: %s\n", argv[i]);
                test_print_help(argv[0]);
                return 2;
            }
            selected[(size_t)(entry - test_entries)] = true;
        }
    }

    if (run_all) {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            test_run_entry(&test_entries[i]);
        }
    } else {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (selected[i]) test_run_entry(&test_entries[i]);
        }
    }

#ifndef DS4_NO_METAL
    test_close_engines();
#endif

    if (test_failures) {
        fprintf(stderr, "ds4 tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4 tests: ok");
    return 0;
}
