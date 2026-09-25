/* GPU kernel tests for the MiMo DFlash block drafter (metal/dflash.metal):
 * target feature capture, per-head q/k RMSNorm with partial rotate-half
 * RoPE, f16 ring stores with the value scale, non-causal block attention
 * over a windowed ring with sinks, and the per-row argmax with its softmax
 * probability.  Everything is checked against a double-precision reference.
 * Build and run: make test-dflash-kernels */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint32_t g_rng = 0x9e3779b9u;

static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xffffffu) / 8388608.0f) - 1.0f;
}

static void require(bool ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
}

static double f16r(float x) {
    return (double)(float)(__fp16)x;
}

static void check_close(const char *what, const float *got, const double *ref, uint64_t n, double tol) {
    double worst = 0.0, scale = 1e-6;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "FAIL %s: non-finite value at %llu\n", what, (unsigned long long)i);
            exit(1);
        }
        const double d = fabs((double)got[i] - ref[i]);
        if (d > worst) { worst = d; worst_i = i; }
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    if (worst > tol * scale) {
        fprintf(stderr, "FAIL %s: max|d| %.3e (rel %.3e) at %llu: got %.7f ref %.7f\n",
                what, worst, worst / scale, (unsigned long long)worst_i, got[worst_i], ref[worst_i]);
        exit(1);
    }
    printf("  %-60s ok  max|d|=%.2e (scale %.2e)\n", what, worst, scale);
}

static ds4_gpu_tensor *upload(const void *data, uint64_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    require(t != NULL, "GPU allocation");
    if (data) require(ds4_gpu_tensor_write(t, 0, data, bytes), "GPU upload");
    return t;
}

static void test_capture(void) {
    enum { SRC_ROWS = 6, DST_ROWS = 4, E = 96, SLOTS = 3 };
    float src[SRC_ROWS * E], feat[DST_ROWS * SLOTS * E];
    for (int i = 0; i < SRC_ROWS * E; i++) src[i] = frand();
    for (int i = 0; i < DST_ROWS * SLOTS * E; i++) feat[i] = 777.0f;
    ds4_gpu_tensor *gs = upload(src, sizeof(src)), *gf = upload(feat, sizeof(feat));
    require(ds4_gpu_dflash_capture_rows_tensor(gf, gs, 2, 1, 3, E, SLOTS, 2), "capture dispatch");
    require(ds4_gpu_tensor_read(gf, 0, feat, sizeof(feat)), "features read");
    for (int r = 0; r < DST_ROWS; r++) {
        for (int s = 0; s < SLOTS; s++) {
            for (int c = 0; c < E; c++) {
                const float got = feat[(r * SLOTS + s) * E + c];
                const bool written = s == 2 && r >= 1 && r < 4;
                const float want = written ? src[(r - 1 + 2) * E + c] : 777.0f;
                require(got == want, "capture copies exactly the requested rows into the slot");
            }
        }
    }
    printf("  %-60s ok\n", "feature capture into slot 2, rows 1..3");
    ds4_gpu_tensor_free(gs);
    ds4_gpu_tensor_free(gf);
}

static void rope_freqs(float *freq, uint32_t rot, double base) {
    for (uint32_t i = 0; i < rot / 2u; i++) freq[i] = (float)pow(base, -2.0 * i / rot);
}

static void ref_head_norm_rope(const float *x, const float *w, double *out, uint32_t rows,
                               uint32_t heads, uint32_t D, uint32_t rot, uint32_t pos0,
                               double eps, const float *freq) {
    double *y = malloc(D * sizeof(double));
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t h = 0; h < heads; h++) {
            const float *src = x + ((uint64_t)r * heads + h) * D;
            double ss = 0.0;
            for (uint32_t d = 0; d < D; d++) ss += (double)src[d] * src[d];
            const double inv = 1.0 / sqrt(ss / D + eps);
            for (uint32_t d = 0; d < D; d++) y[d] = src[d] * inv * w[d];
            double *o = out + ((uint64_t)r * heads + h) * D;
            for (uint32_t d = 0; d < D; d++) o[d] = y[d];
            const uint32_t half = rot / 2u;
            for (uint32_t i = 0; i < half; i++) {
                const double theta = (double)((float)(pos0 + r) * freq[i]);
                o[i] = y[i] * cos(theta) - y[i + half] * sin(theta);
                o[i + half] = y[i + half] * cos(theta) + y[i] * sin(theta);
            }
        }
    }
    free(y);
}

static void test_head_norm_rope(uint32_t heads, uint32_t pos0, const char *what) {
    const uint32_t rows = 8, D = 128, rot = 64;
    const uint64_t n = (uint64_t)rows * heads * D;
    float *x = malloc(n * sizeof(float)), *w = malloc(D * sizeof(float)), freq[64];
    double *ref = malloc(n * sizeof(double));
    for (uint64_t i = 0; i < n; i++) x[i] = frand() * 3.0f;
    for (uint32_t d = 0; d < D; d++) w[d] = 0.5f + frand();
    rope_freqs(freq, rot, 1e4);
    ref_head_norm_rope(x, w, ref, rows, heads, D, rot, pos0, 1e-6, freq);
    ds4_gpu_tensor *gx = upload(x, n * sizeof(float)), *gw = upload(w, D * sizeof(float));
    require(ds4_gpu_dflash_head_norm_rope_tensor(gx, gw, rows, heads, D, rot, pos0, 1e-6f, freq),
            "head norm/rope dispatch");
    require(ds4_gpu_tensor_read(gx, 0, x, n * sizeof(float)), "head norm/rope read");
    check_close(what, x, ref, n, 2e-5);
    ds4_gpu_tensor_free(gx);
    ds4_gpu_tensor_free(gw);
    free(x); free(w); free(ref);
}

static void test_store_kv(void) {
    enum { ROWS = 4, W = 40, CAP = 5 };
    float k[ROWS * W], v[ROWS * W];
    for (int i = 0; i < ROWS * W; i++) { k[i] = frand() * 4.0f; v[i] = frand() * 4.0f; }
    uint16_t kc[CAP * W], vc[CAP * W];
    for (int i = 0; i < CAP * W; i++) { kc[i] = 0x7bffu; vc[i] = 0x7bffu; }
    ds4_gpu_tensor *gk = upload(k, sizeof(k)), *gv = upload(v, sizeof(v));
    ds4_gpu_tensor *gkc = upload(kc, sizeof(kc)), *gvc = upload(vc, sizeof(vc));
    const float scale = 0.612f;
    require(ds4_gpu_dflash_store_kv_tensor(gkc, gvc, gk, gv, ROWS, 3, CAP, W, scale), "store dispatch");
    require(ds4_gpu_tensor_read(gkc, 0, kc, sizeof(kc)) && ds4_gpu_tensor_read(gvc, 0, vc, sizeof(vc)),
            "cache read");
    for (int slot = 0; slot < CAP; slot++) {
        int row = -1;
        for (int r = 0; r < ROWS; r++) if ((3 + r) % CAP == slot) row = r;
        for (int c = 0; c < W; c++) {
            uint16_t wk = 0x7bffu, wv = 0x7bffu;
            if (row >= 0) {
                __fp16 hk = (__fp16)k[row * W + c], hv = (__fp16)(v[row * W + c] * scale);
                memcpy(&wk, &hk, 2);
                memcpy(&wv, &hv, 2);
            }
            require(kc[slot * W + c] == wk && vc[slot * W + c] == wv,
                    "store writes f16 rows at (pos0 + r) % cap with v scaled, nothing else");
        }
    }
    printf("  %-60s ok\n", "f16 ring store with wrap and value scale");
    ds4_gpu_tensor *t[] = { gk, gv, gkc, gvc };
    for (size_t i = 0; i < 4; i++) ds4_gpu_tensor_free(t[i]);
}

typedef struct {
    const char *name;
    uint32_t H, KV, D, cap, window, rows;
    uint32_t ctx_lo, ctx_hi;   /* ring context positions */
    bool sinks;
} attn_case;

/* Reference: K/V as the f16 caches hold them. */
static void ref_attention(const attn_case *c, const float *q, const float *ring_k, const float *ring_v,
                          const float *blk_k, const float *blk_v, const float *sinks, double *out) {
    const uint32_t H = c->H, KV = c->KV, D = c->D, pos0 = c->ctx_hi;
    const uint32_t max_keys = c->window + c->rows;
    double *s = malloc(max_keys * sizeof(double));
    for (uint32_t t = 0; t < c->rows; t++) {
        const uint32_t p = pos0 + t;
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t kvh = h / (H / KV);
            const float *qr = q + ((uint64_t)t * H + h) * D;
            uint32_t n = 0;
            const float *krows[4096], *vrows[4096];
            for (uint32_t pos = c->ctx_lo; pos < c->ctx_hi; pos++) {
                if (p - pos >= c->window) continue;
                krows[n] = ring_k + ((uint64_t)(pos % c->cap) * KV + kvh) * D;
                vrows[n] = ring_v + ((uint64_t)(pos % c->cap) * KV + kvh) * D;
                n++;
            }
            for (uint32_t j = 0; j < c->rows; j++) {
                krows[n] = blk_k + ((uint64_t)j * KV + kvh) * D;
                vrows[n] = blk_v + ((uint64_t)j * KV + kvh) * D;
                n++;
            }
            double m = c->sinks ? sinks[h] : -1e300;
            for (uint32_t j = 0; j < n; j++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < D; d++) dot += (double)qr[d] * f16r(krows[j][d]);
                s[j] = dot / sqrt((double)D);
                if (s[j] > m) m = s[j];
            }
            double total = c->sinks ? exp(sinks[h] - m) : 0.0;
            for (uint32_t j = 0; j < n; j++) { s[j] = exp(s[j] - m); total += s[j]; }
            double *o = out + ((uint64_t)t * H + h) * D;
            for (uint32_t d = 0; d < D; d++) {
                double acc = 0.0;
                for (uint32_t j = 0; j < n; j++) acc += s[j] * f16r(vrows[j][d]);
                o[d] = acc / total;
            }
        }
    }
    free(s);
}

static void test_attention(const attn_case *c) {
    const uint32_t H = c->H, KV = c->KV, D = c->D;
    const uint64_t qn = (uint64_t)c->rows * H * D, rn = (uint64_t)c->cap * KV * D, bn = (uint64_t)c->rows * KV * D;
    float *q = malloc(qn * 4), *rk = malloc(rn * 4), *rv = malloc(rn * 4), *bk = malloc(bn * 4), *bv = malloc(bn * 4);
    float *sinks = malloc(H * 4), *out = malloc(qn * 4);
    double *ref = malloc(qn * 8);
    for (uint64_t i = 0; i < qn; i++) q[i] = frand() * 2.0f;
    for (uint64_t i = 0; i < rn; i++) { rk[i] = frand() * 2.0f; rv[i] = frand(); }
    for (uint64_t i = 0; i < bn; i++) { bk[i] = frand() * 2.0f; bv[i] = frand(); }
    for (uint32_t h = 0; h < H; h++) sinks[h] = frand() * 3.0f;
    /* Ring and block rows as f16, the way the caches store them. */
    uint16_t *rk16 = malloc(rn * 2), *rv16 = malloc(rn * 2), *bk16 = malloc(bn * 2), *bv16 = malloc(bn * 2);
    for (uint64_t i = 0; i < rn; i++) { __fp16 a = (__fp16)rk[i], b = (__fp16)rv[i]; memcpy(rk16 + i, &a, 2); memcpy(rv16 + i, &b, 2); }
    for (uint64_t i = 0; i < bn; i++) { __fp16 a = (__fp16)bk[i], b = (__fp16)bv[i]; memcpy(bk16 + i, &a, 2); memcpy(bv16 + i, &b, 2); }
    ds4_gpu_tensor *gq = upload(q, qn * 4), *grk = upload(rk16, rn * 2), *grv = upload(rv16, rn * 2);
    ds4_gpu_tensor *gbk = upload(bk16, bn * 2), *gbv = upload(bv16, bn * 2), *gs = upload(sinks, H * 4);
    ds4_gpu_tensor *go = upload(NULL, qn * 4);
    require(ds4_gpu_dflash_attention_tensor(go, gq, grk, grv, c->cap, c->ctx_lo, c->ctx_hi, gbk, gbv,
                                            c->sinks ? gs : NULL, c->rows, c->ctx_hi, H, KV, D,
                                            c->window, 1.0f / sqrtf((float)D)),
            "attention dispatch");
    require(ds4_gpu_tensor_read(go, 0, out, qn * 4), "attention read");
    ref_attention(c, q, rk, rv, bk, bv, sinks, ref);
    check_close(c->name, out, ref, qn, 2e-5);
    ds4_gpu_tensor *t[] = { gq, grk, grv, gbk, gbv, gs, go };
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) ds4_gpu_tensor_free(t[i]);
    free(q); free(rk); free(rv); free(bk); free(bv); free(sinks); free(out); free(ref);
    free(rk16); free(rv16); free(bk16); free(bv16);
}

/* A causal mask would hide later block rows from row 0; perturbing the last
 * block key/value must change row 0's output. */
static void test_block_is_noncausal(void) {
    const uint32_t H = 4, KV = 2, D = 32, rows = 8, cap = 16;
    const uint64_t qn = (uint64_t)rows * H * D, rn = (uint64_t)cap * KV * D, bn = (uint64_t)rows * KV * D;
    float *q = malloc(qn * 4), *out0 = malloc(qn * 4), *out1 = malloc(qn * 4);
    uint16_t *ring = calloc(rn, 2), *bk = malloc(bn * 2), *bv = malloc(bn * 2);
    for (uint64_t i = 0; i < qn; i++) q[i] = frand();
    for (uint64_t i = 0; i < bn; i++) { __fp16 a = (__fp16)frand(), b = (__fp16)frand(); memcpy(bk + i, &a, 2); memcpy(bv + i, &b, 2); }
    ds4_gpu_tensor *gq = upload(q, qn * 4), *gr = upload(ring, rn * 2), *go = upload(NULL, qn * 4);
    ds4_gpu_tensor *gbk = upload(bk, bn * 2), *gbv = upload(bv, bn * 2);
    require(ds4_gpu_dflash_attention_tensor(go, gq, gr, gr, cap, 0, 0, gbk, gbv, NULL, rows, 0, H, KV, D, 16,
                                            0.2f), "attention dispatch");
    require(ds4_gpu_tensor_read(go, 0, out0, qn * 4), "attention read");
    for (uint64_t i = (uint64_t)(rows - 1) * KV * D; i < bn; i++) { __fp16 b = (__fp16)5.0f; memcpy(bv + i, &b, 2); }
    require(ds4_gpu_tensor_write(gbv, 0, bv, bn * 2), "block v upload");
    require(ds4_gpu_dflash_attention_tensor(go, gq, gr, gr, cap, 0, 0, gbk, gbv, NULL, rows, 0, H, KV, D, 16,
                                            0.2f), "attention dispatch");
    require(ds4_gpu_tensor_read(go, 0, out1, qn * 4), "attention read");
    double moved = 0.0;
    for (uint32_t i = 0; i < H * D; i++) moved += fabs((double)out1[i] - out0[i]);
    require(moved > 1e-3, "row 0 attends to the last block row (non-causal block)");
    printf("  %-60s ok\n", "block attention is non-causal (row 0 sees row 7)");
    ds4_gpu_tensor *t[] = { gq, gr, go, gbk, gbv };
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) ds4_gpu_tensor_free(t[i]);
    free(q); free(out0); free(out1); free(ring); free(bk); free(bv);
}

static void test_argmax_prob(void) {
    const uint32_t rows = 5, V = 152576;
    float *l = malloc((uint64_t)rows * V * 4);
    for (uint64_t i = 0; i < (uint64_t)rows * V; i++) l[i] = frand() * 4.0f;
    const uint32_t want[5] = { 151675, 0, 77777, 3, 152575 };
    for (uint32_t r = 0; r < rows; r++) l[(uint64_t)r * V + want[r]] = 9.5f + (float)r;
    l[(uint64_t)3 * V + 90000] = 9.5f + 3.0f;   /* tie: the lower index wins */
    ds4_gpu_tensor *gl = upload(l, (uint64_t)rows * V * 4), *gi = upload(NULL, rows * 4), *gp = upload(NULL, rows * 4);
    require(ds4_gpu_dflash_argmax_prob_tensor(gi, gp, gl, rows, V), "argmax dispatch");
    int32_t idx[5];
    float prob[5];
    require(ds4_gpu_tensor_read(gi, 0, idx, sizeof(idx)) && ds4_gpu_tensor_read(gp, 0, prob, sizeof(prob)),
            "argmax read");
    double ref[5];
    for (uint32_t r = 0; r < rows; r++) {
        require(idx[r] == (int32_t)want[r], "argmax returns the lowest index of the row maximum");
        const float *row = l + (uint64_t)r * V;
        const double m = row[want[r]];
        double s = 0.0;
        for (uint32_t i = 0; i < V; i++) s += exp((double)row[i] - m);
        ref[r] = 1.0 / s;
    }
    check_close("argmax softmax probability", prob, ref, rows, 1e-4);
    ds4_gpu_tensor_free(gl);
    ds4_gpu_tensor_free(gi);
    ds4_gpu_tensor_free(gp);
    free(l);
}

int main(void) {
    require(ds4_gpu_init(), "GPU initialization");
    printf("capture\n");
    test_capture();
    printf("q/k head norm + partial rope\n");
    test_head_norm_rope(64, 0, "q 64 heads @0");
    test_head_norm_rope(8, 1048000, "k 8 heads @1048000");
    printf("ring store\n");
    test_store_kv();
    printf("attention\n");
    const attn_case cases[] = {
        { "sinks, ring wrapped, window cuts rows differently", 64, 8, 128, 16, 12, 8, 30, 46, true },
        { "no sinks, ctx_lo above the window", 64, 8, 128, 16, 16, 8, 41, 46, false },
        { "empty context (block only)", 64, 8, 128, 16, 16, 8, 46, 46, true },
        { "release window 1024, ring 1024, ctx 1024", 64, 8, 128, 1024, 1024, 8, 2000, 3024, true },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) test_attention(&cases[i]);
    test_block_is_noncausal();
    printf("argmax\n");
    test_argmax_prob();
    fprintf(stderr, "(expected refusal message follows)\n");
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(1u << 20);
    require(!ds4_gpu_dflash_attention_tensor(t, t, t, t, 1024, 0, 0, t, t, NULL, 8, 0, 64, 8, 128, 2048, 0.1f),
            "a window longer than the score buffer is refused");
    ds4_gpu_tensor_free(t);
    ds4_gpu_cleanup();
    printf("dflash kernels: all checks passed\n");
    return 0;
}
