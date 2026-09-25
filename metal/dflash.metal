/* DFlash block drafter for MiMo V2.6 Flash (--dflash sidecar).
 *
 * The drafter is a small non-causal Qwen3-style decoder.  Its context is the
 * target's hidden rows after a fixed set of layers, concatenated per token,
 * projected by fc + hidden_norm and turned into per-layer K/V rows that live
 * in an f16 ring indexed by absolute position (row p % cap).  A draft block
 * of B rows (the verified anchor token followed by B-1 mask rows) attends
 * over the ring keys inside its sliding window plus all B block keys.
 * Dense projections, norms and SwiGLU use the shared kernels; this file holds
 * the data movement, the per-head q/k norm with partial RoPE, the attention
 * and the per-row argmax with its softmax probability. */

#define DFLASH_ATTN_MAX_KEYS 2048u

/* Copy rows of one target layer's residual into its slot of the
 * concatenated feature rows [row][n_slots][n_embd]. */
struct ds4_metal_args_dflash_capture {
    uint32_t n_rows;
    uint32_t n_embd;
    uint32_t n_slots;
    uint32_t slot;
    uint32_t src_row0;
    uint32_t dst_row0;
};

kernel void kernel_dflash_capture_rows(
        constant ds4_metal_args_dflash_capture & args,
        device const float *src,       /* [rows][n_embd] */
        device float       *features,  /* [rows][n_slots][n_embd] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint row = tgpig.y;
    const uint col = tgpig.x * ntg.x + tpitg.x;
    if (row >= args.n_rows || col >= args.n_embd) return;
    features[((uint64_t)(args.dst_row0 + row) * args.n_slots + args.slot) * args.n_embd + col] =
        src[(uint64_t)(args.src_row0 + row) * args.n_embd + col];
}

/* Qwen3 per-head RMSNorm (weight over head_dim) followed by rotate-half RoPE
 * on the first rot_dim dims at position pos0 + row, in place.  One
 * threadgroup per (head, row), one thread per dim (rounded up to 32). */
struct ds4_metal_args_dflash_head_norm_rope {
    uint32_t n_rows;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t pos0;
    uint32_t pad0;
    float    eps;
    float    pad1;
    float    freq[64];      /* rot_dim / 2 entries */
};

kernel void kernel_dflash_head_norm_rope(
        constant ds4_metal_args_dflash_head_norm_rope & args,
        device float       *x,        /* [rows][n_heads][head_dim] */
        device const float *weight,   /* [head_dim] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint3 ntg3 [[threads_per_threadgroup]]) {
    const uint ntg = ntg3.x;
    threadgroup float normed[256];
    threadgroup float partial[8];
    const uint head = tgpig.x, row = tgpig.y, D = args.head_dim;
    if (head >= args.n_heads || row >= args.n_rows) return;
    device float *p = x + ((uint64_t)row * args.n_heads + head) * D;
    const float v = tid < D ? p[tid] : 0.0f;
    const float ss = simd_sum(v * v);
    if (tiisg == 0) partial[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < (uint)(ntg + 31u) / 32u; i++) total += partial[i];
    const float y = tid < D ? v * rsqrt(total / (float)D + args.eps) * weight[tid] : 0.0f;
    normed[tid] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= D) return;
    float out = y;
    const uint half_rot = args.rot_dim / 2u;
    if (tid < args.rot_dim) {
        const uint i = tid < half_rot ? tid : tid - half_rot;
        const float theta = (float)(args.pos0 + row) * args.freq[i];
        const float c = precise::cos(theta), s = precise::sin(theta);
        out = tid < half_rot ? normed[tid] * c - normed[tid + half_rot] * s
                             : normed[tid] * c + normed[tid - half_rot] * s;
    }
    p[tid] = out;
}

/* Store f32 k/v rows as f16 at (pos0 + row) % cache_cap; v is multiplied by
 * the value scale first. */
struct ds4_metal_args_dflash_store_kv {
    uint32_t n_rows;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t width;
    float    value_scale;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

kernel void kernel_dflash_store_kv(
        constant ds4_metal_args_dflash_store_kv & args,
        device const float *k,            /* [rows][width] */
        device const float *v,            /* [rows][width] */
        device half        *key_cache,    /* [cap][width] */
        device half        *value_cache,  /* [cap][width] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint row = tgpig.y;
    const uint col = tgpig.x * ntg.x + tpitg.x;
    if (row >= args.n_rows || col >= args.width) return;
    const uint slot = (args.pos0 + row) % args.cache_cap;
    const uint64_t src = (uint64_t)row * args.width + col;
    const uint64_t dst = (uint64_t)slot * args.width + col;
    key_cache[dst] = (half)k[src];
    value_cache[dst] = (half)(v[src] * args.value_scale);
}

/* Non-causal block attention.  Query row t sits at position pos0 + t.  Its
 * keys are the ring rows at positions [max(ctx_lo, pos0 + t - window + 1),
 * ctx_hi) and every block row (positions pos0 .. pos0 + n_rows - 1, all
 * inside the window).  GQA: query head h reads kv head h / (H / Hkv).  The
 * optional sink is one more softmax logit per head with no value.  One
 * threadgroup per (head, row); scores live in threadgroup memory. */
struct ds4_metal_args_dflash_attn {
    uint32_t n_rows;
    uint32_t n_head;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t ctx_lo;
    uint32_t ctx_hi;
    uint32_t cache_cap;
    uint32_t pos0;
    uint32_t window;
    uint32_t has_sinks;
    float    scale;
    uint32_t pad0;
};

kernel void kernel_dflash_attention(
        constant ds4_metal_args_dflash_attn & args,
        device const float *q,            /* [rows][H][D] */
        device const half  *key_ring,     /* [cap][Hkv][D] */
        device const half  *value_ring,   /* [cap][Hkv][D] */
        device const half  *block_k,      /* [rows][Hkv][D] */
        device const half  *block_v,      /* [rows][Hkv][D] */
        device const float *sinks,        /* [H] when has_sinks */
        device float       *heads,        /* [rows][H][D] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint3 ntg3 [[threads_per_threadgroup]]) {
    const uint ntg = ntg3.x;
    threadgroup float qs[256];
    threadgroup float sc[DFLASH_ATTN_MAX_KEYS];
    threadgroup float red_max[32];
    threadgroup float red_sum[32];
    const uint h = tgpig.x, t = tgpig.y;
    const uint H = args.n_head, Hkv = args.n_kv, D = args.head_dim;
    if (h >= H || t >= args.n_rows) return;
    const uint kvh = h / (H / Hkv);
    const uint p = args.pos0 + t;
    uint lo = args.ctx_lo;
    if (p + 1u >= args.window && p + 1u - args.window > lo) lo = p + 1u - args.window;
    const uint n_ctx = args.ctx_hi > lo ? args.ctx_hi - lo : 0u;
    const uint n_keys = n_ctx + args.n_rows;
    const uint nsg = ((uint)ntg + 31u) / 32u;

    device const float *qrow = q + ((uint64_t)t * H + h) * D;
    for (uint d = tid; d < D; d += ntg) qs[d] = qrow[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lmax = -3.0e38f;
    for (uint j = tid; j < n_keys; j += ntg) {
        device const half *kr = j < n_ctx ?
            key_ring + ((uint64_t)((lo + j) % args.cache_cap) * Hkv + kvh) * D :
            block_k + ((uint64_t)(j - n_ctx) * Hkv + kvh) * D;
        float s = 0.0f;
        for (uint d = 0; d < D; d += 4u) {
            const float4 kk = float4(*(device const half4 *)(kr + d));
            s += dot(kk, float4(qs[d], qs[d + 1u], qs[d + 2u], qs[d + 3u]));
        }
        s *= args.scale;
        sc[j] = s;
        lmax = max(lmax, s);
    }
    lmax = simd_max(lmax);
    if (tiisg == 0) red_max[sgitg] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -3.0e38f;
    for (uint i = 0; i < nsg; i++) m = max(m, red_max[i]);
    const float sink = args.has_sinks ? sinks[h] : -3.0e38f;
    m = max(m, sink);

    float lsum = 0.0f;
    for (uint j = tid; j < n_keys; j += ntg) {
        const float e = exp(sc[j] - m);
        sc[j] = e;
        lsum += e;
    }
    lsum = simd_sum(lsum);
    if (tiisg == 0) red_sum[sgitg] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = args.has_sinks ? exp(sink - m) : 0.0f;
    for (uint i = 0; i < nsg; i++) total += red_sum[i];
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;

    device float *out = heads + ((uint64_t)t * H + h) * D;
    for (uint d = tid; d < D; d += ntg) {
        float acc = 0.0f;
        for (uint j = 0; j < n_keys; j++) {
            device const half *vr = j < n_ctx ?
                value_ring + ((uint64_t)((lo + j) % args.cache_cap) * Hkv + kvh) * D :
                block_v + ((uint64_t)(j - n_ctx) * Hkv + kvh) * D;
            acc += sc[j] * (float)vr[d];
        }
        out[d] = acc * inv;
    }
}

/* Per row: the first index of the largest logit and its softmax
 * probability.  One threadgroup per row. */
struct ds4_metal_args_dflash_argmax {
    uint32_t n_rows;
    uint32_t n_vocab;
};

kernel void kernel_dflash_argmax_prob(
        constant ds4_metal_args_dflash_argmax & args,
        device const float *logits,   /* [rows][n_vocab] */
        device int32_t     *index,    /* [rows] */
        device float       *prob,     /* [rows] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint3 ntg3 [[threads_per_threadgroup]]) {
    const uint ntg = ntg3.x;
    threadgroup float best_v[32];
    threadgroup uint best_i[32];
    threadgroup float part_sum[32];
    const uint row = tgpig.x, V = args.n_vocab;
    if (row >= args.n_rows) return;
    const uint nsg = ((uint)ntg + 31u) / 32u;
    device const float *l = logits + (uint64_t)row * V;
    float bv = -3.0e38f;
    uint bi = 0xffffffffu;
    for (uint i = tid; i < V; i += ntg) {
        const float x = l[i];
        if (x > bv) { bv = x; bi = i; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(bv, off);
        const uint oi = simd_shuffle_down(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (tiisg == 0) { best_v[sgitg] = bv; best_i[sgitg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = best_v[0];
    uint mi = best_i[0];
    for (uint i = 1; i < nsg; i++) {
        if (best_v[i] > m || (best_v[i] == m && best_i[i] < mi)) { m = best_v[i]; mi = best_i[i]; }
    }
    float s = 0.0f;
    if (m > -3.0e38f) {
        for (uint i = tid; i < V; i += ntg) s += exp(l[i] - m);
    }
    s = simd_sum(s);
    if (tiisg == 0) part_sum[sgitg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < nsg; i++) total += part_sum[i];
        index[row] = mi < V ? (int32_t)mi : -1;
        prob[row] = total > 0.0f ? 1.0f / total : 0.0f;
    }
}
