/* MiMo V2.6 Flash attention: fused-QKV split with partial rotate-half RoPE
 * and value scaling, f16 ring caches, and GQA attention with K/V head dims
 * that differ (192/128 at release, 64/32 in the mini shape), an optional
 * sliding window and an optional per-head sink logit.
 *
 * Caches are [cap][n_kv][dim] f16; absolute position p lives in row p % cap.
 * Sliding-window layers keep a ring of about one window, so a prefill chunk
 * is staged (f32 k/v of the chunk), attended over the ring plus the staged
 * rows, and only then committed.  Staged rows are rounded to f16 when read
 * so they score exactly like the ring rows decode reads.  Transients are f32. */

struct ds4_metal_args_mimo2_qkv {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_kv;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t rot_dim;
    float    value_scale;
    uint32_t store_from;    /* tokens >= store_from write their cache rows */
    uint32_t pad0;
    uint32_t pad1;
    float    freq[64];      /* rot_dim / 2 entries */
};

/* One threadgroup per (q/k/v head, token), one thread per dim.  Slots
 * [0, n_head) are query heads, [n_head, n_head + n_kv) key heads and the
 * rest value heads.  RoPE follows HF rotate_half on the first rot_dim dims:
 * out[i] = x[i] cos - x[i+r/2] sin, out[i+r/2] = x[i+r/2] cos + x[i] sin. */
kernel void kernel_mimo2_qkv_rope_cache(
        constant ds4_metal_args_mimo2_qkv & args,
        device const float *qkv,         /* [T][H*K + Hkv*K + Hkv*V] */
        device float       *q,           /* [T][H][K] */
        device float       *k,           /* [T][Hkv][K] */
        device float       *v,           /* [T][Hkv][V] */
        device half        *key_cache,   /* [cap][Hkv][K] */
        device half        *value_cache, /* [cap][Hkv][V] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint slot = tgpig.x;
    const uint tok = tgpig.y;
    const uint H = args.n_head, Hkv = args.n_kv, K = args.key_dim, V = args.value_dim;
    if (tok >= args.n_tokens || slot >= H + 2u * Hkv) return;
    const uint d = tid;
    const uint pos = args.pos0 + tok;
    const bool store = tok >= args.store_from;
    const uint row = pos % args.cache_cap;
    device const float *src_row = qkv + (uint64_t)tok * (H * K + Hkv * K + Hkv * V);

    if (slot >= H + Hkv) {
        const uint kvh = slot - H - Hkv;
        if (d >= V) return;
        const float x = src_row[H * K + Hkv * K + kvh * V + d] * args.value_scale;
        v[((uint64_t)tok * Hkv + kvh) * V + d] = x;
        if (store) value_cache[((uint64_t)row * Hkv + kvh) * V + d] = (half)x;
        return;
    }
    if (d >= K) return;
    const bool is_key = slot >= H;
    device const float *src = is_key ? src_row + H * K + (slot - H) * K : src_row + slot * K;
    float x = src[d];
    const uint half_rot = args.rot_dim / 2u;
    if (d < args.rot_dim) {
        const uint i = d < half_rot ? d : d - half_rot;
        const float theta = (float)pos * args.freq[i];
        const float c = precise::cos(theta), s = precise::sin(theta);
        x = d < half_rot ? src[d] * c - src[d + half_rot] * s
                         : src[d] * c + src[d - half_rot] * s;
    }
    if (is_key) {
        const uint kvh = slot - H;
        k[((uint64_t)tok * Hkv + kvh) * K + d] = x;
        if (store) key_cache[((uint64_t)row * Hkv + kvh) * K + d] = (half)x;
    } else {
        q[((uint64_t)tok * H + slot) * K + d] = x;
    }
}

/* Store staged k/v rows of tokens [store_from, n_tokens) into the ring:
 * one threadgroup per (key or value head, token). */
kernel void kernel_mimo2_kv_commit(
        constant ds4_metal_args_mimo2_qkv & args,
        device const float *k,           /* [T][Hkv][K] */
        device const float *v,           /* [T][Hkv][V] */
        device half        *key_cache,
        device half        *value_cache,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint slot = tgpig.x;
    const uint tok = args.store_from + tgpig.y;
    const uint Hkv = args.n_kv, K = args.key_dim, V = args.value_dim;
    if (tok >= args.n_tokens || slot >= 2u * Hkv) return;
    const uint row = (args.pos0 + tok) % args.cache_cap;
    const uint d = tid;
    if (slot < Hkv) {
        if (d < K) key_cache[((uint64_t)row * Hkv + slot) * K + d] = (half)k[((uint64_t)tok * Hkv + slot) * K + d];
    } else {
        const uint kvh = slot - Hkv;
        if (d < V) value_cache[((uint64_t)row * Hkv + kvh) * V + d] = (half)v[((uint64_t)tok * Hkv + kvh) * V + d];
    }
}

struct ds4_metal_args_mimo2_attn {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_kv;
    uint32_t window;          /* 0: every earlier position */
    uint32_t n_splits;        /* key ranges per (token, kv head) */
    uint32_t keys_per_split;
    float    scale;
    uint32_t has_sinks;
    uint32_t has_cur;         /* positions >= pos0 come from k_cur/v_cur */
    uint32_t pad0;
};

#define MIMO2_ATTN_NSG 4      /* simdgroups per threadgroup, each owning q heads of one kv group */
#define MIMO2_ATTN_HPS 4      /* q heads per simdgroup: group <= NSG * HPS */

/* Keys visible to query position qpos: the last `window` positions up to and
 * including qpos (HF sliding-window mask kv > q - window), or all of them. */
static inline uint2 mimo2_attn_key_range(constant ds4_metal_args_mimo2_attn &args, uint tok) {
    const uint qpos = args.pos0 + tok;
    const uint n = args.window != 0u ? min(qpos + 1u, args.window) : qpos + 1u;
    return uint2(qpos + 1u - n, n);
}

/* Attention for one (key split, kv head, token).  The simdgroups share the
 * K/V rows and own disjoint query heads; lane j owns key dims
 * j*KPT..+KPT-1 and value dims j*VPT..+VPT-1.  With one split the sink is
 * folded in and heads are written; otherwise each head leaves (m, l, acc)
 * for kernel_mimo2_attn_merge. */
template <uint KPT, uint VPT>
kernel void kernel_mimo2_attn(
        constant ds4_metal_args_mimo2_attn & args,
        device const float *q,           /* [T][H][K] */
        device const half  *key_cache,   /* [cap][Hkv][K] */
        device const half  *value_cache, /* [cap][Hkv][V] */
        device const float *k_cur,       /* [T][Hkv][K] when has_cur */
        device const float *v_cur,       /* [T][Hkv][V] when has_cur */
        device const float *sinks,       /* [H] when has_sinks */
        device float       *heads,       /* [T][H][V] */
        device float       *part,        /* [T][Hkv][n_splits][group][2+V] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint K = KPT * 32u, V = VPT * 32u;
    const uint split = tgpig.x, kvh = tgpig.y, tok = tgpig.z;
    const uint H = args.n_head, Hkv = args.n_kv;
    if (split >= args.n_splits || kvh >= Hkv || tok >= args.n_tokens) return;
    const uint group = H / Hkv;
    const uint hps = (group + MIMO2_ATTN_NSG - 1u) / MIMO2_ATTN_NSG;
    const uint g0 = (uint)sgitg * hps;
    if (g0 >= group) return;
    const uint ng = min(hps, group - g0);
    const uint2 range = mimo2_attn_key_range(args, tok);
    const uint j0 = min(range.y, split * args.keys_per_split);
    const uint j1 = min(range.y, j0 + args.keys_per_split);
    const uint cur_from = args.has_cur ? args.pos0 : 0xffffffffu;

    float qv[MIMO2_ATTN_HPS][KPT];
    float m[MIMO2_ATTN_HPS], l[MIMO2_ATTN_HPS], acc[MIMO2_ATTN_HPS][VPT];
#pragma unroll
    for (uint g = 0; g < MIMO2_ATTN_HPS; g++) {
        const uint h = kvh * group + g0 + min(g, ng - 1u);
        device const float *qh = q + ((uint64_t)tok * H + h) * K + tiisg * KPT;
#pragma unroll
        for (uint i = 0; i < KPT; i++) qv[g][i] = qh[i] * args.scale;
        m[g] = -3.0e38f;
        l[g] = 0.0f;
#pragma unroll
        for (uint i = 0; i < VPT; i++) acc[g][i] = 0.0f;
    }
    for (uint j = j0; j < j1; j++) {
        const uint p = range.x + j;
        float kv[KPT], vv[VPT];
        if (p >= cur_from) {
            device const float *kr = k_cur + ((uint64_t)(p - args.pos0) * Hkv + kvh) * K + tiisg * KPT;
            device const float *vr = v_cur + ((uint64_t)(p - args.pos0) * Hkv + kvh) * V + tiisg * VPT;
#pragma unroll
            for (uint i = 0; i < KPT; i++) kv[i] = (float)(half)kr[i];
#pragma unroll
            for (uint i = 0; i < VPT; i++) vv[i] = (float)(half)vr[i];
        } else {
            const uint row = p % args.cache_cap;
            device const half *kr = key_cache + ((uint64_t)row * Hkv + kvh) * K + tiisg * KPT;
            device const half *vr = value_cache + ((uint64_t)row * Hkv + kvh) * V + tiisg * VPT;
#pragma unroll
            for (uint i = 0; i < KPT; i++) kv[i] = (float)kr[i];
#pragma unroll
            for (uint i = 0; i < VPT; i++) vv[i] = (float)vr[i];
        }
#pragma unroll
        for (uint g = 0; g < MIMO2_ATTN_HPS; g++) {
            if (g < ng) {
                float s = 0.0f;
#pragma unroll
                for (uint i = 0; i < KPT; i++) s += qv[g][i] * kv[i];
                s = simd_sum(s);
                const float m_new = max(m[g], s);
                const float corr = exp(m[g] - m_new);
                const float w = exp(s - m_new);
                l[g] = l[g] * corr + w;
#pragma unroll
                for (uint i = 0; i < VPT; i++) acc[g][i] = acc[g][i] * corr + w * vv[i];
                m[g] = m_new;
            }
        }
    }
#pragma unroll
    for (uint g = 0; g < MIMO2_ATTN_HPS; g++) {
        if (g >= ng) break;
        const uint h = kvh * group + g0 + g;
        if (args.n_splits == 1u) {
            float mm = m[g], ll = l[g], c = 1.0f;
            if (args.has_sinks) {
                const float sink = sinks[h];
                mm = max(m[g], sink);
                c = exp(m[g] - mm);
                ll = l[g] * c + exp(sink - mm);
            }
            const float inv = ll > 0.0f ? c / ll : 0.0f;
            device float *dst = heads + ((uint64_t)tok * H + h) * V + tiisg * VPT;
#pragma unroll
            for (uint i = 0; i < VPT; i++) dst[i] = acc[g][i] * inv;
        } else {
            device float *dst = part +
                ((((uint64_t)tok * Hkv + kvh) * args.n_splits + split) * group + g0 + g) * (2u + V);
            if (tiisg == 0) { dst[0] = m[g]; dst[1] = l[g]; }
#pragma unroll
            for (uint i = 0; i < VPT; i++) dst[2u + tiisg * VPT + i] = acc[g][i];
        }
    }
}

/* Merge the split partials of one (token, head), folding in the sink logit. */
template <uint VPT>
kernel void kernel_mimo2_attn_merge(
        constant ds4_metal_args_mimo2_attn & args,
        device const float *part,
        device const float *sinks,
        device float       *heads,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint V = VPT * 32u;
    const uint h = tgpig.x, tok = tgpig.y;
    const uint H = args.n_head, Hkv = args.n_kv;
    if (h >= H || tok >= args.n_tokens) return;
    const uint group = H / Hkv;
    const uint kvh = h / group, g = h % group;
    const uint64_t stride = (uint64_t)group * (2u + V);
    device const float *base = part + (((uint64_t)tok * Hkv + kvh) * args.n_splits * group + g) * (2u + V);
    float mm = -3.0e38f;
    for (uint s = 0; s < args.n_splits; s++) {
        device const float *p = base + s * stride;
        if (p[1] > 0.0f) mm = max(mm, p[0]);
    }
    const float sink = args.has_sinks ? sinks[h] : -3.0e38f;
    mm = max(mm, sink);
    float ll = args.has_sinks ? exp(sink - mm) : 0.0f;
    float o[VPT];
#pragma unroll
    for (uint i = 0; i < VPT; i++) o[i] = 0.0f;
    for (uint s = 0; s < args.n_splits; s++) {
        device const float *p = base + s * stride;
        const float c = p[1] > 0.0f ? exp(p[0] - mm) : 0.0f;
        ll += p[1] * c;
#pragma unroll
        for (uint i = 0; i < VPT; i++) o[i] += p[2u + tiisg * VPT + i] * c;
    }
    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
    device float *dst = heads + ((uint64_t)tok * H + h) * V + tiisg * VPT;
#pragma unroll
    for (uint i = 0; i < VPT; i++) dst[i] = o[i] * inv;
}

#define MIMO2_MM_NSG  4       /* simdgroups per prefill threadgroup, 8 query rows each */
#define MIMO2_MM_ROWS 32      /* (token, head) query rows per threadgroup */
#define MIMO2_MM_C    32      /* keys per shared tile */

/* Prefill attention with simdgroup matrices for one block of 32 query rows
 * of one kv head.  Row r is (token tok0 + r / group, head kvh*group +
 * r % group), so a block holds 32 / group tokens.  Each tile of 32 keys (ring
 * rows before pos0, staged rows from pos0 when has_cur) is loaded once as
 * f16 into threadgroup memory; each simdgroup computes S = Q K^T for its 8
 * rows (Q in f16, pre-scaled, f32 accumulation), applies the per-row causal /
 * window mask and the running softmax, rescales O through a diagonal matrix
 * and accumulates O += P V (P in f16).  The sink is folded in at the end. */
template <uint K, uint V>
kernel void kernel_mimo2_attn_mm(
        constant ds4_metal_args_mimo2_attn & args,
        device const float *q,           /* [T][H][K] */
        device const half  *key_cache,   /* [cap][Hkv][K] */
        device const half  *value_cache, /* [cap][Hkv][V] */
        device const float *k_cur,       /* [T][Hkv][K] when has_cur */
        device const float *v_cur,       /* [T][Hkv][V] when has_cur */
        device const float *sinks,       /* [H] when has_sinks */
        device float       *heads,       /* [T][H][V] */
        threadgroup char   *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint C = MIMO2_MM_C;
    constexpr uint KS = K + 8u, VS = V + 8u;
    threadgroup half  *ks = (threadgroup half *)shmem;                  /* [C][KS], Q staging [ROWS][KS] */
    threadgroup half  *vs = ks + C * KS;                                /* [C][VS] */
    threadgroup float *ss = (threadgroup float *)(vs + C * VS) + sgitg * 8u * C;  /* [8][C] per simdgroup */
    threadgroup half  *ps = (threadgroup half *)((threadgroup float *)(vs + C * VS) + MIMO2_MM_NSG * 8u * C)
                            + sgitg * 8u * C;                           /* [8][C] per simdgroup */
    threadgroup float *dg = (threadgroup float *)((threadgroup half *)((threadgroup float *)(vs + C * VS) +
                            MIMO2_MM_NSG * 8u * C) + MIMO2_MM_NSG * 8u * C) + sgitg * 64u;  /* [8][8] */

    const uint H = args.n_head, Hkv = args.n_kv;
    const uint group = H / Hkv;
    const uint tpb = MIMO2_MM_ROWS / group;
    const uint kvh = tgpig.y;
    const uint tok0 = tgpig.x * tpb;
    if (kvh >= Hkv || tok0 >= args.n_tokens) return;
    const uint tok_last = min(args.n_tokens, tok0 + tpb) - 1u;
    const uint p_begin = mimo2_attn_key_range(args, tok0).x;
    const uint p_end = args.pos0 + tok_last;
    const uint cur_from = args.has_cur ? args.pos0 : 0xffffffffu;

    /* Q rows, pre-scaled, as f16 through the K tile area. */
    for (uint i = tid; i < MIMO2_MM_ROWS * K; i += MIMO2_MM_NSG * 32u) {
        const uint r = i / K, d = i % K;
        const uint tok = tok0 + r / group;
        float x = 0.0f;
        if (tok < args.n_tokens)
            x = q[((uint64_t)tok * H + kvh * group + r % group) * K + d] * args.scale;
        ks[r * KS + d] = (half)x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 mq[K / 8u];
#pragma unroll
    for (uint i = 0; i < K / 8u; i++) simdgroup_load(mq[i], ks + sgitg * 8u * KS + i * 8u, KS);
    for (uint i = tiisg; i < 64u; i += 32u) dg[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 mo[V / 8u];
#pragma unroll
    for (uint j = 0; j < V / 8u; j++) mo[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    /* Softmax lanes: 4 lanes per row, 8 keys each. */
    const uint my_row = tiisg / 4u, quarter = tiisg % 4u;
    const uint r_global = sgitg * 8u + my_row;
    const uint my_tok = tok0 + r_global / group;
    const bool row_active = my_tok < args.n_tokens;
    const uint my_h = kvh * group + r_global % group;
    const uint my_lo = row_active ? mimo2_attn_key_range(args, my_tok).x : 0u;
    const uint my_hi = args.pos0 + my_tok;
    float m = -3.0e38f, l = 0.0f;

    for (uint p0 = p_begin; p0 <= p_end; p0 += C) {
        const uint n_keys = min(C, p_end + 1u - p0);
        for (uint i = tid; i < C * (K / 4u); i += MIMO2_MM_NSG * 32u) {
            const uint j = i / (K / 4u), c = i % (K / 4u);
            half4 val = half4(0.0h);
            if (j < n_keys) {
                const uint p = p0 + j;
                if (p >= cur_from)
                    val = half4(((device const float4 *)(k_cur + ((uint64_t)(p - args.pos0) * Hkv + kvh) * K))[c]);
                else
                    val = ((device const half4 *)(key_cache + ((uint64_t)(p % args.cache_cap) * Hkv + kvh) * K))[c];
            }
            *((threadgroup half4 *)(ks + j * KS) + c) = val;
        }
        for (uint i = tid; i < C * (V / 4u); i += MIMO2_MM_NSG * 32u) {
            const uint j = i / (V / 4u), c = i % (V / 4u);
            half4 val = half4(0.0h);
            if (j < n_keys) {
                const uint p = p0 + j;
                if (p >= cur_from)
                    val = half4(((device const float4 *)(v_cur + ((uint64_t)(p - args.pos0) * Hkv + kvh) * V))[c]);
                else
                    val = ((device const half4 *)(value_cache + ((uint64_t)(p % args.cache_cap) * Hkv + kvh) * V))[c];
            }
            *((threadgroup half4 *)(vs + j * VS) + c) = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* S = Q K^T for this simdgroup's 8 rows. */
#pragma unroll
        for (uint kb = 0; kb < C / 8u; kb++) {
            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
#pragma unroll
            for (uint i = 0; i < K / 8u; i++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ks + kb * 8u * KS + i * 8u, KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq[i], mk, mqk);
            }
            simdgroup_store(mqk, ss + kb * 8u, C);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float s[8];
        float tmax = -3.0e38f;
#pragma unroll
        for (uint e = 0; e < 8u; e++) {
            const uint p = p0 + quarter * 8u + e;
            const bool valid = row_active && p >= my_lo && p <= my_hi;
            s[e] = valid ? ss[my_row * C + quarter * 8u + e] : -3.0e38f;
            tmax = max(tmax, s[e]);
        }
        tmax = max(tmax, simd_shuffle_xor(tmax, 1));
        tmax = max(tmax, simd_shuffle_xor(tmax, 2));
        const float m_new = max(m, tmax);
        const float corr = exp(m - m_new);
        float sum = 0.0f;
#pragma unroll
        for (uint e = 0; e < 8u; e++) {
            const float pe = s[e] > -1.0e38f ? exp(s[e] - m_new) : 0.0f;
            sum += pe;
            ps[my_row * C + quarter * 8u + e] = (half)pe;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * corr + sum;
        m = m_new;
        if (quarter == 0) dg[my_row * 9u] = corr;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        /* O = diag(corr) O + P V. */
        simdgroup_float8x8 md;
        simdgroup_load(md, dg, 8);
#pragma unroll
        for (uint j = 0; j < V / 8u; j++) simdgroup_multiply(mo[j], md, mo[j]);
#pragma unroll
        for (uint kb = 0; kb < C / 8u; kb++) {
            simdgroup_half8x8 mp;
            simdgroup_load(mp, ps + kb * 8u, C);
#pragma unroll
            for (uint j = 0; j < V / 8u; j++) {
                simdgroup_half8x8 mv;
                simdgroup_load(mv, vs + kb * 8u * VS + j * 8u, VS);
                simdgroup_multiply_accumulate(mo[j], mp, mv, mo[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* heads = O * c / l with the sink folded into the normalizer. */
    float mm = m, ll = l, c = 1.0f;
    if (args.has_sinks && row_active) {
        const float sink = sinks[my_h];
        mm = max(m, sink);
        c = exp(m - mm);
        ll = l * c + exp(sink - mm);
    }
    const float f = ll > 0.0f ? c / ll : 0.0f;
    for (uint jc = 0; jc < V / C; jc++) {
#pragma unroll
        for (uint b = 0; b < C / 8u; b++) simdgroup_store(mo[jc * (C / 8u) + b], ss + b * 8u, C);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (row_active) {
            device float *dst = heads + ((uint64_t)my_tok * H + my_h) * V + jc * C + quarter * 8u;
#pragma unroll
            for (uint e = 0; e < 8u; e++) dst[e] = ss[my_row * C + quarter * 8u + e] * f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define MIMO2_ATTN_INSTANTIATE(KPT_, VPT_) \
template [[host_name("kernel_mimo2_attn_k" #KPT_ "_v" #VPT_)]] \
kernel void kernel_mimo2_attn<KPT_, VPT_>(constant ds4_metal_args_mimo2_attn &, device const float *, \
    device const half *, device const half *, device const float *, device const float *, \
    device const float *, device float *, device float *, uint3, ushort, ushort);

MIMO2_ATTN_INSTANTIATE(6, 4)   /* release: K 192, V 128 */
MIMO2_ATTN_INSTANTIATE(2, 1)   /* mini: K 64, V 32 */

#define MIMO2_MERGE_INSTANTIATE(VPT_) \
template [[host_name("kernel_mimo2_attn_merge_v" #VPT_)]] \
kernel void kernel_mimo2_attn_merge<VPT_>(constant ds4_metal_args_mimo2_attn &, device const float *, \
    device const float *, device float *, uint3, ushort);

MIMO2_MERGE_INSTANTIATE(4)
MIMO2_MERGE_INSTANTIATE(1)

#define MIMO2_MM_INSTANTIATE(K_, V_) \
template [[host_name("kernel_mimo2_attn_mm_k" #K_ "_v" #V_)]] \
kernel void kernel_mimo2_attn_mm<K_, V_>(constant ds4_metal_args_mimo2_attn &, device const float *, \
    device const half *, device const half *, device const float *, device const float *, \
    device const float *, device float *, threadgroup char *, uint3, ushort, ushort, ushort);

MIMO2_MM_INSTANTIATE(192, 128)
MIMO2_MM_INSTANTIATE(64, 32)
