#pragma GCC optimize("O3")
#pragma GCC target("avx2")
#include <immintrin.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;
using u128 = __uint128_t;

using i128 = __m128i;
using i256 = __m256i;
using u32x4 = u32 __attribute__((vector_size(16)));
using u64x2 = u64 __attribute__((vector_size(16)));
using u32x8 = u32 __attribute__((vector_size(32)));
using u64x4 = u64 __attribute__((vector_size(32)));

#define RC(type, val) reinterpret_cast<type>(val)
#define INL inline __attribute__((always_inline))

namespace simd {
    u32x8 load_u32x8(u32 *ptr) {
        return RC(u32x8, _mm256_load_si256((i256 *)ptr));
    }
    u32x8 loadu_u32x8(u32 *ptr) {
        return RC(u32x8, _mm256_loadu_si256((i256 *)ptr));
    }
    void store_u32x8(u32 *ptr, u32x8 val) {
        _mm256_store_si256((i256 *)ptr, RC(i256, val));
    }
    void storeu_u32x8(u32 *ptr, u32x8 val) {
        _mm256_storeu_si256((i256 *)ptr, RC(i256, val));
    }

    u32x8 set1_u32x8(u32 val) {
        return RC(u32x8, _mm256_set1_epi32(val));
    }
    u64x4 set1_u64x4(u64 val) {
        return RC(u64x4, _mm256_set1_epi64x(val));
    }
};  // namespace simd

using namespace simd;

// works for 998'244'353, not sure about other numbers
struct Barrett {
    u32 mod, s, q;
    u32 mod2, s_1;

    Barrett(u32 m) : mod(m) {
        s = std::__lg(mod);
        mod2 = 2 * mod;
        s_1 = s + 1;

        u64 q0 = (((__uint128_t(1) << 64 + s) - 1) / mod + 1);
        if (mod == (1u << s)) {
            q0--;
        }

        assert(mod < (1 << 30) && "this shit won't work");
        if (q0 & ((1ULL << 31) | (1ULL << 62))) {
            std::cerr << "warning improper mod  (line: " << __LINE__ << ")" << std::endl;
        }
        // assert(!(q0 & (1ULL << 31)));  // ! wtf
        // assert(!(q0 & (1ULL << 62)));  // ! wtf

        q = q0 >> 32;
    }

    u32 shrink(u32 val) {
        return std::min<u32>(val, val - mod);
    }

    u32 shrink_2(u32 val) {
        return std::min<u32>(val, val - mod2);
    }

    u32 shrink_21(u32 val) {
        return shrink(shrink_2(val));
    }

    // from [0, 2 * mod * mod) to [0, 2 * mod)
    u32 mod_22(u64 val) {
        u32 a = u32(val >> s) * u64(q) >> 32;
        u32 res = u32(val) - a * mod;
        return res;
    }

    // from [0, 2 * mod * mod) to [0, mod)
    // should work for every mod
    u32 mod_21(u64 val) {
        u32 res = mod_22(val);
        res -= mod * (res >= mod) + mod * (res >= mod2);
        return res;
    }

    // from [0, 4 * mod * mod) to [0, 4 * mod)
    u32 mod_44(u64 val) {
        u32 a = u32(val >> s_1) * u64(q) >> 32;
        u32 res = u32(val) - a * mod2;
        return res;
    }

    // from [0, 4 * mod * mod) to [0, 2 * mod)
    u32 mod_42(u64 val) {
        u32 res = mod_44(val);
        res = shrink_2(res);
        return res;
    }
};

struct Barrett_simd {
    Barrett bt;

    u32 s, s_1;
    u32x8 v_mod, v_q;
    u32x8 v_mod2;

    Barrett_simd(u32 m) : bt(m) {
        s = bt.s;
        s_1 = bt.s_1;
        v_q = set1_u32x8(bt.q);
        v_mod = set1_u32x8(bt.mod);
        v_mod2 = set1_u32x8(bt.mod2);
    }

    u32x8 shrink(u32x8 vec) {
        i256 res = _mm256_min_epu32(RC(i256, vec), _mm256_sub_epi32(RC(i256, vec), RC(i256, v_mod)));
        return RC(u32x8, res);
    }

    u32x8 shrink_2(u32x8 vec) {
        i256 res = _mm256_min_epu32(RC(i256, vec), _mm256_sub_epi32(RC(i256, vec), RC(i256, v_mod2)));
        return RC(u32x8, res);
    }

    u32x8 shrink_21(u32x8 vec) {
        return shrink(shrink_2(vec));
    }

    // from [0, 2 * mod * mod) to [0, 2 * mod)
    u64x4 mod_22(u64x4 vec) {
        i256 a = _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(RC(i256, vec), s), RC(i256, v_q)), 32);
        i256 res = _mm256_sub_epi64(RC(i256, vec), _mm256_mul_epu32(a, RC(i256, v_mod)));
        return RC(u64x4, res);
    }

    // from [0, 4 * mod * mod) to [0, 4 * mod)
    u64x4 mod_44(u64x4 vec) {
        i256 a = _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(RC(i256, vec), s_1), RC(i256, v_q)), 32);
        i256 res = _mm256_sub_epi64(RC(i256, vec), _mm256_mul_epu32(a, RC(i256, v_mod2)));
        return RC(u64x4, res);
    }

    // from [0, 4 * mod * mod) to [0, 2 * mod)
    i256 mod_42(u64x4 vec) {
        i256 res = RC(i256, mod_44(vec));
        res = _mm256_min_epu32(res, _mm256_sub_epi32(res, RC(i256, v_mod2)));
        return res;
    }

    // product in [0, 4 * mod * mod), result in [0, 2 * mod)
    u32x8 mul_mod_42(u32x8 a, u32x8 b) {
        const u32 shuflle_mask = 0b10'11'00'01;
        i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
        i256 x1357 = _mm256_mul_epu32(_mm256_shuffle_epi32(RC(i256, a), shuflle_mask), _mm256_shuffle_epi32(RC(i256, b), shuflle_mask));
        x0246 = RC(i256, mod_44(RC(u64x4, x0246)));
        x1357 = RC(i256, mod_44(RC(u64x4, x1357)));
        i256 res = _mm256_blend_epi32(x0246, _mm256_shuffle_epi32(x1357, shuflle_mask), 0b10'10'10'10);
        res = _mm256_min_epu32(res, _mm256_sub_epi32(res, RC(i256, v_mod2)));
        return RC(u32x8, res);
    }

    // product in [0, 2 * mod * mod), result in [0, 2 * mod)
    u64x4 mul_mod_22(u64x4 a, u64x4 b) {
        i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
        i256 res = RC(i256, mod_22(RC(u64x4, x0246)));
        return RC(u64x4, res);
    }

    // product in [0, 4 * mod * mod), result in [0, 2 * mod)
    u64x4 mul_mod_42(u64x4 a, u64x4 b) {
        const u32 shuflle_mask = 0b10'11'00'01;
        i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
        i256 res = RC(i256, mod_44(RC(u64x4, x0246)));
        res = _mm256_min_epu32(res, _mm256_sub_epi32(res, RC(i256, v_mod2)));
        return RC(u64x4, res);
    }
};

#include <array>
#include <cstring>

namespace FFT {
    constexpr u32 mod = 998'244'353;
    constexpr u32 pr_root = 3;

    Barrett bt(mod);
    Barrett_simd bts(mod);

    u32 power(u32 base, u32 exp) {
        u32 res = 1;
        for (; exp > 0; exp >>= 1) {
            if (exp & 1) {
                res = bt.mod_42(u64(res) * base);
            }
            base = bt.mod_42(u64(base) * base);
        }
        res = bt.shrink_21(res);
        return res;
    }

    std::vector<u32 *> w, w_rb;

    void expand_w(int k) {
        while (w.size() < k) {
            int r = w.size();
            w.push_back((u32 *)aligned_alloc(64, std::max<int>(64, 4 << r)));
            if (r == 0) {
                w.back()[0] = 1;
            } else {
                u32 f = power(pr_root, (mod - 1) >> r + 1);
                for (int j = 0; j < (1 << r - 1); j++) {
                    w[r][2 * j] = w[r - 1][j];
                    w[r][2 * j + 1] = bt.shrink(bt.mod_22(u64(f) * w[r - 1][j]));
                }
            }
        }
    }

    void expand_w_rb(int k) {
        while (w_rb.size() < k) {
            int r = w_rb.size();
            w_rb.push_back((u32 *)aligned_alloc(64, std::max<int>(64, 4 << r)));

            if (r == 0) {
                w_rb.back()[0] = 1;
            } else {
                u32 f = power(pr_root, (mod - 1) >> r + 1);
                for (int j = 0; j < (1 << r - 1); j++) {
                    w_rb[r][j] = w_rb[r - 1][j];
                    w_rb[r][j + (1 << r - 1)] = bt.shrink(bt.mod_22(u64(f) * w_rb[r - 1][j]));
                }
            }
        }
    }

    INL void butterfly_u32_x2(u32 a, u32 b, u32 wj, u32 *ap, u32 *bp) {
        a = bt.shrink_2(a);
        u32 c = bt.mod_42(u64(b) * wj);
        ap[0] = a + c;
        bp[0] = a + bt.mod2 - c;
    }

    INL std::array<u32x8, 2> butterfly_u32x8_x2(u32x8 a, u32x8 b, u32x8 wj) {
        a = bts.shrink_2(a);
        u32x8 c = bts.mul_mod_42(b, wj);
        return {a + c, a + bts.v_mod2 - c};
    }

    INL void butterfly_u32x8_x2(u32x8 a, u32x8 b, u32x8 wj, u32 *ap, u32 *bp) {
        a = bts.shrink_2(a);
        u32x8 c = bts.mul_mod_42(b, wj);
        store_u32x8(ap, a + c);
        store_u32x8(bp, a + bts.v_mod2 - c);
    }

    INL std::array<u32x8, 4> butterfly_u32x8_x4(u32x8 a, u32x8 b, u32x8 c, u32x8 d, u32x8 wj, u32x8 wj1, u32x8 wj2) {
        a = bts.shrink_2(a);
        c = bts.shrink_2(c);
        u32x8 p = bts.mul_mod_42(b, wj);
        u32x8 q = bts.mul_mod_42(d, wj);

        u32x8 a2 = a + p;
        u32x8 b2 = a + bts.v_mod2 - p;
        u32x8 c2 = c + q;
        u32x8 d2 = c + bts.v_mod2 - q;

        auto [r0, r2] = butterfly_u32x8_x2(a2, c2, wj1);
        auto [r1, r3] = butterfly_u32x8_x2(b2, d2, wj2);
        return {r0, r1, r2, r3};
    }

    INL void butterfly_u32x8_x4(u32x8 a, u32x8 b, u32x8 c, u32x8 d, u32x8 wj, u32x8 wj1, u32x8 wj2, u32 *ap, u32 *bp, u32 *cp, u32 *dp) {
        a = bts.shrink_2(a);
        c = bts.shrink_2(c);
        u32x8 p = bts.mul_mod_42(b, wj);
        u32x8 q = bts.mul_mod_42(d, wj);

        u32x8 a2 = a + p;
        u32x8 b2 = a + bts.v_mod2 - p;
        u32x8 c2 = c + q;
        u32x8 d2 = c + bts.v_mod2 - q;

        butterfly_u32x8_x2(a2, c2, wj1, ap, cp);
        butterfly_u32x8_x2(b2, d2, wj2, bp, dp);
    }

    INL void butterfly_u32x8_x8(u32x8 a0, u32x8 a1, u32x8 a2, u32x8 a3, u32x8 a4, u32x8 a5, u32x8 a6, u32x8 a7,
                                u32x8 wj, u32x8 wj0, u32x8 wj1, u32x8 wj_0, u32x8 wj_1, u32x8 wj_2, u32x8 wj_3,
                                u32 *ap0, u32 *ap1, u32 *ap2, u32 *ap3, u32 *ap4, u32 *ap5, u32 *ap6, u32 *ap7) {
        a0 = bts.shrink_2(a0);
        a2 = bts.shrink_2(a2);
        a4 = bts.shrink_2(a4);
        a6 = bts.shrink_2(a6);

        u32x8 c0 = bts.mul_mod_42(a1, wj);
        u32x8 c1 = bts.mul_mod_42(a3, wj);
        u32x8 c2 = bts.mul_mod_42(a5, wj);
        u32x8 c3 = bts.mul_mod_42(a7, wj);

        u32x8 d0 = a0 + c0;
        u32x8 d1 = a0 + bts.v_mod2 - c0;
        u32x8 d2 = a2 + c1;
        u32x8 d3 = a2 + bts.v_mod2 - c1;
        u32x8 d4 = a4 + c2;
        u32x8 d5 = a4 + bts.v_mod2 - c2;
        u32x8 d6 = a6 + c3;
        u32x8 d7 = a6 + bts.v_mod2 - c3;

        butterfly_u32x8_x4(d0, d2, d4, d6, wj0, wj_0, wj_2, ap0, ap2, ap4, ap6);
        butterfly_u32x8_x4(d1, d3, d5, d7, wj1, wj_1, wj_3, ap1, ap3, ap5, ap7);
    }

    // template <int lg, std::array ar>
    // void t_fft(u32 *data) {
    //     static_assert(std::accumulate(ar.begin(), ar.end(), 0) == lg);
    // }

    // input data[i] in [0, 4 * mod)
    // result data[i] in [0, 4 * mod)
    void fft(int lg, u32 *data) {
        expand_w_rb(lg);
        int n = 1 << lg;

        // if (lg == 20) {
        //     t_fft<20, std::array<int, 9>{4, 2, 2, 2, 2, 2, 2, 2, 2}>(data);
        // }

        if (lg <= 5) {
            for (int k = lg - 1; k >= 0; k--) {
                for (int i = 0; i < n; i += (1 << k + 1)) {
                    u32 wj = w_rb[lg - k - 1][i >> k + 1];
                    for (int j = 0; j < (1 << k); j++) {
                        u32 a = data[i + j], b = data[i + (1 << k) + j];
                        butterfly_u32_x2(a, b, wj, data + i + j, data + i + (1 << k) + j);
                    }
                }
            }
        } else {
            int k = lg - 1;
            for (; k % 2; k--) {
                for (int i = 0; i < n; i += (1 << k + 1)) {
                    u32x8 wj = set1_u32x8(w_rb[lg - k - 1][i >> k + 1]);
                    for (int j = 0; j < (1 << k); j += 8) {
                        u32x8 a = load_u32x8(data + i + j), b = load_u32x8(data + i + (1 << k) + j);
                        butterfly_u32x8_x2(a, b, wj, data + i + j, data + i + (1 << k) + j);
                    }
                }
            }
            assert(k % 2 == 0);

            constexpr int B = 4;  // B should be either 2 or 4

            for (; k > B; k -= 2) {
                for (int i = 0; i < n; i += (1 << k + 1)) {
                    u32x8 wj = set1_u32x8(w_rb[lg - k - 1][i >> k + 1]);
                    u32x8 wj1 = set1_u32x8(w_rb[lg - k][i >> k]);
                    u32x8 wj2 = set1_u32x8(w_rb[lg - k][(i >> k) + 1]);
                    for (int j = 0; j < (1 << k - 1); j += 8) {
                        u32x8 a = load_u32x8(data + i + 0 * (1 << k - 1) + j);
                        u32x8 b = load_u32x8(data + i + 1 * (1 << k - 1) + j);
                        u32x8 c = load_u32x8(data + i + 2 * (1 << k - 1) + j);
                        u32x8 d = load_u32x8(data + i + 3 * (1 << k - 1) + j);

                        butterfly_u32x8_x4(a, c, b, d, wj, wj1, wj2,
                                           data + i + 0 * (1 << k - 1) + j,
                                           data + i + 2 * (1 << k - 1) + j,
                                           data + i + 1 * (1 << k - 1) + j,
                                           data + i + 3 * (1 << k - 1) + j);
                    }
                }
            }

            // for (; k >= 0; k--) {
            //     for (int i = 0; i < n; i += (1 << k + 1)) {
            //         u32 wj = w_rb[lg - k - 1][i >> k + 1];
            //         for (int j = 0; j < (1 << k); j++) {
            //             u32 a = data[i + j], b = data[i + (1 << k) + j];
            //             butterfly_u32_x2(a, b, wj, data + i + j, data + i + (1 << k) + j);
            //         }
            //     }
            // }
            // return;
            if constexpr (B == 2) {
                assert(k == 2);
                for (int i = 0; i < n; i += 8) {
                    // u32x4 wj = RC(u32x4, _mm_load_si128((i128 *)(w_rb[lg - 1] + (i >> 1))));

                    auto cum = [&](u32x8 val, int i) {
                        u64x4 wj0 = set1_u64x4(w_rb[lg - 3][i >> 3]);
                        u64x4 wj1 = RC(u64x4, _mm256_setr_epi64x(w_rb[lg - 2][i >> 2], w_rb[lg - 2][i >> 2], w_rb[lg - 2][i + 4 >> 2], w_rb[lg - 2][i + 4 >> 2]));
                        u64x4 wj2 = RC(u64x4, _mm256_setr_epi64x(w_rb[lg - 1][i >> 1], w_rb[lg - 1][i + 2 >> 1], w_rb[lg - 1][i + 4 >> 1], w_rb[lg - 1][i + 6 >> 1]));
                        {
                            val = bts.shrink_2(val);

                            u64x4 x4567 = RC(u64x4, _mm256_permutevar8x32_epi32(RC(i256, val), _mm256_setr_epi32(4, -1, 5, -1, 6, -1, 7, -1)));
                            u32x8 c = RC(u32x8, _mm256_permutevar8x32_epi32(RC(i256, bts.mul_mod_42(x4567, wj0)), _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)));

                            u32x8 sm = val + c;
                            u32x8 df = RC(u32x8, _mm256_permute2x128_si256(RC(i256, val), RC(i256, val), 0)) + bts.v_mod2 - c;
                            val = RC(u32x8, _mm256_blend_epi32(RC(i256, sm), RC(i256, df), 0b11'11'00'00));
                        }
                        {
                            val = bts.shrink_2(val);
                            u64x4 x2367 = RC(u64x4, _mm256_shuffle_epi32(RC(i256, val), 0b00'11'00'10));
                            u32x8 c = RC(u32x8, _mm256_shuffle_epi32(RC(i256, bts.mul_mod_22(x2367, wj1)), 0b10'00'10'00));
                            u32x8 sm = val + c;
                            u32x8 df = RC(u32x8, _mm256_bslli_epi128(RC(i256, val), 8)) + bts.v_mod2 - c;
                            val = RC(u32x8, _mm256_blend_epi32(RC(i256, sm), RC(i256, df), 0b11'00'11'00));
                        }
                        {
                            val = bts.shrink_2(val);
                            u64x4 x1357 = RC(u64x4, _mm256_shuffle_epi32(RC(i256, val), 0b00'11'00'01));
                            u32x8 c = RC(u32x8, _mm256_shuffle_epi32(RC(i256, bts.mul_mod_22(x1357, wj2)), 0b10'10'00'00));
                            u32x8 sm = val + c;
                            u32x8 df = RC(u32x8, _mm256_bslli_epi128(RC(i256, val), 4)) + bts.v_mod2 - c;
                            val = RC(u32x8, _mm256_blend_epi32(RC(i256, sm), RC(i256, df), 0b10'10'10'10));
                        }

                        return val;
                    };
                    u32x8 a = load_u32x8(data + i);
                    a = cum(a, i);
                    store_u32x8(data + i, a);
                }
            } else {
                assert(k == 4);
                for (int i = 0; i < n; i += 32) {
                    auto cum = [&](u32x8 val, int i) {
                        u64x4 wj0 = set1_u64x4(w_rb[lg - 3][i >> 3]);
                        u64x4 wj1 = RC(u64x4, _mm256_setr_epi64x(w_rb[lg - 2][i >> 2], w_rb[lg - 2][i >> 2], w_rb[lg - 2][i + 4 >> 2], w_rb[lg - 2][i + 4 >> 2]));
                        u64x4 wj2 = RC(u64x4, _mm256_setr_epi64x(w_rb[lg - 1][i >> 1], w_rb[lg - 1][i + 2 >> 1], w_rb[lg - 1][i + 4 >> 1], w_rb[lg - 1][i + 6 >> 1]));
                        {
                            val = bts.shrink_2(val);

                            u64x4 x4567 = RC(u64x4, _mm256_permutevar8x32_epi32(RC(i256, val), _mm256_setr_epi32(4, -1, 5, -1, 6, -1, 7, -1)));
                            u32x8 c = RC(u32x8, _mm256_permutevar8x32_epi32(RC(i256, bts.mul_mod_42(x4567, wj0)), _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)));

                            u32x8 sm = val + c;
                            u32x8 df = RC(u32x8, _mm256_permute2x128_si256(RC(i256, val), RC(i256, val), 0)) + bts.v_mod2 - c;
                            val = RC(u32x8, _mm256_blend_epi32(RC(i256, sm), RC(i256, df), 0b11'11'00'00));
                        }
                        {
                            val = bts.shrink_2(val);
                            u64x4 x2367 = RC(u64x4, _mm256_shuffle_epi32(RC(i256, val), 0b00'11'00'10));
                            u32x8 c = RC(u32x8, _mm256_shuffle_epi32(RC(i256, bts.mul_mod_22(x2367, wj1)), 0b10'00'10'00));
                            u32x8 sm = val + c;
                            u32x8 df = RC(u32x8, _mm256_bslli_epi128(RC(i256, val), 8)) + bts.v_mod2 - c;
                            val = RC(u32x8, _mm256_blend_epi32(RC(i256, sm), RC(i256, df), 0b11'00'11'00));
                        }
                        {
                            val = bts.shrink_2(val);
                            u64x4 x1357 = RC(u64x4, _mm256_shuffle_epi32(RC(i256, val), 0b00'11'00'01));
                            u32x8 c = RC(u32x8, _mm256_shuffle_epi32(RC(i256, bts.mul_mod_22(x1357, wj2)), 0b10'10'00'00));
                            u32x8 sm = val + c;
                            u32x8 df = RC(u32x8, _mm256_bslli_epi128(RC(i256, val), 4)) + bts.v_mod2 - c;
                            val = RC(u32x8, _mm256_blend_epi32(RC(i256, sm), RC(i256, df), 0b10'10'10'10));
                        }

                        return val;
                    };
                    u32x8 a0 = load_u32x8(data + i + 0 * 8);
                    u32x8 a1 = load_u32x8(data + i + 1 * 8);
                    u32x8 a2 = load_u32x8(data + i + 2 * 8);
                    u32x8 a3 = load_u32x8(data + i + 3 * 8);

                    u32x8 wj = set1_u32x8(w_rb[lg - 4 - 1][i >> 4 + 1]);
                    u32x8 wj1 = set1_u32x8(w_rb[lg - 4][i >> 4]);
                    u32x8 wj2 = set1_u32x8(w_rb[lg - 4][(i >> 4) + 1]);

                    auto [b0, b2, b1, b3] = butterfly_u32x8_x4(a0, a2, a1, a3, wj, wj1, wj2);

                    b0 = cum(b0, i + 0 * 8);
                    b1 = cum(b1, i + 1 * 8);
                    b2 = cum(b2, i + 2 * 8);
                    b3 = cum(b3, i + 3 * 8);

                    store_u32x8(data + i + 0 * (1 << 4 - 1), b0);
                    store_u32x8(data + i + 1 * (1 << 4 - 1), b1);
                    store_u32x8(data + i + 2 * (1 << 4 - 1), b2);
                    store_u32x8(data + i + 3 * (1 << 4 - 1), b3);
                }
            }
        }
    }

    // input data[i] in [0, 2 * mod)
    // result data[i] in [0, mod)
    void ifft(int lg, u32 *data) {
        expand_w(lg);
        int n = 1 << lg;

        if (lg <= 5) {
            for (int k = 0; k < lg; k++) {
                for (int i = 0; i < n; i += (1 << k + 1)) {
                    for (int j = 0; j < (1 << k); j++) {
                        u32 a = data[i + j], b = data[i + (1 << k) + j];
                        u32 wj = w[k][j];
                        butterfly_u32_x2(a, b, wj, data + i + j, data + i + (1 << k) + j);
                    }
                }
            }
            std::reverse(data + 1, data + n);
            u32 rv = power(n, (mod - 2));
            for (int i = 0; i < n; i++) {
                data[i] = bt.shrink(bt.mod_42(u64(rv) * data[i]));
            }
            return;
        }

        int k = 0;
        {
            // u32x8 wj0 = set1_u32x8(w[0][0]); // * w[0][0] == 1

            u64x4 wj1 = RC(u64x4, _mm256_setr_epi64x(w[1][0], w[1][1], w[1][0], w[1][1]));
            u64x4 wj2 = RC(u64x4, _mm256_setr_epi64x(w[2][0], w[2][1], w[2][2], w[2][3]));
            u32x8 wj3 = load_u32x8(w[3]);
            u32x8 wj4_0 = load_u32x8(w[4]), wj4_1 = load_u32x8(w[4] + 8);

            u32x8 wj5_0 = load_u32x8(w[5]), wj5_1 = load_u32x8(w[5] + 8);
            u32x8 wj5_2 = load_u32x8(w[5] + 16), wj5_3 = load_u32x8(w[5] + 24);

            auto cum = [&](u32x8 val) {
                {
                    u32x8 val2 = RC(u32x8, _mm256_shuffle_epi32(RC(i256, val), 0b10'11'00'01));
                    u32x8 sm = val + val2;
                    u32x8 df = val2 + (bts.v_mod2 - val);
                    val = RC(u32x8, _mm256_blend_epi32(RC(i256, sm), RC(i256, df), 0b10'10'10'10));
                    val = bts.shrink_2(val);
                }
                {
                    u64x4 x2367 = RC(u64x4, _mm256_shuffle_epi32(RC(i256, val), 0b00'11'00'10));
                    u32x8 c = RC(u32x8, _mm256_shuffle_epi32(RC(i256, bts.mul_mod_22(x2367, wj1)), 0b10'00'10'00));
                    u32x8 sm = val + c;
                    u32x8 df = RC(u32x8, _mm256_bslli_epi128(RC(i256, val), 8)) + bts.v_mod2 - c;
                    val = RC(u32x8, _mm256_blend_epi32(RC(i256, sm), RC(i256, df), 0b11'00'11'00));
                    val = bts.shrink_2(val);
                }

                {
                    u64x4 x4567 = RC(u64x4, _mm256_permutevar8x32_epi32(RC(i256, val), _mm256_setr_epi32(4, -1, 5, -1, 6, -1, 7, -1)));
                    u32x8 c = RC(u32x8, _mm256_permutevar8x32_epi32(RC(i256, bts.mul_mod_22(x4567, wj2)), _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)));

                    u32x8 sm = val + c;
                    u32x8 df = RC(u32x8, _mm256_permute2x128_si256(RC(i256, val), RC(i256, val), 0)) + bts.v_mod2 - c;
                    val = RC(u32x8, _mm256_blend_epi32(RC(i256, sm), RC(i256, df), 0b11'11'00'00));
                }

                return val;
            };

            constexpr int BT = 2;
            if constexpr (BT == 0) {
                for (int i = 0; i < n; i += 16) {
                    u32x8 a = load_u32x8(data + i), b = load_u32x8(data + i + 8);

                    a = cum(a);
                    b = cum(b);

                    butterfly_u32x8_x2(a, b, wj3, data + i, data + i + 8);
                }
                k += 4;
            } else if constexpr (BT == 1) {
                for (int i = 0; i < n; i += 32) {
                    u32x8 a = load_u32x8(data + i + 0 * 8);
                    u32x8 b = load_u32x8(data + i + 1 * 8);
                    u32x8 c = load_u32x8(data + i + 2 * 8);
                    u32x8 d = load_u32x8(data + i + 3 * 8);

                    a = cum(a);
                    b = cum(b);
                    c = cum(c);
                    d = cum(d);

                    butterfly_u32x8_x4(a, b, c, d, wj3, wj4_0, wj4_1, data + i, data + i + 8, data + i + 16, data + i + 24);
                }
                k += 5;
            } else {
                for (int i = 0; i < n; i += 64) {
                    u32x8 a0 = load_u32x8(data + i + 0 * 8);
                    u32x8 a1 = load_u32x8(data + i + 1 * 8);
                    u32x8 a2 = load_u32x8(data + i + 2 * 8);
                    u32x8 a3 = load_u32x8(data + i + 3 * 8);
                    u32x8 a4 = load_u32x8(data + i + 4 * 8);
                    u32x8 a5 = load_u32x8(data + i + 5 * 8);
                    u32x8 a6 = load_u32x8(data + i + 6 * 8);
                    u32x8 a7 = load_u32x8(data + i + 7 * 8);

                    a0 = cum(a0);
                    a1 = cum(a1);
                    a2 = cum(a2);
                    a3 = cum(a3);
                    a4 = cum(a4);
                    a5 = cum(a5);
                    a6 = cum(a6);
                    a7 = cum(a7);

                    butterfly_u32x8_x8(a0, a1, a2, a3, a4, a5, a6, a7,
                                       wj3, wj4_0, wj4_1, wj5_0, wj5_1, wj5_2, wj5_3,
                                       data + i + 0, data + i + 8, data + i + 16, data + i + 24, data + i + 32, data + i + 40, data + i + 48, data + i + 56);
                }
                k += 6;
            }
        }

        for (; k + 1 < lg; k += 2) {
            for (int i = 0; i < n; i += (1 << k + 2)) {
                for (int j = 0; j < (1 << k); j += 8) {
                    u32x8 a = load_u32x8(data + i + 0 * (1 << k) + j);
                    u32x8 b = load_u32x8(data + i + 1 * (1 << k) + j);
                    u32x8 c = load_u32x8(data + i + 2 * (1 << k) + j);
                    u32x8 d = load_u32x8(data + i + 3 * (1 << k) + j);

                    u32x8 wj = load_u32x8(w[k] + j);
                    u32x8 wj1 = load_u32x8(w[k + 1] + j);
                    u32x8 wj2 = load_u32x8(w[k + 1] + (1 << k) + j);

                    butterfly_u32x8_x4(a, b, c, d, wj, wj1, wj2,
                                       data + i + 0 * (1 << k) + j,
                                       data + i + 1 * (1 << k) + j,
                                       data + i + 2 * (1 << k) + j,
                                       data + i + 3 * (1 << k) + j);
                }
            }
        }
        for (; k < lg; k++) {
            for (int i = 0; i < n; i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j += 8) {
                    u32x8 a = load_u32x8(data + i + j), b = load_u32x8(data + i + (1 << k) + j);
                    u32x8 wj = load_u32x8(w[k] + j);
                    butterfly_u32x8_x2(a, b, wj, data + i + j, data + i + (1 << k) + j);
                }
            }
        }

        std::reverse(data + 1, data + n);
        u32x8 rv = set1_u32x8(power(n, mod - 2));
        for (int i = 0; i < n; i += 8) {
            // data[i] = bt.shrink(bt.mod_42(u64(rv) * data[i]));
            u32x8 a = load_u32x8(data + i);
            store_u32x8(data + i, bts.shrink(bts.mul_mod_42(a, rv)));
        }
    }

    std::vector<u32> convolve_slow(std::vector<u32> a, std::vector<u32> b) {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        std::vector<u32> c(sz);
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                // c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % mod;
                c[i + j] = bt.shrink(c[i + j] + bt.mod_21(u64(a[i]) * b[j]));
            }
        }

        return c;
    }

    void convolve(int sz, __restrict__ u32 *a, __restrict__ u32 *b) {
        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        assert(sz <= (1 << lg));

        fft(lg, a);
        fft(lg, b);

        if (lg < 3) {
            for (int i = 0; i < (1 << lg); i++) {
                a[i] = bt.mod_42(u64(bt.shrink_2(a[i])) * bt.shrink_2(b[i]));
            }
        } else {
            for (int i = 0; i < (1 << lg); i += 8) {
                // a[i] = bt.mod_42(u64(bt.shrink_2(a[i])) * bt.shrink_2(b[i]));
                u32x8 ai = load_u32x8(a + i);
                u32x8 bi = load_u32x8(b + i);
                store_u32x8(a + i, bts.mul_mod_42(bts.shrink_2(ai), bts.shrink_2(bi)));
            }
        }

        ifft(lg, a);
    }

    std::vector<u32> convolve(std::vector<u32> a, std::vector<u32> b) {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        u32 *ap = (u32 *)std::aligned_alloc(64, std::max(64, 4 << lg));
        u32 *bp = (u32 *)std::aligned_alloc(64, std::max(64, 4 << lg));
        memset(ap, 0, 4 << lg);
        memset(bp, 0, 4 << lg);
        std::copy(a.begin(), a.end(), ap);
        std::copy(b.begin(), b.end(), bp);

        convolve(1 << lg, ap, bp);

        std::vector<u32> res(ap, ap + sz);
        std::free(ap);
        std::free(bp);
        return res;
    }

    void free_mem() {
        for (u32 *ptr : w) {
            free(ptr);
        }
        w.clear();
        w.shrink_to_fit();
        for (u32 *ptr : w_rb) {
            free(ptr);
        }
        w_rb.clear();
        w_rb.shrink_to_fit();
    }
};  // namespace FFT

#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <iostream>

using u32 = uint32_t;
using u64 = uint64_t;

// io from https://judge.yosupo.jp/submission/142782

namespace QIO_base {
    constexpr int O_buffer_default_size = 1 << 18;
    constexpr int O_buffer_default_flush_threshold = 40;
    struct _int_to_char_tab {
        char tab[40000];
        constexpr _int_to_char_tab() : tab() {
            for (int i = 0; i != 10000; ++i) {
                for (int j = 3, n = i; ~j; --j) {
                    tab[i * 4 + j] = n % 10 + 48, n /= 10;
                }
            }
        }
    } constexpr _otab;
}  // namespace QIO_base
namespace QIO_I {
    using namespace QIO_base;
    struct Qinf {
        FILE *f;
        char *bg, *ed, *p;
        struct stat Fl;
        Qinf(FILE *fi) : f(fi) {
            int fd = fileno(f);
            fstat(fd, &Fl);
            bg = (char *)mmap(0, Fl.st_size + 1, PROT_READ, MAP_PRIVATE, fd, 0);
            p = bg, ed = bg + Fl.st_size;
            madvise(p, Fl.st_size + 1, MADV_SEQUENTIAL);
        }
        ~Qinf() { munmap(bg, Fl.st_size + 1); }
        void skip_space() {
            while (*p <= ' ') {
                ++p;
            }
        }
        char get() { return *p++; }
        char seek() { return *p; }
        bool eof() { return p == ed; }
        Qinf &read(char *s, size_t count) { return memcpy(s, p, count), p += count, *this; }
        Qinf &operator>>(u32 &x) {
            skip_space(), x = 0;
            for (; *p > ' '; ++p) {
                x = x * 10 + (*p & 0xf);
            }
            return *this;
        }
        Qinf &operator>>(int &x) {
            skip_space();
            if (*p == '-') {
                for (++p, x = 48 - *p++; *p > ' '; ++p) {
                    x = x * 10 - (*p ^ 48);
                }
            } else {
                for (x = *p++ ^ 48; *p > ' '; ++p) {
                    x = x * 10 + (*p ^ 48);
                }
            }
            return *this;
        }
    } qin(stdin);
}  // namespace QIO_I
namespace QIO_O {
    using namespace QIO_base;
    struct Qoutf {
        FILE *f;
        char *bg, *ed, *p;
        char *ed_thre;
        int fp;
        u64 _fpi;
        Qoutf(FILE *fo, size_t sz = O_buffer_default_size) : f(fo), bg(new char[sz]), ed(bg + sz), p(bg), ed_thre(ed - O_buffer_default_flush_threshold), fp(6), _fpi(1000000ull) {}
        void flush() { fwrite_unlocked(bg, 1, p - bg, f), p = bg; }
        void chk() {
            if (__builtin_expect(p > ed_thre, 0)) {
                flush();
            }
        }
        ~Qoutf() {
            flush();
            delete[] bg;
        }
        void put4(u32 x) {
            if (x > 99u) {
                if (x > 999u) {
                    memcpy(p, _otab.tab + (x << 2) + 0, 4), p += 4;
                } else {
                    memcpy(p, _otab.tab + (x << 2) + 1, 3), p += 3;
                }
            } else {
                if (x > 9u) {
                    memcpy(p, _otab.tab + (x << 2) + 2, 2), p += 2;
                } else {
                    *p++ = x ^ 48;
                }
            }
        }
        void put2(u32 x) {
            if (x > 9u) {
                memcpy(p, _otab.tab + (x << 2) + 2, 2), p += 2;
            } else {
                *p++ = x ^ 48;
            }
        }
        Qoutf &write(const char *s, size_t count) {
            if (count > 1024 || p + count > ed_thre) {
                flush(), fwrite_unlocked(s, 1, count, f);
            } else {
                memcpy(p, s, count), p += count, chk();
            }
            return *this;
        }
        Qoutf &operator<<(char ch) { return *p++ = ch, *this; }
        Qoutf &operator<<(u32 x) {
            if (x > 99999999u) {
                put2(x / 100000000u), x %= 100000000u;
                memcpy(p, _otab.tab + ((x / 10000u) << 2), 4), p += 4;
                memcpy(p, _otab.tab + ((x % 10000u) << 2), 4), p += 4;
            } else if (x > 9999u) {
                put4(x / 10000u);
                memcpy(p, _otab.tab + ((x % 10000u) << 2), 4), p += 4;
            } else {
                put4(x);
            }
            return chk(), *this;
        }
        Qoutf &operator<<(int x) {
            if (x < 0) {
                *p++ = '-', x = -x;
            }
            return *this << static_cast<u32>(x);
        }
    } qout(stdout);
}  // namespace QIO_O
namespace QIO {
    using QIO_I::qin;
    using QIO_I::Qinf;
    using QIO_O::qout;
    using QIO_O::Qoutf;
}  // namespace QIO
using namespace QIO;

alignas(64) u32 a[1 << 20];
alignas(64) u32 b[1 << 20];

int32_t main() {
    int n, m;
    qin >> n >> m;

    // u32 *a = (u32 *)aligned_alloc(4 << 20, 64);
    // u32 *b = (u32 *)aligned_alloc(4 << 20, 64);

    // u32 *a = (u32 *)malloc((4 << 20) + 64);
    // u32 *b = (u32 *)malloc((4 << 20) + 64);
    // a += 64 - (RC(u64, a) & 63) & 63;
    // b += 64 - (RC(u64, b) & 63) & 63;

    for (int i = 0; i < n; i++) {
        qin >> a[i];
    }
    for (int i = 0; i < m; i++) {
        qin >> b[i];
    }
    FFT::convolve(n + m - 1, a, b);
    for (int i = 0; i < (n + m - 1); i++) {
        qout << a[i] << " \n"[i + 1 == (n + m - 1)];
    }

    return 0;
}
