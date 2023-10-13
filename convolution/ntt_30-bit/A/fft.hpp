#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>

#include "barrett.hpp"

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
                if (r <= 3) {
                    for (int j = 0; j < (1 << r - 1); j++) {
                        w[r][2 * j] = w[r - 1][j];
                        w[r][2 * j + 1] = bt.shrink(bt.mod_22(u64(f) * w[r - 1][j]));
                    }
                } else {
                    u64x4 ff = set1_u64x4(f);
                    for (int j = 0; j < (1 << r - 1); j += 4) {
                        u32x4 val = load_u32x4(w[r - 1] + j);
                        u32x8 a = RC(u32x8, _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(RC(i128, val)), _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3)));
                        u32x8 b = bts.shrink(RC(u32x8, bts.mul_mod_22(RC(u64x4, a), ff)));
                        b = shuffle_u32x8<0b10'00'00'00>(b);
                        u32x8 c = blend_u32x8<0b10'10'10'10>(a, b);
                        store_u32x8(w[r] + 2 * j, c);
                    }
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
                if (r <= 3) {
                    for (int j = 0; j < (1 << r - 1); j++) {
                        w_rb[r][j] = w_rb[r - 1][j];
                        w_rb[r][j + (1 << r - 1)] = bt.shrink(bt.mod_22(u64(f) * w_rb[r - 1][j]));
                    }
                } else {
                    u32x8 ff = set1_u32x8(f);
                    for (int j = 0; j < (1 << r - 1); j += 8) {
                        u32x8 val = load_u32x8(w_rb[r - 1] + j);
                        u32x8 val2 = bts.shrink(bts.mul_mod_22(val, ff));
                        store_u32x8(w_rb[r] + j, val);
                        store_u32x8(w_rb[r] + (1 << r - 1) + j, val2);
                    }
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
