#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>

#include "barrett.hpp"

namespace FFT {
    const u32 mod = 998'244'353;
    const u32 pr_root = 3;

    Barrett bt(mod);
    Barrett_simd bts(mod);

    u32 power(u32 base, u32 exp) {
        u32 res = 1;
        for (; exp > 0; exp >>= 1) {
            if (exp & 1) {
                res = bt.mul_mod_21(res, base);
            }
            base = bt.mul_mod_21(base, base);
        }
        res = bt.shrink(res);
        return res;
    }

    std::vector<u32 *> w;
    u32 *w_rb = nullptr;
    size_t w_rb_sz;

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
                        w[r][2 * j + 1] = bt.mul_mod_21(f, w[r - 1][j]);
                    }
                } else {
                    u64x4 ff = set1_u64x4(f);
                    for (int j = 0; j < (1 << r - 1); j += 4) {
                        u32x4 val = load_u32x4(w[r - 1] + j);
                        u32x8 a = RC(u32x8, _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(RC(i128, val)), _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3)));
                        u32x8 b = RC(u32x8, bts.mul_mod_21(RC(u64x4, a), ff));
                        b = shuffle_u32x8<0b10'00'00'00>(b);
                        u32x8 c = blend_u32x8<0b10'10'10'10>(a, b);
                        store_u32x8(w[r] + 2 * j, c);
                    }
                }
            }
        }
    }

    void expand_w_rb(int k) {
        if (w_rb_sz < (1 << k - 1)) {
            if (w_rb != nullptr) {
                free(w_rb);
            }
            w_rb_sz = 1 << k - 1;
            w_rb = (u32 *)std::aligned_alloc(64, std::max(64, 4 << k - 1));
            w_rb[0] = 1;
            for (int r = 1; r < k; r++) {
                u32 f = power(pr_root, (mod - 1) >> r + 1);
                if (r <= 3) {
                    for (int j = 0; j < (1 << r - 1); j++) {
                        w_rb[j + (1 << r - 1)] = bt.mul_mod_21(f, w_rb[j]);
                    }
                } else {
                    u32x8 ff = set1_u32x8(f);
                    for (int j = 0; j < (1 << r - 1); j += 8) {
                        u32x8 val = load_u32x8(w_rb + j);
                        val = bts.mul_mod_21(val, ff);
                        store_u32x8(w_rb + (1 << r - 1) + j, val);
                    }
                }
            }
        }
    }

    std::vector<u32> w_cum, w_cum2, w_cum4;

    void expand_cum(int k) {
        while (w_cum.size() < k) {
            int r = w_cum.size();
            u32 f = power(pr_root, (mod - 1) >> r + 2);
            u32 res = bt.mul_mod_21(f, power(power(f, (1 << r + 1) - 2), mod - 2));
            w_cum.push_back(res);
        }
        while (w_cum2.size() < k) {
            int r = w_cum2.size();
            u32 f = power(pr_root, (mod - 1) >> r + 3);
            u32 res = bt.mul_mod_21(f, power(power(f, (1 << r + 1) - 2), mod - 2));
            w_cum2.push_back(res);
        }
        while (w_cum4.size() < k) {
            int r = w_cum4.size();
            u32 f = power(pr_root, (mod - 1) >> r + 4);
            u32 res = bt.mul_mod_21(f, power(power(f, (1 << r + 1) - 2), mod - 2));
            w_cum4.push_back(res);
        }
        // expand_w_rb(k);
        // for (int i = 0; i + 1 < (1 << k - 1); i++) {
        //     int a = __builtin_ctz(~i);
        //     u32 x = bt.mul_mod_21(w_rb[i + 1], power(w_rb[i], mod - 2));
        //     assert(x == w_cum[a]);
        // }
    }

    INL void butterfly_u32_x2(u32 a, u32 b, u32 wj, u32 *ap, u32 *bp) {
        a = bt.shrink(a);
        u32 c = bt.mul_mod_21(b, wj);
        ap[0] = a + c;
        bp[0] = a + bt.mod - c;
    }

    INL std::array<u32x8, 2> butterfly_u32x8_x2(u32x8 a, u32x8 b, u32x8 wj) {
        a = bts.shrink(a);
        u32x8 c = bts.mul_mod_21(b, wj);
        return {a + c, a + bts.v_mod - c};
    }

    INL void butterfly_u32x8_x2(u32x8 a, u32x8 b, u32x8 wj, u32 *ap, u32 *bp) {
        a = bts.shrink(a);
        u32x8 c = bts.mul_mod_21(b, wj);
        store_u32x8(ap, a + c);
        store_u32x8(bp, a + bts.v_mod - c);
    }

    INL void butterfly_u32x8_x4(u32x8 a, u32x8 b, u32x8 c, u32x8 d, u32x8 wj, u32x8 wj1, u32x8 wj2, u32 *ap, u32 *bp, u32 *cp, u32 *dp) {
        a = bts.shrink(a);
        c = bts.shrink(c);
        u32x8 p = bts.mul_mod_21(b, wj);
        u32x8 q = bts.mul_mod_21(d, wj);

        u32x8 a2 = a + p;
        u32x8 b2 = a + bts.v_mod - p;
        u32x8 c2 = c + q;
        u32x8 d2 = c + bts.v_mod - q;

        butterfly_u32x8_x2(a2, c2, wj1, ap, cp);
        butterfly_u32x8_x2(b2, d2, wj2, bp, dp);
    }

    [[gnu::noinline]] void fft_cum(int lg, u32 *data) {
        const int n = 1 << lg;

        u32 f0 = 1;
        std::array<u32, 4> w_rb;
        w_rb[0] = 1;
        w_rb[1] = power(pr_root, (mod - 1) / 4);
        w_rb[2] = power(pr_root, (mod - 1) / 8);
        w_rb[3] = bt.mul_mod_21(w_rb[1], w_rb[2]);
        u64x4 wj1 = RC(u64x4, _mm256_setr_epi64x(w_rb[0], w_rb[0], w_rb[1], w_rb[1]));
        u64x4 wj2 = RC(u64x4, _mm256_setr_epi64x(w_rb[0], w_rb[1], w_rb[2], w_rb[3]));

        for (int i = 0; i < n; i += 8) {
            // u64x4 wj0 = set1_u64x4(w_rb[i >> 3]);
            u64x4 wj0 = set1_u64x4(f0);
            const int c = __builtin_ctz(~(i >> 3));
            f0 = bt.mod_22(u64(f0) * w_cum[c]);

            // u64x4 wj1 = RC(u64x4, _mm256_setr_epi64x(w_rb[i >> 2], w_rb[i >> 2], w_rb[i + 4 >> 2], w_rb[i + 4 >> 2]));

            // u64x4 wj2 = RC(u64x4, _mm256_setr_epi64x(w_rb[i >> 1], w_rb[i + 2 >> 1], w_rb[i + 4 >> 1], w_rb[i + 6 >> 1]));

            auto cum = [&](u32x8 val, int i) {
                {
                    val = bts.shrink(val);
                    u64x4 x4567 = RC(u64x4, permute_u32x8(val, setr_u32x8(4, -1, 5, -1, 6, -1, 7, -1)));
                    u32x8 c = RC(u32x8, permute_u32x8(RC(u32x8, bts.mul_mod_21(x4567, wj0)), setr_u32x8(0, 2, 4, 6, 0, 2, 4, 6)));

                    c = blend_u32x8<0xf0>(c, bts.v_mod - c);
                    val = permute_u32x8_epi128<4>(val, val) + c;
                }
                {
                    val = bts.shrink(val);
                    u64x4 x2367 = RC(u64x4, shuffle_u32x8<0b00'11'00'10>(val));
                    u32x8 c = shuffle_u32x8<0b10'00'10'00>(RC(u32x8, bts.mul_mod_21(x2367, wj1)));

                    c = blend_u32x8<0b11'00'11'00>(c, bts.v_mod - c);
                    val = shuffle_u32x8<0b01'00'01'00>(val) + c;
                }
                {
                    val = bts.shrink(val);
                    u64x4 x1357 = RC(u64x4, shuffle_u32x8<0b00'11'00'01>(val));
                    u32x8 c = shuffle_u32x8<0b10'10'00'00>(RC(u32x8, bts.mul_mod_21(x1357, wj2)));

                    c = blend_u32x8<0b10'10'10'10>(c, bts.v_mod - c);
                    val = shuffle_u32x8<0b10'10'00'00>(val) + c;
                }

                return val;
            };

            u32x8 a = load_u32x8(data + i);
            a = cum(a, i);
            store_u32x8(data + i, a);

            wj1 = RC(u64x4, RC(u32x8, bts.mul_mod_22(wj1, set1_u64x4(w_cum2[c]))));
            wj2 = RC(u64x4, RC(u32x8, bts.mul_mod_22(wj2, set1_u64x4(w_cum4[c]))));
        }
    }

    // input data[i] in [0, 2 * mod)
    // result data[i] in [0, 2 * mod)
    void fft(int lg, u32 *data) {
        const int n = 1 << lg;

        if (lg <= 5) {
            expand_w_rb(lg);
            for (int k = lg - 1; k >= 0; k--) {
                for (int i = 0; i < n; i += (1 << k + 1)) {
                    u32 wj = w_rb[i >> k + 1];
                    for (int j = 0; j < (1 << k); j++) {
                        u32 a = data[i + j], b = data[i + (1 << k) + j];
                        butterfly_u32_x2(a, b, wj, data + i + j, data + i + (1 << k) + j);
                    }
                }
            }
        } else {
            expand_cum(lg);

            u32 w_rb1 = power(pr_root, (mod - 1) / 4);

            int k = lg - 1;
            for (; k > 2; k--) {
                for (int i = 0; i < (1 << k); i += 8) {
                    u32x8 a = load_u32x8(data + i);
                    u32x8 b = load_u32x8(data + (1 << k) + i);
                    a = bts.shrink(a);
                    b = bts.shrink(b);
                    store_u32x8(data + i, a + b);
                    store_u32x8(data + (1 << k) + i, a + bts.v_mod - b);
                }
                u32 f = w_cum[0];
                for (int i = (1 << k + 1); i < n; i += (1 << k + 1)) {
                    // u32x8 wj = set1_u32x8(w_rb[i >> k + 1]);
                    u32x8 wj = set1_u32x8(f);
                    f = bt.mul_mod_21(f, w_cum[__builtin_ctz(~(i >> k + 1))]);
                    for (int j = 0; j < (1 << k); j += 8) {
                        u32x8 a = load_u32x8(data + i + j), b = load_u32x8(data + i + (1 << k) + j);
                        butterfly_u32x8_x2(a, b, wj, data + i + j, data + i + (1 << k) + j);
                    }
                }
            }

            assert(k == 2);
            fft_cum(lg, data);
        }
    }

    // input data[i] in [0, 2 * mod)
    // result data[i] in [0, mod)
    void ifft(int lg, u32 *data) {
        int n = 1 << lg;
        expand_w(lg);

        if (lg <= 6) {
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
                data[i] = bt.shrink(data[i]);
                // data[i] = bt.mul_mod_21(rv, data[i]);
            }

            return;
        }

        int k = 0;
        {
            // u32x8 wj0 = set1_u32x8(w[0][0]); // * w[0][0] == 1

            u64x4 wj1 = RC(u64x4, _mm256_setr_epi64x(w[1][0], w[1][1], w[1][0], w[1][1]));
            u64x4 wj2 = RC(u64x4, _mm256_setr_epi64x(w[2][0], w[2][1], w[2][2], w[2][3]));
            u32x8 wj3 = load_u32x8(w[3]);

            auto cum = [&](u32x8 val) {
                {
                    val = bts.shrink(val);

                    u32x8 c = blend_u32x8<0b10'10'10'10>(val, bts.v_mod - val);
                    val = shuffle_u32x8<0b10'11'00'01>(val) + c;
                }
                {
                    val = bts.shrink(val);
                    u64x4 x2367 = RC(u64x4, shuffle_u32x8<0b00'11'00'10>(val));
                    u32x8 c = shuffle_u32x8<0b10'00'10'00>(RC(u32x8, bts.mul_mod_21(x2367, wj1)));

                    c = blend_u32x8<0b11'00'11'00>(c, bts.v_mod - c);
                    val = shuffle_u32x8<0b01'00'01'00>(val) + c;
                }
                {
                    val = bts.shrink(val);
                    u64x4 x4567 = RC(u64x4, permute_u32x8(val, setr_u32x8(4, -1, 5, -1, 6, -1, 7, -1)));
                    u32x8 c = permute_u32x8(RC(u32x8, bts.mul_mod_21(x4567, wj2)), setr_u32x8(0, 2, 4, 6, 0, 2, 4, 6));

                    c = blend_u32x8<0xf0>(c, bts.v_mod - c);
                    val = permute_u32x8_epi128<4>(val, val) + c;
                }

                return val;
            };

            for (int i = 0; i < n; i += 16) {
                u32x8 a = load_u32x8(data + i), b = load_u32x8(data + i + 8);

                a = cum(a);
                b = cum(b);

                butterfly_u32x8_x2(a, b, wj3, data + i, data + i + 8);
            }
            k += 4;
        }

        for (; k < lg - 1; k++) {
            for (int i = 0; i < n; i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j += 8) {
                    u32x8 a = load_u32x8(data + i + j), b = load_u32x8(data + i + (1 << k) + j);
                    u32x8 wj = load_u32x8(w[k] + j);

                    butterfly_u32x8_x2(a, b, wj, data + i + j, data + i + (1 << k) + j);
                }
            }
        }
        assert(k == lg - 1);
        for (int i = 0; i < n; i += (1 << k + 1)) {
            for (int j = 0; j < (1 << k); j += 8) {
                u32x8 a = load_u32x8(data + i + j), b = load_u32x8(data + i + (1 << k) + j);
                u32x8 wj = load_u32x8(w[k] + j);

                // butterfly_u32x8_x2(a, b, wj, data + i + j, data + i + (1 << k) + j);
                auto [p, q] = butterfly_u32x8_x2(a, b, wj);

                store_u32x8(data + i + j, bts.shrink(p));
                store_u32x8(data + i + (1 << k) + j, bts.shrink(q));
            }
        }

        std::reverse(data + 1, data + n);

        // ! this part is done right in the "convolve" function
        // u32x8 rv = set1_u32x8(power(n, mod - 2));
        // for (int i = 0; i < n; i += 8) {
        //     u32x8 a = load_u32x8(data + i);
        //     store_u32x8(data + i, bts.mul_mod_21(a, rv));
        // }
    }

    std::vector<u32> convolve_slow(std::vector<u32> a, std::vector<u32> b) {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        std::vector<u32> c(sz);
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                // c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % mod;
                c[i + j] = bt.shrink(c[i + j] + bt.mul_mod_21(u64(a[i]), b[j]));
            }
        }

        return c;
    }

    [[gnu::noinline]] void convolve(int sz, __restrict__ u32 *a, __restrict__ u32 *b) {
        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        assert(sz <= (1 << lg));

        fft(lg, a);
        fft(lg, b);

        u32x8 rv = set1_u32x8(power(1 << lg, mod - 2));
        if (lg <= 3) {
            for (int i = 0; i < (1 << lg); i++) {
                a[i] = bt.mul_mod_21(bt.shrink(a[i]), b[i]);
                a[i] = bt.mul_mod_21(a[i], rv[0]);
            }
        } else {
            // #pragma GCC unroll 2
            for (int i = 0; i < (1 << lg); i += 8) {
                u32x8 ai = load_u32x8(a + i);
                u32x8 bi = load_u32x8(b + i);
                u32x8 res = bts.mul_mod_21(bts.shrink(ai), bi);
                res = bts.mul_mod_22(res, rv);
                store_u32x8(a + i, res);
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
        if (w_rb) {
            free(w_rb);
            w_rb = nullptr;
            w_rb_sz = 0;
        }
    }
};  // namespace FFT
