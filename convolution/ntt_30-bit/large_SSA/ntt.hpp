#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "montgomery.hpp"

struct NTT {
    alignas(32) Montgomery_simd mts;
    Montgomery mt;
    u32 mod, g;

    [[gnu::noinline]] u32 power(u32 base, u32 exp) const {
        const auto mt = this->mt;  // ! to put Montgomery constants in registers
        u32 res = mt.r;
        for (; exp > 0; exp >>= 1) {
            if (exp & 1) {
                res = mt.mul(res, base);
            }
            base = mt.mul(base, base);
        }
        return mt.shrink(res);
    }

    // mod should be prime
    u32 find_pr_root(u32 mod) const {
        u32 m = mod - 1;
        std::vector<u32> vec;
        for (u32 i = 2; u64(i) * i <= m; i++) {
            if (m % i == 0) {
                vec.push_back(i);
                do {
                    m /= i;
                } while (m % i == 0);
            }
        }
        if (m != 1) {
            vec.push_back(m);
        }
        for (u32 i = 2;; i++) {
            if (std::all_of(vec.begin(), vec.end(), [&](u32 f) { return mt.r != power(mt.mul(i, mt.r2), (mod - 1) / f); })) {
                return i;
            }
        }
    }

    u32x8 get_powers_u32x8(u32 x) const {
        u32 x2 = mt.mul<true>(x, x);
        u32 x3 = mt.mul<true>(x, x2);
        u32 x4 = mt.mul<true>(x2, x2);
        return setr_u32x8(mt.r, x, x2, x3, x4, mt.mul<true>(x2, x3), mt.mul<true>(x3, x3), mt.mul<true>(x4, x3));
    }

    static constexpr int LG = 30;
    alignas(32) u32 w[4], w_r[4];
    alignas(32) u64x4 w_cum_x4[LG], w_rcum_x4[LG];
    alignas(32) u32x8 w_cum_x8[LG], w_rcum_x8[LG];

    NTT(u32 mod = 998'244'353) : mt(mod), mts(mod), mod(mod) {
        const auto mt = this->mt;  // ! to put Montgomery constants in registers

        g = mt.mul<true>(mt.r2, find_pr_root(mod));

        for (int i = 0; i < LG; i++) {
            u32 f = power(g, (mod - 1) >> i + 3);
            u32 res = mt.mul<true>(f, power(power(f, (1 << i + 1) - 2), mod - 2));
            u32 res_r = power(res, mod - 2);
            w_cum_x4[i] = setr_u64x4(res, power(res, 2), res, power(res, 3));
            w_rcum_x4[i] = setr_u64x4(res_r, power(res_r, 2), res_r, power(res_r, 3));
        }
        for (int i = 0; i < LG; i++) {
            u32 f = power(g, (mod - 1) >> i + 4);
            f = mt.mul<true>(f, power(power(f, (1 << i + 1) - 2), mod - 2));
            u32 f1 = f;
            u32 f2 = mt.mul<true>(f1, f1);
            u32 f4 = mt.mul<true>(f2, f2);
            w_cum_x8[i] = setr_u32x8(f, f2, f, f4, f, f2, f, f4);

            u32 f_r = power(f, mod - 2);
            w_rcum_x8[i][0] = mt.r;
            for (int j = 1; j < 8; j++) {
                w_rcum_x8[i][j] = mt.mul<true>(w_rcum_x8[i][j - 1], f_r);
            }
        }

        u32 w18 = power(g, (mod - 1) / 8);
        w[0] = mt.r, w[1] = power(w18, 2), w[2] = w18, w[3] = power(w18, 3);
        u32 w78 = power(w18, 7);  // == w18 ^ (-1)
        w_r[0] = mt.r, w_r[1] = power(w78, 2), w_r[2] = w78, w_r[3] = power(w78, 3);
    }

    // input data[i] in [0, 2 * mod)
    // output data[i] in [0, 4 * mod)
    [[gnu::noinline]] __attribute__((optimize("O3"))) void ntt(int lg, u32 *data) const {
        const auto mt = this->mt;    // ! to put Montgomery constants in registers
        const auto mts = this->mts;  // ! to put Montgomery constants in registers

        int n = 1 << lg;
        int k = lg;

        if (lg % 2 == 0) {
            for (int i = 0; i < n / 2; i += 8) {
                u32x8 a = load_u32x8(data + i);
                u32x8 b = load_u32x8(data + n / 2 + i);

                store_u32x8(data + i, mts.shrink2(a + b));
                store_u32x8(data + n / 2 + i, mts.shrink2_n(a - b));
            }
            k--;
        }

        assert(k % 2 == 1);
        for (; k > 4; k -= 2) {
            u64x4 wj_cum = set1_u64x4(mt.r);
            u32x8 w_1 = set1_u32x8(w[1]);
            for (int i = 0; i < n; i += (1 << k)) {
                u32x8 w1 = shuffle_u32x8<0b00'00'00'00>((u32x8)wj_cum);
                u32x8 w2 = permute_u32x8((u32x8)wj_cum, set1_u32x8(2));
                u32x8 w3 = permute_u32x8((u32x8)wj_cum, set1_u32x8(6));
                wj_cum = mts.mul<true>(wj_cum, w_cum_x4[__builtin_ctz(~(i >> k))]);

                for (int j = 0; j < (1 << k - 2); j += 8) {
                    u32x8 a = load_u32x8(data + i + 0 * (1 << k - 2) + j);
                    u32x8 b = load_u32x8(data + i + 1 * (1 << k - 2) + j);
                    u32x8 c = load_u32x8(data + i + 2 * (1 << k - 2) + j);
                    u32x8 d = load_u32x8(data + i + 3 * (1 << k - 2) + j);

                    a = mts.shrink2(a);
                    b = mts.mul<false, true>(b, w1),
                    c = mts.mul<false, true>(c, w2),
                    d = mts.mul<false, true>(d, w3);

                    u32x8 a1 = mts.shrink2(a + c), b1 = mts.shrink2(b + d),
                          c1 = mts.shrink2_n(a - c),
                          d1 = mts.mul<false, true>(b + mts.mod2 - d, w_1);

                    store_u32x8(data + i + 0 * (1 << k - 2) + j, a1 + b1);
                    store_u32x8(data + i + 1 * (1 << k - 2) + j, a1 + mts.mod2 - b1);
                    store_u32x8(data + i + 2 * (1 << k - 2) + j, c1 + d1);
                    store_u32x8(data + i + 3 * (1 << k - 2) + j, c1 + mts.mod2 - d1);
                }
            }
        }

        assert(k == 3);

        {
            // *  { w, w^2, w * w_1, w^4,  |  w * w_2, w^2 * w_1, w * w_3, w^4 }
            u32x8 cum = setr_u32x8(w[0], w[0], w[1], w[0], w[2], w[1], w[3], w[0]);

            int n_8 = n / 8;

            for (int i = 0; i < n_8; i++) {
                u32x8 vec = load_u32x8(data + i * 8);

                u32x8 wj0 = shuffle_u32x8<0b11'11'11'11>(cum);
                u32x8 wj1 = shuffle_u32x8<0b01'01'01'01>(cum);
                u32x8 wj2 = cum;  // no shuffle needed

                u32x8 bw;

                vec = mts.shrink2(vec);
                bw = permute_u32x8((u32x8)mts.mul_to_hi((u64x4)wj0, (u64x4)permute_u32x8(vec, setr_u32x8(4, -1, 5, -1, 6, -1, 7, -1))), setr_u32x8(1, 3, 5, 7, 1, 3, 5, 7));
                vec = permute_u32x8_epi128<0>(vec, vec) + blend_u32x8<0b11'11'00'00>(bw, mts.mod2 - bw);

                vec = mts.shrink2(vec);
                bw = shuffle_u32x8<0b11'01'11'01>((u32x8)mts.mul_to_hi((u64x4)wj1, (u64x4)shuffle_u32x8<0b00'11'00'10>(vec)));
                vec = shuffle_u32x8<0b01'00'01'00>(vec) + blend_u32x8<0b11'00'11'00>(bw, mts.mod2 - bw);

                vec = mts.shrink2(vec);
                bw = shuffle_u32x8<0b11'11'01'01>((u32x8)mts.mul_to_hi((u64x4)wj2, (u64x4)shuffle_u32x8<0b00'11'00'01>(vec)));
                vec = shuffle_u32x8<0b10'10'00'00>(vec) + blend_u32x8<0b10'10'10'10>(bw, mts.mod2 - bw);

                store_u32x8(data + i * 8, vec);

                cum = mts.mul(cum, w_cum_x8[__builtin_ctz(~i)]);
            }
        }
    }

    // input data[i] in [0, 2 * mod)
    // output data[i] in [0, mod)
    // fc (if specified) should be in [0, mod)
    // if fc is specified everything is multiplied by fc
    [[gnu::noinline]] __attribute__((optimize("O3"))) void intt(int lg, u32 *data, u32 fc = -1u) const {
        const auto mt = this->mt;    // ! to put Montgomery constants in registers
        const auto mts = this->mts;  // ! to put Montgomery constants in registers

        if (fc == -1u) {
            fc = mt.r;
        }

        int n = 1 << lg;
        int k = 1;
        {
            u32x8 cum0 = setr_u32x8(w_r[0], w_r[0], w_r[0], w_r[0], w_r[0], w_r[0], w_r[1], w_r[1]);
            u32x8 cum1 = setr_u32x8(w_r[0], w_r[0], w_r[0], w_r[1], w_r[0], w_r[2], w_r[0], w_r[3]);

            const u32 inv_2 = mt.mul<true>(mt.r2, (mod + 1) / 2);
            u32x8 cum = set1_u32x8(mt.mul<true>(fc, power(inv_2, lg)));

            int n_8 = n / 8;
            for (int i = 0; i < n_8; i++) {
                u32x8 vec = load_u32x8(data + i * 8);

                vec = mts.mul(cum1, blend_u32x8<0b10'10'10'10>(vec, mts.mod2 - vec) + shuffle_u32x8<0b10'11'00'01>(vec));
                vec = mts.mul(cum0, blend_u32x8<0b11'00'11'00>(vec, mts.mod2 - vec) + shuffle_u32x8<0b01'00'11'10>(vec));
                vec = mts.mul(cum, blend_u32x8<0b11'11'00'00>(vec, mts.mod2 - vec) + permute_u32x8_epi128<1>(vec, vec));

                store_u32x8(data + i * 8, vec);

                cum = mts.mul<true>(cum, w_rcum_x8[__builtin_ctz(~i)]);
            }
            k += 3;
        }

        for (; k + 1 <= lg; k += 2) {
            u64x4 wj_cum = set1_u64x4(mt.r);
            u32x8 w_1 = set1_u32x8(w_r[1]);

            for (int i = 0; i < n; i += (1 << k + 1)) {
                u32x8 w1 = shuffle_u32x8<0b00'00'00'00>((u32x8)wj_cum);
                u32x8 w2 = permute_u32x8((u32x8)wj_cum, set1_u32x8(2));
                u32x8 w3 = permute_u32x8((u32x8)wj_cum, set1_u32x8(6));
                wj_cum = mts.mul<true>(wj_cum, w_rcum_x4[__builtin_ctz(~(i >> k + 1))]);

                for (int j = 0; j < (1 << k - 1); j += 8) {
                    u32x8 a = load_u32x8(data + i + 0 * (1 << k - 1) + j);
                    u32x8 b = load_u32x8(data + i + 1 * (1 << k - 1) + j);
                    u32x8 c = load_u32x8(data + i + 2 * (1 << k - 1) + j);
                    u32x8 d = load_u32x8(data + i + 3 * (1 << k - 1) + j);

                    u32x8 a1 = mts.shrink2(a + b), b1 = mts.shrink2_n(a - b),
                          c1 = mts.shrink2(c + d), d1 = mts.mul<false, true>(c + mts.mod2 - d, w_1);

                    store_u32x8(data + i + 0 * (1 << k - 1) + j, mts.shrink2(a1 + c1));
                    store_u32x8(data + i + 1 * (1 << k - 1) + j, mts.mul<false, true>(b1 + d1, w1));
                    store_u32x8(data + i + 2 * (1 << k - 1) + j, mts.mul<false, true>(a1 + mts.mod2 - c1, w2));
                    store_u32x8(data + i + 3 * (1 << k - 1) + j, mts.mul<false, true>(b1 + mts.mod2 - d1, w3));
                }
            }
        }
        if (k == lg) {
            for (int i = 0; i < n / 2; i += 8) {
                u32x8 a = load_u32x8(data + i);
                u32x8 b = load_u32x8(data + n / 2 + i);

                store_u32x8(data + i, mts.shrink(mts.shrink2(a + b)));
                store_u32x8(data + n / 2 + i, mts.shrink(mts.shrink2_n(a - b)));
            }
        } else {
            for (int i = 0; i < n; i += 8) {
                u32x8 ai = load_u32x8(data + i);
                store_u32x8(data + i, mts.shrink(ai));
            }
        }
    }

    __attribute__((optimize("O3"))) std::vector<u32> convolve_slow(std::vector<u32> a, std::vector<u32> b) const {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);
        const auto mt = this->mt;  // ! to put Montgomery constants in registers

        std::vector<u32> c(sz);

        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                // c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % mod;
                c[i + j] = mt.shrink(c[i + j] + mt.mul<true>(mt.r2, mt.mul(a[i], b[j])));
            }
        }

        return c;
    }

    // a and b should be 32-byte aligned
    // writes (a * b) to a
    __attribute__((optimize("O3"))) [[gnu::noinline]] void convolve(int lg, __restrict__ u32 *a, __restrict__ u32 *b) const {
        if (lg <= 4) {
            int n = (1 << lg);
            __restrict__ u32 *c = (u32 *)_mm_malloc(n * 4, 4);
            memset(c, 0, 4 * n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    c[(i + j) & (n - 1)] = mt.shrink2(c[(i + j) & (n - 1)] + mt.mul(a[i], b[j]));
                }
            }
            for (int i = 0; i < n; i++) {
                a[i] = mt.mul<true>(mt.r2, c[i]);
            }
            _mm_free(c);
            return;
        }

        ntt(lg, a);
        ntt(lg, b);

        const auto mts = this->mts;  // ! to put Montgomery constants in registers
        for (int i = 0; i < (1 << lg); i += 8) {
            u32x8 ai = load_u32x8(a + i), bi = load_u32x8(b + i);
            store_u32x8(a + i, mts.mul(mts.shrink2(ai), mts.shrink2(bi)));
        }

        intt(lg, a, mt.r2);
    }

    __attribute__((optimize("O3"))) [[gnu::noinline]] void convolve_neg(int lg, u32 *__restrict__ a, u32 *__restrict__ b, u32 fc = 1u) const {
        auto mt = this->mt;
        auto mts = this->mts;

        fc = mt.mul<true>(mt.r2, fc);

        u32 w2 = power(g, mod - 1 >> lg + 1);
        if (lg <= 4) {
            u32 f = mt.r;
            for (int i = 0; i < (1 << lg); i++) {
                a[i] = mt.mul(a[i], f);
                b[i] = mt.mul(b[i], f);
                f = mt.mul(f, w2);
            }
        } else {
            u32x8 w2_8 = set1_u32x8(power(w2, 8));

            u32x8 fv[4];
            fv[0] = get_powers_u32x8(w2);

            fv[1] = mts.mul<true>(fv[0], w2_8);
            fv[2] = mts.mul<true>(fv[1], w2_8);
            fv[3] = mts.mul<true>(fv[2], w2_8);

            u32x8 w2_32 = set1_u32x8(power(w2, 32));

            for (int i = 0; i < (1 << lg); i += 32) {
                for (int j = 0; j < 4; j++) {
                    u32x8 aj = load_u32x8(a + i + 8 * j);
                    u32x8 bj = load_u32x8(b + i + 8 * j);
                    store_u32x8(a + i + 8 * j, mts.mul(aj, fv[j]));
                    store_u32x8(b + i + 8 * j, mts.mul(bj, fv[j]));
                    fv[j] = mts.mul(fv[j], w2_32);
                }
            }
        }
        convolve(lg, a, b);

        w2 = power(w2, mod - 2);
        if (lg <= 4) {
            u32 f = fc;
            for (int i = 0; i < (1 << lg); i++) {
                a[i] = mt.mul<true>(a[i], f);
                f = mt.mul(f, w2);
            }
        } else {
            u32x8 w2_8 = set1_u32x8(power(w2, 8));

            u32x8 fv[4];
            fv[0] = get_powers_u32x8(w2);
            fv[0] = mts.mul<true>(fv[0], set1_u32x8(fc));
            fv[1] = mts.mul<true>(fv[0], w2_8);
            fv[2] = mts.mul<true>(fv[1], w2_8);
            fv[3] = mts.mul<true>(fv[2], w2_8);

            u32x8 w2_32 = set1_u32x8(power(w2, 32));

            for (int i = 0; i < (1 << lg); i += 32) {
                for (int j = 0; j < 4; j++) {
                    u32x8 aj = load_u32x8(a + i + 8 * j);
                    store_u32x8(a + i + 8 * j, mts.mul<true>(aj, fv[j]));
                    fv[j] = mts.mul(fv[j], w2_32);
                }
            }
        }
    }

    __attribute__((optimize("O3"))) std::vector<u32> convolve(const std::vector<u32> &a, const std::vector<u32> &b) const {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        u32 *ap = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
        u32 *bp = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
        memset(ap, 0, 4 << lg);
        memset(bp, 0, 4 << lg);

        std::copy(a.begin(), a.end(), ap);
        std::copy(b.begin(), b.end(), bp);

        convolve(lg, ap, bp);

        std::vector<u32> res(ap, ap + sz);
        _mm_free(ap);
        _mm_free(bp);
        return res;
    }
};
