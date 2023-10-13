#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "montgomery.hpp"

// thanks https://judge.yosupo.jp/submission/142782
struct FFT {
    Montgomery mt;
    Montgomery_simd mts;
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

    u32 find_pr_root(u32 mod) const {
        assert(mod == 998'244'353);
        return 3;
    }

    FFT(u32 mod = 998'244'353) : mt(mod), mts(mod), mod(mod) {
        g = mt.mul<true>(mt.r2, find_pr_root(mod));
    }

    [[gnu::noinline]] std::pair<std::vector<u32>, u32x8 *> make_cum(int lg, bool inv = false) const {
        lg -= 2;

        const auto mt = this->mt;  // ! to put Montgomery constants in registers
        std::vector<u32> w_cum(lg);
        for (int i = 0; i < lg; i++) {
            u32 f = power(g, (mod - 1) >> i + 3);
            if (inv) {
                f = power(f, mod - 2);
            }
            u32 res = mt.mul<true>(f, power(power(f, (1 << i + 1) - 2), mod - 2));
            w_cum[i] = res;
        }
        u32x8 *w_cum_x8 = (u32x8 *)_mm_malloc(32 * lg, 32);

        for (int i = 0; i < lg; i++) {
            u32 f = power(g, (mod - 1) >> i + 4);
            if (inv) {
                f = power(f, mod - 2);
            }
            f = mt.mul<true>(f, power(power(f, (1 << i + 1) - 2), mod - 2));
            w_cum_x8[i][0] = mt.r;
            for (int j = 1; j < 8; j++) {
                w_cum_x8[i][j] = mt.mul<true>(w_cum_x8[i][j - 1], f);
            }
        }
        return {w_cum, w_cum_x8};
    }

    // input data[i] in [0, 4 * mod)
    // output data[i] in [0, 4 * mod)
    [[gnu::noinline]] __attribute__((optimize("O3"))) void fft(int lg, u32 *data) const {
        auto [w_cum, w_cum_x8] = make_cum(lg, false);
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
            u32 wj = mt.r;
            u32x8 w_1 = set1_u32x8(power(g, (mod - 1) / 4));
            for (int i = 0; i < n; i += (1 << k)) {
                u32 wj2 = mt.mul<true>(wj, wj);
                u32x8 w1 = set1_u32x8(wj);
                u32x8 w2 = set1_u32x8(wj2);
                u32x8 w3 = set1_u32x8(mt.mul<true>(wj, wj2));
                wj = mt.mul<true>(wj, w_cum[__builtin_ctz(~(i >> k))]);

                for (int j = 0; j < (1 << k - 2); j += 8) {
                    u32x8 a = load_u32x8(data + i + 0 * (1 << k - 2) + j);
                    u32x8 b = load_u32x8(data + i + 1 * (1 << k - 2) + j);
                    u32x8 c = load_u32x8(data + i + 2 * (1 << k - 2) + j);
                    u32x8 d = load_u32x8(data + i + 3 * (1 << k - 2) + j);

                    a = mts.shrink2(a);
                    b = mts.mul(b, w1), c = mts.mul(c, w2), d = mts.mul(d, w3);

                    u32x8 a1 = mts.shrink2(a + c), b1 = mts.shrink2(b + d),
                          c1 = mts.shrink2_n(a - c), d1 = mts.mul(b + mts.mod2 - d, w_1);

                    store_u32x8(data + i + 0 * (1 << k - 2) + j, a1 + b1);
                    store_u32x8(data + i + 1 * (1 << k - 2) + j, a1 + mts.mod2 - b1);
                    store_u32x8(data + i + 2 * (1 << k - 2) + j, c1 + d1);
                    store_u32x8(data + i + 3 * (1 << k - 2) + j, c1 + mts.mod2 - d1);
                }
            }
        }

        assert(k == 3);

        std::array<u32, 4> w;
        w[0] = mt.r;
        w[1] = power(g, (mod - 1) / 4);
        w[2] = power(g, (mod - 1) / 8);
        w[3] = mt.mul<true>(w[1], w[2]);

        u32x8 cum0 = setr_u32x8(w[0], w[0], w[0], w[0], w[0], w[0], w[1], w[1]);
        u32x8 cum1 = setr_u32x8(w[0], w[0], w[0], w[1], w[0], w[2], w[0], w[3]);
        u32x8 cum = set1_u32x8(mt.r);
        for (int i = 0; i < n; i += 8) {
            u32x8 vec = load_u32x8(data + i);

            vec = mts.mul(vec, cum);
            vec = mts.mul(blend_u32x8<0b11'11'00'00>(vec, mts.mod2 - vec) + permute_u32x8_epi128<1>(vec, vec), cum0);
            vec = mts.mul(blend_u32x8<0b11'00'11'00>(vec, mts.mod2 - vec) + shuffle_u32x8<0b01'00'11'10>(vec), cum1);
            vec = blend_u32x8<0b10'10'10'10>(vec, mts.mod2 - vec) + shuffle_u32x8<0b10'11'00'01>(vec);

            store_u32x8(data + i, vec);

            cum = mts.mul<true>(cum, w_cum_x8[__builtin_ctz(~(i >> 3))]);
        }

        _mm_free(w_cum_x8);
    }

    // input data[i] in [0, 4 * mod)
    // output data[i] in [0, mod)
    [[gnu::noinline]] __attribute__((optimize("O3"))) void ifft(int lg, u32 *data) const {
        auto [w_cum, w_cum_x8] = make_cum(lg, true);
        const auto mt = this->mt;    // ! to put Montgomery constants in registers
        const auto mts = this->mts;  // ! to put Montgomery constants in registers

        int n = 1 << lg;
        int k = 1;
        {
            std::array<u32, 4> w;
            w[0] = mt.r;
            w[1] = power(power(g, mod - 2), (mod - 1) / 4);
            w[2] = power(power(g, mod - 2), (mod - 1) / 8);
            w[3] = mt.mul<true>(w[1], w[2]);

            u32x8 cum0 = setr_u32x8(w[0], w[0], w[0], w[0], w[0], w[0], w[1], w[1]);
            u32x8 cum1 = setr_u32x8(w[0], w[0], w[0], w[1], w[0], w[2], w[0], w[3]);

            u32 rv = mt.mul<true>(mt.r2, power(mt.mul<true>(mt.r2, 1 << lg), mod - 2));
            u32x8 cum = set1_u32x8(rv);

            for (int i = 0; i < n; i += 8) {
                u32x8 vec = load_u32x8(data + i);

                vec = mts.mul(cum1, blend_u32x8<0b10'10'10'10>(vec, mts.mod2 - vec) + shuffle_u32x8<0b10'11'00'01>(vec));
                vec = mts.mul(cum0, blend_u32x8<0b11'00'11'00>(vec, mts.mod2 - vec) + shuffle_u32x8<0b01'00'11'10>(vec));
                vec = mts.mul(cum, blend_u32x8<0b11'11'00'00>(vec, mts.mod2 - vec) + permute_u32x8_epi128<1>(vec, vec));

                store_u32x8(data + i, vec);

                cum = mts.mul<true>(cum, w_cum_x8[__builtin_ctz(~(i >> 3))]);
            }

            _mm_free(w_cum_x8);

            k += 3;
        }

        for (; k + 1 <= lg; k += 2) {
            u32 wj = mt.r;
            u32x8 w_1 = set1_u32x8(power(power(g, mod - 2), (mod - 1) / 4));

            for (int i = 0; i < n; i += (1 << k + 1)) {
                u32 wj2 = mt.mul<true>(wj, wj);
                u32x8 w1 = set1_u32x8(wj);
                u32x8 w2 = set1_u32x8(wj2);
                u32x8 w3 = set1_u32x8(mt.mul<true>(wj, wj2));
                wj = mt.mul<true>(wj, w_cum[__builtin_ctz(~(i >> k + 1))]);

                for (int j = 0; j < (1 << k - 1); j += 8) {
                    u32x8 a = load_u32x8(data + i + 0 * (1 << k - 1) + j);
                    u32x8 b = load_u32x8(data + i + 1 * (1 << k - 1) + j);
                    u32x8 c = load_u32x8(data + i + 2 * (1 << k - 1) + j);
                    u32x8 d = load_u32x8(data + i + 3 * (1 << k - 1) + j);

                    u32x8 a1 = mts.shrink2(a + b), b1 = mts.shrink2_n(a - b),
                          c1 = mts.shrink2(c + d), d1 = mts.mul(c + mts.mod2 - d, w_1);

                    store_u32x8(data + i + 0 * (1 << k - 1) + j, mts.shrink2(a1 + c1));
                    store_u32x8(data + i + 1 * (1 << k - 1) + j, mts.mul(w1, b1 + d1));
                    store_u32x8(data + i + 2 * (1 << k - 1) + j, mts.mul(w2, a1 + mts.mod2 - c1));
                    store_u32x8(data + i + 3 * (1 << k - 1) + j, mts.mul(w3, b1 + mts.mod2 - d1));
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

    __attribute__((optimize("O3"))) [[gnu::noinline]] void convolve(int sz, __restrict__ u32 *a, __restrict__ u32 *b) const {
        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        if (lg <= 4) {
            auto c = convolve_slow(std::vector<u32>(a, a + sz), std::vector<u32>(b, b + sz));
            memcpy(a, c.data(), 4 * sz);
            return;
        }

        assert(sz <= (1 << lg));

        fft(lg, a);
        fft(lg, b);

        const auto mts = this->mts;  // ! to put Montgomery constants in registers
        for (int i = 0; i < (1 << lg); i += 8) {
            u32x8 ai = load_u32x8(a + i), bi = load_u32x8(b + i);
            store_u32x8(a + i, mts.mul(mts.shrink2(ai), mts.shrink2(bi)));
        }

        ifft(lg, a);
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

        convolve(1 << lg, ap, bp);

        std::vector<u32> res(ap, ap + sz);
        _mm_free(ap);
        _mm_free(bp);
        return res;
    }
};
