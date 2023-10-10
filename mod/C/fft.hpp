#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "montgomery.hpp"

struct FFT {
    Montgomery mt;
    Montgomery_simd mts;
    u32 mod, g;

    [[gnu::noinline]] u32 power(u32 base, u32 exp) {
        u32 res = mt.r;
        for (; exp > 0; exp >>= 1) {
            if (exp & 1) {
                res = mt.mul(res, base);
            }
            base = mt.mul(base, base);
        }
        return mt.shrink(res);
    }

    u32 find_pr_root(u32 mod) {
        assert(mod == 998'244'353);
        return 3;
    }

    FFT(u32 mod = 998'244'353) : mt(mod), mts(mod), mod(mod) {
        g = mt.mul<true>(mt.r2, find_pr_root(mod));
    }

    [[gnu::noinline]] std::pair<std::vector<u32>, u32x8 *> make_cum(int lg, bool inv = false) {
        std::vector<u32> w_cum(lg);
        for (int i = 0; i < lg; i++) {
            u32 f = power(g, (mod - 1) >> i + 2);
            if (inv) {
                f = power(f, mod - 2);
            }
            u32 res = mt.mul<true>(f, power(power(f, (1 << i + 1) - 2), mod - 2));
            w_cum[i] = res;
        }
        u32x8 *w_cum_x8 = new u32x8[lg];

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

    [[gnu::noinline]] __attribute__((optimize("O3"))) void fft(int lg, u32 *data) {
        auto [w_cum, w_cum_x8] = make_cum(lg);

        int n = 1 << lg;
        int k = lg;

        for (; k > 3; k--) {
            u32 wj = mt.r;
            for (int i = 0; i < (1 << k - 1); i += 8) {
                u32x8 a = load_u32x8(data + i);
                u32x8 b = load_u32x8(data + (1 << k - 1) + i);

                store_u32x8(data + i, mts.shrink2(a + b));
                store_u32x8(data + (1 << k - 1) + i, mts.shrink2_n(a - b));
            }
            wj = w_cum[0];
            for (int i = (1 << k); i < n; i += (1 << k)) {
                u32x8 wj_x8 = set1_u32x8(wj);
                wj = mt.mul<true>(wj, w_cum[__builtin_ctz(~(i >> k))]);

                for (int j = 0; j < (1 << k - 1); j += 8) {
                    u32x8 a = load_u32x8(data + i + j);
                    u32x8 b = load_u32x8(data + i + (1 << k - 1) + j);

                    a = mts.shrink2(a);
                    u32x8 c = mts.mul(b, wj_x8);
                    store_u32x8(data + i + j, a + c);
                    store_u32x8(data + i + (1 << k - 1) + j, a + mts.mod2 - c);
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

        delete[] w_cum_x8;
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void ifft(int lg, u32 *data) {
        auto [w_cum, w_cum_x8] = make_cum(lg, true);

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

            delete[] w_cum_x8;
            k += 3;
        }

        for (; k <= lg; k++) {
            u32 wj = mt.r;
            for (int i = 0; i < (1 << k - 1); i += 8) {
                u32x8 a = load_u32x8(data + i);
                u32x8 b = load_u32x8(data + (1 << k - 1) + i);

                store_u32x8(data + i, mts.shrink2(a + b));
                store_u32x8(data + (1 << k - 1) + i, mts.shrink2_n(a - b));
            }
            wj = w_cum[0];
            for (int i = (1 << k); i < n; i += (1 << k)) {
                u32x8 wj_x8 = set1_u32x8(wj);
                wj = mt.mul<true>(wj, w_cum[__builtin_ctz(~(i >> k))]);

                for (int j = 0; j < (1 << k - 1); j += 8) {
                    u32x8 a = load_u32x8(data + i + j);
                    u32x8 b = load_u32x8(data + i + (1 << k - 1) + j);

                    store_u32x8(data + i + j, mts.shrink2(a + b));
                    store_u32x8(data + i + (1 << k - 1) + j, mts.mul(a + mt.mod2 - b, wj_x8));
                }
            }
        }
    }

    __attribute__((optimize("O3"))) std::vector<u32> convolve_slow(std::vector<u32> a, std::vector<u32> b) {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        std::vector<u32> c(sz);
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                // c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % mod;
                c[i + j] = mt.shrink(c[i + j] + mt.mul<true>(mt.r2, mt.mul(a[i], b[j])));
            }
        }

        return c;
    }

    __attribute__((optimize("O3"))) [[gnu::noinline]] void convolve(int sz, __restrict__ u32 *a, __restrict__ u32 *b) {
        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        if (lg <= 4) {
            auto c = convolve_slow(std::vector<u32>(a, a + sz), std::vector<u32>(b, b + sz));
            memcpy(a, c.data(), 4 * sz);
            return;
        }

        assert(sz <= (1 << lg));

        fft(lg, a);
        fft(lg, b);

        for (int i = 0; i < (1 << lg); i += 8) {
            u32x8 ai = load_u32x8(a + i), bi = load_u32x8(b + i);
            store_u32x8(a + i, mts.mul(mts.shrink2(ai), mts.shrink2(bi)));
        }

        ifft(lg, a);
        for (int i = 0; i < (1 << lg); i += 8) {
            u32x8 ai = load_u32x8(a + i);
            store_u32x8(a + i, mts.shrink(ai));
        }
    }

    __attribute__((optimize("O3"))) std::vector<u32> convolve(std::vector<u32> a, std::vector<u32> b) {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        u32 *ap = (u32 *)new u32x8[std::max(1, (1 << lg) / 8)];
        u32 *bp = (u32 *)new u32x8[std::max(1, (1 << lg) / 8)];
        memset(ap, 0, 4 << lg);
        memset(bp, 0, 4 << lg);
        std::copy(a.begin(), a.end(), ap);
        std::copy(b.begin(), b.end(), bp);

        convolve(1 << lg, ap, bp);

        std::vector<u32> res(ap, ap + sz);
        delete[] (u32x8 *)ap;
        delete[] (u32x8 *)bp;
        return res;
    }
};
