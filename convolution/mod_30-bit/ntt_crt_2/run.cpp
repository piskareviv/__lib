#include <cassert>
#include <iostream>

#define qin std::cin
#define qout std::cout
// #include "IO.hpp"

#include "ntt.hpp"

int32_t main() {
    cum_timer __total("total");

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    constexpr u32 mod = 1e9 + 7;

    Montgomery mt(mod);
    Montgomery_simd mts(mod);

    const std::array<u32, 3> mods = {754974721, 897581057, 998'244'353};
    // ! mods should be in non decreasing order
    // ! using these primes because ntt input should be in [0, 2 * mod)
    constexpr int K = mods.size();

    std::array<NTT, K> ntt;
    std::array<Montgomery, K> mt_ar;
    std::array<Montgomery_simd, K> mts_ar;
    for (int i = 0; i < K; i++) {
        ntt[i] = NTT(mods[i]);
        mt_ar[i] = Montgomery(mods[i]);
        mts_ar[i] = Montgomery_simd(mods[i]);
    }

    int n, m;
    n = m = 5e5;
    qin >> n >> m;

    int lg = std::__lg(std::max(1, n + m - 2)) + 1;
    lg = std::max(lg, 3);

    std::array<u32 *, 2 + K> data;
    for (int i = 0; i < data.size(); i++) {
        data[i] = (u32 *)_mm_malloc(std::max(64, (1 << lg) * 4), 64);
    }

    {
        cum_timer __("input");
        for (int i = 0; i < n; i++) {
            qin >> data[0][i];
        }
        memset(data[0] + n, 0, (4 << lg) - 4 * n);
        for (int i = 0; i < m; i++) {
            qin >> data[1][i];
        }
        memset(data[1] + m, 0, (4 << lg) - 4 * m);
    }

    {
        cum_timer __("work");
        {
            cum_timer __("convolution");
            for (int i = K - 1; i > 0; i--) {
                memcpy(data[i + 2], data[0], (4 << lg));
                memcpy(data[i + 1], data[1], (4 << lg));
                ntt[i].convolve2(lg, data[i + 2], data[i + 1]);
            }
            ntt[0].convolve2(lg, data[0], data[1]);
            std::rotate(data.begin() + 1, data.begin() + 3, data.end());
        }
        {
            cum_timer __("CRT");
            assert(K == 3);

            u32x8 m12 = set1_u32x8(mt_ar[1].mul(mt_ar[1].r2, mt_ar[1].power(mods[0], mods[1] - 2)));
            u32x8 m23 = set1_u32x8(mt_ar[2].mul(mt_ar[2].r2, mt_ar[2].power(mods[1], mods[2] - 2)));
            u32x8 m123 = set1_u32x8(mt_ar[2].mul(mt_ar[2].r2, mt_ar[2].power(mt_ar[2].mul(mt_ar[2].r2, mt_ar[2].mul(mods[0], mods[1])), mods[2] - 2)));

            u32x8 m1m = set1_u32x8(mt.mul<true>(mt.r2, mods[0]));
            u32x8 m12m = set1_u32x8(mt.mul<true>(mt.r3, mt.mul(mods[0], mods[1])));

            for (int i = 0; i < (1 << lg); i += 8) {
                u32x8 a1 = load_u32x8(data[0] + i);
                u32x8 a2 = load_u32x8(data[1] + i);
                u32x8 a3 = load_u32x8(data[2] + i);
                u32x8 b1 = a1;
                u32x8 b2 = mts_ar[1].mul<true>(mts_ar[1].mod + a2 - b1, m12);
                u32x8 b3 = mts_ar[2].shrink(mts_ar[2].mul<true>(mts_ar[2].mod + a3 - b1, m123) + mts_ar[2].mul<true>(mts_ar[2].mod - b2, m23));

                u32x8 sum = b1 + mts.mul<true>(b2, m1m) + mts.mul<true>(b3, m12m);
                sum = mts.shrink(mts.shrink2(sum));

                store_u32x8(data[0] + i, sum);
            }
        }
    }

    {
        cum_timer __("output");
        for (int i = 0; i < (n + m - 1); i++) {
            qout << data[0][i] << " \n"[i + 1 == (n + m - 1)];
        }
    }
    return 0;
}
