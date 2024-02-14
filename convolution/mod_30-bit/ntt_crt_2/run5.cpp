#include <cassert>
#include <iostream>

#define qin std::cin
#define qout std::cout
// #include "IO.hpp"

#include "ntt.hpp"

// #define CUM

int32_t main() {
    cum_timer __total("total");

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    constexpr u32 mod = 1e9 + 7;

    Montgomery mt(mod);
    Montgomery_simd mts(mod);

    const std::array<u32, 5> mods = {1007681537,
                                     1012924417,
                                     1045430273,
                                     1051721729,
                                     1053818881};
    // ! mods should be in non decreasing order
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
#ifndef CUM
    qin >> n >> m;
#endif

    int lg = std::__lg(std::max(1, n + m - 2)) + 1;
    lg = std::max(lg, 3);

    u64 *input_a = (u64 *)_mm_malloc(std::max(64, (8 << lg)), 64);
    u64 *input_b = (u64 *)_mm_malloc(std::max(64, (8 << lg)), 64);

    {
#ifndef CUM
        cum_timer __("input");
        for (int i = 0; i < n; i++) {
            qin >> input_a[i];
        }
        memset(input_a + n, 0, (8 << lg) - 8 * n);
        for (int i = 0; i < m; i++) {
            qin >> input_b[i];
        }
        memset(input_b + m, 0, (8 << lg) - 8 * m);
#endif
    }
    {
        cum_timer __("work");

        std::array<u32 *, 1 + K> data;
        for (int i = 0; i < data.size(); i++) {
            data[i] = (u32 *)_mm_malloc(std::max(64, (1 << lg) * 4), 64);
        }

        // for (int i = 0; i < 100; i++)
        //
        {
            {
                auto trans = [&](int n, const u64 *data, u32 *dest, const Montgomery mt, const Montgomery_simd mts) {
                    while (n % 8) {
                        n++;
                    }
                    // u32x8 mul_vec = setr_u32x8(mt.r, mt.r2, mt.r, mt.r2, mt.r, mt.r2, mt.r, mt.r2);
                    u32x8 mul_vec = set1_u32x8(mt.r2);
                    u32x8 hint = mul64_u32x8(mul_vec, mts.n_inv);
                    for (int i = 0; i < n; i += 8) {
                        u32x8 dt1 = load_u32x8((u32 *)(data + i));
                        u32x8 dt2 = load_u32x8((u32 *)(data + i + 4));

                        u32x8 p1 = blend_u32x8<0b10'10'10'10>(dt1, shift_left_u32x8_epi128<4>(dt2));
                        p1 = mts.shrink2(p1);  // ! from [0, 2^32) to [0, 3 * mod)  (works since mod > 10^9)

                        u32x8 p2 = blend_u32x8<0b10'10'10'10>(shift_right_u32x8_epi128<4>(dt1), dt2);
                        u32x8 res = mts.shrink2(p1 + mts.mul_hint<true>(p2, mul_vec, hint));

                        res = permute_u32x8(res, setr_u32x8(0, 2, 4, 6, 1, 3, 5, 7));
                        store_u32x8(dest + i, res);
                    }
                    memset(dest + n, 0, (4 << lg) - 4 * n);
                };
                cum_timer __("convolution");
                for (int i = 0; i < K; i++) {
                    {
                        cum_timer __("  transform");
                        trans(n, input_a, data[i], mt_ar[i], mts_ar[i]);
                        trans(m, input_b, data[i + 1], mt_ar[i], mts_ar[i]);
                    }
                    ntt[i].convolve2(lg, data[i], data[i + 1]);
                }
            }
            {
                cum_timer __("CRT");

                std::array<std::array<u32x8, K - 1>, K - 1> inv;
                for (int i = 0; i < K - 1; i++) {
                    std::array<u32, K> pr;
                    auto mt = mt_ar[i + 1];
                    pr[0] = 1;
                    for (int j = 0; j <= i; j++) {
                        pr[j + 1] = mt._mul(pr[j], mods[j]);
                    }
                    for (int j = 0; j <= i; j++) {
                        inv[i][j] = set1_u32x8(mt._mul(mt.r, mt._mul(pr[j], mt.inv(pr[i + 1]))));
                    }
                }

                std::array<u64x4, K> mul_m;
                {
                    std::array<u64, K> pr;
                    pr[0] = 1;
                    for (int j = 0; j + 1 < K; j++) {
                        pr[j + 1] = pr[j] * u64(mods[j]);
                    }
                    for (int j = 1; j < K; j++) {
                        mul_m[j] = set1_u64x4(pr[j]);
                    }
                }

                for (int i = 0; i < (1 << lg); i += 8) {
                    u32x8 a[K];
                    for (int j = 0; j < K; j++) {
                        a[j] = load_u32x8(data[j] + i);
                    }

                    u32x8 b[K];
                    b[0] = a[0];

                    auto get_cum = [&](u32x8 vec) {
                        struct Cum {
                            u64x4 a, b;
                        } cum;

                        cum.a = (u64x4)permute_u32x8(vec, setr_u32x8(0, -1, 1, -1, 2, -1, 3, -1));
                        cum.b = (u64x4)permute_u32x8(vec, setr_u32x8(4, -1, 5, -1, 6, -1, 7, -1));
                        return cum;
                    };
                    auto get_cum2 = [&](u32x8 vec) {
                        auto cum = get_cum(vec);
                        cum.a = (u64x4)(blend_u32x8<0b01'01'01'01>(set1_u32x8(0), (u32x8)cum.a));
                        cum.b = (u64x4)(blend_u32x8<0b01'01'01'01>(set1_u32x8(0), (u32x8)cum.b));
                        return cum;
                    };
                    // u32x8 sum = b[0];
                    auto [sum1, sum2] = get_cum2(b[0]);

                    for (int j = 1; j < K; j++) {
                        u32x8 bi = mts_ar[j].mul(mts_ar[j].mod + a[j] - b[0], inv[j - 1][0]);
                        for (int t = 1; t < j; t++) {
                            bi = mts_ar[j].shrink2_n(bi - mts_ar[j].mul(b[t], inv[j - 1][t]));
                        }
                        b[j] = mts_ar[j].shrink(bi);

                        auto [dlt1, dlt2] = get_cum(b[j]);
                        sum1 += mul64_u64x4_cum(mul_m[j], dlt1);
                        sum2 += mul64_u64x4_cum(mul_m[j], dlt2);
                    }

                    store_u64x4(input_a + i + 0, sum1);
                    store_u64x4(input_a + i + 4, sum2);
                }
            }
        }
    }

    {
#ifndef CUM
        cum_timer __("output");
        for (int i = 0; i < (n + m - 1); i++) {
            qout << input_a[i] << " \n"[i + 1 == (n + m - 1)];
        }
#endif
    }
    return 0;
}
