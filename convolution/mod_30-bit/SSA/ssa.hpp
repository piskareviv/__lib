#include <stdint.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <vector>

#include "aux.hpp"

struct SSA_20 {
    u32 mod;
    Montgomery mt;
    Montgomery_simd mts;

    SSA_20(u32 mod = 1'000'000'007) : mod(mod), mt(mod), mts(mod) {
        ;
    }

    struct Cum_6 {
        u32* ptr;
    };

    // // writes a * b to [a:b]
    // __attribute__((optimize("O3"))) void karatsuba_5(u32* a, u32* b) {
    //     constexpr int sz = 32;
    //     alignas(64) u64 aux_a[sz * 2], aux_b[sz];
    //     for (int i = 0; i < sz; i++) {
    //         aux_b[i] = b[i];
    //     }
    //     for (int i = 1; i <= sz; i++) {
    //         aux_a[i - 1] = a[sz - i];
    //     }
    //     for (int i = 0; i < sz; i++) {
    //         aux_a[i + sz] = mod - aux_a[i];
    //     }

    //     u64x4 mod2_8 = set1_u64x4(mod * 1ULL * mod * 8);
    //     u64x4 mod2_4 = set1_u64x4(mod * 1ULL * mod * 4);
    //     u64x4 mod2_2 = set1_u64x4(mod * 1ULL * mod * 2);
    //     u64x4 mod2_1 = set1_u64x4(mod * 1ULL * mod * 1);

    //     for (int i = 0; i < sz; i++) {
    //         int sh = sz - 1 - i;
    //         u64x4 sum[4];
    //         memset(sum, 0, sizeof(sum));
    //         for (int j = 0; j < sz; j += 16) {
    //             for (int t = 0; t < 4; t++) {
    //                 sum[t] += (u64x4)mul64_u32x8((u32x8)loadu_u64x4(aux_a + sh + j + t * 4),
    //                                              (u32x8)load_u64x4(aux_b + j + t * 4));
    //             }
    //         }
    //         u64x4 sm = (sum[0] + sum[1]) + (sum[2] + sum[3]);
    //         sm = sm < mod2_4 ? sm : sm - mod2_4;
    //         sm = sm + (u64x4)permute_u32x8_epi128<1>((u32x8)sm, (u32x8)sm);
    //         sm = sm + (u64x4)shuffle_u32x8<0b00'00'11'10>((u32x8)sm);
    //         u64 res = _mm256_extract_epi64((i256)sm, 0);

    //         res = res % u32(1e9 + 7);  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING  CUM !!!!!!!,  sorry for this
    //         a[i] = res;
    //     }
    // }

    // writes a * b to a
    __attribute__((optimize("O3"))) void convolve_6(u32* a, u32* b) {
        // return;
        if (0) {
            alignas(64) u32 aux[1 << 6 + 1];

            memset(aux, 0, sizeof(aux));
            for (int i = 0; i < 64; i++) {
                for (int j = 0; j < 64; j++) {
                    aux[i + j] = mt.shrink(aux[i + j] + mt.mul<true>(a[i], b[j]));
                }
            }
            for (int i = 0; i < 64; i++) {
                a[i] = mt.shrink_n(aux[i] - aux[i + 64]);
                a[i] = mt.mul<true>(a[i], mt.r2);
            }
            return;
        } else if (1) {
            constexpr int sz = 64;
            alignas(64) u64 aux_a[sz * 2], aux_b[sz];
            for (int i = 0; i < sz; i++) {
                aux_b[i] = b[i];
            }
            for (int i = 1; i <= sz; i++) {
                aux_a[i - 1] = a[sz - i];
            }
            for (int i = 0; i < sz; i++) {
                aux_a[i + sz] = mod - aux_a[i];
            }

            u64x4 mod2_8 = set1_u64x4(mod * 1ULL * mod * 8);
            u64x4 mod2_4 = set1_u64x4(mod * 1ULL * mod * 4);
            u64x4 mod2_2 = set1_u64x4(mod * 1ULL * mod * 2);
            u64x4 mod2_1 = set1_u64x4(mod * 1ULL * mod * 1);

            for (int i = 0; i < sz; i++) {
                int sh = sz - 1 - i;
                u64x4 sum[4];
                memset(sum, 0, sizeof(sum));
                for (int j = 0; j < sz; j += 16) {
                    for (int t = 0; t < 4; t++) {
                        sum[t] += (u64x4)mul64_u32x8((u32x8)loadu_u64x4(aux_a + sh + j + t * 4),
                                                     (u32x8)load_u64x4(aux_b + j + t * 4));
                    }
                }
                u64x4 sm = (sum[0] + sum[1]) + (sum[2] + sum[3]);
                sm = sm < mod2_8 ? sm : sm - mod2_8;
                sm = sm < mod2_4 ? sm : sm - mod2_4;
                // sm = sm < mod2_2 ? sm : sm - mod2_2;
                // sm = sm < mod2_1 ? sm : sm - mod2_1;
                sm = sm + (u64x4)permute_u32x8_epi128<1>((u32x8)sm, (u32x8)sm);
                sm = sm + (u64x4)shuffle_u32x8<0b00'00'11'10>((u32x8)sm);
                u64 res = _mm256_extract_epi64((i256)sm, 0);

                res = res % u32(1e9 + 7);  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING  CUM !!!!!!!,  sorry for this
                a[i] = res;
            }
        } else {
            auto mts = this->mts;
            constexpr int sz = 64;
            alignas(64) u32 aux_a[sz / 2], aux_b[sz / 2];
            for (int i = 0; i < sz / 2; i++) {
                aux_a[i] = mt.shrink(a[i] + a[i + sz / 2]);
                aux_b[i] = mt.shrink(b[i] + a[i + sz / 2]);
            }
        }
    }

    // writes a * b to a
    __attribute__((optimize("O3"))) void convolve_6(Cum_6 a, Cum_6 b) {
        convolve_6(a.ptr, b.ptr);
    }

    // [a, b] -> [a + b, a - b]
    [[gnu::noinline]] __attribute__((optimize("O3"))) void add_sub_6(Cum_6 a, Cum_6 b) const {
        const auto mts = this->mts;
        constexpr int sz = 1ULL << 6;
        for (int i = 0; i < sz; i += 8) {
            u32x8 va = load_u32x8(a.ptr + i);
            u32x8 vb = load_u32x8(b.ptr + i);
            store_u32x8(a.ptr + i, mts.shrink(va + vb));
            store_u32x8(b.ptr + i, mts.shrink_n(va - vb));
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void mul_6_by_w(int w, Cum_6 a) const {
        const auto mt = this->mt;
        constexpr int sz = 1ULL << 6;
        assert(w < 2 * sz);
        if (w >= sz) {
            w -= sz;
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (int i = w; i < sz; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        } else {
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (int i = 0; i < w; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void ssa10_all_rec(int k, int w, Cum_6* cum_a, Cum_6* cum_b) {
        if (k == 0) {
            convolve_6(cum_a[0], cum_b[0]);
        } else {
            int dt = 5 - k;
            int sz = 1ULL << 5 + 1;

            int tf = (sz >> dt) * w;
            int tf_r = (sz * 2 - tf) & (sz * 2 - 1);

            for (auto cum : std::array<Cum_6*, 2>{cum_a, cum_b}) {
                for (int i = 0; i < (1 << k - 1); i++) {
                    mul_6_by_w(tf, cum[i + (1 << k - 1)]);
                    add_sub_6(cum[i], cum[i + (1 << k - 1)]);
                }
            }

            ssa10_all_rec(k - 1, w, cum_a, cum_b);
            ssa10_all_rec(k - 1, w + (1ULL << dt), cum_a + (1 << k - 1), cum_b + (1 << k - 1));

            for (int i = 0; i < (1 << k - 1); i++) {
                add_sub_6(cum_a[i], cum_a[i + (1 << k - 1)]);
                mul_6_by_w(tf_r, cum_a[i + (1 << k - 1)]);
            }
        }
    }

    struct Cum_10 {
        u32* ptr;
        int sh;
    };

    // writes a * b to a
    __attribute__((optimize("O3"))) void convolve_10(Cum_10 a, Cum_10 b) {
        alignas(128) u32 aux_a[1 << 10 + 1];
        alignas(128) u32 aux_b[1 << 10 + 1];
        Cum_6 cum_a[1 << 5], cum_b[1 << 5];
        {
            auto fuck = [&](u32* aux, Cum_10 a, Cum_6* cum) {
                memset(aux, 0, 4 << 10 + 1);
                for (int i = 0; i < (1 << 5); i++) {
                    memcpy(aux + i * 64, a.ptr + i * 32, 4 << 5);
                    cum[i] = Cum_6{aux + i * 64};

                    mul_6_by_w(i * 2, cum[i]);
                }
            };
            fuck(aux_a, a, cum_a);
            fuck(aux_b, b, cum_b);
        }

        ssa10_all_rec(5, 0, cum_a, cum_b);

        u32 f = mt.mul<true>(mt.r2, mt.power(mod + 1 >> 1, 16));
        u32x8 fx8 = set1_u32x8(f);

        for (int i = 0; i < (1 << 5); i++) {
            mul_6_by_w(((1 << 7) - i * 2) % (1 << 7), cum_a[i]);
        }
        for (int i = 0; i < (1 << 5); i++) {
            int i2 = i == 0 ? (1 << 5) - 1 : i - 1;
            for (int j = 0; j < 4; j++) {
                u32x8 a0 = load_u32x8(aux_a + i * 64 + j * 8);
                u32x8 a1 = load_u32x8(aux_a + i2 * 64 + 32 + j * 8);
                if (i == 0) {
                    a1 = mod - a1;
                }
                store_u32x8(a.ptr + i * 32 + j * 8, mts.mul<true>(a0 + a1, fx8));
            }
        }
    }

    // [a, b] -> [a + b, a - b]
    [[gnu::noinline]] __attribute__((optimize("O3"))) void add_sub_10(Cum_10 a, Cum_10 b) const {
        const auto mts = this->mts;
        constexpr int sz = 1ULL << 10;
        for (int i = 0; i < sz; i += 8) {
            u32x8 va = load_u32x8(a.ptr + i);
            u32x8 vb = load_u32x8(b.ptr + i);
            store_u32x8(a.ptr + i, mts.shrink(va + vb));
            store_u32x8(b.ptr + i, mts.shrink_n(va - vb));
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void mul_10_by_w(int w, Cum_10 a) const {
        const auto mt = this->mt;
        constexpr int sz = 1ULL << 10;
        assert(w < 2 * sz);
        if (w >= sz) {
            w -= sz;
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (int i = w; i < sz; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        } else {
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (int i = 0; i < w; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void ssa20_all_rec(int k, int w, Cum_10* cum_a, Cum_10* cum_b) {
        if (k == 0) {
            convolve_10(cum_a[0], cum_b[0]);
        } else {
            int dt = 11 - k;
            int sz = 1ULL << 9 + 1;

            int tf = (sz >> dt) * w;
            int tf_r = (sz * 2 - tf) & (sz * 2 - 1);

            for (auto cum : std::array{cum_a, cum_b}) {
                for (int i = 0; i < (1 << k - 1); i++) {
                    mul_10_by_w(tf, cum[i + (1 << k - 1)]);
                    add_sub_10(cum[i], cum[i + (1 << k - 1)]);
                }
            }

            ssa20_all_rec(k - 1, w, cum_a, cum_b);
            ssa20_all_rec(k - 1, w + (1ULL << dt), cum_a + (1 << k - 1), cum_b + (1 << k - 1));

            for (int i = 0; i < (1 << k - 1); i++) {
                add_sub_10(cum_a[i], cum_a[i + (1 << k - 1)]);
                mul_10_by_w(tf_r, cum_a[i + (1 << k - 1)]);
            }
        }
    }

    // writes a * b to a
    [[gnu::noinline]] __attribute__((optimize("O3"))) void convolve_20(std::vector<Cum_10>& a, std::vector<Cum_10>& b) {
        ssa20_all_rec(11, 0, a.data(), b.data());
        for (int i = 0; i < (1 << 11); i++) {
            int j = (i + 1) % (1 << 11);
            for (int t = 0; t < (1 << 9); t++) {
                a[j].ptr[t] = mt.shrink(a[j].ptr[t] + a[i].ptr[t + (1 << 9)]);
            }
        }
    }
};

// 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
