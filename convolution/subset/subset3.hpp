#pragma GCC optimize("O3")
#pragma GCC target("avx2,bmi2")

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

struct Montgomery {
    u32 mod;    //
    u32 mod2;   // mod * 2
    u32 n_inv;  // n_inv * mod == -1 (mod 2^32)
    u32 r;      // 2^32 % mod
    u32 r2;     // r^2 % mod;

    Montgomery() = default;
    Montgomery(u32 mod) {
        assert(mod % 2);
        assert(mod < (1 << 30));
        this->mod = mod;
        mod2 = 2 * mod;
        n_inv = 1;
        for (int i = 0; i < 5; i++) {
            n_inv *= 2 + n_inv * mod;
        }
        assert(n_inv * mod == u32(-1));
        r = (u64(1) << 32) % mod;
        r2 = u64(r) * r % mod;
    }

    u32 shrink(u32 val) const {
        return std::min(val, val - mod);
    }
    u32 shrink_n(u32 val) const {
        return std::min(val, val + mod);
    }

    // result * 2^32 == val
    template <bool strict = true>
    u32 reduce(u64 val) const {
        u32 res = val + u32(val) * n_inv * u64(mod) >> 32;
        if constexpr (strict) {
            res = shrink(res);
        }
        return res;
    }

    // result * 2^32 == a * b
    template <bool strict = true>
    u32 mul(u32 a, u32 b) const {
        return reduce<strict>(u64(a) * b);
    }
};

using u32x8 = __attribute__((vector_size(32))) u32;
using u64x4 = __attribute__((vector_size(32))) u64;
using i256 = __m256i;

u32x8 load_u32x8(const u32* data) {
    return u32x8(_mm256_load_si256((i256*)data));
}
u32x8 load_unaligned_u32x8(const u32* data) {
    return u32x8(_mm256_loadu_si256((i256*)data));
}
void store_u32x8(u32* data, u32x8 vec) {
    _mm256_store_si256((i256*)data, (i256)vec);
}
void store_unaligned_u32x8(u32* data, u32x8 vec) {
    _mm256_storeu_si256((i256*)data, (i256)vec);
}

struct Montgomery_simd {
    u32x8 mod;
    u32x8 mod2;
    u32x8 n_inv;
    u32x8 r, r2;

    Montgomery_simd(u32 mod) {
        Montgomery mt(mod);
        this->mod = (u32x8)_mm256_set1_epi32(mt.mod);
        this->mod2 = (u32x8)_mm256_set1_epi32(mt.mod2);
        this->n_inv = (u32x8)_mm256_set1_epi32(mt.n_inv);
        this->r = (u32x8)_mm256_set1_epi32(mt.r);
        this->r2 = (u32x8)_mm256_set1_epi32(mt.r2);
    }

    u32x8 shrink(u32x8 val) const {
        return (u32x8)_mm256_min_epu32((i256)val, _mm256_sub_epi32((i256)val, (i256)mod));
    }
    u32x8 shrink_n(u32x8 val) const {
        return (u32x8)_mm256_min_epu32((i256)val, _mm256_add_epi32((i256)val, (i256)mod));
    }
    u32x8 shrink2(u32x8 val) const {
        return (u32x8)_mm256_min_epu32((i256)val, _mm256_sub_epi32((i256)val, (i256)mod2));
    }

    template <bool strict = true>
    u32x8 reduce(u64x4 x0246, u64x4 x1357) const {
        u64x4 x0246_ninv = (u64x4)_mm256_mul_epu32((i256)x0246, (i256)n_inv);
        u64x4 x1357_ninv = (u64x4)_mm256_mul_epu32((i256)x1357, (i256)n_inv);
        u64x4 x0246_res = (u64x4)_mm256_add_epi64((i256)x0246, _mm256_mul_epu32((i256)x0246_ninv, (i256)mod));
        u64x4 x1357_res = (u64x4)_mm256_add_epi64((i256)x1357, _mm256_mul_epu32((i256)x1357_ninv, (i256)mod));
        u32x8 result = (u32x8)_mm256_or_si256(_mm256_bsrli_epi128((i256)x0246_res, 4), (i256)x1357_res);
        if constexpr (strict) {
            result = shrink(result);
        }
        return result;
    }

    template <bool strict = true, bool b_use_only_even = false>
    u32x8 mul(u32x8 a, u32x8 b) const {
        u32x8 a_sh = (u32x8)_mm256_bsrli_epi128((i256)a, 4);
        u32x8 b_sh = b_use_only_even ? b : (u32x8)_mm256_bsrli_epi128((i256)b, 4);
        u64x4 x0246 = (u64x4)_mm256_mul_epu32((i256)a, (i256)b);
        u64x4 x1357 = (u64x4)_mm256_mul_epu32((i256)a_sh, (i256)b_sh);
        return reduce<strict>(x0246, x1357);
    }
};

template <typename Functor, size_t... S>
__attribute__((always_inline)) constexpr void static_foreach_seq(Functor function, std::index_sequence<S...>) {
    ((function(std::integral_constant<size_t, S>())), ...);
}

template <size_t Size, typename Functor>
__attribute__((always_inline)) constexpr void static_for(Functor functor) {
    return static_foreach_seq(functor, std::make_index_sequence<Size>());
}

struct WTF {
    u32 mod;
    Montgomery mt;
    Montgomery_simd mts;

    WTF() = default;
    WTF(u32 mod) : mod(mod), mt(mod), mts(mod) { ; }

    template <bool inverse = false>
    void SOS(int lg, u32* data) const {
        // return;
        // const auto mt = this->mt;
        // auto add = [&](u32 a, u32 b) {
        //     return !inverse ? mt.shrink(a + b) : mt.shrink_n(a - b);
        // };
        // for (int k = lg - 1; k >= 0; k--) {
        //     for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
        //         for (int j = 0; j < (1 << k); j++) {
        //             data[i + (1 << k) + j] = add(data[i + (1 << k) + j], data[i + j]);
        //         }
        //     }
        // }

        const auto mts = this->mts;
        auto add = [&](u32x8 a, u32x8 b) {
            return !inverse ? mts.shrink(a + b) : mts.shrink_n(a - b);
        };
        int k = lg;

#define FUCK(r)                                                                                     \
    while (k - r >= 3) {                                                                            \
        k -= r;                                                                                     \
        alignas(64) u32x8 dt[1 << r];                                                               \
        for (size_t i = 0; i < (1ULL << lg); i += (1ULL << k + r)) {                                \
            for (size_t j = 0; j < (1ULL << k); j += 8) {                                           \
                static_for<1 << r>([&](auto it) {                                                   \
                    dt[it] = load_u32x8(data + i + j + (1ULL << k) * it);                           \
                });                                                                                 \
                static_for<r>([&](auto k) {                                                         \
                    static_for<1 << r - k - 1>([&](auto i) {                                        \
                        static_for<1 << k>([&](auto j) {                                            \
                            u32x8 a = dt[(i << k + 1) + j], b = dt[(i << k + 1) + (1ULL << k) + j]; \
                            dt[(i << k + 1) + (1ULL << k) + j] = add(b, a);                         \
                        });                                                                         \
                    });                                                                             \
                });                                                                                 \
                static_for<1 << r>([&](auto it) {                                                   \
                    store_u32x8(data + i + j + (1ULL << k) * it, dt[it]);                           \
                });                                                                                 \
            }                                                                                       \
        }                                                                                           \
        if (lg >= 15) {                                                                             \
            for (int i = 0; i < (1ULL << lg); i += (1ULL << k)) {                                   \
                SOS<inverse>(k, data + i);                                                          \
            }                                                                                       \
            return;                                                                                 \
        }                                                                                           \
    }

        FUCK(3);
        FUCK(2);
        FUCK(1);

#undef FUCK

        for (int i = 0; i < (1 << lg); i += 8) {
            u32x8 val = load_u32x8(data + i);
            val = mts.shrink((u32x8)_mm256_blend_epi32((i256)val, (i256)add(val, (u32x8)_mm256_shuffle_epi32((i256)val, 0b10'11'00'01)), 0b10'10'10'10));
            val = mts.shrink((u32x8)_mm256_blend_epi32((i256)val, (i256)add(val, (u32x8)_mm256_shuffle_epi32((i256)val, 0b01'00'11'10)), 0b11'00'11'00));
            val = mts.shrink((u32x8)_mm256_blend_epi32((i256)val, (i256)add(val, (u32x8)_mm256_permute2x128_si256((i256)val, (i256)val, 0x01)), 0b11'11'00'00));

            store_u32x8(data + i, val);
        }
    }

    void convolve_subset(int lg, const u32* ptr1, const u32* ptr2, u32* ptr_out) const {
        assert(lg >= 3);
        const auto mt = this->mt;
        const auto mts = this->mts;

        u32* p_out = (u32*)_mm_malloc(4 << lg, 64);
        std::vector<u32*> vec1(lg - 1), vec2(lg - 1);
        for (int i = 0; i < lg - 1; i++) {
            vec1[i] = (u32*)_mm_malloc(4 << lg, 64);
            vec2[i] = (u32*)_mm_malloc(4 << lg, 64);
            memset(vec1[i], 0, 4 << lg);
            memset(vec2[i], 0, 4 << lg);
        }
        {
            u32x8 val1 = (u32x8)_mm256_set1_epi32(mt.mul<true>(ptr1[0], mt.r2));
            u32x8 val2 = (u32x8)_mm256_set1_epi32(mt.mul<true>(ptr2[0], mt.r2));
            u32x8 sum = (u32x8)_mm256_setzero_si256();
            for (int i = 0; i < (1 << lg); i += 8) {
                int p_cnt = __builtin_popcount(i);
                u32x8 vc1 = mts.mul<true, true>(load_unaligned_u32x8(ptr1 + i), mts.r2);
                u32x8 vc2 = mts.mul<true, true>(load_unaligned_u32x8(ptr2 + i), mts.r2);
                u32x8 vc3 = mts.mul<false, true>(load_unaligned_u32x8(ptr2 + ((1 << lg) - i - 8)), mts.r2);
                vc3 = (u32x8)_mm256_permutevar8x32_epi32((i256)vc3, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                sum = mts.shrink2(sum + mts.mul<false>(vc1, vc3));

                store_u32x8(p_out + i, mts.shrink(mts.shrink2(mts.mul<false, true>(vc1, val2) + mts.mul<false, true>(vc2, val1))));
                static_for<8>([&](auto j) {
                    constexpr std::array<int, 8> table = {0, 1, 1, 2, 1, 2, 2, 3};
                    int pc = p_cnt + table[j];
                    if (0 < pc && pc <= lg - 1) {
                        vec1[pc - 1][i + j] = vc1[(size_t)j];
                        vec2[pc - 1][i + j] = vc2[(size_t)j];
                    }
                });
            }
            p_out[0] = mt.shrink_n(p_out[0] - mt.mul<true>(val1[0], val2[0]));

            sum = mts.shrink(sum);
            sum = mts.shrink(sum + (u32x8)_mm256_permute2x128_si256((i256)sum, (i256)sum, 1));
            sum = mts.shrink(sum + (u32x8)_mm256_bsrli_epi128((i256)sum, 8));
            sum = mts.shrink(sum + (u32x8)_mm256_bsrli_epi128((i256)sum, 4));
            u32 sm = sum[0];
            p_out[(1 << lg) - 1] = sm;
        }

        for (int i = 0; i < lg - 1; i++) {
            SOS<false>(lg, vec1[i]);
            SOS<false>(lg, vec2[i]);
        }
        constexpr int K = 1;
        u64x4* help = (u64x4*)_mm_malloc((2 * K) * 32 * (lg - 1), 64);
        for (int i = 0; i < (1 << lg); i += 8 * K) {
            memset(help, 0, (2 * K) * 32 * (lg - 1));
            for (int x = 1; x < lg - 1; x++) {
                if (x >= 16 && x % 8 == 0) {
                    for (int j = 0; j < (2 * K) * (lg - 1); j++) {
                        help[j] = (u64x4)mts.shrink2((u32x8)help[j]);
                    }
                }
                for (int y = 1; x + y < lg; y++) {
                    int ind = x + y - 1;
                    u32x8 a = load_u32x8(vec1[x - 1] + i), b = load_u32x8(vec2[y - 1] + i);
                    u32x8 a_sh = (u32x8)_mm256_bsrli_epi128((i256)a, 4), b_sh = (u32x8)_mm256_bsrli_epi128((i256)b, 4);
                    u64x4 c = (u64x4)_mm256_mul_epu32((i256)a, (i256)b), c_sh = (u64x4)_mm256_mul_epu32((i256)a_sh, (i256)b_sh);
                    help[2 * ind] += c;
                    help[2 * ind + 1] += c_sh;
                }
            }
            for (int j = 0; j < (2 * K) * (lg - 1); j += 2) {
                u64x4 a = (u64x4)mts.shrink2((u32x8)help[j]);
                u64x4 b = (u64x4)mts.shrink2((u32x8)help[j + 1]);
                u32x8 c = mts.shrink(mts.reduce<true>(a, b));
                store_u32x8(vec1[j >> 1] + i, c);
            }
        }

        _mm_free(help);
        for (int i = 0; i < lg - 1; i++) {
            _mm_free(vec2[i]);
        }

        for (int i = 0; i < lg - 1; i++) {
            SOS<true>(lg, vec1[i]);
        }
        for (int i = 0; i < (1 << lg); i++) {
            int p_cnt = __builtin_popcount(i);
            p_out[i] = mt.reduce<true>(mt.shrink(p_out[i] + ((p_cnt == 0 || p_cnt == lg) ? 0 : vec1[p_cnt - 1][i])));
        }
        for (int i = 0; i < lg - 1; i++) {
            _mm_free(vec1[i]);
        }
        memcpy(ptr_out, p_out, 4 << lg);
        _mm_free(p_out);
    }
};
