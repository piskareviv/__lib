#pragma GCC target("avx2,bmi")
#include <immintrin.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

namespace simd {
    using i128 = __m128i;
    using i256 = __m256i;
    using u32x8 = u32 __attribute__((vector_size(32)));
    using u64x4 = u64 __attribute__((vector_size(32)));

    u32x8 load_u32x8(u32* ptr) {
        return (u32x8)(_mm256_load_si256((i256*)ptr));
    }
    u32x8 loadu_u32x8(u32* ptr) {
        return (u32x8)(_mm256_loadu_si256((i256*)ptr));
    }
    void store_u32x8(u32* ptr, u32x8 val) {
        _mm256_store_si256((i256*)ptr, (i256)(val));
    }
    void storeu_u32x8(u32* ptr, u32x8 val) {
        _mm256_storeu_si256((i256*)ptr, (i256)(val));
    }

    u32x8 set1_u32x8(u32 val) {
        return (u32x8)(_mm256_set1_epi32(val));
    }
    u64x4 set1_u64x4(u64 val) {
        return (u64x4)(_mm256_set1_epi64x(val));
    }

    u32x8 setr_u32x8(u32 a0, u32 a1, u32 a2, u32 a3, u32 a4, u32 a5, u32 a6, u32 a7) {
        return (u32x8)(_mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7));
    }
    u64x4 setr_u64x4(u64 a0, u64 a1, u64 a2, u64 a3) {
        return (u64x4)(_mm256_setr_epi64x(a0, a1, a2, a3));
    }

    template <int imm8>
    u32x8 shuffle_u32x8(u32x8 val) {
        return (u32x8)(_mm256_shuffle_epi32((i256)(val), imm8));
    }
    u32x8 permute_u32x8(u32x8 val, u32x8 p) {
        return (u32x8)(_mm256_permutevar8x32_epi32((i256)(val), (i256)(p)));
    }

    template <int imm8>
    u32x8 permute_u32x8_epi128(u32x8 a, u32x8 b) {
        return (u32x8)(_mm256_permute2x128_si256((i256)(a), (i256)(b), imm8));
    }

    template <int imm8>
    u32x8 blend_u32x8(u32x8 a, u32x8 b) {
        return (u32x8)(_mm256_blend_epi32((i256)(a), (i256)(b), imm8));
    }

    u32x8 shift_left_u32x8_epi64(u32x8 val, int imm8) {
        return (u32x8)(_mm256_slli_epi64((i256)(val), imm8));
    }
    u32x8 shift_right_u32x8_epi64(u32x8 val, int imm8) {
        return (u32x8)(_mm256_srli_epi64((i256)(val), imm8));
    }

    u32x8 min_u32x8(u32x8 a, u32x8 b) {
        return (u32x8)(_mm256_min_epu32((i256)(a), (i256)(b)));
    }
    u32x8 mul64_u32x8(u32x8 a, u32x8 b) {
        return (u32x8)(_mm256_mul_epu32((i256)(a), (i256)(b)));
    }

};  // namespace simd
using namespace simd;

// Montgomery32
struct Montgomery {
    u32 mod;
    u32 mod2;   // 2 * mod
    u32 n_inv;  // n_inv * mod == -1 (mod 2^32)
    u32 r;      // 2^32 % mod
    u32 r2;     // (2^32) ^ 2 % mod

    Montgomery() = default;
    Montgomery(u32 mod) : mod(mod) {
        assert(mod % 2);
        assert(mod < (1 << 30));
        n_inv = 1;
        for (int i = 0; i < 5; i++) {
            n_inv *= 2u + n_inv * mod;
        }
        assert(n_inv * mod == -1u);

        mod2 = 2 * mod;
        r = (1ULL << 32) % mod;
        r2 = r * u64(r) % mod;
    }

    u32 shrink(u32 val) const {
        return std::min(val, val - mod);
    }
    u32 shrink2(u32 val) const {
        return std::min(val, val - mod2);
    }
    u32 shrink_n(u32 val) const {
        return std::min(val, val + mod);
    }
    u32 shrink2_n(u32 val) const {
        return std::min(val, val + mod2);
    }

    // a * b should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u32 mul(u32 a, u32 b) const {
        u64 val = u64(a) * b;
        u32 res = (val + u32(val) * n_inv * u64(mod)) >> 32;
        if constexpr (strict)
            res = shrink(res);
        return res;
    }

    [[gnu::noinline]] u32 power(u32 b, u32 e) const {
        b = mul(b, r2);
        u32 r = 1;
        for (; e > 0; e >>= 1) {
            if (e & 1) {
                r = mul(r, b);
            }
            b = mul(b, b);
        }
        r = shrink(r);
        return r;
    }
};

// Montgomery32
struct Montgomery_simd {
    alignas(32) u32x8 mod;
    alignas(32) u32x8 mod2;   // 2 * mod
    alignas(32) u32x8 n_inv;  // n_inv * mod == -1 (mod 2^32)
    alignas(32) u32x8 r;      // 2^32 % mod
    alignas(32) u32x8 r2;     // (2^32) ^ 2 % mod

    Montgomery_simd() = default;
    Montgomery_simd(u32 md) {
        Montgomery mt(md);
        mod = set1_u32x8(mt.mod);
        mod2 = set1_u32x8(mt.mod2);
        n_inv = set1_u32x8(mt.n_inv);
        r = set1_u32x8(mt.r);
        r2 = set1_u32x8(mt.r2);
    }

    u32x8 shrink(u32x8 val) const {
        return min_u32x8(val, val - mod);
    }
    u32x8 shrink2(u32x8 val) const {
        return min_u32x8(val, val - mod2);
    }
    u32x8 shrink_n(u32x8 val) const {
        return min_u32x8(val, val + mod);
    }
    u32x8 shrink2_n(u32x8 val) const {
        return min_u32x8(val, val + mod2);
    }

    // a * b should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u32x8 mul(u32x8 a, u32x8 b) const {
        u32x8 x0246 = mul64_u32x8(a, b);
        u32x8 x1357 = mul64_u32x8(shift_right_u32x8_epi64(a, 32), shift_right_u32x8_epi64(b, 32));
        u32x8 x0246_ninv = mul64_u32x8(x0246, n_inv);
        u32x8 x1357_ninv = mul64_u32x8(x1357, n_inv);
        u32x8 res = blend_u32x8<0b10'10'10'10>(shift_right_u32x8_epi64(u32x8((u64x4)x0246 + (u64x4)mul64_u32x8(x0246_ninv, mod)), 32),
                                               u32x8((u64x4)x1357 + (u64x4)mul64_u32x8(x1357_ninv, mod)));
        if constexpr (strict)
            res = shrink(res);
        return res;
    }
};

// template <typename F, std::size_t... S>
// constexpr void static_for(F&& function, std::index_sequence<S...>) {
//     int unpack[] = {0,
//                     ((function(std::integral_constant<std::size_t, S>{})), 0)...};

//     (void)unpack;
// }

// template <std::size_t iterations, typename F>
// constexpr void static_for(F&& function) {
//     static_for(std::forward<F>(function), std::make_index_sequence<iterations>());
// }

// template <int First, int Last>
// struct static_for2 {
//     template <typename Fn>
//     constexpr void operator()(Fn fn) const {
//         if (First < Last) {
//             fn(std::integral_constant<int, First>());
//             static_for2<First + 1, Last>()(fn);
//         }
//     }
// };

// template <int N>
// struct static_for2<N, N> {
//     template <typename Fn>
//     constexpr void operator()(Fn fn) const {}
// };

// template <int N>
// using static_for = static_for2<0, N>;

// template <typename T>
// void static_consume(std::initializer_list<T>) {}

template <typename Functor, size_t... S>
__attribute__((always_inline)) constexpr void static_foreach_seq(Functor function, std::index_sequence<S...>) {
    // return static_consume({(function(std::integral_constant<std::size_t, S>{}), 0)...});
    ((function(std::integral_constant<size_t, S>())), ...);
}

template <size_t Size, typename Functor>
__attribute__((always_inline)) constexpr void static_for(Functor functor) {
    return static_foreach_seq(functor, std::make_index_sequence<Size>());
}

struct Cum {
    Montgomery mt;
    Montgomery_simd mts;

    Cum() = default;
    Cum(u32 mod) : mt(mod), mts(mod) { ; }

    // data must be 64 bytes aligned
    [[gnu::noinline]] __attribute__((optimize("O3"))) void fwht_(int lg, u32* data) const {
        if (lg <= 3) {
            for (int k = 0; k < lg; k++) {
                for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                    for (int j = 0; j < (1 << k); j++) {
                        u32 a = data[i + j], b = data[i + (1 << k) + j];
                        data[i + j] = mt.shrink(a + b);
                        data[i + (1 << k) + j] = mt.shrink_n(a - b);
                    }
                }
            }
        } else {
            auto mts = this->mts;
            u32 mod = mt.mod;

            auto proc = [&](u32x8& val) {
                val = mts.shrink(blend_u32x8<0b10'10'10'10>(val, mod - val) + shuffle_u32x8<0b10'11'00'01>(val));
                val = mts.shrink(blend_u32x8<0b11'00'11'00>(val, mod - val) + shuffle_u32x8<0b01'00'11'10>(val));
                val = mts.shrink(blend_u32x8<0b11'11'00'00>(val, mod - val) + permute_u32x8_epi128<0x01>(val, val));
            };
            for (int i = 0; i < (1 << lg); i += 16) {
                u32x8 val0 = load_u32x8(data + i);
                u32x8 val1 = load_u32x8(data + i + 8);
                proc(val0);
                proc(val1);
                store_u32x8(data + i, mts.shrink(val0 + val1));
                store_u32x8(data + i + 8, mts.shrink_n(val0 - val1));
            }
            for (int k = 4; k < lg; k++) {
                if (k + 1 < lg) {
                    for (int i = 0; i < (1 << lg); i += (1 << k + 2)) {
                        for (int j = 0; j < (1 << k); j += 8) {
                            u32x8 a0 = load_u32x8(data + i + j + (1 << k) * 0);
                            u32x8 a1 = load_u32x8(data + i + j + (1 << k) * 1);
                            u32x8 a2 = load_u32x8(data + i + j + (1 << k) * 2);
                            u32x8 a3 = load_u32x8(data + i + j + (1 << k) * 3);

                            u32x8 b0 = mts.shrink(a0 + a1);
                            u32x8 b1 = mts.shrink_n(a0 - a1);
                            u32x8 b2 = mts.shrink(a2 + a3);
                            u32x8 b3 = mts.shrink_n(a2 - a3);

                            u32x8 c0 = mts.shrink(b0 + b2);
                            u32x8 c1 = mts.shrink(b1 + b3);
                            u32x8 c2 = mts.shrink_n(b0 - b2);
                            u32x8 c3 = mts.shrink_n(b1 - b3);

                            store_u32x8(data + i + j + (1 << k) * 0, c0);
                            store_u32x8(data + i + j + (1 << k) * 1, c1);
                            store_u32x8(data + i + j + (1 << k) * 2, c2);
                            store_u32x8(data + i + j + (1 << k) * 3, c3);
                        }
                    }
                    k++;
                    continue;
                }
                for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                    for (int j = 0; j < (1 << k); j += 8) {
                        u32x8 a = load_u32x8(data + i + j), b = load_u32x8(data + i + (1 << k) + j);
                        store_u32x8(data + i + j, mts.shrink(a + b));
                        store_u32x8(data + i + (1 << k) + j, mts.shrink_n(a - b));
                    }
                }
            }
        }
    }

    // data must be 64 bytes aligned
    [[gnu::noinline]] __attribute__((optimize("O3"))) void fwht(size_t lg, u32* data) const {
        if (lg <= 3) {
            auto mt = this->mt;
            for (int k = 0; k < lg; k++) {
                for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                    for (int j = 0; j < (1 << k); j++) {
                        u32 a = data[i + j], b = data[i + (1 << k) + j];
                        data[i + j] = mt.shrink(a + b);
                        data[i + (1 << k) + j] = mt.shrink_n(a - b);
                    }
                }
            }
        } else {
            const auto mts = this->mts;
            u32 mod = mt.mod;

            auto proc = [&](u32x8& val) {
                val = mts.shrink(blend_u32x8<0b10'10'10'10>(val, mod - val) + shuffle_u32x8<0b10'11'00'01>(val));
                val = mts.shrink(blend_u32x8<0b11'00'11'00>(val, mod - val) + shuffle_u32x8<0b01'00'11'10>(val));
                val = mts.shrink(blend_u32x8<0b11'11'00'00>(val, mod - val) + permute_u32x8_epi128<0x01>(val, val));
            };
            for (int i = 0; i < (1 << lg); i += 16) {
                u32x8 val0 = load_u32x8(data + i);
                u32x8 val1 = load_u32x8(data + i + 8);
                proc(val0);
                proc(val1);
                store_u32x8(data + i, mts.shrink(val0 + val1));
                store_u32x8(data + i + 8, mts.shrink_n(val0 - val1));
            }

            for (int k = 4; k < lg; k++) {
                if (k + 8 <= lg && 0) {  // ! turned off
                    alignas(64) u32x8 dt[1 << 8 + 1];
                    u32* const dt_ptr = (u32*)dt;
                    // static u32* dt_ptr = (u32*)_mm_malloc(32 << 4, 64);

                    auto cum = [&](std::array<u32*, 4> input, std::array<u32*, 4> output) {
                        u32x8 a0 = load_u32x8(input[0]);
                        u32x8 a1 = load_u32x8(input[1]);
                        u32x8 a2 = load_u32x8(input[2]);
                        u32x8 a3 = load_u32x8(input[3]);

                        u32x8 b0 = mts.shrink(a0 + a1);
                        u32x8 b1 = mts.shrink_n(a0 - a1);
                        u32x8 b2 = mts.shrink(a2 + a3);
                        u32x8 b3 = mts.shrink_n(a2 - a3);

                        u32x8 c0 = mts.shrink(b0 + b2);
                        u32x8 c1 = mts.shrink(b1 + b3);
                        u32x8 c2 = mts.shrink_n(b0 - b2);
                        u32x8 c3 = mts.shrink_n(b1 - b3);

                        store_u32x8(output[0], c0);
                        store_u32x8(output[1], c1);
                        store_u32x8(output[2], c2);
                        store_u32x8(output[3], c3);
                    };

                    for (int i = 0; i < (1 << lg); i += (1 << k + 8)) {
                        for (int j = 0; j < (1 << k); j += 16) {
                            auto radix2 = [&](int k2, auto addr_in, auto addr_out) {
                                for (int i = 0; i < (1 << 8); i += (1 << k2 + 2)) {
                                    for (int j = 0; j < (1 << k2); j++) {
                                        std::array<int, 4> ar = {
                                            i + j + (1 << k2) * 0,
                                            i + j + (1 << k2) * 1,
                                            i + j + (1 << k2) * 2,
                                            i + j + (1 << k2) * 3};

                                        cum({addr_in(ar[0]), addr_in(ar[1]),
                                             addr_in(ar[2]), addr_in(ar[3])},
                                            {addr_out(ar[0]), addr_out(ar[1]),
                                             addr_out(ar[2]), addr_out(ar[3])});

                                        cum({addr_in(ar[0]) + 8, addr_in(ar[1]) + 8,
                                             addr_in(ar[2]) + 8, addr_in(ar[3]) + 8},
                                            {addr_out(ar[0]) + 8, addr_out(ar[1]) + 8,
                                             addr_out(ar[2]) + 8, addr_out(ar[3]) + 8});
                                    }
                                }
                            };

                            auto addr_gen0 = [&](int ind) {
                                return data + i + j + (1 << k) * ind;
                            };
                            auto addr_gen1 = [&](int ind) {
                                return dt_ptr + 16 * ind;
                            };

                            radix2(0, addr_gen0, addr_gen1);
                            radix2(2, addr_gen1, addr_gen1);
                            radix2(4, addr_gen1, addr_gen1);
                            radix2(6, addr_gen1, addr_gen0);
                        }
                    }
                    k += 8 - 1;
                    continue;
                }

#define FUCK(r)                                                                                     \
    if (k + r <= lg) {                                                                              \
        alignas(64) u32x8 dt[1 << r];                                                               \
        for (size_t i = 0; i < (1ULL << lg); i += (1ULL << k + r)) {                                \
            for (size_t j = 0; j < (1ULL << k); j += 8) {                                           \
                static_for<1 << r>([&](auto it) {                                                   \
                    dt[it] = load_u32x8(data + i + j + (1ULL << k) * it);                           \
                });                                                                                 \
                static_for<r>([&](auto k) {                                                         \
                    static_for<1 << r - k - 1>([&](auto i) {                                        \
                        static_for<1 << decltype(k)()>([&](auto j) {                                \
                            u32x8 a = dt[(i << k + 1) + j], b = dt[(i << k + 1) + (1ULL << k) + j]; \
                            dt[(i << k + 1) + j] = mts.shrink(a + b);                               \
                            dt[(i << k + 1) + (1ULL << k) + j] = mts.shrink_n(a - b);               \
                        });                                                                         \
                    });                                                                             \
                });                                                                                 \
                static_for<1 << r>([&](auto it) {                                                   \
                    store_u32x8(data + i + j + (1ULL << k) * it, dt[it]);                           \
                });                                                                                 \
            }                                                                                       \
        }                                                                                           \
        k += r - 1;                                                                                 \
        continue;                                                                                   \
    }

                FUCK(3);
                FUCK(2);
                FUCK(1);
#undef FUCK
            }
        }
    }

    void fwht_rec(size_t lg, u32* data) const {
        if (lg <= 13) {
            fwht(lg, data);
        } else {
            {
                for (int i = 0; i < (1ULL << lg - 2); i += 8) {
                    auto cum = [&](std::array<u32*, 4> input, std::array<u32*, 4> output) {
                        u32x8 a0 = load_u32x8(input[0]);
                        u32x8 a1 = load_u32x8(input[1]);
                        u32x8 a2 = load_u32x8(input[2]);
                        u32x8 a3 = load_u32x8(input[3]);

                        u32x8 b0 = mts.shrink(a0 + a1);
                        u32x8 b1 = mts.shrink_n(a0 - a1);
                        u32x8 b2 = mts.shrink(a2 + a3);
                        u32x8 b3 = mts.shrink_n(a2 - a3);

                        u32x8 c0 = mts.shrink(b0 + b2);
                        u32x8 c1 = mts.shrink(b1 + b3);
                        u32x8 c2 = mts.shrink_n(b0 - b2);
                        u32x8 c3 = mts.shrink_n(b1 - b3);

                        store_u32x8(output[0], c0);
                        store_u32x8(output[1], c1);
                        store_u32x8(output[2], c2);
                        store_u32x8(output[3], c3);
                    };
                    std::array<u32*, 4> ar = {data + i + (1ULL << lg - 2) * 0, data + i + (1ULL << lg - 2) * 1,
                                              data + i + (1ULL << lg - 2) * 2, data + i + (1ULL << lg - 2) * 3};
                    cum(ar, ar);
                }
                fwht_rec(lg - 2, data + (1ULL << lg - 2) * 0);
                fwht_rec(lg - 2, data + (1ULL << lg - 2) * 1);
                fwht_rec(lg - 2, data + (1ULL << lg - 2) * 2);
                fwht_rec(lg - 2, data + (1ULL << lg - 2) * 3);
                return;
            }

            {
#define FUCK(r)                                                                                 \
    alignas(64) u32x8 dt[1 << r];                                                               \
    for (size_t i = 0; i < (1ULL << lg); i += (1ULL << k + r)) {                                \
        for (size_t j = 0; j < (1ULL << k); j += 8) {                                           \
            static_for<1 << r>([&](auto it) {                                                   \
                dt[it] = load_u32x8(data + i + j + (1ULL << k) * it);                           \
            });                                                                                 \
            static_for<r>([&](auto k) {                                                         \
                static_for<1 << r - k - 1>([&](auto i) {                                        \
                    static_for<1 << decltype(k)()>([&](auto j) {                                \
                        u32x8 a = dt[(i << k + 1) + j], b = dt[(i << k + 1) + (1ULL << k) + j]; \
                        dt[(i << k + 1) + j] = mts.shrink(a + b);                               \
                        dt[(i << k + 1) + (1ULL << k) + j] = mts.shrink_n(a - b);               \
                    });                                                                         \
                });                                                                             \
            });                                                                                 \
            static_for<1 << r>([&](auto it) {                                                   \
                store_u32x8(data + i + j + (1ULL << k) * it, dt[it]);                           \
            });                                                                                 \
        }                                                                                       \
    }
                int k = lg - 3;
                FUCK(3);
#undef FUCK
                for (int i = 0; i < 8; i++) {
                    fwht_rec(lg - 3, data + (1ULL << lg - 3) * i);
                }
            }
        }
    }

    // a and b must be 64 bytes aligned
    __attribute__((optimize("O3"))) void
    convolve_xor(size_t lg, __restrict__ u32* a, __restrict__ u32* b) const {
        fwht_rec(lg, a);
        fwht_rec(lg, b);

        u32 f = mt.mul<true>(mt.mul(mt.r2, mt.r2), mt.power(mt.mod + 1 >> 1, lg));
        if (lg < 3) {
            for (size_t i = 0; i < (1 << lg); i++) {
                a[i] = mt.mul<true>(f, mt.mul(a[i], b[i]));
            }
        } else {
            auto mts = this->mts;
            u32x8 f_x8 = set1_u32x8(f);
            for (size_t i = 0; i < (1ULL << lg); i += 8) {
                u32x8 ai = load_u32x8(a + i), bi = load_u32x8(b + i);
                ai = mts.mul<true>(f_x8, mts.mul(ai, bi));
                store_u32x8(a + i, ai);
            }
        }
        fwht_rec(lg, a);
    }
};
