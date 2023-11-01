#pragma GCC target("avx2,bmi,bmi2")
#include <immintrin.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iostream>

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
    u32x8 blendv_u32x8(u32x8 a, u32x8 b, u32x8 mask) {
        return (u32x8)(_mm256_blendv_epi8((i256)(a), (i256)(b), (i256)mask));
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

    u32x8 get_compress_perm_epi32(u32x8 mask) {
        u32 msk = _mm256_movemask_epi8((i256)mask);
        u32 cum = _pext_u32(0x76543210, msk);
        u64 cum64 = _pdep_u64(cum, 0x0F'0F'0F'0F'0F'0F'0F'0F);
        return (u32x8)_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(cum64));
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
        n_inv = -mod & 3;
        for (int i = 0; i < 4; i++) {
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

    // val should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u32 reduce(u64 val) const {
        u32 res = (val + u32(val) * n_inv * u64(mod)) >> 32;
        if constexpr (strict)
            res = shrink(res);
        return res;
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
    alignas(32) u32x8 mod, mod_sh;
    alignas(32) u32x8 mod2;             // 2 * mod
    alignas(32) u32x8 n_inv, n_inv_sh;  // n_inv * mod == -1 (mod 2^32)
    alignas(32) u32x8 r;                // 2^32 % mod
    alignas(32) u32x8 r2;               // (2^32) ^ 2 % mod

    Montgomery_simd() = default;
    Montgomery_simd(u32x8 md) {
        n_inv = -md & 3;
        mod = md;
        mod2 = 2 * md;
        for (int i = 0; i < 4; i++) {
            n_inv *= 2 + n_inv * mod;
        }
        n_inv_sh = shift_right_u32x8_epi64(n_inv, 32);
        mod_sh = shift_right_u32x8_epi64(mod, 32);
        for (int i = 0; i < 8; i++) {
            r2[i] = -1ULL % mod[i] + 1;  // !!!!!!!!! CUM
        }
        r = mul<true>(set1_u32x8(1), r2);

        // mod = set1_u32x8(mt.mod);
        // mod2 = set1_u32x8(mt.mod2);
        // n_inv = set1_u32x8(mt.n_inv);
        // r = set1_u32x8(mt.r);
        // r2 = set1_u32x8(mt.r2);
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

    // val should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u32x8 reduce(u32x8 x0246, u32x8 x1357) const {
        u32x8 x0246_ninv = mul64_u32x8(x0246, n_inv);
        u32x8 x1357_ninv = mul64_u32x8(x1357, n_inv_sh);
        u32x8 res = blend_u32x8<0b10'10'10'10>(shift_right_u32x8_epi64(u32x8((u64x4)x0246 + (u64x4)mul64_u32x8(x0246_ninv, mod)), 32),
                                               u32x8((u64x4)x1357 + (u64x4)mul64_u32x8(x1357_ninv, mod_sh)));
        if constexpr (strict)
            res = shrink(res);
        return res;
    }

    // a * b should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u32x8 mul(u32x8 a, u32x8 b) const {
        u32x8 x0246 = mul64_u32x8(a, b);
        u32x8 x1357 = mul64_u32x8(shift_right_u32x8_epi64(a, 32), shift_right_u32x8_epi64(b, 32));
        u32x8 res = reduce<strict>(x0246, x1357);
        return res;
    }

    // multiplies mod x^2 - d
    std::pair<u32x8, u32x8> mul_mod(u32x8 a0, u32x8 a1, u32x8 b0, u32x8 b1, u32x8 d) {
        u32x8 a0_sh = shift_right_u32x8_epi64(a0, 32);
        u32x8 a1_sh = shift_right_u32x8_epi64(a1, 32);
        u32x8 b0_sh = shift_right_u32x8_epi64(b0, 32);
        u32x8 b1_sh = shift_right_u32x8_epi64(b1, 32);

        u32x8 c0 = mul(a1, d);
        u32x8 c0_sh = shift_right_u32x8_epi64(c0, 32);
        u32x8 res0 = reduce<true>(u32x8((u64x4)mul64_u32x8(a0, b0) + (u64x4)mul64_u32x8(c0, b1)),
                                  u32x8((u64x4)mul64_u32x8(a0_sh, b0_sh) + (u64x4)mul64_u32x8(c0_sh, b1_sh)));
        u32x8 res1 = reduce<true>(u32x8((u64x4)mul64_u32x8(a0, b1) + (u64x4)mul64_u32x8(a1, b0)),
                                  u32x8((u64x4)mul64_u32x8(a0_sh, b1_sh) + (u64x4)mul64_u32x8(a1_sh, b0_sh)));
        return {res0, res1};
    }

    // multiplies mod x^2 - d
    std::pair<u32x8, u32x8> sq_mod(u32x8 a0, u32x8 a1, u32x8 d) {
        u32x8 a0_sh = shift_right_u32x8_epi64(a0, 32);
        u32x8 a1_sh = shift_right_u32x8_epi64(a1, 32);

        u32x8 c0 = mul(a1, d);
        u32x8 c0_sh = shift_right_u32x8_epi64(c0, 32);
        u32x8 res0 = reduce<true>(u32x8((u64x4)mul64_u32x8(a0, a0) + (u64x4)mul64_u32x8(c0, a1)),
                                  u32x8((u64x4)mul64_u32x8(a0_sh, a0_sh) + (u64x4)mul64_u32x8(c0_sh, a1_sh)));
        u32x8 res1 = reduce<true>(u32x8((u64x4)mul64_u32x8(a0, a1) << 1),
                                  u32x8((u64x4)mul64_u32x8(a0_sh, a1_sh) << 1));
        return {res0, res1};
    }

    // lg - number of bits in exp
    template <u32 lg = 30>
    u32x8 power(u32x8 base, u32x8 exp) const {
        u32x8 res = set1_u32x8(1);
        base = mul(base, r2);
        for (u32 i = 0; i < lg; i++, exp >>= 1) {
            res = mul(res, (exp & 1) ? base : r);
            base = mul(base, base);
        }
        res = shrink(res);

        return res;
    }
};
