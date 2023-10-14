#pragma GCC target("avx512f,bmi")
#include <immintrin.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iostream>

using u32 = uint32_t;
using u64 = uint64_t;

namespace simd {
    using i512 = __m512i;
    using u32x16 = u32 __attribute__((vector_size(64)));
    using u64x8 = u64 __attribute__((vector_size(64)));

    u32x16 load_u32x16(u32* ptr) {
        return (u32x16)(_mm512_load_si512((i512*)ptr));
    }
    u32x16 loadu_u32x16(u32* ptr) {
        return (u32x16)(_mm512_loadu_si512((i512*)ptr));
    }
    void store_u32x16(u32* ptr, u32x16 val) {
        _mm512_store_si512((i512*)ptr, (i512)(val));
    }
    void storeu_u32x16(u32* ptr, u32x16 val) {
        _mm512_storeu_si512((i512*)ptr, (i512)(val));
    }

    u32x16 set1_u32x16(u32 val) {
        return (u32x16)(_mm512_set1_epi32(val));
    }
    u64x8 set1_u64x8(u64 val) {
        return (u64x8)(_mm512_set1_epi64(val));
    }

    u32x16 setr_u32x16_si256(u32 a0, u32 a1, u32 a2, u32 a3, u32 a4, u32 a5, u32 a6, u32 a7) {
        return (u32x16)_mm512_broadcast_i64x4(_mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7));
    }
    u64x8 setr_u64x8_si256(u64 a0, u64 a1, u64 a2, u64 a3) {
        return (u64x8)_mm512_broadcast_i64x4(_mm256_setr_epi64x(a0, a1, a2, a3));
    }

    u64x8 setr_u64x8(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7) {
        return (u64x8)_mm512_setr_epi64(a0, a1, a2, a3, a4, a5, a6, a7);
    }

    template <int imm8>
    u32x16 shuffle_u32x16(u32x16 val) {
        return (u32x16)(_mm512_shuffle_epi32((i512)(val), (_MM_PERM_ENUM)imm8));
    }
    u32x16 permute_u32x16(u32x16 val, u32x16 p) {
        return (u32x16)(_mm512_permutexvar_epi32((i512)(p), (i512)(val)));
    }

    template <int imm8>
    u32x16 permute_u32x16_epi128(u32x16 a) {
        return (u32x16)(_mm512_shuffle_i64x2((i512)(a), (i512)(a), imm8));
    }

    template <int imm16>
    u32x16 blend_u32x16(u32x16 a, u32x16 b) {
        return (u32x16)(_mm512_mask_blend_epi32(imm16, (i512)(a), (i512)(b)));
    }
    template <int imm8>
    u32x16 blend_u32x16_si256(u32x16 a, u32x16 b) {
        return (u32x16)(_mm512_mask_blend_epi32(imm8 | (imm8 << 8), (i512)(a), (i512)(b)));
    }

    u32x16 shift_left_u32x16_epi64(u32x16 val, int imm8) {
        return (u32x16)(_mm512_slli_epi64((i512)(val), imm8));
    }
    u32x16 shift_right_u32x16_epi64(u32x16 val, int imm8) {
        return (u32x16)(_mm512_srli_epi64((i512)(val), imm8));
    }

    u32x16 min_u32x16(u32x16 a, u32x16 b) {
        return (u32x16)(_mm512_min_epu32((i512)(a), (i512)(b)));
    }
    u32x16 mul64_u32x16(u32x16 a, u32x16 b) {
        return (u32x16)(_mm512_mul_epu32((i512)(a), (i512)(b)));
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
};

// Montgomery32
struct Montgomery_simd {
    alignas(64) u32x16 mod;
    alignas(64) u32x16 mod2;   // 2 * mod
    alignas(64) u32x16 n_inv;  // n_inv * mod == -1 (mod 2^32)
    alignas(64) u32x16 r;      // 2^32 % mod
    alignas(64) u32x16 r2;     // (2^32) ^ 2 % mod

    Montgomery_simd() = default;
    Montgomery_simd(u32 md) {
        Montgomery mt(md);
        mod = set1_u32x16(mt.mod);
        mod2 = set1_u32x16(mt.mod2);
        n_inv = set1_u32x16(mt.n_inv);
        r = set1_u32x16(mt.r);
        r2 = set1_u32x16(mt.r2);
    }

    u32x16 shrink(u32x16 val) const {
        return min_u32x16(val, val - mod);
    }
    u32x16 shrink2(u32x16 val) const {
        return min_u32x16(val, val - mod2);
    }
    u32x16 shrink_n(u32x16 val) const {
        return min_u32x16(val, val + mod);
    }
    u32x16 shrink2_n(u32x16 val) const {
        return min_u32x16(val, val + mod2);
    }

    // a * b should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u32x16 mul(u32x16 a, u32x16 b) const {
        u32x16 x0246 = mul64_u32x16(a, b);
        u32x16 x1357 = mul64_u32x16(shift_right_u32x16_epi64(a, 32), shift_right_u32x16_epi64(b, 32));
        u32x16 x0246_ninv = mul64_u32x16(x0246, n_inv);
        u32x16 x1357_ninv = mul64_u32x16(x1357, n_inv);
        u32x16 res = blend_u32x16_si256<0b10'10'10'10>(shift_right_u32x16_epi64(u32x16((u64x8)x0246 + (u64x8)mul64_u32x16(x0246_ninv, mod)), 32),
                                                       u32x16((u64x8)x1357 + (u64x8)mul64_u32x16(x1357_ninv, mod)));
        if constexpr (strict)
            res = shrink(res);
        return res;
    }

    // a * b should be in [0, 2**32 * mod)
    // puts result in high 32-bit of each 64-bit word
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u64x8 mul_to_hi(u64x8 a, u64x8 b) const {
        u32x16 val = mul64_u32x16((u32x16)a, (u32x16)b);
        u32x16 val_ninv = mul64_u32x16(val, n_inv);
        u32x16 res = u32x16(u64x8(val) + u64x8(mul64_u32x16(val_ninv, mod)));
        if constexpr (strict)
            res = shrink(res);
        return (u64x8)res;
    }

    // a * b should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u64x8 mul(u64x8 a, u64x8 b) const {
        return (u64x8)shift_right_u32x16_epi64((u32x16)mul_to_hi<strict>(a, b), 32);
    }
};
