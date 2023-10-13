#pragma GCC target("avx2,bmi")
#include <immintrin.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;
using u128 = __uint128_t;

using i128 = __m128i;
using i256 = __m256i;
using u32x4 = u32 __attribute__((vector_size(16)));
using u64x2 = u64 __attribute__((vector_size(16)));
using u32x8 = u32 __attribute__((vector_size(32)));
using u64x4 = u64 __attribute__((vector_size(32)));

#define RC(type, val) reinterpret_cast<type>(val)

namespace simd {
    u32x4 load_u32x4(u32* ptr) {
        return RC(u32x4, _mm_load_si128((i128*)ptr));
    }
    u32x8 load_u32x8(u32* ptr) {
        return RC(u32x8, _mm256_load_si256((i256*)ptr));
    }
    u32x8 loadu_u32x8(u32* ptr) {
        return RC(u32x8, _mm256_loadu_si256((i256*)ptr));
    }
    void store_u32x8(u32* ptr, u32x8 val) {
        _mm256_store_si256((i256*)ptr, RC(i256, val));
    }
    void storeu_u32x8(u32* ptr, u32x8 val) {
        _mm256_storeu_si256((i256*)ptr, RC(i256, val));
    }

    u32x8 set1_u32x8(u32 val) {
        return RC(u32x8, _mm256_set1_epi32(val));
    }
    u64x4 set1_u64x4(u64 val) {
        return RC(u64x4, _mm256_set1_epi64x(val));
    }

    u32x8 setr_u32x8(u32 a0, u32 a1, u32 a2, u32 a3, u32 a4, u32 a5, u32 a6, u32 a7) {
        return RC(u32x8, _mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7));
    }

    template <int imm8>
    u32x8 shuffle_u32x8(u32x8 val) {
        return RC(u32x8, _mm256_shuffle_epi32(RC(i256, val), imm8));
    }
    u32x8 permute_u32x8(u32x8 val, u32x8 p) {
        return RC(u32x8, _mm256_permutevar8x32_epi32(RC(i256, val), RC(i256, p)));
    }

    template <int imm8>
    u32x8 permute_u32x8_epi128(u32x8 a, u32x8 b) {
        return RC(u32x8, _mm256_permute2x128_si256(RC(i256, a), RC(i256, b), imm8));
    }

    template <int imm8>
    u32x8 blend_u32x8(u32x8 a, u32x8 b) {
        return RC(u32x8, _mm256_blend_epi32(RC(i256, a), RC(i256, b), imm8));
    }

    template <int imm8>
    u32x8 shift_left_u32x8_epi128(u32x8 val) {
        return RC(u32x8, _mm256_bslli_epi128(RC(i256, val), imm8));
    }
    template <int imm8>
    u32x8 shift_right_u32x8_epi128(u32x8 val) {
        return RC(u32x8, _mm256_bsrli_epi128(RC(i256, val), imm8));
    }

    u32x8 shift_left_u32x8_epi64(u32x8 val, int imm8) {
        return RC(u32x8, _mm256_slli_epi64(RC(i256, val), imm8));
    }
    u32x8 shift_right_u32x8_epi64(u32x8 val, int imm8) {
        return RC(u32x8, _mm256_srli_epi64(RC(i256, val), imm8));
    }

};  // namespace simd

using namespace simd;

// works for 998'244'353, not sure about other numbers
struct Barrett {
    u32 mod, s, q;

    Barrett(u32 m) : mod(m) {
        s = std::__lg(mod);

        u64 q0 = (((__uint128_t(1) << 64 + s) - 1) / mod + 1);
        if (mod == (1u << s)) {
            q0--;
        }

        assert(mod < (1 << 30) && "this shit won't work");
        if (q0 & ((1ULL << 31) | (1ULL << 62))) {
            std::cerr << "warning improper mod  (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
        }
        // assert(!(q0 & (1ULL << 31)));  // ! wtf
        // assert(!(q0 & (1ULL << 62)));  // ! wtf

        q = q0 >> 32;
    }

    u32 shrink(u32 val) {
        return std::min<u32>(val, val - mod);
    }

    // from [0, 2 * mod^2) to [0, 2 * mod)
    u32 mod_22(u64 val) {
        u32 a = u32(val >> s) * u64(q) >> 32;
        u32 res = u32(val) - a * mod;
        return res;
    }

    // product in [0, 2 * mod^2), result in [0, mod)
    u32 mul_mod_21(u32 a, u32 b) {
        return shrink(mod_22(u64(a) * b));
    }
};

struct Barrett_simd {
    u32x8 v_q;
    u32x8 v_mod;
    Barrett bt;
    u32 s;

    Barrett_simd(u32 m) : bt(m) {
        sizeof(Barrett_simd);

        s = bt.s;
        v_q = set1_u32x8(bt.q);
        v_mod = set1_u32x8(bt.mod);

        // assert(RC(u64, &v_q) % 32 == 0);
        // assert(RC(u64, &v_mod) % 32 == 0);
    }

    u32x8 shrink(u32x8 vec) {
        i256 res = _mm256_min_epu32(RC(i256, vec), _mm256_sub_epi32(RC(i256, vec), RC(i256, v_mod)));
        return RC(u32x8, res);
    }

    // from [0, 2 * mod^2) to [0, 2 * mod)
    u64x4 mod_22(u64x4 vec) {
        i256 a = _mm256_bsrli_epi128(_mm256_mul_epu32(_mm256_srli_epi64(RC(i256, vec), s), RC(i256, v_q)), 4);
        i256 res = _mm256_sub_epi32(RC(i256, vec), _mm256_mul_epu32(a, RC(i256, v_mod)));
        return RC(u64x4, res);
    }

    // product in [0, 2 * mod^2), result in [0, 2 * mod)
    u64x4 mul_mod_22(u64x4 a, u64x4 b) {
        i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
        i256 res = RC(i256, mod_22(RC(u64x4, x0246)));
        return RC(u64x4, res);
    }

    // // product in [0, 2 * mod^2), result in [0, 2 * mod)
    // u32x8 mul_mod_22(u32x8 a, u32x8 b) {
    //     i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
    //     i256 x1357 = _mm256_mul_epu32(_mm256_bsrli_epi128(RC(i256, a), 4), _mm256_bsrli_epi128(RC(i256, b), 4));
    //     x0246 = RC(i256, mod_22(RC(u64x4, x0246)));
    //     x1357 = RC(i256, mod_22(RC(u64x4, x1357)));
    //     i256 res = _mm256_blend_epi32(x0246, _mm256_bslli_epi128(x1357, 4), 0b10'10'10'10);
    //     return RC(u32x8, res);
    // }

    // product in [0, 2 * mod^2), result in [0, 2 * mod)
    u32x8 mul_mod_22(u32x8 a, u32x8 b) {
        i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
        i256 x1357 = _mm256_mul_epu32(_mm256_bsrli_epi128(RC(i256, a), 4), _mm256_bsrli_epi128(RC(i256, b), 4));
        i256 val = _mm256_blend_epi32(x0246, _mm256_bslli_epi128(x1357, 4), 0b10'10'10'10);
        x0246 = _mm256_bsrli_epi128(_mm256_mul_epu32(_mm256_srli_epi64(RC(i256, x0246), s), RC(i256, v_q)), 4);
        x1357 = _mm256_mul_epu32(_mm256_srli_epi64(RC(i256, x1357), s), RC(i256, v_q));
        i256 qt = _mm256_blend_epi32(x0246, x1357, 0b10'10'10'10);
        i256 res = _mm256_sub_epi32(val, _mm256_mullo_epi32(qt, RC(i256, v_mod)));
        return RC(u32x8, res);
    }

    // product in [0, 2 * mod^2), result in [0, mod)
    u64x4 mul_mod_21(u64x4 a, u64x4 b) {
        return RC(u64x4, shrink(RC(u32x8, mul_mod_22(a, b))));
    }

    // product in [0, 2 * mod^2), result in [0, mod)
    u32x8 mul_mod_21(u32x8 a, u32x8 b) {
        return shrink(mul_mod_22(a, b));
    }
};
