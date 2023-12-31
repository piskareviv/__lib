#pragma GCC target("avx2")
#pragma GCC target("bmi")
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
#define INL inline __attribute__((always_inline))

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

    template <int imm8>
    u32x8 shuffle_u32x8(u32x8 val) {
        return RC(u32x8, _mm256_shuffle_epi32(RC(i256, val), imm8));
    }
    u32x8 permute_u32x8(u32x8 val, u32x8 p) {
        return RC(u32x8, _mm256_permutevar8x32_epi32(RC(i256, val), RC(i256, p)));
    }

    template <int imm8>
    u32x8 blend_u32x8(u32x8 a, u32x8 b) {
        return RC(u32x8, _mm256_blend_epi32(RC(i256, a), RC(i256, b), imm8));
    }
    u32x8 blendv_u32x8(u32x8 a, u32x8 b, u32x8 mask) {
        return RC(u32x8, _mm256_blendv_epi8(RC(i256, a), RC(i256, b), RC(i256, mask)));
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
    u32 mod2, s_1;

    Barrett(u32 m) : mod(m) {
        s = std::__lg(mod);
        mod2 = 2 * mod;
        s_1 = s + 1;

        u64 q0 = (((__uint128_t(1) << 64 + s) - 1) / mod + 1);
        if (mod == (1u << s)) {
            q0--;
        }

        assert(mod < (1 << 30) && "this shit won't work");
        if (q0 & ((1ULL << 31) | (1ULL << 62))) {
            std::cerr << "warning improper mod  (line: " << __LINE__ << ")" << std::endl;
        }
        // assert(!(q0 & (1ULL << 31)));  // ! wtf
        // assert(!(q0 & (1ULL << 62)));  // ! wtf

        q = q0 >> 32;
    }

    u32 shrink(u32 val) {
        return std::min<u32>(val, val - mod);
    }

    u32 shrink_2(u32 val) {
        return std::min<u32>(val, val - mod2);
    }

    u32 shrink_21(u32 val) {
        return shrink(shrink_2(val));
    }

    // from [0, 2 * mod * mod) to [0, 2 * mod)
    u32 mod_22(u64 val) {
        u32 a = u32(val >> s) * u64(q) >> 32;
        u32 res = u32(val) - a * mod;
        return res;
    }

    // // // from [0, 2 * mod * mod) to [0, mod)
    // // // should work for every mod
    // // u32 mod_21(u64 val) {
    // //     u32 res = mod_22(val);
    // //     res -= mod * (res >= mod) + mod * (res >= mod2);
    // //     return res;
    // // }

    // from [0, 4 * mod * mod) to [0, 4 * mod)
    u32 mod_44(u64 val) {
        u32 a = u32(val >> s_1) * u64(q) >> 32;
        u32 res = u32(val) - a * mod2;
        return res;
    }

    // from [0, 4 * mod * mod) to [0, 2 * mod)
    u32 mod_42(u64 val) {
        u32 res = mod_44(val);
        res = shrink_2(res);
        return res;
    }

    // product in [0, 2 * mod * mod), result in [0, mod)
    u32 mul_mod_21(u32 a, u32 b) {
        return shrink(mod_22(u64(a) * b));
    }
};

struct Barrett_simd {
    Barrett bt;

    u32 s, s_1;
    u32x8 v_mod, v_q;
    u32x8 v_mod2;

    Barrett_simd(u32 m) : bt(m) {
        s = bt.s;
        s_1 = bt.s_1;
        v_q = set1_u32x8(bt.q);
        v_mod = set1_u32x8(bt.mod);
        v_mod2 = set1_u32x8(bt.mod2);
    }

    u32x8 shrink(u32x8 vec) {
        i256 res = _mm256_min_epu32(RC(i256, vec), _mm256_sub_epi32(RC(i256, vec), RC(i256, v_mod)));
        return RC(u32x8, res);
    }

    u32x8 shrink_2(u32x8 vec) {
        i256 res = _mm256_min_epu32(RC(i256, vec), _mm256_sub_epi32(RC(i256, vec), RC(i256, v_mod2)));
        return RC(u32x8, res);
    }

    u32x8 shrink_21(u32x8 vec) {
        return shrink(shrink_2(vec));
    }

    // from [0, 2 * mod * mod) to [0, 2 * mod)
    u64x4 mod_22(u64x4 vec) {
        i256 a = _mm256_bsrli_epi128(_mm256_mul_epu32(_mm256_srli_epi64(RC(i256, vec), s), RC(i256, v_q)), 4);
        i256 res = _mm256_sub_epi64(RC(i256, vec), _mm256_mul_epu32(a, RC(i256, v_mod)));
        return RC(u64x4, res);
    }

    // from [0, 4 * mod * mod) to [0, 4 * mod)
    u64x4 mod_44(u64x4 vec) {
        i256 a = _mm256_bsrli_epi128(_mm256_mul_epu32(_mm256_srli_epi64(RC(i256, vec), s_1), RC(i256, v_q)), 4);
        i256 res = _mm256_sub_epi32(RC(i256, vec), _mm256_mul_epu32(a, RC(i256, v_mod2)));
        return RC(u64x4, res);
    }

    // from [0, 4 * mod * mod) to [0, 2 * mod)
    i256 mod_42(u64x4 vec) {
        i256 res = RC(i256, mod_44(vec));
        res = _mm256_min_epu32(res, _mm256_sub_epi32(res, RC(i256, v_mod2)));
        return res;
    }

    // product in [0, 4 * mod * mod), result in [0, 2 * mod)
    u32x8 mul_mod_42(u32x8 a, u32x8 b) {
        i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
        i256 x1357 = _mm256_mul_epu32(_mm256_bsrli_epi128(RC(i256, a), 4), _mm256_bsrli_epi128(RC(i256, b), 4));
        x0246 = RC(i256, mod_44(RC(u64x4, x0246)));
        x1357 = RC(i256, mod_44(RC(u64x4, x1357)));
        i256 res = _mm256_blend_epi32(x0246, _mm256_bslli_epi128(x1357, 4), 0b10'10'10'10);
        res = _mm256_min_epu32(res, _mm256_sub_epi32(res, RC(i256, v_mod2)));
        return RC(u32x8, res);
    }

    // product in [0, 2 * mod * mod), result in [0, 2 * mod)
    u32x8 mul_mod_22(u32x8 a, u32x8 b) {
        i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
        i256 x1357 = _mm256_mul_epu32(_mm256_bsrli_epi128(RC(i256, a), 4), _mm256_bsrli_epi128(RC(i256, b), 4));
        x0246 = RC(i256, mod_22(RC(u64x4, x0246)));
        x1357 = RC(i256, mod_22(RC(u64x4, x1357)));
        i256 res = _mm256_blend_epi32(x0246, _mm256_bslli_epi128(x1357, 4), 0b10'10'10'10);
        return RC(u32x8, res);
    }

    // product in [0, 2 * mod * mod), result in [0, 2 * mod)
    u64x4 mul_mod_22(u64x4 a, u64x4 b) {
        i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
        i256 res = RC(i256, mod_22(RC(u64x4, x0246)));
        return RC(u64x4, res);
    }

    // product in [0, 4 * mod * mod), result in [0, 2 * mod)
    u64x4 mul_mod_42(u64x4 a, u64x4 b) {
        const u32 shuflle_mask = 0b10'11'00'01;
        i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
        i256 res = RC(i256, mod_44(RC(u64x4, x0246)));
        res = _mm256_min_epu32(res, _mm256_sub_epi32(res, RC(i256, v_mod2)));
        return RC(u64x4, res);
    }
};
