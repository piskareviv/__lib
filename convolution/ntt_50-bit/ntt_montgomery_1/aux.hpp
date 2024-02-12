#pragma GCC target("avx512ifma,bmi")
#include <immintrin.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iostream>

// #define debug(x) std::cerr << #x << ": " << (x) << std::endl;

using u64 = uint64_t;
using u128 = __uint128_t;

namespace simd {
    using i512 = __m512i;
    using u64x8 = u64 __attribute__((vector_size(64)));

    u64x8 load_u64x8(u64* ptr) {
        return (u64x8)(_mm512_load_si512((i512*)ptr));
    }
    u64x8 loadu_u64x8(u64* ptr) {
        return (u64x8)(_mm512_loadu_si512((i512*)ptr));
    }
    void store_u64x8(u64* ptr, u64x8 val) {
        _mm512_store_si512((i512*)ptr, (i512)(val));
    }
    void storeu_u64x8(u64* ptr, u64x8 val) {
        _mm512_storeu_si512((i512*)ptr, (i512)(val));
    }

    u64x8 set1_u64x8(u64 val) {
        return (u64x8)(_mm512_set1_epi64(val));
    }

    u64x8 setr_u64x8(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7) {
        return (u64x8)(_mm512_setr_epi64(a0, a1, a2, a3, a4, a5, a6, a7));
    }

    template <int imm8>
    u64x8 shuffle_u64x8(u64x8 val) {
        return (u64x8)(_mm512_permutex_epi64((i512)(val), imm8));
    }
    u64x8 permute_u64x8(u64x8 val, u64x8 p) {
        return (u64x8)(_mm512_permutexvar_epi64((i512)(val), (i512)(p)));
    }

    template <int imm8>
    u64x8 permute_u64x8_i64x2(u64x8 a) {
        return (u64x8)(_mm512_shuffle_i64x2((i512)(a), (i512)(a), imm8));
    }

    template <int imm8>
    u64x8 blend_u64x8(u64x8 a, u64x8 b) {
        return (u64x8)(_mm512_mask_blend_epi64(imm8, (i512)(a), (i512)(b)));
    }

    template <int imm8>
    u64x8 shift_left_u64x8_epi128(u64x8 val) {
        return (u64x8)(_mm512_bslli_epi128((i512)(val), imm8));
    }
    template <int imm8>
    u64x8 shift_right_u64x8_epi128(u64x8 val) {
        return (u64x8)(_mm512_bsrli_epi128((i512)(val), imm8));
    }

    u64x8 shift_left_u64x8_epi64(u64x8 val, int imm8) {
        return (u64x8)(_mm512_slli_epi64((i512)(val), imm8));
    }
    u64x8 shift_right_u64x8_epi64(u64x8 val, int imm8) {
        return (u64x8)(_mm512_srli_epi64((i512)(val), imm8));
    }

    u64x8 min_u64x8(u64x8 a, u64x8 b) {
        return (u64x8)(_mm512_min_epu64((i512)(a), (i512)(b)));
    }

    // a + low52(b * c)
    u64x8 madd52lo_u64x8(u64x8 a, u64x8 b, u64x8 c) {
        return (u64x8)(_mm512_madd52lo_epu64((i512)(a), (i512)(b), (i512)(c)));
    }

    // a + high52(b * c)
    u64x8 madd52hi_u64x8(u64x8 a, u64x8 b, u64x8 c) {
        return (u64x8)(_mm512_madd52hi_epu64((i512)(a), (i512)(b), (i512)(c)));
    }

};  // namespace simd
using namespace simd;

// Montgomery52
struct Montgomery {
    u64 mod;
    u64 mod2;   // 2 * mod
    u64 n_inv;  // n_inv * mod == -1 (mod 2^52)
    u64 r;      // 2^52 % mod
    u64 r2;     // (2^52) ^ 2 % mod

    Montgomery() = default;
    Montgomery(u64 mod) : mod(mod) {
        assert(mod % 2);
        assert(mod < (1ULL << 50));
        n_inv = 1;
        for (int i = 0; i < 6; i++) {
            n_inv *= u64(2) + n_inv * mod;
        }
        assert(n_inv * mod == u64(-1));
        n_inv %= u64(1) << 52;
        assert((n_inv * mod + 1U) % (1ULL << 52) == 0);

        mod2 = 2 * mod;
        r = (1ULL << 52) % mod;
        r2 = r * u128(r) % mod;
    }

    u64 shrink(u64 val) const {
        return std::min(val, val - mod);
    }
    u64 shrink2(u64 val) const {
        return std::min(val, val - mod2);
    }
    u64 shrink_n(u64 val) const {
        return std::min(val, val + mod);
    }
    u64 shrink2_n(u64 val) const {
        return std::min(val, val + mod2);
    }

    // a * b should be in [0, 2**52 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u64 mul(u64 a, u64 b) const {
        u128 val = u128(a) * b;
        u64 res = (val + (u64(val) * n_inv & ((1ULL << 52) - 1)) * u128(mod)) >> 52;
        if constexpr (strict)
            res = shrink(res);
        return res;
    }
};

// Montgomery52
struct Montgomery_simd {
    alignas(64) u64x8 mod;
    alignas(64) u64x8 mod2;   // 2 * mod
    alignas(64) u64x8 inv;    // n_inv * mod == -1 (mod 2^52)
    alignas(64) u64x8 n_inv;  // n_inv * mod == -1 (mod 2^52)
    alignas(64) u64x8 r;      // 2^52 % mod
    alignas(64) u64x8 r2;     // (2^52) ^ 2 % mod

    Montgomery_simd() = default;
    Montgomery_simd(u64 md) {
        Montgomery mt(md);
        mod = set1_u64x8(mt.mod);
        mod2 = set1_u64x8(mt.mod2);
        inv = set1_u64x8((1ULL << 52) - mt.n_inv);
        n_inv = set1_u64x8(mt.n_inv);
        r = set1_u64x8(mt.r);
        r2 = set1_u64x8(mt.r2);
    }

    u64x8 shrink(u64x8 val) const {
        return min_u64x8(val, val - mod);
    }
    u64x8 shrink2(u64x8 val) const {
        return min_u64x8(val, val - mod2);
    }
    u64x8 shrink_n(u64x8 val) const {
        return min_u64x8(val, val + mod);
    }
    u64x8 shrink2_n(u64x8 val) const {
        return min_u64x8(val, val + mod2);
    }

    // a * b should be in [0, 2**52 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u64x8 mul(u64x8 a, u64x8 b) const {
        const u64x8 zero = set1_u64x8(0);
        u64x8 low = madd52lo_u64x8(zero, a, b);
        u64x8 high = madd52hi_u64x8(mod, a, b);
        u64x8 low_ninv = madd52lo_u64x8(zero, low, inv);
        u64x8 res = high - madd52hi_u64x8(zero, low_ninv, mod);

        if constexpr (strict) {
            res = shrink(res);
        }

        return res;
    }
};

struct cum_timer {
    clock_t beg;
    std::string s;

    cum_timer(std::string s) : s(s) {
        reset();
    }

    void reset() {
        beg = clock();
    }

    double elapsed(bool reset = false) {
        clock_t clk = clock();
        double res = (clk - beg) * 1.0 / CLOCKS_PER_SEC;
        if (reset) {
            beg = clk;
        }
        return res;
    }

    void print() {
        std::cerr << s << ": " << elapsed() << std::endl;
    }

    ~cum_timer() {
        print();
    }
};