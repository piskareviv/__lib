#pragma GCC optimize("O3")
#pragma GCC target("avx2,bmi")
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

    u32x8 load_u32x8(u32 *ptr) {
        return (u32x8)(_mm256_load_si256((i256 *)ptr));
    }
    u32x8 loadu_u32x8(u32 *ptr) {
        return (u32x8)(_mm256_loadu_si256((i256 *)ptr));
    }
    void store_u32x8(u32 *ptr, u32x8 val) {
        _mm256_store_si256((i256 *)ptr, (i256)(val));
    }
    void storeu_u32x8(u32 *ptr, u32x8 val) {
        _mm256_storeu_si256((i256 *)ptr, (i256)(val));
    }

    u64x4 load_u64x4(u64 *ptr) {
        return (u64x4)(_mm256_load_si256((i256 *)ptr));
    }
    u64x4 loadu_u64x4(u64 *ptr) {
        return (u64x4)(_mm256_loadu_si256((i256 *)ptr));
    }
    void store_u64x4(u64 *ptr, u64x4 val) {
        _mm256_store_si256((i256 *)ptr, (i256)(val));
    }
    void storeu_u64x4(u64 *ptr, u64x4 val) {
        _mm256_storeu_si256((i256 *)ptr, (i256)(val));
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

    template <int imm8>
    u32x8 shift_left_u32x8_epi128(u32x8 val) {
        return (u32x8)(_mm256_bslli_epi128((i256)(val), imm8));
    }
    template <int imm8>
    u32x8 shift_right_u32x8_epi128(u32x8 val) {
        return (u32x8)(_mm256_bsrli_epi128((i256)(val), imm8));
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

    template <bool strict = false>
    u64x4 reduce(u64x4 val) const {
        val = (u64x4)shift_right_u32x8_epi64(u32x8(val + (u64x4)mul64_u32x8(mul64_u32x8((u32x8)val, n_inv), mod)), 32);
        if constexpr (strict) {
            val = (u64x4)shrink((u32x8)val);
        }
        return val;
    }

    template <bool strict = false>
    u32x8 reduce(u64x4 x0246, u64x4 x1357) const {
        u32x8 x0246_ninv = mul64_u32x8((u32x8)x0246, n_inv);
        u32x8 x1357_ninv = mul64_u32x8((u32x8)x1357, n_inv);
        u32x8 res = blend_u32x8<0b10'10'10'10>(shift_right_u32x8_epi128<4>(u32x8((u64x4)x0246 + (u64x4)mul64_u32x8(x0246_ninv, mod))),
                                               u32x8((u64x4)x1357 + (u64x4)mul64_u32x8(x1357_ninv, mod)));
        if constexpr (strict)
            res = shrink(res);
        return res;
    }

    // a * b should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    //
    // set eq_b to <true> to slightly improve performance, if all elements of b are equal
    template <bool strict = false, bool eq_b = false>
    u32x8 mul(u32x8 a, u32x8 b) const {
        u32x8 x0246 = mul64_u32x8(a, b);
        u32x8 b_sh = b;
        if constexpr (!eq_b) {
            b_sh = shift_right_u32x8_epi128<4>(b);
        }
        u32x8 x1357 = mul64_u32x8(shift_right_u32x8_epi128<4>(a), b_sh);

        return reduce<strict>((u64x4)x0246, (u64x4)x1357);
    }

    // all elements of b should be equal
    template <bool strict = false>
    u32x8 mul_hint(u32x8 a, u32x8 b, u32x8 b_ni) const {
        u32x8 a_sh = shift_right_u32x8_epi128<4>(a);
        u32x8 x0246 = mul64_u32x8(a, b);
        u32x8 x1357 = mul64_u32x8(a_sh, b);
        u32x8 x0246_ninv = mul64_u32x8(a, b_ni);
        u32x8 x1357_ninv = mul64_u32x8(a_sh, b_ni);
        u32x8 res = blend_u32x8<0b10'10'10'10>(shift_right_u32x8_epi128<4>(u32x8((u64x4)x0246 + (u64x4)mul64_u32x8(x0246_ninv, mod))),
                                               u32x8((u64x4)x1357 + (u64x4)mul64_u32x8(x1357_ninv, mod)));
        if constexpr (strict)
            res = shrink(res);
        return res;
    }

    // a * b should be in [0, 2**32 * mod)
    // puts result in high 32-bit of each 64-bit word
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u64x4 mul_to_hi(u64x4 a, u64x4 b) const {
        u32x8 val = mul64_u32x8((u32x8)a, (u32x8)b);
        u32x8 val_ninv = mul64_u32x8(val, n_inv);
        u32x8 res = u32x8(u64x4(val) + u64x4(mul64_u32x8(val_ninv, mod)));
        if constexpr (strict)
            res = shrink(res);
        return (u64x4)res;
    }

    // a * b should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u64x4 mul(u64x4 a, u64x4 b) const {
        u32x8 val = mul64_u32x8((u32x8)a, (u32x8)b);
        return reduce<strict>((u64x4)val);
    }
};

template <size_t scale, size_t... S, typename Functor>
__attribute__((always_inline)) constexpr void static_foreach_seq(Functor function, std::index_sequence<S...>) {
    ((function(std::integral_constant<size_t, S * scale>())), ...);
}

template <size_t Size, size_t scale = 1, typename Functor>
__attribute__((always_inline)) constexpr void static_for(Functor functor) {
    return static_foreach_seq<scale>(functor, std::make_index_sequence<Size>());
}

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
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

struct NTT {
    alignas(32) Montgomery_simd mts;
    Montgomery mt;
    u32 mod, g;

    [[gnu::noinline]] u32 power(u32 base, u32 exp) const {
        const auto mt = this->mt;  // ! to put Montgomery constants in registers
        u32 res = mt.r;
        for (; exp > 0; exp >>= 1) {
            if (exp & 1) {
                res = mt.mul(res, base);
            }
            base = mt.mul(base, base);
        }
        return mt.shrink(res);
    }

    // mod should be prime
    u32 find_pr_root(u32 mod) const {
        u32 m = mod - 1;
        std::vector<u32> vec;
        for (u32 i = 2; u64(i) * i <= m; i++) {
            if (m % i == 0) {
                vec.push_back(i);
                do {
                    m /= i;
                } while (m % i == 0);
            }
        }
        if (m != 1) {
            vec.push_back(m);
        }
        for (u32 i = 2;; i++) {
            if (std::all_of(vec.begin(), vec.end(), [&](u32 f) { return mt.r != power(mt.mul(i, mt.r2), (mod - 1) / f); })) {
                return i;
            }
        }
    }

    u32x8 get_powers_u32x8(u32 x) const {
        u32 x2 = mt.mul<true>(x, x);
        u32 x3 = mt.mul<true>(x, x2);
        u32 x4 = mt.mul<true>(x2, x2);
        return setr_u32x8(mt.r, x, x2, x3, x4, mt.mul<true>(x2, x3), mt.mul<true>(x3, x3), mt.mul<true>(x4, x3));
    }

    static constexpr int LG = 30;
    alignas(32) u32 w[4], w_r[4];
    alignas(32) u64x4 w_cum_x4[LG], w_rcum_x4[LG];
    alignas(32) u32x8 w_cum_x8[LG], w_rcum_x8[LG];

    NTT(u32 mod = 998'244'353) : mt(mod), mts(mod), mod(mod) {
        const auto mt = this->mt;  // ! to put Montgomery constants in registers

        g = mt.mul<true>(mt.r2, find_pr_root(mod));

        for (int i = 0; i < LG; i++) {
            u32 f = power(g, (mod - 1) >> i + 3);
            u32 res = mt.mul<true>(f, power(power(f, (1 << i + 1) - 2), mod - 2));
            u32 res_r = power(res, mod - 2);
            w_cum_x4[i] = setr_u64x4(res, power(res, 2), res, power(res, 3));
            w_rcum_x4[i] = setr_u64x4(res_r, power(res_r, 2), res_r, power(res_r, 3));
        }
        for (int i = 0; i < LG; i++) {
            u32 f = power(g, (mod - 1) >> i + 4);
            f = mt.mul<true>(f, power(power(f, (1 << i + 1) - 2), mod - 2));
            u32 f1 = f;
            u32 f2 = mt.mul<true>(f1, f1);
            u32 f4 = mt.mul<true>(f2, f2);
            w_cum_x8[i] = setr_u32x8(f, f2, f, f4, f, f2, f, f4);

            u32 f_r = power(f, mod - 2);
            w_rcum_x8[i][0] = mt.r;
            for (int j = 1; j < 8; j++) {
                w_rcum_x8[i][j] = mt.mul<true>(w_rcum_x8[i][j - 1], f_r);
            }
        }

        u32 w18 = power(g, (mod - 1) / 8);
        w[0] = mt.r, w[1] = power(w18, 2), w[2] = w18, w[3] = power(w18, 3);
        u32 w78 = power(w18, 7);  // == w18 ^ (-1)
        w_r[0] = mt.r, w_r[1] = power(w78, 2), w_r[2] = w78, w_r[3] = power(w78, 3);
    }

    template <bool trivial = false, bool hint = false>
    static inline void butterfly_forward_x4(u32 *addr_inp_0, u32 *addr_inp_1, u32 *addr_inp_2, u32 *addr_inp_3,
                                            u32 *addr_out_0, u32 *addr_out_1, u32 *addr_out_2, u32 *addr_out_3,
                                            u32x8 w_1, u32x8 w1, u32x8 w2, u32x8 w3,
                                            const Montgomery_simd &mts,
                                            u32x8 w_1_h = u32x8(), u32x8 w1_h = u32x8(), u32x8 w2_h = u32x8(), u32x8 w3_h = u32x8()) {
        u32x8 a = load_u32x8(addr_inp_0);
        u32x8 b = load_u32x8(addr_inp_1);
        u32x8 c = load_u32x8(addr_inp_2);
        u32x8 d = load_u32x8(addr_inp_3);

        a = mts.shrink2(a);
        if constexpr (!trivial) {
            if constexpr (hint) {
                b = mts.mul_hint(b, w1, w1_h),
                c = mts.mul_hint(c, w2, w2_h),
                d = mts.mul_hint(d, w3, w3_h);
            } else {
                b = mts.mul<false, true>(b, w1),
                c = mts.mul<false, true>(c, w2),
                d = mts.mul<false, true>(d, w3);
            }
        } else {
            b = mts.shrink2(b),
            c = mts.shrink2(c),
            d = mts.shrink2(d);
        }

        u32x8 a1 = mts.shrink2(a + c),
              b1 = mts.shrink2(b + d),
              c1 = mts.shrink2_n(a - c),
              d1 = !hint
                       ? mts.mul<false, true>(b + mts.mod2 - d, w_1)
                       : mts.mul_hint(b + mts.mod2 - d, w_1, w_1_h);
        // d1 = mts.mul<false, true>(b + mts.mod2 - d, w_1);

        store_u32x8(addr_out_0, a1 + b1);
        store_u32x8(addr_out_1, a1 + mts.mod2 - b1);
        store_u32x8(addr_out_2, c1 + d1);
        store_u32x8(addr_out_3, c1 + mts.mod2 - d1);
    }

    template <bool trivial = false, bool multiply_by_cum = false, bool hint = false>
    static inline void butterfly_inverse_x4(u32 *addr_inp_0, u32 *addr_inp_1, u32 *addr_inp_2, u32 *addr_inp_3,
                                            u32 *addr_out_0, u32 *addr_out_1, u32 *addr_out_2, u32 *addr_out_3,
                                            u32x8 w_1, u32x8 w1, u32x8 w2, u32x8 w3,
                                            const Montgomery_simd &mts, u32x8 cum = u32x8(),
                                            u32x8 w_1_h = u32x8(), u32x8 w1_h = u32x8(), u32x8 w2_h = u32x8(), u32x8 w3_h = u32x8(), u32x8 cum_h = u32x8()) {
        u32x8 a = load_u32x8(addr_inp_0);
        u32x8 b = load_u32x8(addr_inp_1);
        u32x8 c = load_u32x8(addr_inp_2);
        u32x8 d = load_u32x8(addr_inp_3);

        u32x8 a1 = mts.shrink2(a + b),
              b1 = mts.shrink2_n(a - b),
              c1 = mts.shrink2(c + d),
              d1 = !hint
                       ? mts.mul<false, true>(c + mts.mod2 - d, w_1)
                       : mts.mul_hint(c + mts.mod2 - d, w_1, w_1_h);

        if constexpr (!trivial || multiply_by_cum) {
            if constexpr (!multiply_by_cum) {
                store_u32x8(addr_out_0, mts.shrink2(a1 + c1));
            } else {
                if constexpr (!hint) {
                    store_u32x8(addr_out_0, mts.mul<trivial, true>(a1 + c1, cum));
                } else {
                    store_u32x8(addr_out_0, mts.mul_hint<trivial>(a1 + c1, cum, cum_h));
                }
            }
            if constexpr (!hint) {
                store_u32x8(addr_out_1, mts.mul<trivial, true>(b1 + d1, w1));
                store_u32x8(addr_out_2, mts.mul<trivial, true>(a1 + mts.mod2 - c1, w2));
                store_u32x8(addr_out_3, mts.mul<trivial, true>(b1 + mts.mod2 - d1, w3));
            } else {
                store_u32x8(addr_out_1, mts.mul_hint<trivial>(b1 + d1, w1, w1_h));
                store_u32x8(addr_out_2, mts.mul_hint<trivial>(a1 + mts.mod2 - c1, w2, w2_h));
                store_u32x8(addr_out_3, mts.mul_hint<trivial>(b1 + mts.mod2 - d1, w3, w3_h));
            }
        } else {
            store_u32x8(addr_out_0, mts.shrink(mts.shrink2(a1 + c1)));
            store_u32x8(addr_out_1, mts.shrink(mts.shrink2(b1 + d1)));
            store_u32x8(addr_out_2, mts.shrink(mts.shrink2(a1 + mts.mod2 - c1)));
            store_u32x8(addr_out_3, mts.shrink(mts.shrink2(b1 + mts.mod2 - d1)));
        }
    }

    template <auto... t_args>
    static inline void butterfly_forward_x4(u32 *addr_0, u32 *addr_1, u32 *addr_2, u32 *addr_3, const auto &...args) {
        butterfly_forward_x4<t_args...>(addr_0, addr_1, addr_2, addr_3, addr_0, addr_1, addr_2, addr_3, args...);
    }

    template <auto... t_args>
    static inline void butterfly_inverse_x4(u32 *addr_0, u32 *addr_1, u32 *addr_2, u32 *addr_3, const auto &...args) {
        butterfly_inverse_x4<t_args...>(addr_0, addr_1, addr_2, addr_3, addr_0, addr_1, addr_2, addr_3, args...);
    }

    // input data[i] in [0, 2 * mod)
    // output data[i] in [0, 4 * mod)
    [[gnu::noinline]] __attribute__((optimize("O3"))) void ntt(int lg, u32 *data) const {
        const auto mt = this->mt;    // ! to put Montgomery constants in registers
        const auto mts = this->mts;  // ! to put Montgomery constants in registers

        int n = 1 << lg;
        int k = lg;

        if (lg % 2 == 0) {
            for (int i = 0; i < n / 2; i += 8) {
                u32x8 a = load_u32x8(data + i);
                u32x8 b = load_u32x8(data + n / 2 + i);

                store_u32x8(data + i, mts.shrink2(a + b));
                store_u32x8(data + n / 2 + i, mts.shrink2_n(a - b));
            }
            k--;
        }

        assert(k % 2 == 1);
        for (; k > 4; k -= 2) {
            u64x4 wj_cum = set1_u64x4(mt.r);
            u32x8 w_1 = set1_u32x8(w[1]);
            for (int i = 0; i < n; i += (1 << k)) {
                u32x8 w1 = shuffle_u32x8<0b00'00'00'00>((u32x8)wj_cum);
                u32x8 w2 = permute_u32x8((u32x8)wj_cum, set1_u32x8(2));
                u32x8 w3 = permute_u32x8((u32x8)wj_cum, set1_u32x8(6));
                wj_cum = mts.mul<true>(wj_cum, w_cum_x4[__builtin_ctz(~(i >> k))]);

                for (int j = 0; j < (1 << k - 2); j += 8) {
                    butterfly_forward_x4(data + i + 0 * (1 << k - 2) + j,
                                         data + i + 1 * (1 << k - 2) + j,
                                         data + i + 2 * (1 << k - 2) + j,
                                         data + i + 3 * (1 << k - 2) + j,
                                         w_1, w1, w2, w3, mts);
                }
            }
        }

        assert(k == 3);
        {
            // *  { w, w^2, w * w_1, w^4,  |  w * w_2, w^2 * w_1, w * w_3, w^4 }
            u32x8 cum = setr_u32x8(w[0], w[0], w[1], w[0], w[2], w[1], w[3], w[0]);

            int n_8 = n / 8;

            for (int i = 0; i < n_8; i++) {
                u32x8 vec = load_u32x8(data + i * 8);

                u32x8 wj0 = shuffle_u32x8<0b11'11'11'11>(cum);
                u32x8 wj1 = shuffle_u32x8<0b01'01'01'01>(cum);
                u32x8 wj2 = cum;  // no shuffle needed

                u32x8 bw;

                vec = mts.shrink2(vec);
                bw = permute_u32x8((u32x8)mts.mul_to_hi((u64x4)wj0, (u64x4)permute_u32x8(vec, setr_u32x8(4, -1, 5, -1, 6, -1, 7, -1))), setr_u32x8(1, 3, 5, 7, 1, 3, 5, 7));
                vec = permute_u32x8_epi128<0>(vec, vec) + blend_u32x8<0b11'11'00'00>(bw, mts.mod2 - bw);

                vec = mts.shrink2(vec);
                bw = shuffle_u32x8<0b11'01'11'01>((u32x8)mts.mul_to_hi((u64x4)wj1, (u64x4)shuffle_u32x8<0b00'11'00'10>(vec)));
                vec = shuffle_u32x8<0b01'00'01'00>(vec) + blend_u32x8<0b11'00'11'00>(bw, mts.mod2 - bw);

                vec = mts.shrink2(vec);
                bw = shuffle_u32x8<0b11'11'01'01>((u32x8)mts.mul_to_hi((u64x4)wj2, (u64x4)shuffle_u32x8<0b00'11'00'01>(vec)));
                vec = shuffle_u32x8<0b10'10'00'00>(vec) + blend_u32x8<0b10'10'10'10>(bw, mts.mod2 - bw);

                store_u32x8(data + i * 8, vec);

                cum = mts.mul(cum, w_cum_x8[__builtin_ctz(~i)]);
            }
        }
    }

    // input data[i] in [0, 2 * mod)
    // output data[i] in [0, mod)
    // fc (if specified) should be in [0, mod)
    // if fc is specified everything is multiplied by fc
    [[gnu::noinline]] __attribute__((optimize("O3"))) void intt(int lg, u32 *data, u32 fc = -1u) const {
        const auto mt = this->mt;    // ! to put Montgomery constants in registers
        const auto mts = this->mts;  // ! to put Montgomery constants in registers

        if (fc == -1u) {
            fc = mt.r;
        }

        int n = 1 << lg;
        int k = 1;
        {
            u32x8 cum0 = setr_u32x8(w_r[0], w_r[0], w_r[0], w_r[0], w_r[0], w_r[0], w_r[1], w_r[1]);
            u32x8 cum1 = setr_u32x8(w_r[0], w_r[0], w_r[0], w_r[1], w_r[0], w_r[2], w_r[0], w_r[3]);

            const u32 inv_2 = mt.mul<true>(mt.r2, (mod + 1) / 2);
            u32x8 cum = set1_u32x8(mt.mul<true>(fc, power(inv_2, lg)));

            int n_8 = n / 8;
            for (int i = 0; i < n_8; i++) {
                u32x8 vec = load_u32x8(data + i * 8);

                vec = mts.mul(cum1, blend_u32x8<0b10'10'10'10>(vec, mts.mod2 - vec) + shuffle_u32x8<0b10'11'00'01>(vec));
                vec = mts.mul(cum0, blend_u32x8<0b11'00'11'00>(vec, mts.mod2 - vec) + shuffle_u32x8<0b01'00'11'10>(vec));
                vec = mts.mul(cum, blend_u32x8<0b11'11'00'00>(vec, mts.mod2 - vec) + permute_u32x8_epi128<1>(vec, vec));

                store_u32x8(data + i * 8, vec);

                cum = mts.mul<true>(cum, w_rcum_x8[__builtin_ctz(~i)]);
            }
            k += 3;
        }

        for (; k + 1 <= lg; k += 2) {
            u64x4 wj_cum = set1_u64x4(mt.r);
            u32x8 w_1 = set1_u32x8(w_r[1]);

            for (int i = 0; i < n; i += (1 << k + 1)) {
                u32x8 w1 = shuffle_u32x8<0b00'00'00'00>((u32x8)wj_cum);
                u32x8 w2 = permute_u32x8((u32x8)wj_cum, set1_u32x8(2));
                u32x8 w3 = permute_u32x8((u32x8)wj_cum, set1_u32x8(6));
                wj_cum = mts.mul<true>(wj_cum, w_rcum_x4[__builtin_ctz(~(i >> k + 1))]);

                for (int j = 0; j < (1 << k - 1); j += 8) {
                    butterfly_inverse_x4(data + i + 0 * (1 << k - 1) + j,
                                         data + i + 1 * (1 << k - 1) + j,
                                         data + i + 2 * (1 << k - 1) + j,
                                         data + i + 3 * (1 << k - 1) + j,
                                         w_1, w1, w2, w3, mts);
                }
            }
        }
        if (k == lg) {
            for (int i = 0; i < n / 2; i += 8) {
                u32x8 a = load_u32x8(data + i);
                u32x8 b = load_u32x8(data + n / 2 + i);

                store_u32x8(data + i, mts.shrink(mts.shrink2(a + b)));
                store_u32x8(data + n / 2 + i, mts.shrink(mts.shrink2_n(a - b)));
            }
        } else {
            for (int i = 0; i < n; i += 8) {
                u32x8 ai = load_u32x8(data + i);
                store_u32x8(data + i, mts.shrink(ai));
            }
        }
    }

    struct Cum_info {
        alignas(64) u64x4 w_cum[LG], w_cum_r[LG];
    };

    static constexpr int MUL_HIGH_RADIX = 16;
    static constexpr int MUL_ITER = 10;
    static constexpr int MUL_QUAD = 4;

    template <int X, int lg, bool mul_by_fc = true>
    __attribute__((optimize("O3"))) void aux_mod_mul_x(u32 *__restrict__ a, u32 *__restrict__ b, std::array<u32, X> w, u32 fc, const Montgomery_simd &mts) const {
        static_assert(3 <= lg);
        constexpr int sz = 1 << lg;

        alignas(64) u64 aux_a[X][sz * 2];
        if constexpr (mul_by_fc) {
            Montgomery mt;
            mt.mod = mts.mod[0];
            mt.n_inv = mts.n_inv[0];
            mt.r2 = mts.r2[0];
            fc = mt.mul<true>(fc, mt.r2);
        }
        u32x8 fc_x8;
        if constexpr (mul_by_fc) {
            fc_x8 = set1_u32x8(fc);
        }

        const u32x8 perm_0 = (u32x8)setr_u64x4(0, 1, 2, 3);
        const u32x8 perm_1 = (u32x8)setr_u64x4(4, 5, 6, 7);
        for (int it = 0; it < X; it++) {
            for (int i = 0; i < sz; i += 8) {
                u64x4 w_x4 = (u64x4)set1_u32x8(w[it]);
                {
                    u32x8 vec = load_u32x8(b + sz * it + i);
                    if constexpr (mul_by_fc) {
                        vec = mts.mul<true>(vec, fc_x8);
                    } else {
                        vec = mts.shrink(mts.shrink2(vec));
                    }
                    store_u32x8(b + sz * it + i, vec);
                }

                u32x8 vec = load_u32x8(a + sz * it + i);
                vec = mts.shrink(mts.shrink2(vec));

                u64x4 a_lw = (u64x4)permute_u32x8(vec, perm_0);
                u64x4 a_hg = (u64x4)permute_u32x8(vec, perm_1);

                u64x4 aw_lw = mts.mul<true>(a_lw, w_x4);
                u64x4 aw_hg = mts.mul<true>(a_hg, w_x4);

                store_u64x4(aux_a[it] + i + 0, aw_lw);
                store_u64x4(aux_a[it] + i + 4, aw_hg);
                store_u64x4(aux_a[it] + i + sz + 0, a_lw);
                store_u64x4(aux_a[it] + i + sz + 4, a_hg);
            }
        }

        auto reduce_cum = [&](u64x4 val) {
            val = (u64x4)mts.shrink2((u32x8)val);
            return val;
        };

        alignas(64) u64x4 aux[X * sz / 4];
        memset(aux, 0, sizeof(aux));

        for (int i = 0; i < sz; i++) {
            if (i >= 16 && i % 8 == 0) {
                for (int it = 0; it < X; it++) {
                    for (int j = 0; j < sz / 4; j++) {
                        aux[it * sz / 4 + j] = reduce_cum(aux[it * sz / 4 + j]);
                    }
                }
            }

            for (int it = 0; it < X; it++) {
                for (int j = 0; j < sz; j += 4) {
                    u64x4 bi = (u64x4)set1_u32x8(b[i + sz * it]);
                    u64x4 aj = loadu_u64x4(aux_a[it] + sz - i + j);
                    aux[it * sz / 4 + j / 4] += (u64x4)mul64_u32x8((u32x8)bi, (u32x8)aj);
                }
            }
        }

        for (int i = 0; i < sz * X; i += 8) {
            u64x4 a0 = aux[i / 4];
            u64x4 a1 = aux[i / 4 + 1];

            if constexpr (lg >= 4) {
                a0 = reduce_cum(a0);
                a1 = reduce_cum(a1);
            }

            u32x8 b = permute_u32x8(mts.reduce(a0, a1), setr_u32x8(0, 2, 4, 6, 1, 3, 5, 7));
            b = mts.shrink(mts.shrink2(b));
            store_u32x8(a + i, b);
        }
    }

    // multiplies mod x^2^lg - w
    // writes result to a
    // multiplies everything by fc
    template <bool mul_by_fc = true>
    __attribute__((optimize("O3"))) inline void mul_mod(int lg, u32 *__restrict__ a, u32 *__restrict__ b, u32 w, u32 fc, const Montgomery_simd &mts) const {
        const auto mt = this->mt;  // ! to put Montgomery constants in registers

        if (lg >= 3) {
            assert(lg <= 12);
            static_for<10>([&](auto i) {
                if (i + 3 == lg) {
                    aux_mod_mul_x<1, i + 3, mul_by_fc>(a, b, std::array<u32, 1>{w}, fc, mts);
                }
            });
        } else {
            u32 buf[1 << 6];
            memset(buf, 0, sizeof(buf));
            for (int i = 0; i < (1 << lg); i++) {
                a[i] = mt.shrink2(a[i]);
                b[i] = mt.shrink2(b[i]);
            }
            for (int i = 0; i < (1 << lg); i++) {
                for (int j = 0; j < (1 << lg); j++) {
                    int ind = i + j;
                    u32 pr = mt.mul(a[i], b[j]);
                    if (ind >= (1 << lg)) {
                        pr = mt.mul(pr, w);
                        ind -= 1 << lg;
                    }
                    buf[ind] = mt.shrink2(buf[ind] + pr);
                }
            }

            if constexpr (mul_by_fc) {
                fc = mt.mul<true>(fc, mt.r2);
                for (int i = 0; i < (1 << lg); i++) {
                    buf[i] = mt.mul<true>(buf[i], fc);
                }
            } else {
                for (int i = 0; i < (1 << lg); i++) {
                    buf[i] = mt.shrink(mt.shrink2(buf[i]));
                }
            }

            memcpy(a, buf, 4 << lg);
        }
    }

    template <bool mul_by_fc = true>
    __attribute__((optimize("O3"))) inline void mul_mod_x4(int lg, u32 *__restrict__ a, u32 *__restrict__ b,
                                                           std::array<u32, 4> w, u32 fc, const Montgomery_simd &mts) const {
        if (lg >= 3) {
            assert(lg <= 12);
            static_for<10>([&](auto i) {
                if (i + 3 == lg) {
                    aux_mod_mul_x<4, i + 3, mul_by_fc>(a, b, w, fc, mts);
                }
            });
        } else {
            for (int it = 0; it < 4; it++) {
                mul_mod<mul_by_fc>(lg, a + (1 << lg) * it, b + (1 << lg) * it, w[it], fc, mts);
            }
        }
    }

    template <bool first = true>
    __attribute__((optimize("O3"))) void ntt_mul_rec_aux(int lg, int ind, u32 *data_a, u32 *data_b, Cum_info &cum_info, u32 fc) const {
        const auto mts = this->mts;  // ! to put Montgomery constants in registers
        const auto mt = this->mt;

        if (lg <= MUL_ITER) {
            // return;
            int k = lg;

            for (; k > MUL_QUAD; k -= 2) {
                u32x8 w_1 = set1_u32x8(w[1]);
                u32x8 w_1_h = mul64_u32x8(w_1, mts.n_inv);

                u64x4 wj_cum = first ? (u64x4)mts.r : cum_info.w_cum[k];

                for (int i = 0; i < (1 << lg); i += (1 << k)) {
                    u32x8 w1 = shuffle_u32x8<0b00'00'00'00>((u32x8)wj_cum);
                    u32x8 w2 = permute_u32x8((u32x8)wj_cum, set1_u32x8(2));
                    u32x8 w3 = permute_u32x8((u32x8)wj_cum, set1_u32x8(6));

                    u32x8 w1_h = mul64_u32x8(w1, mts.n_inv);
                    u32x8 w2_h = mul64_u32x8(w2, mts.n_inv);
                    u32x8 w3_h = mul64_u32x8(w3, mts.n_inv);

                    wj_cum = mts.mul<true>(wj_cum, w_cum_x4[__builtin_ctz(~(ind + i >> k))]);

                    for (auto data : std::array{data_a, data_b}) {
                        for (int j = 0; j < (1 << k - 2); j += 8) {
                            std::array<u32 *, 4> dt = {data + i + 0 * (1 << k - 2) + j,
                                                       data + i + 1 * (1 << k - 2) + j,
                                                       data + i + 2 * (1 << k - 2) + j,
                                                       data + i + 3 * (1 << k - 2) + j};

                            if (first && i == 0) {
                                butterfly_forward_x4<true, true>(dt[0], dt[1], dt[2], dt[3],
                                                                 w_1, w1, w2, w3, mts,
                                                                 w_1_h, w1_h, w2_h, w3_h);
                                continue;
                            }
                            butterfly_forward_x4<false, true>(dt[0], dt[1], dt[2], dt[3],
                                                              w_1, w1, w2, w3, mts,
                                                              w_1_h, w1_h, w2_h, w3_h);
                        }
                    }
                    if (k - 2 <= MUL_QUAD) {
                        u32 f0 = w1[0];
                        u32 f1 = mt.mul<true>(f0, w[1]);

                        std::array<u32, 4> wi = {f0, mod - f0, f1, mod - f1};
                        mul_mod_x4<false>(k - 2, data_a + i, data_b + i, wi, fc, mts);
                    }
                }
                cum_info.w_cum[k] = wj_cum;
            }

            fc = mt.mul<true>(fc, mt.r2);
            u32x8 fc_x8 = set1_u32x8(fc);
            u32x8 fc_x8_h = mul64_u32x8(fc_x8, mts.n_inv);

            for (; k + 1 < lg; k += 2) {
                u64x4 wj_cum = cum_info.w_cum_r[k];
                if (first) {
                    wj_cum = (u64x4)mts.r;
                    if (k <= MUL_QUAD) {
                        wj_cum = (u64x4)fc_x8;
                    }
                }
                u32x8 w_1 = set1_u32x8(w_r[1]);

                for (int i = 0; i < (1 << lg); i += (1 << k + 2)) {
                    u32x8 w1 = shuffle_u32x8<0b00'00'00'00>((u32x8)wj_cum);
                    u32x8 w2 = permute_u32x8((u32x8)wj_cum, set1_u32x8(2));
                    u32x8 w3 = permute_u32x8((u32x8)wj_cum, set1_u32x8(6));

                    wj_cum = mts.mul<true>(wj_cum, w_rcum_x4[__builtin_ctz(~(ind + i >> k + 2))]);

                    for (int j = 0; j < (1 << k); j += 8) {
                        std::array<u32 *, 4> dt = {data_a + i + 0 * (1 << k) + j,
                                                   data_a + i + 1 * (1 << k) + j,
                                                   data_a + i + 2 * (1 << k) + j,
                                                   data_a + i + 3 * (1 << k) + j};

                        if (k <= MUL_QUAD) {
                            if (first && i == 0) {
                                butterfly_inverse_x4<true, true, 0>(dt[0], dt[1], dt[2], dt[3], w_1, w1, w2, w3, mts, fc_x8);
                                continue;
                            }
                            butterfly_inverse_x4<false, true>(dt[0], dt[1], dt[2], dt[3], w_1, w1, w2, w3, mts, fc_x8);
                            continue;
                        }
                        if (first && i == 0) {
                            butterfly_inverse_x4<true>(dt[0], dt[1], dt[2], dt[3], w_1, w1, w2, w3, mts);
                            continue;
                        }
                        butterfly_inverse_x4(dt[0], dt[1], dt[2], dt[3], w_1, w1, w2, w3, mts);
                        continue;
                    }
                }
                cum_info.w_cum_r[k] = wj_cum;
            }
            return;
        }

        auto recurse = [&](auto RD) {
            ntt_mul_rec_aux<first>(lg - RD, ind, data_a, data_b, cum_info, fc);
            for (int i = 1; i < (1 << RD); i++) {
                ntt_mul_rec_aux<false>(lg - RD, ind + (1 << lg - RD) * i,
                                       data_a + (1 << lg - RD) * i,
                                       data_b + (1 << lg - RD) * i,
                                       cum_info, fc);
            }
        };

        if (lg < MUL_HIGH_RADIX || 1) {
            {
                int k = lg;
                u64x4 wj_cum = first ? (u64x4)mts.r : cum_info.w_cum[k];

                u32x8 w_1 = set1_u32x8(w[1]);
                u32x8 w1 = shuffle_u32x8<0b00'00'00'00>((u32x8)wj_cum);
                u32x8 w2 = permute_u32x8((u32x8)wj_cum, set1_u32x8(2));
                u32x8 w3 = permute_u32x8((u32x8)wj_cum, set1_u32x8(6));

                u32x8 w_1_h = mul64_u32x8(w_1, mts.n_inv);
                u32x8 w1_h = mul64_u32x8(w1, mts.n_inv);
                u32x8 w2_h = mul64_u32x8(w2, mts.n_inv);
                u32x8 w3_h = mul64_u32x8(w3, mts.n_inv);

                wj_cum = mts.mul<true>(wj_cum, w_cum_x4[__builtin_ctz(~(ind >> k))]);

                for (auto data : std::array{data_a, data_b}) {
                    for (int j = 0; j < (1 << k - 2); j += 8) {
                        butterfly_forward_x4<first, true>(data + 0 * (1 << k - 2) + j,
                                                          data + 1 * (1 << k - 2) + j,
                                                          data + 2 * (1 << k - 2) + j,
                                                          data + 3 * (1 << k - 2) + j,
                                                          w_1, w1, w2, w3, mts,
                                                          w_1_h, w1_h, w2_h, w3_h);
                    }
                }

                cum_info.w_cum[k] = wj_cum;
            }

            recurse(std::integral_constant<int64_t, 2>());

            {
                int k = lg;
                u64x4 wj_cum = first ? (u64x4)mts.r : cum_info.w_cum_r[k];

                u32x8 w_1 = set1_u32x8(w_r[1]);
                u32x8 w1 = shuffle_u32x8<0b00'00'00'00>((u32x8)wj_cum);
                u32x8 w2 = permute_u32x8((u32x8)wj_cum, set1_u32x8(2));
                u32x8 w3 = permute_u32x8((u32x8)wj_cum, set1_u32x8(6));

                // u32x8 w_1_h = mul64_u32x8(w_1, mts.n_inv);
                // u32x8 w1_h = mul64_u32x8(w1, mts.n_inv);
                // u32x8 w2_h = mul64_u32x8(w2, mts.n_inv);
                // u32x8 w3_h = mul64_u32x8(w3, mts.n_inv);

                wj_cum = mts.mul<true>(wj_cum, w_rcum_x4[__builtin_ctz(~(ind >> k))]);

                for (int j = 0; j < (1 << k - 2); j += 8) {
                    butterfly_inverse_x4<first>(data_a + 0 * (1 << k - 2) + j,
                                                data_a + 1 * (1 << k - 2) + j,
                                                data_a + 2 * (1 << k - 2) + j,
                                                data_a + 3 * (1 << k - 2) + j,
                                                w_1, w1, w2, w3, mts);
                    // continue;
                    // butterfly_inverse_x4<first, false, true>(data_a + 0 * (1 << k - 2) + j,
                    //                                          data_a + 1 * (1 << k - 2) + j,
                    //                                          data_a + 2 * (1 << k - 2) + j,
                    //                                          data_a + 3 * (1 << k - 2) + j,
                    //                                          w_1, w1, w2, w3, mts, u32x8(),
                    //                                          w_1_h, w1_h, w2_h, w3_h);
                }

                cum_info.w_cum_r[k] = wj_cum;
            }
        } else {
            constexpr int RD = 6;
            constexpr int BLC = 1024;
            {
                u32x8 w_1 = set1_u32x8(w[1]);
                alignas(64) u32x8 wx_0[3];
                alignas(64) u32x8 wx_1[4][3];
                alignas(64) u32x8 wx_2[16][3];
                for (int t = 0; t < 3; t++) {
                    u64x4 wj_cum = first ? (u64x4)mts.r : cum_info.w_cum[lg - 2 * t];
                    for (int i = 0; i < (1 << 2 * t); i++) {
                        u32x8 *wi = std::array{wx_0, wx_1[i], wx_2[i]}[t];

                        wi[0] = shuffle_u32x8<0b00'00'00'00>((u32x8)wj_cum);
                        wi[1] = permute_u32x8((u32x8)wj_cum, set1_u32x8(2));
                        wi[2] = permute_u32x8((u32x8)wj_cum, set1_u32x8(6));
                        wj_cum = mts.mul<true>(wj_cum, w_cum_x4[__builtin_ctz(~((ind >> lg - 2 * t) + i))]);
                    }
                    cum_info.w_cum[lg - 2 * t] = wj_cum;
                }

                for (auto data : std::array{data_a, data_b}) {
                    const int sz = 1 << lg - 6;
                    for (int j0 = 0; j0 < sz; j0 += BLC) {
                        for (int t = 0; t <= 2; t++) {
                            const int sz_it = 1 << lg - 2 * t - 2;

                            for (int i = 0, i0 = 0; i < (1 << lg); i += 4 * sz_it, i0++) {
                                u32x8 *wi = std::array{wx_0, wx_1[i0], wx_2[i0]}[t];
                                u32x8 w1 = wi[0], w2 = wi[1], w3 = wi[2];
                                for (int j1 = j0, j2 = 0; j1 < sz_it; j1 += sz, j2++) {
                                    for (int j = j1; j < j1 + BLC; j += 8) {
                                        std::array<u32 *, 4> dt = {data + i + 0 * sz_it + j, data + i + 1 * sz_it + j,
                                                                   data + i + 2 * sz_it + j, data + i + 3 * sz_it + j};
                                        if (first && i == 0) {
                                            butterfly_forward_x4<true>(dt[0], dt[1], dt[2], dt[3],
                                                                       w_1, w1, w2, w3, mts);
                                        } else {
                                            butterfly_forward_x4(dt[0], dt[1], dt[2], dt[3],
                                                                 w_1, w1, w2, w3, mts);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            recurse(std::integral_constant<int64_t, 6>());

            {
                u32x8 w_1 = set1_u32x8(w_r[1]);
                u32x8 wx_0[3];
                u32x8 wx_1[4][3];
                u32x8 wx_2[16][3];
                for (int t = 0; t < 3; t++) {
                    u64x4 wj_cum = first ? (u64x4)mts.r : cum_info.w_cum_r[lg - 2 * t];
                    for (int i = 0; i < (1 << 2 * t); i++) {
                        u32x8 *wi = std::array{wx_0, wx_1[i], wx_2[i]}[t];

                        wi[0] = shuffle_u32x8<0b00'00'00'00>((u32x8)wj_cum);
                        wi[1] = permute_u32x8((u32x8)wj_cum, set1_u32x8(2));
                        wi[2] = permute_u32x8((u32x8)wj_cum, set1_u32x8(6));
                        wj_cum = mts.mul<true>(wj_cum, w_rcum_x4[__builtin_ctz(~((ind >> lg - 2 * t) + i))]);
                    }
                    cum_info.w_cum_r[lg - 2 * t] = wj_cum;
                }
                const int sz = 1 << lg - 6;
                for (int j0 = 0; j0 < sz; j0 += BLC) {
                    for (int t = 2; t >= 0; t--) {
                        const int sz_it = 1 << lg - 2 * t - 2;
                        const int sz_bf = BLC << RD - 2 * t - 2;

                        for (int i = 0, i0 = 0; i < (1 << lg); i += 4 * sz_it, i0++) {
                            u32x8 *wi = std::array{wx_0, wx_1[i0], wx_2[i0]}[t];
                            u32x8 w1 = wi[0], w2 = wi[1], w3 = wi[2];
                            for (int j1 = j0; j1 < sz_it; j1 += sz) {
                                for (int j = j1; j < j1 + BLC; j += 8) {
                                    std::array<u32 *, 4> dt = {data_a + i + 0 * sz_it + j, data_a + i + 1 * sz_it + j,
                                                               data_a + i + 2 * sz_it + j, data_a + i + 3 * sz_it + j};
                                    if (first && i == 0) {
                                        butterfly_inverse_x4<true>(dt[0], dt[1], dt[2], dt[3],
                                                                   w_1, w1, w2, w3, mts);
                                    } else {
                                        butterfly_inverse_x4(dt[0], dt[1], dt[2], dt[3],
                                                             w_1, w1, w2, w3, mts);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // a and b should be 32-byte aligned
    // writes (a * b) to a
    [[gnu::noinline]] __attribute__((optimize("O3"))) void convolve(int lg, u32 *__restrict__ a, u32 *__restrict__ b) const {
        if (lg <= 4) {
            int n = (1 << lg);
            u32 *__restrict__ c = (u32 *)_mm_malloc(n * 4, 4);
            memset(c, 0, 4 * n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    c[(i + j) & (n - 1)] = mt.shrink2(c[(i + j) & (n - 1)] + mt.mul(a[i], b[j]));
                }
            }
            for (int i = 0; i < n; i++) {
                a[i] = mt.mul<true>(mt.r2, c[i]);
            }
            _mm_free(c);
            return;
        }

        ntt(lg, a);
        ntt(lg, b);

        const auto mts = this->mts;  // ! to put Montgomery constants in registers
        for (int i = 0; i < (1 << lg); i += 8) {
            u32x8 ai = load_u32x8(a + i), bi = load_u32x8(b + i);
            store_u32x8(a + i, mts.mul(mts.shrink2(ai), mts.shrink2(bi)));
        }

        intt(lg, a, mt.r2);
    }

    // a and b should be 32-byte aligned
    // writes (a * b) to a
    [[gnu::noinline]] __attribute__((optimize("O3"))) void convolve2(int lg, u32 *__restrict__ a, u32 *__restrict__ b) const {
        if (lg <= MUL_QUAD) {
            mul_mod(lg, a, b, mt.r, mt.r, mts);
            for (int i = 0; i < (1 << lg); i++) {
                a[i] = mt.shrink(mt.shrink2(a[i]));
            }
            return;
        }

        Cum_info cum_info;
        u32 f = power(mt.mul(mt.r2, mod + 1 >> 1), (lg - (MUL_QUAD - 1)) / 2 * 2);
        ntt_mul_rec_aux(lg, 0, a, b, cum_info, f);
    }

    __attribute__((optimize("O3"))) std::vector<u32> convolve_slow(std::vector<u32> a, std::vector<u32> b) const {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);
        const auto mt = this->mt;  // ! to put Montgomery constants in registers

        std::vector<u32> c(sz);

        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                // c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % mod;
                c[i + j] = mt.shrink(c[i + j] + mt.mul<true>(mt.r2, mt.mul(a[i], b[j])));
            }
        }

        return c;
    }

    __attribute__((optimize("O3"))) std::vector<u32> convolve(const std::vector<u32> &a, const std::vector<u32> &b) const {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        u32 *ap = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
        u32 *bp = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
        memset(ap, 0, 4 << lg);
        memset(bp, 0, 4 << lg);

        std::copy(a.begin(), a.end(), ap);
        std::copy(b.begin(), b.end(), bp);

        convolve(lg, ap, bp);

        std::vector<u32> res(ap, ap + sz);
        _mm_free(ap);
        _mm_free(bp);
        return res;
    }

    __attribute__((optimize("O3"))) std::vector<u32> convolve2(const std::vector<u32> &a, const std::vector<u32> &b) const {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        u32 *ap = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
        u32 *bp = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
        memset(ap, 0, 4 << lg);
        memset(bp, 0, 4 << lg);

        std::copy(a.begin(), a.end(), ap);
        std::copy(b.begin(), b.end(), bp);

        convolve2(lg, ap, bp);

        std::vector<u32> res(ap, ap + sz);
        _mm_free(ap);
        _mm_free(bp);
        return res;
    }
};

#include <sys/mman.h>
#include <sys/stat.h>

#include <cstring>
#include <iostream>

// io from https://judge.yosupo.jp/submission/142782

namespace __io {
    using u32 = uint32_t;
    using u64 = uint64_t;

    namespace QIO_base {
        constexpr int O_buffer_default_size = 1 << 18;
        constexpr int O_buffer_default_flush_threshold = 40;
        struct _int_to_char_tab {
            char tab[40000];
            constexpr _int_to_char_tab() : tab() {
                for (int i = 0; i != 10000; ++i) {
                    for (int j = 3, n = i; ~j; --j) {
                        tab[i * 4 + j] = n % 10 + 48, n /= 10;
                    }
                }
            }
        } constexpr _otab;
    }  // namespace QIO_base
    namespace QIO_I {
        using namespace QIO_base;
        struct Qinf {
            FILE *f;
            char *bg, *ed, *p;
            struct stat Fl;
            Qinf(FILE *fi) : f(fi) {
                int fd = fileno(f);
                fstat(fd, &Fl);
                bg = (char *)mmap(0, Fl.st_size + 1, PROT_READ, MAP_PRIVATE, fd, 0);
                p = bg, ed = bg + Fl.st_size;
                madvise(p, Fl.st_size + 1, MADV_SEQUENTIAL);
            }
            ~Qinf() { munmap(bg, Fl.st_size + 1); }
            void skip_space() {
                while (*p <= ' ') {
                    ++p;
                }
            }
            char get() { return *p++; }
            char seek() { return *p; }
            bool eof() { return p == ed; }
            Qinf &read(char *s, size_t count) { return memcpy(s, p, count), p += count, *this; }
            Qinf &operator>>(u32 &x) {
                skip_space(), x = 0;
                for (; *p > ' '; ++p) {
                    x = x * 10 + (*p & 0xf);
                }
                return *this;
            }
            Qinf &operator>>(int &x) {
                skip_space();
                if (*p == '-') {
                    for (++p, x = 48 - *p++; *p > ' '; ++p) {
                        x = x * 10 - (*p ^ 48);
                    }
                } else {
                    for (x = *p++ ^ 48; *p > ' '; ++p) {
                        x = x * 10 + (*p ^ 48);
                    }
                }
                return *this;
            }
        } qin(stdin);
    }  // namespace QIO_I
    namespace QIO_O {
        using namespace QIO_base;
        struct Qoutf {
            FILE *f;
            char *bg, *ed, *p;
            char *ed_thre;
            int fp;
            u64 _fpi;
            Qoutf(FILE *fo, size_t sz = O_buffer_default_size) : f(fo), bg(new char[sz]), ed(bg + sz), p(bg), ed_thre(ed - O_buffer_default_flush_threshold), fp(6), _fpi(1000000ull) {}
            void flush() { fwrite_unlocked(bg, 1, p - bg, f), p = bg; }
            void chk() {
                if (__builtin_expect(p > ed_thre, 0)) {
                    flush();
                }
            }
            ~Qoutf() {
                flush();
                delete[] bg;
            }
            void put4(u32 x) {
                if (x > 99u) {
                    if (x > 999u) {
                        memcpy(p, _otab.tab + (x << 2) + 0, 4), p += 4;
                    } else {
                        memcpy(p, _otab.tab + (x << 2) + 1, 3), p += 3;
                    }
                } else {
                    if (x > 9u) {
                        memcpy(p, _otab.tab + (x << 2) + 2, 2), p += 2;
                    } else {
                        *p++ = x ^ 48;
                    }
                }
            }
            void put2(u32 x) {
                if (x > 9u) {
                    memcpy(p, _otab.tab + (x << 2) + 2, 2), p += 2;
                } else {
                    *p++ = x ^ 48;
                }
            }
            Qoutf &write(const char *s, size_t count) {
                if (count > 1024 || p + count > ed_thre) {
                    flush(), fwrite_unlocked(s, 1, count, f);
                } else {
                    memcpy(p, s, count), p += count, chk();
                }
                return *this;
            }
            Qoutf &operator<<(char ch) { return *p++ = ch, *this; }
            Qoutf &operator<<(u32 x) {
                if (x > 99999999u) {
                    put2(x / 100000000u), x %= 100000000u;
                    memcpy(p, _otab.tab + ((x / 10000u) << 2), 4), p += 4;
                    memcpy(p, _otab.tab + ((x % 10000u) << 2), 4), p += 4;
                } else if (x > 9999u) {
                    put4(x / 10000u);
                    memcpy(p, _otab.tab + ((x % 10000u) << 2), 4), p += 4;
                } else {
                    put4(x);
                }
                return chk(), *this;
            }
            Qoutf &operator<<(int x) {
                if (x < 0) {
                    *p++ = '-', x = -x;
                }
                return *this << static_cast<u32>(x);
            }
        } qout(stdout);
    }  // namespace QIO_O
    namespace QIO {
        using QIO_I::qin;
        using QIO_I::Qinf;
        using QIO_O::qout;
        using QIO_O::Qoutf;
    }  // namespace QIO
    using namespace QIO;
};  // namespace __io
using namespace __io;

#include <cassert>
#include <iostream>

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    auto __total = cum_timer("total");

    NTT fft(998'244'353);

    int n, m;
    n = m = 5e5;
    qin >> n >> m;
    int lg = std::__lg(std::max(1, n + m - 2)) + 1;

    // int lg = 20;
    // int n = 1 << lg - 1;
    // int m = 1 << lg - 1;

    u32 *a = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
    u32 *b = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);

    {
        auto __ = cum_timer("input");

        for (int i = 0; i < n; i++) {
            qin >> a[i];
        }
        memset(a + n, 0, (4 << lg) - 4 * n);
        for (int i = 0; i < m; i++) {
            qin >> b[i];
        }
        memset(b + m, 0, (4 << lg) - 4 * m);
    }
    {
        auto __ = cum_timer("work");

        // for (int i = 0; i < 100; i++)
        //
        {
            // fft.convolve(lg, a, b);
            fft.convolve2(lg, a, b);
        }
    }
    {
        auto __ = cum_timer("output");
        for (int i = 0; i < (n + m - 1); i++) {
            qout << a[i] << " \n"[i + 1 == (n + m - 1)];
        }
    }

    return 0;
}
