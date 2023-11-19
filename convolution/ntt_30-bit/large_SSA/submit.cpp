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
        u32x8 x0246_ninv = mul64_u32x8(x0246, n_inv);
        u32x8 x1357_ninv = mul64_u32x8(x1357, n_inv);
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
        return (u64x4)shift_right_u32x8_epi128<4>((u32x8)mul_to_hi<strict>(a, b));
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
                    u32x8 a = load_u32x8(data + i + 0 * (1 << k - 2) + j);
                    u32x8 b = load_u32x8(data + i + 1 * (1 << k - 2) + j);
                    u32x8 c = load_u32x8(data + i + 2 * (1 << k - 2) + j);
                    u32x8 d = load_u32x8(data + i + 3 * (1 << k - 2) + j);

                    a = mts.shrink2(a);
                    b = mts.mul<false, true>(b, w1),
                    c = mts.mul<false, true>(c, w2),
                    d = mts.mul<false, true>(d, w3);

                    u32x8 a1 = mts.shrink2(a + c), b1 = mts.shrink2(b + d),
                          c1 = mts.shrink2_n(a - c),
                          d1 = mts.mul<false, true>(b + mts.mod2 - d, w_1);

                    store_u32x8(data + i + 0 * (1 << k - 2) + j, a1 + b1);
                    store_u32x8(data + i + 1 * (1 << k - 2) + j, a1 + mts.mod2 - b1);
                    store_u32x8(data + i + 2 * (1 << k - 2) + j, c1 + d1);
                    store_u32x8(data + i + 3 * (1 << k - 2) + j, c1 + mts.mod2 - d1);
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
                    u32x8 a = load_u32x8(data + i + 0 * (1 << k - 1) + j);
                    u32x8 b = load_u32x8(data + i + 1 * (1 << k - 1) + j);
                    u32x8 c = load_u32x8(data + i + 2 * (1 << k - 1) + j);
                    u32x8 d = load_u32x8(data + i + 3 * (1 << k - 1) + j);

                    u32x8 a1 = mts.shrink2(a + b), b1 = mts.shrink2_n(a - b),
                          c1 = mts.shrink2(c + d), d1 = mts.mul<false, true>(c + mts.mod2 - d, w_1);

                    store_u32x8(data + i + 0 * (1 << k - 1) + j, mts.shrink2(a1 + c1));
                    store_u32x8(data + i + 1 * (1 << k - 1) + j, mts.mul<false, true>(b1 + d1, w1));
                    store_u32x8(data + i + 2 * (1 << k - 1) + j, mts.mul<false, true>(a1 + mts.mod2 - c1, w2));
                    store_u32x8(data + i + 3 * (1 << k - 1) + j, mts.mul<false, true>(b1 + mts.mod2 - d1, w3));
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

    // a and b should be 32-byte aligned
    // writes (a * b) to a
    __attribute__((optimize("O3"))) [[gnu::noinline]] void convolve(int lg, __restrict__ u32 *a, __restrict__ u32 *b) const {
        if (lg <= 4) {
            int n = (1 << lg);
            __restrict__ u32 *c = (u32 *)_mm_malloc(n * 4, 4);
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

    __attribute__((optimize("O3"))) [[gnu::noinline]] void convolve_neg(int lg, u32 *__restrict__ a, u32 *__restrict__ b, u32 fc = 1u) const {
        auto mt = this->mt;
        auto mts = this->mts;

        fc = mt.mul<true>(mt.r2, fc);

        u32 w2 = power(g, mod - 1 >> lg + 1);
        if (lg <= 4) {
            u32 f = mt.r;
            for (int i = 0; i < (1 << lg); i++) {
                a[i] = mt.mul(a[i], f);
                b[i] = mt.mul(b[i], f);
                f = mt.mul(f, w2);
            }
        } else {
            u32x8 w2_8 = set1_u32x8(power(w2, 8));

            u32x8 fv[4];
            fv[0] = get_powers_u32x8(w2);

            fv[1] = mts.mul<true>(fv[0], w2_8);
            fv[2] = mts.mul<true>(fv[1], w2_8);
            fv[3] = mts.mul<true>(fv[2], w2_8);

            u32x8 w2_32 = set1_u32x8(power(w2, 32));

            for (int i = 0; i < (1 << lg); i += 32) {
                for (int j = 0; j < 4; j++) {
                    u32x8 aj = load_u32x8(a + i + 8 * j);
                    u32x8 bj = load_u32x8(b + i + 8 * j);
                    store_u32x8(a + i + 8 * j, mts.mul(aj, fv[j]));
                    store_u32x8(b + i + 8 * j, mts.mul(bj, fv[j]));
                    fv[j] = mts.mul(fv[j], w2_32);
                }
            }
        }
        convolve(lg, a, b);

        w2 = power(w2, mod - 2);
        if (lg <= 4) {
            u32 f = fc;
            for (int i = 0; i < (1 << lg); i++) {
                a[i] = mt.mul<true>(a[i], f);
                f = mt.mul(f, w2);
            }
        } else {
            u32x8 w2_8 = set1_u32x8(power(w2, 8));

            u32x8 fv[4];
            fv[0] = get_powers_u32x8(w2);
            fv[0] = mts.mul<true>(fv[0], set1_u32x8(fc));
            fv[1] = mts.mul<true>(fv[0], w2_8);
            fv[2] = mts.mul<true>(fv[1], w2_8);
            fv[3] = mts.mul<true>(fv[2], w2_8);

            u32x8 w2_32 = set1_u32x8(power(w2, 32));

            for (int i = 0; i < (1 << lg); i += 32) {
                for (int j = 0; j < 4; j++) {
                    u32x8 aj = load_u32x8(a + i + 8 * j);
                    store_u32x8(a + i + 8 * j, mts.mul<true>(aj, fv[j]));
                    fv[j] = mts.mul(fv[j], w2_32);
                }
            }
        }
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
};
#include <iostream>
#include <string>
#include <vector>

struct SSA {
    // ring modulo X^(2L) + 1
    struct Cum {
        u32 *ptr;
        size_t sh;

        u32 &operator()(int L, size_t ind) {
            return ptr[ind];
        };
    };

    NTT ntt;

    SSA(u32 mod = 998'244'353) : ntt(mod) {
        sizeof(SSA);
    }

    std::pair<int, int> get_LB(int lg) const {
        int L = (lg - 1) / 2;
        int B = lg - L;
        return {L, B};
    }

    // [a, b] -> [a + b, a - b]
    __attribute__((optimize("O3"))) void add_sub(int L, Cum &a, Cum &b) const {
        const auto mt = ntt.mt;
        const auto mts = ntt.mts;
        size_t sz = 1ULL << L + 1;
        // for (size_t i = 0; i < sz; i++) {
        //     u32 va = a(L, i), vb = b(L, i);
        //     a(L, i) = mt.shrink(va + vb);
        //     b(L, i) = mt.shrink_n(va - vb);
        // }
        for (size_t i = 0; i < sz; i += 8) {
            u32x8 va = load_u32x8(a.ptr + i);
            u32x8 vb = load_u32x8(b.ptr + i);
            store_u32x8(a.ptr + i, mts.shrink(va + vb));
            store_u32x8(b.ptr + i, mts.shrink_n(va - vb));
        }
    }

    __attribute__((optimize("O3"))) void mul(int L, size_t w, Cum &a) const {
        const auto mt = ntt.mt;
        size_t sz = 1ULL << L + 1;
        assert(w < 2 * sz);
        if (w >= sz) {
            w -= sz;
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (size_t i = w; i < sz; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        } else {
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (size_t i = 0; i < w; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        }
    }

    __attribute__((optimize("O3"))) void ssa_ntt_all_rec(int L, int B, int k, size_t w, Cum *cum_a, Cum *cum_b) const {
        if (k == 0) {
            auto mt = ntt.mt;
            u32 rv = mt.power(mt.mod + 1 >> 1, B);
            ntt.convolve_neg(L + 1, cum_a[0].ptr, cum_b[0].ptr, rv);
        } else {
            int dt = B - k;
            size_t sz = 1ULL << L + 1;

            size_t tf = (sz >> dt) * w;
            size_t tf_r = (sz * 2 - tf) & (sz * 2 - 1);

            for (auto cum : std::array{cum_a, cum_b}) {
                for (size_t i = 0; i < (1 << k - 1); i++) {
                    mul(L, tf, cum[i + (1 << k - 1)]);
                    add_sub(L, cum[i], cum[i + (1 << k - 1)]);
                }
            }

            ssa_ntt_all_rec(L, B, k - 1, w, cum_a, cum_b);
            ssa_ntt_all_rec(L, B, k - 1, w + (1ULL << dt), cum_a + (1 << k - 1), cum_b + (1 << k - 1));

            for (size_t i = 0; i < (1 << k - 1); i++) {
                add_sub(L, cum_a[i], cum_a[i + (1 << k - 1)]);
                mul(L, tf_r, cum_a[i + (1 << k - 1)]);
            }
        }
    }

    __attribute__((optimize("O3"))) void ssa_ntt_rec(int L, int B, int k, size_t w, Cum *cum) const {
        if (k == 0) {
            return;
        }
        int dt = B - k;
        size_t sz = 1ULL << L + 1;
        size_t tf = (sz >> dt) * w;
        for (size_t i = 0; i < (1 << k - 1); i++) {
            mul(L, tf, cum[i + (1 << k - 1)]);
            add_sub(L, cum[i], cum[i + (1 << k - 1)]);
        }

        ssa_ntt_rec(L, B, k - 1, w, cum);
        ssa_ntt_rec(L, B, k - 1, w + (1ULL << dt), cum + (1 << k - 1));
    }

    __attribute__((optimize("O3"))) void ssa_intt_rec(int L, int B, int k, size_t w, Cum *cum) const {
        if (k == 0) {
            return;
        }
        int dt = B - k;
        size_t sz = 1ULL << L + 1;
        size_t tf = (sz >> dt) * w;
        size_t tf_r = (sz * 2 - tf) & (sz * 2 - 1);

        ssa_intt_rec(L, B, k - 1, w, cum);
        ssa_intt_rec(L, B, k - 1, w + (1ULL << dt), cum + (1 << k - 1));

        for (size_t i = 0; i < (1 << k - 1); i++) {
            add_sub(L, cum[i], cum[i + (1 << k - 1)]);
            mul(L, tf_r, cum[i + (1 << k - 1)]);
        }
    }

    __attribute__((optimize("O3"))) void convolve(int lg, std::vector<Cum> &a, std::vector<Cum> &b) const {
        size_t sz = 1ULL << lg;
        assert(lg > 6);

        auto [L, B] = get_LB(lg);
        assert(a.size() == (1 << B));
        assert(b.size() == (1 << B));

        if (!"CUM") {
            ssa_ntt_rec(L, B, B, 0, a.data());
            ssa_ntt_rec(L, B, B, 0, b.data());

            auto mt = ntt.mt;
            u32 rv = mt.power(mt.mod + 1 >> 1, B);
            for (int i = 0; i < (1 << B); i++) {
                ntt.convolve_neg(L + 1, a[i].ptr, b[i].ptr, rv);
            }

            ssa_intt_rec(L, B, B, 0, a.data());
        } else {
            ssa_ntt_all_rec(L, B, B, 0, a.data(), b.data());
        }

        const auto mt = ntt.mt;
        const auto mts = ntt.mts;

        for (size_t i = 0; i < (1ULL << B); i++) {
            if (i + 1 < (1ULL << B)) {
                for (size_t j = 0; j < (1ULL << L); j++) {
                    a[i + 1].ptr[j] = mt.shrink(a[i + 1].ptr[j] + a[i].ptr[j + (1ULL << L)]);
                    a[i].ptr[j + (1ULL << L)] = 0;
                }
            }
        }
    }

    __attribute__((optimize("O3"))) std::vector<u32> convolve(const std::vector<u32> &a, const std::vector<u32> &b) const {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);
        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        if (lg <= 6) {
            return ntt.convolve(a, b);
        }

        auto [L, B] = get_LB(lg);
        std::vector<Cum> cum_a(1 << B), cum_b(1 << B);
        auto fill_cum = [&](std::vector<Cum> &cum, const std::vector<u32> &vec) {
            for (size_t i = 0; i < (1ULL << B); i++) {
                cum[i] = Cum{(u32 *)_mm_malloc(4ULL << L + 1, 64), 0};
                memset(cum[i].ptr, 0, 4ULL << L + 1);
                for (size_t j = 0; j < (1ULL << L); j++) {
                    size_t ind = i * (1 << L) + j;
                    if (ind < vec.size())
                        cum[i].ptr[j] = vec[ind];
                }
            }
        };
        fill_cum(cum_b, b);
        fill_cum(cum_a, a);

        convolve(lg, cum_a, cum_b);
        std::vector<u32> res(sz);
        for (size_t i = 0; i < (1ULL << B); i++) {
            for (size_t j = 0; j < (1ULL << L); j++) {
                size_t ind = i * (1ULL << L) + j;
                if (ind < sz)
                    res[ind] = cum_a[i].ptr[j];
            }
        }
        return res;
    }
};
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <iostream>

using u32 = uint32_t;
using u64 = uint64_t;

// io from https://judge.yosupo.jp/submission/142782

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
#pragma GCC target("avx2")
#include <cassert>
#include <iostream>

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    n = m = 1 << 24;
    qin >> n >> m;

    int sz = std::max(0, n + m - 1);
    int lg = std::__lg(std::max(1, sz - 1)) + 1;

    if (lg <= 6) {
        NTT ntt;
        u32 *a = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
        u32 *b = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
        for (int i = 0; i < n; i++) {
            qin >> a[i];
        }
        memset(a + n, 0, (4 << lg) - 4 * n);
        for (int i = 0; i < m; i++) {
            qin >> b[i];
        }
        memset(b + m, 0, (4 << lg) - 4 * m);
        ntt.convolve(lg, a, b);
        for (int i = 0; i < (n + m - 1); i++) {
            qout << a[i] << ' ';  // " \n"[i + 1 == (n + m - 1)];
        }
    } else {
        SSA ssa(998'244'353);
        auto [L, B] = ssa.get_LB(lg);

        clock_t beg = clock();

        std::vector<SSA::Cum> a(1 << B), b(1 << B);
        auto read = [L, B](auto &a, int n) {
            for (int i = 0; i < (1 << B); i++) {
                a[i] = SSA::Cum{(u32 *)_mm_malloc(4 << L + 1, 64), 0};
                int ind = i * (1 << L);
                int d = std::max(0, std::min(1 << L, n - ind));
                for (int j = 0; j < d; j++) {
                    qin >> a[i].ptr[j];
                }
                memset(a[i].ptr + d, 0, 4 * ((1 << L + 1) - d));
            }
        };
        read(a, n);
        read(b, m);

        std::cerr << "input " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
        beg = clock();

        ssa.convolve(lg, a, b);

        std::cerr << "convolution " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
        beg = clock();

        for (int i = 0; i < (1 << B); i++) {
            int ind = i * (1 << L);
            int d = std::max(0, std::min(1 << L, sz - ind));
            if (d == 0) {
                break;
            }
            for (int j = 0; j < d; j++) {
                qout << a[i].ptr[j] << ' ';
            }
        }
        std::cerr << "output " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
        beg = clock();
    }
    return 0;
}
