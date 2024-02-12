#include <iostream>
#pragma GCC optimize("O3")
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

    u64x8 load_u64x8(u64 *ptr) {
        return (u64x8)(_mm512_load_si512((i512 *)ptr));
    }
    u64x8 loadu_u64x8(u64 *ptr) {
        return (u64x8)(_mm512_loadu_si512((i512 *)ptr));
    }
    void store_u64x8(u64 *ptr, u64x8 val) {
        _mm512_store_si512((i512 *)ptr, (i512)(val));
    }
    void storeu_u64x8(u64 *ptr, u64x8 val) {
        _mm512_storeu_si512((i512 *)ptr, (i512)(val));
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
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

struct NTT {
    Montgomery mt;
    Montgomery_simd mts;
    u64 mod, g;

    [[gnu::noinline]] u64 power(u64 base, u64 exp) const {
        const auto mt = this->mt;  // ! to put Montgomery constants in registers
        u64 res = mt.r;
        for (; exp > 0; exp >>= 1) {
            if (exp & 1) {
                res = mt.mul(res, base);
            }
            base = mt.mul(base, base);
        }
        return mt.shrink(res);
    }

    // mod should be prime
    u64 find_pr_root(u64 mod) const {
        u64 m = mod - 1;
        std::vector<u64> vec;
        for (u64 i = 2; u128(i) * i <= m; i++) {
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
        for (u64 i = 2;; i++) {
            if (std::all_of(vec.begin(), vec.end(), [&](u64 f) { return mt.r != power(mt.mul(i, mt.r2), (mod - 1) / f); })) {
                return i;
            }
        }
    }

    NTT(u64 mod = 998'244'353) : mt(mod), mts(mod), mod(mod) {
        g = mt.mul<true>(mt.r2, find_pr_root(mod));
    }

    [[gnu::noinline]] std::pair<std::vector<u64>, u64x8 *> make_cum(int lg, bool inv = false) const {
        lg -= 2;

        const auto mt = this->mt;  // ! to put Montgomery constants in registers
        std::vector<u64> w_cum(lg);
        for (int i = 0; i < lg; i++) {
            u64 f = power(g, (mod - 1) >> i + 3);
            if (inv) {
                f = power(f, mod - 2);
            }
            u64 res = mt.mul<true>(f, power(power(f, (1 << i + 1) - 2), mod - 2));
            w_cum[i] = res;
        }
        u64x8 *w_cum_x8 = (u64x8 *)_mm_malloc(64 * lg, 64);

        for (int i = 0; i < lg; i++) {
            u64 f = power(g, (mod - 1) >> i + 4);
            if (inv) {
                f = power(f, mod - 2);
            }
            f = mt.mul<true>(f, power(power(f, (1 << i + 1) - 2), mod - 2));
            w_cum_x8[i][0] = mt.r;
            for (int j = 1; j < 8; j++) {
                w_cum_x8[i][j] = mt.mul<true>(w_cum_x8[i][j - 1], f);
            }
        }
        return {w_cum, w_cum_x8};
    }

    // input data[i] in [0, 4 * mod)
    // output data[i] in [0, 4 * mod)
    [[gnu::noinline]] __attribute__((optimize("O3"))) void fft(int lg, u64 *data) const {
        auto [w_cum, w_cum_x8] = make_cum(lg, false);
        const auto mt = this->mt;    // ! to put Montgomery constants in registers
        const auto mts = this->mts;  // ! to put Montgomery constants in registers

        int n = 1 << lg;
        int k = lg;

        if (lg % 2 == 0) {
            for (int i = 0; i < n / 2; i += 8) {
                u64x8 a = load_u64x8(data + i);
                u64x8 b = load_u64x8(data + n / 2 + i);

                store_u64x8(data + i, mts.shrink2(a + b));
                store_u64x8(data + n / 2 + i, mts.shrink2_n(a - b));
            }
            k--;
        }

        assert(k % 2 == 1);
        for (; k > 4; k -= 2) {
            u64 wj = mt.r;
            u64x8 w_1 = set1_u64x8(power(g, (mod - 1) / 4));
            for (int i = 0; i < n; i += (1 << k)) {
                u64 wj2 = mt.mul<true>(wj, wj);
                u64x8 w1 = set1_u64x8(wj);
                u64x8 w2 = set1_u64x8(wj2);
                u64x8 w3 = set1_u64x8(mt.mul<true>(wj, wj2));
                wj = mt.mul<true>(wj, w_cum[__builtin_ctz(~(i >> k))]);

                for (int j = 0; j < (1 << k - 2); j += 8) {
                    u64x8 a = load_u64x8(data + i + 0 * (1 << k - 2) + j);
                    u64x8 b = load_u64x8(data + i + 1 * (1 << k - 2) + j);
                    u64x8 c = load_u64x8(data + i + 2 * (1 << k - 2) + j);
                    u64x8 d = load_u64x8(data + i + 3 * (1 << k - 2) + j);

                    a = mts.shrink2(a);
                    b = mts.mul(b, w1), c = mts.mul(c, w2), d = mts.mul(d, w3);

                    u64x8 a1 = mts.shrink2(a + c), b1 = mts.shrink2(b + d),
                          c1 = mts.shrink2_n(a - c), d1 = mts.mul(b + mts.mod2 - d, w_1);

                    store_u64x8(data + i + 0 * (1 << k - 2) + j, a1 + b1);
                    store_u64x8(data + i + 1 * (1 << k - 2) + j, a1 + mts.mod2 - b1);
                    store_u64x8(data + i + 2 * (1 << k - 2) + j, c1 + d1);
                    store_u64x8(data + i + 3 * (1 << k - 2) + j, c1 + mts.mod2 - d1);
                }
            }
        }

        assert(k == 3);

        std::array<u64, 4> w;
        w[0] = mt.r;
        w[1] = power(g, (mod - 1) / 4);
        w[2] = power(g, (mod - 1) / 8);
        w[3] = mt.mul<true>(w[1], w[2]);

        u64x8 cum0 = setr_u64x8(w[0], w[0], w[0], w[0], w[0], w[0], w[1], w[1]);
        u64x8 cum1 = setr_u64x8(w[0], w[0], w[0], w[1], w[0], w[2], w[0], w[3]);
        u64x8 cum = set1_u64x8(mt.r);
        for (int i = 0; i < n; i += 8) {
            u64x8 vec = load_u64x8(data + i);

            vec = mts.mul(vec, cum);
            vec = mts.mul(blend_u64x8<0b11'11'00'00>(vec, mts.mod2 - vec) + permute_u64x8_i64x2<0b01'00'11'10>(vec), cum0);
            vec = mts.mul(blend_u64x8<0b11'00'11'00>(vec, mts.mod2 - vec) + shuffle_u64x8<0b01'00'11'10>(vec), cum1);
            vec = blend_u64x8<0b10'10'10'10>(vec, mts.mod2 - vec) + shuffle_u64x8<0b10'11'00'01>(vec);

            store_u64x8(data + i, vec);

            cum = mts.mul<true>(cum, w_cum_x8[__builtin_ctz(~(i >> 3))]);
        }

        _mm_free(w_cum_x8);
    }

    // input data[i] in [0, 4 * mod)
    // output data[i] in [0, mod)
    [[gnu::noinline]] __attribute__((optimize("O3"))) void ifft(int lg, u64 *data) const {
        auto [w_cum, w_cum_x8] = make_cum(lg, true);
        const auto mt = this->mt;    // ! to put Montgomery constants in registers
        const auto mts = this->mts;  // ! to put Montgomery constants in registers

        int n = 1 << lg;
        int k = 1;
        {
            std::array<u64, 4> w;
            w[0] = mt.r;
            w[1] = power(power(g, mod - 2), (mod - 1) / 4);
            w[2] = power(power(g, mod - 2), (mod - 1) / 8);
            w[3] = mt.mul<true>(w[1], w[2]);

            u64x8 cum0 = setr_u64x8(w[0], w[0], w[0], w[0], w[0], w[0], w[1], w[1]);
            u64x8 cum1 = setr_u64x8(w[0], w[0], w[0], w[1], w[0], w[2], w[0], w[3]);

            u64 rv = mt.mul<true>(mt.r2, power(mt.mul<true>(mt.r2, 1 << lg), mod - 2));
            u64x8 cum = set1_u64x8(rv);

            for (int i = 0; i < n; i += 8) {
                u64x8 vec = load_u64x8(data + i);

                vec = mts.mul(cum1, blend_u64x8<0b10'10'10'10>(vec, mts.mod2 - vec) + shuffle_u64x8<0b10'11'00'01>(vec));
                vec = mts.mul(cum0, blend_u64x8<0b11'00'11'00>(vec, mts.mod2 - vec) + shuffle_u64x8<0b01'00'11'10>(vec));
                vec = mts.mul(cum, blend_u64x8<0b11'11'00'00>(vec, mts.mod2 - vec) + permute_u64x8_i64x2<0b01'00'11'10>(vec));

                store_u64x8(data + i, vec);

                cum = mts.mul<true>(cum, w_cum_x8[__builtin_ctz(~(i >> 3))]);
            }

            _mm_free(w_cum_x8);

            k += 3;
        }

        for (; k + 1 <= lg; k += 2) {
            u64 wj = mt.r;
            u64x8 w_1 = set1_u64x8(power(power(g, mod - 2), (mod - 1) / 4));

            for (int i = 0; i < n; i += (1 << k + 1)) {
                u64 wj2 = mt.mul<true>(wj, wj);
                u64x8 w1 = set1_u64x8(wj);
                u64x8 w2 = set1_u64x8(wj2);
                u64x8 w3 = set1_u64x8(mt.mul<true>(wj, wj2));
                wj = mt.mul<true>(wj, w_cum[__builtin_ctz(~(i >> k + 1))]);

                for (int j = 0; j < (1 << k - 1); j += 8) {
                    u64x8 a = load_u64x8(data + i + 0 * (1 << k - 1) + j);
                    u64x8 b = load_u64x8(data + i + 1 * (1 << k - 1) + j);
                    u64x8 c = load_u64x8(data + i + 2 * (1 << k - 1) + j);
                    u64x8 d = load_u64x8(data + i + 3 * (1 << k - 1) + j);

                    u64x8 a1 = mts.shrink2(a + b), b1 = mts.shrink2_n(a - b),
                          c1 = mts.shrink2(c + d), d1 = mts.mul(c + mts.mod2 - d, w_1);

                    store_u64x8(data + i + 0 * (1 << k - 1) + j, mts.shrink2(a1 + c1));
                    store_u64x8(data + i + 1 * (1 << k - 1) + j, mts.mul(w1, b1 + d1));
                    store_u64x8(data + i + 2 * (1 << k - 1) + j, mts.mul(w2, a1 + mts.mod2 - c1));
                    store_u64x8(data + i + 3 * (1 << k - 1) + j, mts.mul(w3, b1 + mts.mod2 - d1));
                }
            }
        }
        if (k == lg) {
            for (int i = 0; i < n / 2; i += 8) {
                u64x8 a = load_u64x8(data + i);
                u64x8 b = load_u64x8(data + n / 2 + i);

                store_u64x8(data + i, mts.shrink(mts.shrink2(a + b)));
                store_u64x8(data + n / 2 + i, mts.shrink(mts.shrink2_n(a - b)));
            }
        } else {
            for (int i = 0; i < n; i += 8) {
                u64x8 ai = load_u64x8(data + i);
                store_u64x8(data + i, mts.shrink(ai));
            }
        }
    }

    __attribute__((optimize("O3"))) std::vector<u64> convolve_slow(std::vector<u64> a, std::vector<u64> b) const {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);
        const auto mt = this->mt;  // ! to put Montgomery constants in registers

        std::vector<u64> c(sz);

        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                // c[i + j] = (c[i + j] + u128(a[i]) * b[j]) % mod;
                c[i + j] = mt.shrink(c[i + j] + mt.mul<true>(mt.r2, mt.mul(a[i], b[j])));
            }
        }

        return c;
    }

    __attribute__((optimize("O3"))) [[gnu::noinline]] void convolve(int lg, __restrict__ u64 *a, __restrict__ u64 *b) const {
        int sz = 1 << lg;
        if (lg <= 4) {
            auto c = convolve_slow(std::vector<u64>(a, a + sz), std::vector<u64>(b, b + sz));
            memcpy(a, c.data(), 8 * sz);
            return;
        }

        fft(lg, a);
        fft(lg, b);

        const auto mts = this->mts;  // ! to put Montgomery constants in registers
        for (int i = 0; i < (1 << lg); i += 8) {
            u64x8 ai = load_u64x8(a + i), bi = load_u64x8(b + i);
            store_u64x8(a + i, mts.mul(mts.shrink2(ai), mts.shrink2(bi)));
        }

        ifft(lg, a);
    }

    __attribute__((optimize("O3"))) std::vector<u64> convolve(const std::vector<u64> &a, const std::vector<u64> &b) const {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        u64 *ap = (u64 *)_mm_malloc(std::max(64, (1 << lg) * 8), 64);
        u64 *bp = (u64 *)_mm_malloc(std::max(64, (1 << lg) * 8), 64);
        memset(ap, 0, 8 << lg);
        memset(bp, 0, 8 << lg);
        std::copy(a.begin(), a.end(), ap);
        std::copy(b.begin(), b.end(), bp);

        convolve(lg, ap, bp);

        std::vector<u64> res(ap, ap + sz);
        _mm_free(ap);
        _mm_free(bp);
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

#include <cassert>
#include <iostream>

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // auto __total = cum_timer("total");

    NTT fft(998'244'353);

    int n, m;
    n = m = 5e5;
    qin >> n >> m;
    int lg = std::__lg(std::max(1, n + m - 2)) + 1;

    // int lg = 20;
    // int n = 1 << lg - 1;
    // int m = 1 << lg - 1;

    u64 *a = (u64 *)_mm_malloc(std::max(64, (1 << lg) * 8), 64);
    u64 *b = (u64 *)_mm_malloc(std::max(64, (1 << lg) * 8), 64);

    {
        // auto __ = cum_timer("input");

        for (int i = 0; i < n; i++) {
            u32 ai;
            qin >> ai;
            a[i] = ai;
            // qin >> a[i];
        }
        memset(a + n, 0, (8 << lg) - 8 * n);
        for (int i = 0; i < m; i++) {
            u32 bi;
            qin >> bi;
            b[i] = bi;
            // qin >> b[i];
        }
        memset(b + m, 0, (8 << lg) - 8 * m);
    }
    {
        auto __ = cum_timer("work");
        // for (int i = 0; i < 100; i++)
        //
        {
            fft.convolve(lg, a, b);
        }
    }
    {
        // auto __ = cum_timer("output");
        for (int i = 0; i < (n + m - 1); i++) {
            qout << (u32)a[i] << " \n"[i + 1 == (n + m - 1)];
        }
    }

    return 0;
}
