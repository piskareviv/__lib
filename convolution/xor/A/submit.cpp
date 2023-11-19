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
//     constexpr void operator()(Fn const& fn) const {
//         if (First < Last) {
//             fn(First);
//             static_for2<First + 1, Last>()(fn);
//         }
//     }
// };

// template <int N>
// struct static_for2<N, N> {
//     template <typename Fn>
//     constexpr void operator()(Fn const& fn) const {}
// };

// template <int N>
// using static_for = static_for2<0, N>;

template <typename T>
void static_consume(std::initializer_list<T>) {}

template <typename Functor, std::size_t... S>
constexpr void static_foreach_seq(Functor &&function, std::index_sequence<S...>) {
    return static_consume({(function(std::integral_constant<std::size_t, S>{}), 0)...});
}

template <std::size_t Size, typename Functor>
constexpr void static_for(Functor &&functor) {
    return static_foreach_seq(std::forward<Functor>(functor), std::make_index_sequence<Size>());
}

struct Cum {
    Montgomery mt;
    Montgomery_simd mts;

    Cum() = default;
    Cum(u32 mod) : mt(mod), mts(mod) { ; }

    // data must be 32 bytes aligned
    __attribute__((optimize("O3"))) void fwht_(int lg, u32 *data) const {
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

            auto proc = [&](u32x8 &val) {
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

    // data must be 32 bytes aligned
    __attribute__((optimize("O3"))) void fwht(int lg, u32 *data) const {
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

            auto proc = [&](u32x8 &val) {
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
#define FUCK(r)                                                            \
    if (k + r <= lg) {                                                     \
        u32x8 dt[1 << r];                                                  \
        for (int i = 0; i < (1 << lg); i += (1 << k + r)) {                \
            for (int j = 0; j < (1 << k); j += 8) {                        \
                static_for<1 << r>([&](auto it) {                          \
                    dt[it] = load_u32x8(data + i + j + (1 << k) * it);     \
                });                                                        \
                static_for<r>([&](auto k) constexpr {                      \
                    static_for<1 << r - k>([&](size_t i) constexpr {       \
                        i *= (1 << k + 1);                                 \
                        static_for<1 << k>([&](auto j) constexpr {         \
                            u32x8 a = dt[i + j], b = dt[i + (1 << k) + j]; \
                            dt[i + j] = mts.shrink(a + b);                 \
                            dt[i + (1 << k) + j] = mts.shrink_n(a - b);    \
                        });                                                \
                    });                                                    \
                });                                                        \
                static_for<1 << r>([&](auto it) {                          \
                    store_u32x8(data + i + j + (1 << k) * it, dt[it]);     \
                });                                                        \
            }                                                              \
        }                                                                  \
        k += r - 1;                                                        \
        continue;                                                          \
    }

                // FUCK(5);
                // FUCK(4);
                // FUCK(3);
                FUCK(2);
                FUCK(1);
#undef FUCK
                // for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                //     for (int j = 0; j < (1 << k); j += 8) {
                //         u32x8 a = load_u32x8(data + i + j), b = load_u32x8(data + i + (1 << k) + j);
                //         store_u32x8(data + i + j, mts.shrink(a + b));
                //         store_u32x8(data + i + (1 << k) + j, mts.shrink_n(a - b));
                //     }
                // }
            }
        }
    }

    // a and b must be 32 bytes aligned
    void convolve_xor(int lg, __restrict__ u32 *a, __restrict__ u32 *b) const {
        fwht(lg, a);
        fwht(lg, b);

        u32 f = mt.mul<true>(mt.mul(mt.r2, mt.r2), mt.power(mt.mod + 1 >> 1, lg));
        if (lg < 3) {
            for (int i = 0; i < (1 << lg); i++) {
                a[i] = mt.mul<true>(f, mt.mul(a[i], b[i]));
            }
        } else {
            auto mts = this->mts;
            u32x8 f_x8 = set1_u32x8(f);
            for (int i = 0; i < (1 << lg); i += 8) {
                u32x8 ai = load_u32x8(a + i), bi = load_u32x8(b + i);
                ai = mts.mul<true>(f_x8, mts.mul(ai, bi));
                store_u32x8(a + i, ai);
            }
        }
        fwht(lg, a);
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
#include <iostream>
#include <vector>

int32_t main() {
    int lg;
    qin >> lg;
    int n = 1 << lg;
    u32 *a = (u32 *)_mm_malloc(n * 4, std::min(n * 4, 64));
    u32 *b = (u32 *)_mm_malloc(n * 4, std::min(n * 4, 64));
    for (int i = 0; i < n; i++) {
        qin >> a[i];
    }
    for (int i = 0; i < n; i++) {
        qin >> b[i];
    }

    Cum cum(998'244'353);

    cum.convolve_xor(lg, a, b);

    for (int i = 0; i < n; i++) {
        qout << a[i] << " \n"[i == n - 1];
    }

    return 0;
}
