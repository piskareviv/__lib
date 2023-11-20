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

    u64x4 load_u64x4(u64* ptr) {
        return (u64x4)(_mm256_load_si256((i256*)ptr));
    }
    u64x4 loadu_u64x4(u64* ptr) {
        return (u64x4)(_mm256_loadu_si256((i256*)ptr));
    }
    void store_u64x4(u64* ptr, u64x4 val) {
        _mm256_store_si256((i256*)ptr, (i256)(val));
    }
    void storeu_u64x4(u64* ptr, u64x4 val) {
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
    template <bool strict = false>
    u32x8 mul(u32x8 a, u32x8 b) const {
        u32x8 x0246 = mul64_u32x8(a, b);
        u32x8 x1357 = mul64_u32x8(shift_right_u32x8_epi128<4>(a), shift_right_u32x8_epi128<4>(b));
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

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <vector>


struct SSA_20 {
    u32 mod;
    Montgomery mt;
    Montgomery_simd mts;

    SSA_20(u32 mod = 1'000'000'007) : mod(mod), mt(mod), mts(mod) {
        ;
    }

    struct Cum_6 {
        u32* ptr;
    };

    // writes a * b to a
    // __attribute__((optimize("O3")))
    void convolve_6(u32* a, u32* b) {
        // return;
        if (0) {
            alignas(64) u32 aux[1 << 6 + 1];

            memset(aux, 0, sizeof(aux));
            for (int i = 0; i < 64; i++) {
                for (int j = 0; j < 64; j++) {
                    aux[i + j] = mt.shrink(aux[i + j] + mt.mul<true>(a[i], b[j]));
                }
            }
            for (int i = 0; i < 64; i++) {
                a[i] = mt.shrink_n(aux[i] - aux[i + 64]);
                a[i] = mt.mul<true>(a[i], mt.r2);
            }
            return;
        } else {
            constexpr int sz = 64;
            alignas(64) u64 aux_a[sz * 2], aux_b[sz];
            for (int i = 0; i < sz; i++) {
                aux_b[i] = b[i];
            }
            for (int i = 1; i <= sz; i++) {
                aux_a[i - 1] = a[sz - i];
            }
            for (int i = 0; i < sz; i++) {
                aux_a[i + sz] = mod - aux_a[i];
            }

            u64x4 mod2_8 = set1_u64x4(mod * 1ULL * mod * 8);
            u64x4 mod2_4 = set1_u64x4(mod * 1ULL * mod * 4);
            u64x4 mod2_2 = set1_u64x4(mod * 1ULL * mod * 2);
            u64x4 mod2_1 = set1_u64x4(mod * 1ULL * mod * 1);

            for (int i = 0; i < sz; i++) {
                int sh = sz - 1 - i;
                u64x4 sum[4];
                memset(sum, 0, sizeof(sum));
                for (int j = 0; j < 64; j += 16) {
                    for (int t = 0; t < 4; t++) {
                        sum[t] += (u64x4)mul64_u32x8((u32x8)loadu_u64x4(aux_a + sh + j + t * 4),
                                                     (u32x8)load_u64x4(aux_b + j + t * 4));
                    }
                }
                u64x4 sm = (sum[0] + sum[1]) + (sum[2] + sum[3]);
                sm = sm < mod2_8 ? sm : sm - mod2_8;
                sm = sm < mod2_4 ? sm : sm - mod2_4;
                // sm = sm < mod2_2 ? sm : sm - mod2_2;
                // sm = sm < mod2_1 ? sm : sm - mod2_1;
                sm = sm + (u64x4)permute_u32x8_epi128<1>((u32x8)sm, (u32x8)sm);
                sm = sm + (u64x4)shuffle_u32x8<0b00'00'11'10>((u32x8)sm);
                u64 res = _mm256_extract_epi64((i256)sm, 0);

                res = res % u32(1e9 + 7);  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING  CUM !!!!!!!,  sorry for this
                a[i] = res;
            }
        }
    }

    // writes a * b to a
    __attribute__((optimize("O3"))) void convolve_6(Cum_6 a, Cum_6 b) {
        convolve_6(a.ptr, b.ptr);
    }

    // [a, b] -> [a + b, a - b]
    [[gnu::noinline]] __attribute__((optimize("O3"))) void add_sub_6(Cum_6 a, Cum_6 b) const {
        const auto mts = this->mts;
        constexpr int sz = 1ULL << 6;
        for (int i = 0; i < sz; i += 8) {
            u32x8 va = load_u32x8(a.ptr + i);
            u32x8 vb = load_u32x8(b.ptr + i);
            store_u32x8(a.ptr + i, mts.shrink(va + vb));
            store_u32x8(b.ptr + i, mts.shrink_n(va - vb));
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void mul_6_by_w(int w, Cum_6 a) const {
        const auto mt = this->mt;
        constexpr int sz = 1ULL << 6;
        assert(w < 2 * sz);
        if (w >= sz) {
            w -= sz;
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (int i = w; i < sz; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        } else {
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (int i = 0; i < w; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void ssa10_all_rec(int k, int w, Cum_6* cum_a, Cum_6* cum_b) {
        if (k == 0) {
            convolve_6(cum_a[0], cum_b[0]);
        } else {
            int dt = 5 - k;
            int sz = 1ULL << 5 + 1;

            int tf = (sz >> dt) * w;
            int tf_r = (sz * 2 - tf) & (sz * 2 - 1);

            for (auto cum : std::array<Cum_6*, 2>{cum_a, cum_b}) {
                for (int i = 0; i < (1 << k - 1); i++) {
                    mul_6_by_w(tf, cum[i + (1 << k - 1)]);
                    add_sub_6(cum[i], cum[i + (1 << k - 1)]);
                }
            }

            ssa10_all_rec(k - 1, w, cum_a, cum_b);
            ssa10_all_rec(k - 1, w + (1ULL << dt), cum_a + (1 << k - 1), cum_b + (1 << k - 1));

            for (int i = 0; i < (1 << k - 1); i++) {
                add_sub_6(cum_a[i], cum_a[i + (1 << k - 1)]);
                mul_6_by_w(tf_r, cum_a[i + (1 << k - 1)]);
            }
        }
    }

    struct Cum_10 {
        u32* ptr;
        int sh;
    };

    // writes a * b to a
    __attribute__((optimize("O3"))) void convolve_10(Cum_10 a, Cum_10 b) {
        alignas(128) u32 aux_a[1 << 10 + 1];
        alignas(128) u32 aux_b[1 << 10 + 1];
        Cum_6 cum_a[1 << 5], cum_b[1 << 5];
        {
            auto fuck = [&](u32* aux, Cum_10 a, Cum_6* cum) {
                memset(aux, 0, 4 << 10 + 1);
                for (int i = 0; i < (1 << 5); i++) {
                    memcpy(aux + i * 64, a.ptr + i * 32, 4 << 5);
                    cum[i] = Cum_6{aux + i * 64};

                    mul_6_by_w(i * 2, cum[i]);
                }
            };
            fuck(aux_a, a, cum_a);
            fuck(aux_b, b, cum_b);
        }

        ssa10_all_rec(5, 0, cum_a, cum_b);

        u32 f = mt.mul<true>(mt.r2, mt.power(mod + 1 >> 1, 16));
        u32x8 fx8 = set1_u32x8(f);

        for (int i = 0; i < (1 << 5); i++) {
            mul_6_by_w(((1 << 7) - i * 2) % (1 << 7), cum_a[i]);
        }
        for (int i = 0; i < (1 << 5); i++) {
            int i2 = i == 0 ? (1 << 5) - 1 : i - 1;
            for (int j = 0; j < 4; j++) {
                u32x8 a0 = load_u32x8(aux_a + i * 64 + j * 8);
                u32x8 a1 = load_u32x8(aux_a + i2 * 64 + 32 + j * 8);
                if (i == 0) {
                    a1 = mod - a1;
                }
                store_u32x8(a.ptr + i * 32 + j * 8, mts.mul<true>(a0 + a1, fx8));
            }
        }
    }

    // [a, b] -> [a + b, a - b]
    [[gnu::noinline]] __attribute__((optimize("O3"))) void add_sub_10(Cum_10 a, Cum_10 b) const {
        const auto mts = this->mts;
        constexpr int sz = 1ULL << 10;
        for (int i = 0; i < sz; i += 8) {
            u32x8 va = load_u32x8(a.ptr + i);
            u32x8 vb = load_u32x8(b.ptr + i);
            store_u32x8(a.ptr + i, mts.shrink(va + vb));
            store_u32x8(b.ptr + i, mts.shrink_n(va - vb));
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void mul_10_by_w(int w, Cum_10 a) const {
        const auto mt = this->mt;
        constexpr int sz = 1ULL << 10;
        assert(w < 2 * sz);
        if (w >= sz) {
            w -= sz;
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (int i = w; i < sz; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        } else {
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (int i = 0; i < w; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void ssa20_all_rec(int k, int w, Cum_10* cum_a, Cum_10* cum_b) {
        if (k == 0) {
            convolve_10(cum_a[0], cum_b[0]);
        } else {
            int dt = 11 - k;
            int sz = 1ULL << 9 + 1;

            int tf = (sz >> dt) * w;
            int tf_r = (sz * 2 - tf) & (sz * 2 - 1);

            for (auto cum : std::array{cum_a, cum_b}) {
                for (int i = 0; i < (1 << k - 1); i++) {
                    mul_10_by_w(tf, cum[i + (1 << k - 1)]);
                    add_sub_10(cum[i], cum[i + (1 << k - 1)]);
                }
            }

            ssa20_all_rec(k - 1, w, cum_a, cum_b);
            ssa20_all_rec(k - 1, w + (1ULL << dt), cum_a + (1 << k - 1), cum_b + (1 << k - 1));

            for (int i = 0; i < (1 << k - 1); i++) {
                add_sub_10(cum_a[i], cum_a[i + (1 << k - 1)]);
                mul_10_by_w(tf_r, cum_a[i + (1 << k - 1)]);
            }
        }
    }

    // writes a * b to a
    [[gnu::noinline]] __attribute__((optimize("O3"))) void convolve_20(std::vector<Cum_10>& a, std::vector<Cum_10>& b) {
        ssa20_all_rec(11, 0, a.data(), b.data());
        for (int i = 0; i < (1 << 11); i++) {
            int j = (i + 1) % (1 << 11);
            for (int t = 0; t < (1 << 9); t++) {
                a[j].ptr[t] = mt.shrink(a[j].ptr[t] + a[i].ptr[t + (1 << 9)]);
            }
        }
    }
};

// 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

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

// #define qin std::cin
// #define qout std::cout


int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    n = m = 1 << 19;
    qin >> n >> m;

    int sz = std::max(0, n + m - 1);

    SSA_20 ssa(1e9 + 7);
    // auto [L, B] = ssa.get_LB(lg);
    const auto L = 9;
    const auto B = 11;

    clock_t beg = clock();

    std::vector<SSA_20::Cum_10> a(1 << B), b(1 << B);
    auto read = [L, B](auto &a, int n) {
        u32 *ptr = (u32 *)_mm_malloc(4 << L + B + 1, 64);

        for (int i = 0; i < (1 << B); i++) {
            a[i] = SSA_20::Cum_10{ptr + i * (1 << L + 1), 0};

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

    ssa.convolve_20(a, b);

    std::cerr << "convolution " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
    beg = clock();

    for (int i = 0; i < (1 << B); i++) {
        int ind = i * (1 << L);
        int d = std::max(0, std::min(1 << L, sz - ind));
        if (d == 0) {
            break;
        }
        for (int j = 0; j < d; j++) {
            assert(a[i].ptr[j] < 1e9 + 7);
            qout << a[i].ptr[j] << ' ';
        }
    }
    qout << '\n';

    std::cerr << "output " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
    beg = clock();

    return 0;
}

