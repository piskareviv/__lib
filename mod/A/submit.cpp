#pragma GCC optimize("O3")
// #pragma GCC target("avx512f,avx512vl,avx512dq")
#pragma GCC target("avx2")
#include <immintrin.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

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

    u32 shrink_4(u32 val) {
        return shrink(shrink_2(val));
    }

    // from [0, 2 * mod * mod) to [0, 2 * mod)
    u32 mod_22(u64 val) {
        u32 a = u32(val >> s) * u64(q) >> 32;
        u32 res = u32(val) - a * mod;
        return res;
    }

    // from [0, 2 * mod * mod) to [0, mod)
    // should work for every mod
    u32 mod_21(u64 val) {
        u32 res = mod_22(val);
        res -= mod * (res >= mod) + mod * (res >= mod2);
        return res;
    }

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
};

struct Barrett_simd {
    Barrett bt;

    u32 s, s_1;
    u32x8 v_mod, v_q;
    u32x8 v_mod2;

    Barrett_simd(u32 m) : bt(m) {
        s = bt.s;
        s_1 = bt.s_1;
        v_q = RC(u32x8, _mm256_set1_epi32(bt.q));
        v_mod = RC(u32x8, _mm256_set1_epi32(bt.mod));
        v_mod2 = RC(u32x8, _mm256_set1_epi32(bt.mod2));
    }

    // from [0, 2 * mod * mod) to [0, 2 * mod)
    u64x4 mod_22(u64x4 vec) {
        i256 a = _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(RC(i256, vec), s), RC(i256, v_q)), 32);
        i256 res = _mm256_sub_epi64(RC(i256, vec), _mm256_mul_epu32(a, RC(i256, v_mod)));
        return RC(u64x4, res);
    }

    // from [0, 4 * mod * mod) to [0, 4 * mod)
    u64x4 mod_44(u64x4 vec) {
        i256 a = _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(RC(i256, vec), s_1), RC(i256, v_q)), 32);
        i256 res = _mm256_sub_epi64(RC(i256, vec), _mm256_mul_epu32(a, RC(i256, v_mod2)));
        return RC(u64x4, res);
    }

    // from [0, 4 * mod * mod) to [0, 2 * mod)
    i256 mod_42(u64x4 vec) {
        i256 res = RC(i256, mod_44(vec));
        res = _mm256_min_epu32(res, _mm256_sub_epi32(res, RC(i256, v_mod2)));
        return res;
    }

    // product [0, 4 * mod * mod), result [0, 2 * mod)
    u32x8 mul_mod_42(u32x8 a, u32x8 b) {
        const u32 shuflle_mask = 0b10'11'00'01;
        i256 x0246 = _mm256_mul_epu32(RC(i256, a), RC(i256, b));
        i256 x1357 = _mm256_mul_epu32(_mm256_shuffle_epi32(RC(i256, a), shuflle_mask), _mm256_shuffle_epi32(RC(i256, b), shuflle_mask));
        x0246 = RC(i256, mod_44(RC(u64x4, x0246)));
        x1357 = RC(i256, mod_44(RC(u64x4, x1357)));
        i256 res = _mm256_blend_epi32(x0246, _mm256_shuffle_epi32(x1357, shuflle_mask), 0b10'10'10'10);
        res = _mm256_min_epu32(res, _mm256_sub_epi32(res, RC(i256, v_mod2)));
        return RC(u32x8, res);
    }
};

namespace FFT {
    constexpr u32 mod = 998'244'353;
    constexpr u32 pr_root = 3;

    Barrett bt(mod);
    Barrett_simd bts(mod);

    u32 power(u32 base, u32 exp) {
        u32 res = 1;
        for (; exp > 0; exp >>= 1) {
            if (exp & 1) {
                res = bt.mod_42(u64(res) * base);
            }
            base = bt.mod_42(u64(base) * base);
        }
        res = bt.shrink_4(res);
        return res;
    }

    std::vector<std::vector<u32>> w, w_rb;
    std::vector<std::vector<int>> rev_bits;

    void expand_w(int k) {
        while (w.size() < k) {
            int r = w.size();
            w.emplace_back(1 << r);
            if (r == 0) {
                w.back()[0] = 1;
            } else {
                u32 f = power(pr_root, (mod - 1) >> r + 1);
                for (int j = 0; j < (1 << r - 1); j++) {
                    w[r][2 * j] = w[r - 1][j];
                    w[r][2 * j + 1] = bt.shrink(bt.mod_22(u64(f) * w[r - 1][j]));
                }
            }
        }
    }

    void expand_w_rb(int k) {
        while (w_rb.size() < k) {
            int r = w_rb.size();
            w_rb.emplace_back(1 << r);
            if (r == 0) {
                w_rb.back()[0] = 1;
            } else {
                u32 f = power(pr_root, (mod - 1) >> r + 1);
                for (int j = 0; j < (1 << r - 1); j++) {
                    w_rb[r][j] = w_rb[r - 1][j];
                    w_rb[r][j + (1 << r - 1)] = bt.shrink(bt.mod_22(u64(f) * w_rb[r - 1][j]));
                }
            }
        }
    }

    void expand_rb(int k) {
        while (rev_bits.size() <= k) {
            int r = rev_bits.size();
            rev_bits.emplace_back(1 << r);
            for (int j = 1; j < (1 << r); j++) {
                rev_bits[r][j] = (rev_bits[r][j >> 1] >> 1) | ((j & 1) << r - 1);
            }
        }
    }

    void fft(int lg, u32 *data) {
        expand_w(lg);
        expand_rb(lg);
        int n = 1 << lg;
        for (int i = 0; i < n; i++) {
            if (rev_bits[lg][i] < i) {
                std::swap(data[i], data[rev_bits[lg][i]]);
            }
        }
        for (int k = 0; k < lg; k++) {
            for (int i = 0; i < n; i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    u32 a = data[i + j], b = data[i + (1 << k) + j];
                    u32 wj = w[k][j];

                    a = bt.shrink_2(a);
                    u32 c = bt.mod_42(u64(b) * wj);
                    data[i + j] = a + c;
                    data[i + (1 << k) + j] = a + bt.mod2 - c;
                }
            }
        }
    }

    void ifft(int lg, u32 *data) {
        int n = 1 << lg;
        fft(lg, data);
        std::reverse(data + 1, data + n);
        u32 rv = power(n, (mod - 2));

        for (int i = 0; i < n; i++) {
            assert(data[i] < 4 * mod);
            data[i] = bt.shrink(bt.mod_42(u64(rv) * data[i]));
        }
    }

    void fft2(int lg, u32 *data) {
        expand_w_rb(lg);
        int n = 1 << lg;

        for (int k = lg - 1; k >= 0; k--) {
            for (int i = 0; i < n; i += (1 << k + 1)) {
                u32 wj = w_rb[lg - k - 1][i >> k + 1];
                for (int j = 0; j < (1 << k); j++) {
                    u32 a = data[i + j], b = data[i + (1 << k) + j];

                    a = bt.shrink_2(a);
                    u32 c = bt.mod_42(u64(b) * wj);
                    data[i + j] = a + c;
                    data[i + (1 << k) + j] = a + bt.mod2 - c;
                }
            }
        }
    }

    void ifft2(int lg, u32 *data) {
        expand_w(lg);
        int n = 1 << lg;

        for (int k = 0; k < lg; k++) {
            for (int i = 0; i < n; i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    u32 a = data[i + j], b = data[i + (1 << k) + j];
                    u32 wj = w[k][j];

                    a = bt.shrink_2(a);
                    u32 c = bt.mod_42(u64(b) * wj);
                    data[i + j] = a + c;
                    data[i + (1 << k) + j] = a + bt.mod2 - c;
                }
            }
        }

        std::reverse(data + 1, data + n);
        u32 rv = power(n, (mod - 2));

        for (int i = 0; i < n; i++) {
            // assert(data[i] < 4 * mod);
            data[i] = bt.shrink(bt.mod_42(u64(rv) * data[i]));
        }
    }

    std::vector<u32> convolve_slow(std::vector<u32> a, std::vector<u32> b) {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        std::vector<u32> c(sz);
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                // c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % mod;
                c[i + j] = bt.shrink(c[i + j] + bt.mod_21(u64(a[i]) * b[j]));
            }
        }

        return c;
    }

    std::vector<u32> convolve(std::vector<u32> a, std::vector<u32> b) {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        assert(sz <= (1 << lg));
        a.resize(1 << lg, 0u);
        b.resize(1 << lg, 0u);

        fft2(lg, a.data());
        fft2(lg, b.data());
        for (int i = 0; i < (1 << lg); i++) {
            a[i] = bt.mod_42(u64(bt.shrink_2(a[i])) * bt.shrink_2(b[i]));
        }
        ifft2(lg, a.data());
        // for (u32 val : a) {
        //     assert(val < mod);
        // }
        a.resize(sz);
        return std::move(a);
    }

};  // namespace FFT

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    qin >> n >> m;
    std::vector<u32> a(n), b(m);
    for (u32 &i : a) {
        qin >> i;
    }
    for (u32 &i : b) {
        qin >> i;
    }
    auto c = FFT::convolve(std::move(a), std::move(b));
    for (int i = 0; i < c.size(); i++) {
        qout << c[i] << " \n"[i + 1 == c.size()];
    }

    return 0;
}
