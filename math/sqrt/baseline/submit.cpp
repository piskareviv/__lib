#include <assert.h>
#include <stdint.h>

#include <iostream>
#include <random>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

// Mongtomery32
struct Montgomery {
    u32 mod;
    u32 mod2;
    u32 n_inv;
    u32 r;
    u32 r2;

    Montgomery(u32 mod) : mod(mod) {
        assert(mod < (1 << 30));
        assert(mod % 2);
        mod2 = 2 * mod;
        n_inv = 1;
        for (int i = 0; i < 5; i++)
            n_inv *= 2 + n_inv * mod;
        assert(n_inv * mod == u32(-1));
        r = (1ULL << 32) % mod;
        r2 = r * 1ULL * r % mod;
    }

    u32 shrink(u32 val) const {
        return std::min(val, val - mod);
    }
    u32 shrink_n(u32 val) const {
        return std::min(val, val + mod);
    }
    u32 shrink2(u32 val) const {
        return std::min(val, val - mod2);
    }
    u32 shrink2_n(u32 val) const {
        return std::min(val, val + mod2);
    }

    template <bool strict = false>
    u32 reduce(u64 val) {
        u32 ans = val + u32(val) * n_inv * u64(mod) >> 32;
        if constexpr (strict)
            ans = shrink(ans);
        return ans;
    }

    template <bool strict = false>
    u32 mul(u32 a, u32 b) {
        u64 res = u64(a) * b;
        return reduce<strict>(res);
    }

    [[gnu::noinline]] u32 power(u32 b, u32 e) {
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

std::mt19937 rnd(2086);

int sqrt(u32 val, u32 mod) {
    if (val <= 1) {
        return val;
    }
    if (mod == 2) {
        return 1;
    }
    assert(mod % 2);
    Montgomery mt(mod);

    if (mt.power(val, (mod - 1) / 2) == mod - 1) {
        return -1;
    }

    const u32 val_mt = mt.mul<true>(val, mt.r2);
    auto mul_cum = [&](std::pair<u32, u32> a, std::pair<u32, u32> b) {
        return std::pair<u32, u32>{mt.reduce<true>(u64(a.first) * b.first + u64(val_mt) * mt.mul(a.second, b.second)), mt.reduce<true>(u64(a.first) * b.second + u64(a.second) * b.first)};
    };
    auto power_cum = [&](std::pair<u32, u32> b, u32 e) {
        std::pair<u32, u32> r(1, 0);
        b.first = mt.mul<true>(b.first, mt.r2);
        b.second = mt.mul<true>(b.second, mt.r2);
        for (; e > 0; e >>= 1) {
            if (e & 1) {
                r = mul_cum(r, b);
            }
            b = mul_cum(b, b);
        }
        r.first = mt.shrink(r.first);
        r.second = mt.shrink(r.second);
        return r;
    };

    while (true) {
        u32 d = rnd() % mod;
        // if (mt.power(mt.shrink((d * 1ULL * d) % mod + mod - val), (mod - 1) / 2) == 1) {
        //     continue;
        // }

        auto [x0, x1] = power_cum({d, 1}, (mod - 1) / 2);
        if (x1 != 0) {
            u32 res = mt.mul<true>(mt.r2, mt.mul(x0 + mod - 1, mt.power(x1, mod - 2)));
            if (mt.mul<true>(mt.r2, mt.mul(res, res)) != val) {
                continue;
            }
            return res;
        }
    }
}

std::vector<int> sqrt(const std::vector<int>& val, const std::vector<int>& mod) {
    int n = val.size();
    std::vector<int> res(n);
    for (int i = 0; i < n; i++) {
        res[i] = sqrt(val[i], mod[i]);
        if (res[i] != -1) {
            assert(res[i] * 1ULL * res[i] % mod[i] == val[i]);
        }
    }
    return res;
}
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

#include <iostream>

// int total = 0;
// 
int32_t main() {
    // std::ios_base::sync_with_stdio(false);
    // std::cin.tie(nullptr);

    int n;
    qin >> n;
    std::vector<int> val(n), mod(n);
    for (int i = 0; i < n; i++) {
        qin >> val[i] >> mod[i];
    }
    auto res = sqrt(val, mod);
    for (int i = 0; i < n; i++) {
        qout << res[i] << " \n"[i == n - 1];
    }

    // std::cerr << total / double(n) << "\n";

    return 0;
}
