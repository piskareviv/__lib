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
