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
        constexpr u64 E16 = 1e16, E12 = 1e12;
        constexpr u32 E8 = 1e8, E4 = 1e4;
        struct ict {
            int num[10000];
            constexpr ict() {
                int j = 0;
                for (int e0 = (48 << 0); e0 < (58 << 0); e0 += (1 << 0)) {
                    for (int e1 = (48 << 8); e1 < (58 << 8); e1 += (1 << 8)) {
                        for (int e2 = (48 << 16); e2 < (58 << 16); e2 += (1 << 16)) {
                            for (int e3 = (48 << 24); e3 < (58 << 24); e3 += (1 << 24)) {
                                num[j] = e0 ^ e1 ^ e2 ^ e3, ++j;
                            }
                        }
                    }
                }
            }
        } constexpr ot;
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
            }
            ~Qinf() {
                munmap(bg, Fl.st_size + 1);
            }
            void skip_space() {
                while (*p <= ' ') {
                    ++p;
                }
            }
            char get() {
                return *p++;
            }
            char seek() const {
                return *p;
            }
            Qinf &read(char *s, size_t count) {
                return memcpy(s, p, count), p += count, *this;
            }
            Qinf &operator>>(u32 &x) {
                skip_space(), x = 0;
                for (; *p > ' '; ++p) {
                    x = x * 10 + (*p & 0xf);
                }
                return *this;
            }
            Qinf &operator>>(u64 &x) {
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
            Qoutf(FILE *fo, size_t sz = O_buffer_default_size) : f(fo),
                                                                 bg(new char[sz]),
                                                                 ed(bg + sz),
                                                                 p(bg),
                                                                 ed_thre(ed - O_buffer_default_flush_threshold),
                                                                 fp(6),
                                                                 _fpi(1000000ull) {
            }
            void flush() {
                fwrite_unlocked(bg, 1, p - bg, f), p = bg;
            }
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
                auto C = (const char *)(ot.num + x);
                if (x > 99u) {
                    if (x > 999u) {
                        memcpy(p, C, 4), p += 4;
                    } else {
                        memcpy(p, C + 1, 3), p += 3;
                    }
                } else {
                    if (x > 9u) {
                        memcpy(p, C + 2, 2), p += 2;
                    } else {
                        *p++ = x ^ 48;
                    }
                }
            }
            void put2(u32 x) {
                if (x > 9u) {
                    memcpy(p, (const char *)(ot.num + x) + 2, 2), p += 2;
                } else {
                    *p++ = x ^ 48;
                }
            }
            Qoutf &write(const char *s, size_t count) {
                if (count > 1024 || p + count > ed_thre)
                    flush(), fwrite_unlocked(s, 1, count, f);
                else
                    memcpy(p, s, count), p += count, chk();

                return *this;
            }
            Qoutf &operator<<(char ch) {
                return *p++ = ch, *this;
            }
            Qoutf &operator<<(u32 x) {
                if (x >= E8) {
                    put2(x / E8), x %= E8;
                    memcpy(p, ot.num + x / E4, 4), p += 4;
                    memcpy(p, ot.num + x % E4, 4), p += 4;
                } else if (x >= E4) {
                    put4(x / E4);
                    memcpy(p, ot.num + x % E4, 4), p += 4;
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
            Qoutf &operator<<(u64 x) {
                if (x >= E8) {
                    u64 q0 = x / E8, r0 = x % E8;
                    if (x >= E16) {
                        u64 q1 = q0 / E8, r1 = q0 % E8;
                        put4(q1);
                        memcpy(p, ot.num + r1 / E4, 4), p += 4;
                        memcpy(p, ot.num + r1 % E4, 4), p += 4;
                    } else if (x >= E12) {
                        put4(q0 / E4);
                        memcpy(p, ot.num + q0 % E4, 4), p += 4;
                    } else {
                        put4(q0);
                    }
                    memcpy(p, ot.num + r0 / E4, 4), p += 4;
                    memcpy(p, ot.num + r0 % E4, 4), p += 4;
                } else {
                    if (x >= E4) {
                        put4(x / E4);
                        memcpy(p, ot.num + x % E4, 4), p += 4;
                    } else {
                        put4(x);
                    }
                }
                return chk(), *this;
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
