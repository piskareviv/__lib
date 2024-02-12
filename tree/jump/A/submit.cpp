#pragma GCC target("avx2,bmi,bmi2")
#include <immintrin.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <numeric>

using i256 = __m256i;

int lower_bound_epi32(i256 vec, i256 val) {
    i256 cmp = _mm256_cmpgt_epi32(val, vec);
    uint32_t mask = _mm256_movemask_ps((__m256)cmp);
    return __builtin_ctz(~mask);
}

int lower_bound_epi32(i256 vec0, i256 vec1, i256 val) {
    i256 cmp0 = _mm256_cmpgt_epi32(val, vec0), cmp1 = _mm256_cmpgt_epi32(val, vec1);
    uint32_t mask = _mm256_movemask_ps((__m256)cmp0) | _mm256_movemask_ps((__m256)cmp1) << 8;
    return __builtin_ctz(~mask);
}

int div_up(int a, int b) {
    return (a - 1) / b + 1;
}

struct Cum {
    static constexpr int K = 16;

    int n;
    int lg;
    const int* data;
    const int** cum;

    Cum() = default;

    // data should be padded with at least 64 bytes
    [[gnu::noinline]] Cum(int n, const int* data) : n(n), data(data) {
        if (n <= K) {
            lg = 0;
            return;
        }
        int m = div_up(n + 1, K + 1);
        for (lg = 0; m > 1; m = div_up(m, K + 1), lg++) {
            ;
        }
        cum = new const int*[lg];
        m = div_up(n + 1, K + 1);
        for (int64_t i = lg - 1, f = 1; i >= 0; i--) {
            int b_cnt = div_up(m, K + 1);
            f *= (K + 1);
            int* ptr = (int*)_mm_malloc(4 * K * b_cnt, 64);

            for (int j = 0; j < b_cnt; j++) {
                for (int t = 0; t < K; t++) {
                    int64_t ind = j * f * (K + 1) + f * t + f - 1;
                    if (ind >= n) {
                        ptr[j * K + t] = std::numeric_limits<int>::max();
                    } else {
                        ptr[j * K + t] = data[ind];
                    }
                }
            }
            cum[i] = ptr;
            m = b_cnt;
        }
        assert(m == 1);
    }

    [[gnu::noinline]] int lower_bound(int val) {
        int ind = 0;
        i256 vec_val = _mm256_set1_epi32(val);
        for (int i = 0; i < lg; i++) {
            i256 vec0 = _mm256_load_si256((__m256i*)(cum[i] + ind * K));
            i256 vec1 = _mm256_load_si256((__m256i*)(cum[i] + ind * K + 8));
            int dt = lower_bound_epi32(vec0, vec1, vec_val);
            ind = ind * (K + 1) + dt;
        }
        {
            // assert(0 <= ind && ind * (K + 1) <= n);
            i256 vec0 = _mm256_loadu_si256((__m256i*)(data + ind * (K + 1)));
            i256 vec1 = _mm256_loadu_si256((__m256i*)(data + ind * (K + 1) + 8));
            int dt = lower_bound_epi32(vec0, vec1, vec_val);
            ind = ind * (K + 1) + dt;

            ind = std::min(ind, n);
        }
        return ind;
    }
};
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>


namespace WTF {

    class ST {
       private:
        int size, layer_cnt, **data;

       public:
        ST(int n = 0, int *base = nullptr) : size(n), layer_cnt(32 - __builtin_clz(n)), data(new int *[32 - __builtin_clz(n)]) {
            for (int i = 0; i < layer_cnt; i++) {
                data[i] = new int[size];
            }
            if (base != nullptr) {
                config(n, base);
            }
        }

        void config(int n, int *base) {
            if (n > size) {
                for (int i = 0; i < layer_cnt; i++) {
                    // delete[] data[i];
                }
                // delete[] data;

                layer_cnt = 32 - __builtin_clz(n);
                data = new int *[layer_cnt];
                for (int i = 0; i < layer_cnt; i++) {
                    data[i] = new int[n];
                }
            }
            std::copy(base, base + n, data[0]);
            for (int k = 1; k < layer_cnt; k++) {
                for (int i = 0; i < (n - (1 << (k - 1))); i++) {
                    data[k][i] = std::min(data[k - 1][i], data[k - 1][i + (1 << (k - 1))]);
                }
            }
        }

        //[l, r)
        int get(int l, int r) {
            if (l == r) {
                return std::numeric_limits<int>::max();
            }
            int lg = 31 - __builtin_clz(r - l);
            int res = std::min(data[lg][l], data[lg][r - (1 << lg)]);
            return res;
        }
        ~ST() {
            for (int i = 0; i < layer_cnt; i++) {
                // delete[] data[i];
            }
            // delete[] data;
        }
    };

    class DS {
       private:
        static constexpr int b = 32;
        ST st;
        int size;
        const int *input;
        uint32_t *data;

        //[l, r)
        int get2(int l, int r) {
            if (l == r) {
                return std::numeric_limits<int>::max();
            }
            if ((r - l) > b) {
                exit(1);
            }
            uint32_t val = data[l];
            if ((r - l) != 32) {
                val &= (1ull << uint32_t(r * 1ull - l * 1ull)) - 1ull;
            }
            int x = 31 - __builtin_clz(val);
            if (x == -1) {
                exit(2);
            }
            return input[l + x];
        }

       public:
        DS() = default;
        DS(int n, const int *base) : size(n) {
            input = base;

            int *help = new int[n];
            {
                std::vector<int> stack(2 * b + 10);
                stack.clear();
                for (int i = 0; i < n; i++) {
                    if (stack.size() > (2 * b + 5)) {
                        for (int j = 0; j < b; j++) {
                            help[stack[j]] = n;
                        }
                        stack.erase(stack.begin(), stack.begin() + b);
                    }
                    while ((!stack.empty()) && (base[stack.back()] > base[i])) {
                        help[stack.back()] = i;
                        stack.pop_back();
                    }
                    stack.push_back(i);
                }
                for (int i : stack) {
                    help[i] = n;
                }
            }

            data = new uint32_t[n];
            for (int i = n - 1; i >= 0; i--) {
                data[i] = 1ull;
                if ((help[i] < n) && ((help[i] - i) < b)) {
                    data[i] |= data[help[i]] << ((uint32_t)(help[i] - i));
                }
            }
            // delete[] help;

            help = nullptr;
            help = new int[(n - 1ll) / b + 1];
            std::fill(help, help + (n - 1ll) / b + 1, std::numeric_limits<int>::max());
            for (int i = 0; i < n; i++) {
                help[i / b] = std::min(help[i / b], base[i]);
            }
            st.config((n - 1ll) / b + 1, help);
            // delete[] help;
        }

        //[l, r)
        int get(int l, int r) {
            if ((r - l) <= b) {
                return get2(l, r);
            }
            int ll = ((l != 0) ? ((l - 1) / b + 1) : 0), rr = r / b;
            return std::min({st.get(ll, rr), get2(l, ll * b), get2(rr * b, r)});
        }

        ~DS() {
            // delete[] data;
        }
    };

    class DS2 {
       private:
        static constexpr int b = 14;
        ST st;
        int size;
        std::pair<int, uint16_t> *data;

        //[l, r)
        int get2(int l, int r) {
            if (l == r) {
                return std::numeric_limits<int>::max();
            }
            if ((r - l) > b) {
                exit(1);
            }
            uint32_t val = data[l].second;
            if ((r - l) != 32) {
                val &= (1ull << uint32_t(r * 1ull - l * 1ull)) - 1ull;
            }
            int x = 31 - __builtin_clz(val);
            if (x == -1) {
                exit(2);
            }
            return data[l + x].first;
        }

       public:
        DS2() = default;
        DS2(int n, const int *base) : size(n) {
            data = new std::pair<int, uint16_t>[n];
            for (int i = 0; i < n; i++) {
                data[i].first = base[i];
            }
            // // delete[] base;

            int *help = new int[n];
            {
                std::vector<int> stack(2 * b + 10);
                stack.clear();
                for (int i = 0; i < n; i++) {
                    if (stack.size() > (2 * b + 5)) {
                        for (int j = 0; j < b; j++) {
                            help[stack[j]] = n;
                        }
                        stack.erase(stack.begin(), stack.begin() + b);
                    }
                    while ((!stack.empty()) && (data[stack.back()].first > data[i].first)) {
                        help[stack.back()] = i;
                        stack.pop_back();
                    }
                    stack.push_back(i);
                }
                for (int i : stack) {
                    help[i] = n;
                }
            }

            for (int i = n - 1; i >= 0; i--) {
                data[i].second = 1ull;
                if ((help[i] < n) && ((help[i] - i) < b)) {
                    data[i].second |= data[help[i]].second << ((uint32_t)(help[i] - i));
                }
            }
            // delete[] help;

            help = new int[(n - 1ll) / b + 1];
            std::fill(help, help + (n - 1ll) / b + 1, std::numeric_limits<int>::max());
            for (int i = 0; i < n; i++) {
                help[i / b] = std::min(help[i / b], data[i].first);
            }
            st.config((n - 1ll) / b + 1, help);
            // delete[] help;
        }

        //[l, r)
        int get(int l, int r) {
            if ((r - l) <= b) {
                return get2(l, r);
            }
            int ll = ((l != 0) ? ((l - 1) / b + 1) : 0), rr = r / b;
            return std::min({st.get(ll, rr), get2(l, ll * b), get2(rr * b, r)});
        }

        ~DS2() {
            // delete[] data;
        }
    };

};  // namespace WTF

struct ST {
    std::vector<std::vector<int>> data;
    int lg;

    ST() = default;

    [[gnu::noinline]] __attribute__((optimize("O3"))) ST(const std::vector<int> &vec) {
        lg = std::__lg(vec.size()) + 1;
        data.resize(lg);
        data[0] = vec;
        for (int k = 1; k < lg; k++) {
            data[k].resize(vec.size() - (1 << k) + 1);
#pragma GCC ivdep
            for (int i = 0; i < vec.size() - (1 << k) + 1; i++) {
                data[k][i] = std::min(data[k - 1][i], data[k - 1][i + (1 << k - 1)]);
            }
        }
    }

    int get_min(int l, int r) {
        int k = std::__lg(r - l);
        return std::min(data[k][l], data[k][r - (1 << k)]);
    }
};

struct Jump {
    ST st;
    // WTF::DS2 ds;

    std::vector<int> depth, tin, depth_eul;
    std::vector<int> eul_ind, depth_ind, eul;
    std::vector<Cum> cum;

    Jump() = default;

    [[gnu::noinline]] Jump(const std::vector<std::pair<int, int>> &edg) {
        int n = edg.size() + 1;
        std::vector<int> ind(n + 1);
        std::vector<int> nxt(2 * n);  // 2 * n instead of 2 * n - 2 for later reuse

        for (int i = 0; i < n - 1; i++) {
            ind[edg[i].first]++;
            ind[edg[i].second]++;
        }
        for (int i = 0, c = 0; i <= n; i++) {
            ind[i] = c += ind[i];
        }
        for (int i = 0; i < n - 1; i++) {
            nxt[--ind[edg[i].first]] = edg[i].second;
            nxt[--ind[edg[i].second]] = edg[i].first;
        }

        int e = 0;
        depth.assign(n, 0);
        tin.assign(n, 0);
        depth_eul.assign(2 * n - 1, 0);
        eul.assign(2 * n, 0);
        auto dfs = [&](auto dfs, int v, int f, int d) -> void {
            const int l0 = ind[v], r0 = ind[v + 1];
            tin[v] = e;
            depth_eul[e] = d;
            depth[v] = d;
            eul[e] = v;
            e++;

            for (int t_id = l0; t_id < r0; t_id++) {
                int t = nxt[t_id];
                if (t != f) {
                    dfs(dfs, t, v, d + 1);
                    eul[e] = v;
                    depth_eul[e] = d;
                    e++;
                }
            }
        };
        dfs(dfs, 0, -1, 0);
        assert(e == 2 * n - 1);

        depth_ind = std::move(ind);
        depth_ind.assign(n + 1, 0);
        for (int i = 0; i < e; i++) {
            depth_ind[depth_eul[i]]++;
        }
        for (int i = 0, c = 0; i <= n; i++) {
            depth_ind[i] = c += depth_ind[i];
        }
        eul_ind.resize(e + 32);
        for (int i = 0; i < e; i++) {
            eul_ind[--depth_ind[depth_eul[i]]] = i;
        }

        st = ST(depth_eul);
        // ds = WTF::DS2(depth_eul.size(), depth_eul.data());
        cum.resize(n);
        for (int i = 0; i < n; i++) {
            std::reverse(eul_ind.begin() + depth_ind[i], eul_ind.begin() + depth_ind[i + 1]);
            cum[i] = Cum(depth_ind[i + 1] - depth_ind[i], eul_ind.data() + depth_ind[i]);
        }
    }

    [[gnu::noinline]] int get_lca_depth(int u, int v) {
        int a = tin[u];
        int b = tin[v];
        int res = st.get_min(std::min(a, b), std::max(a, b) + 1);
        // int res = ds.get(std::min(a, b), std::max(a, b) + 1);
        return res;
    }

    // from vertex v to depth d
    [[gnu::noinline]] int jump(int v, int d) {
        int it = cum[d].lower_bound(tin[v]);
        int res = eul[cum[d].data[it]];
        return res;
    }

    [[gnu::noinline]] int jump_path(int s, int t, int i) {
        int d = get_lca_depth(s, t);
        int len = depth[s] + depth[t] - 2 * d;
        int ans = -1;
        if (i <= len) {
            if (i <= depth[s] - d) {
                ans = jump(s, depth[s] - i);
            } else {
                ans = jump(t, depth[t] - (len - i));
            }
        }
        return ans;
    }
};
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstring>
#include <iostream>

using u32 = uint32_t;
using u64 = uint64_t;

// io from https://judge.yosupo.jp/submission/142782

namespace __io {
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
#pragma GCC target("avx2")

// #define qin std::cin
// #define qout std::cout
//

#include <array>
#include <iostream>

int32_t main() {
    int n, q;
    qin >> n >> q;
    std::vector<std::pair<int, int>> edg(n - 1);
    for (auto& [u, v] : edg) {
        qin >> u >> v;
    }
    std::vector<std::array<int, 3>> quer(q);
    for (auto& [s, t, i] : quer) {
        qin >> s >> t >> i;
    }

    Jump jmp(edg);

    for (auto& [s, t, i] : quer) {
        // int d = jmp.get_lca_depth(s, t);
        // int len = jmp.depth[s] + jmp.depth[t] - 2 * d;
        // int ans = -1;
        // if (i <= len) {
        //     if (i <= jmp.depth[s] - d) {
        //         ans = jmp.jump(s, jmp.depth[s] - i);
        //     } else {
        //         ans = jmp.jump(t, jmp.depth[t] - (len - i));
        //     }
        // }
        int ans = jmp.jump_path(s, t, i);
        qout << ans << '\n';
    }

    return 0;
}
