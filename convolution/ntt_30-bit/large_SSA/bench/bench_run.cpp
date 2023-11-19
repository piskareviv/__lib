#pragma GCC optimize("O3")
#pragma GCC target("avx2")
#include <cassert>
#include <iostream>

#include "IO.hpp"
#include "ssa.hpp"

int32_t main(int argc, char **argv) {
    int lg = 20;
    int cnt = 1000;
    if (argc > 1) {
        lg = std::atoi(argv[1]);
        cnt = std::atoi(argv[2]);
    }

    int n, m;
    n = m = 1 << lg;
    int sz = 1 << lg;

    // n = m = 1 << 24;
    // qin >> n >> m;
    // int sz = std::max(0, n + m - 1);
    // int lg = std::__lg(std::max(1, sz - 1)) + 1;

    if (lg <= 6) {
        NTT ntt;
        u32 *a = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
        u32 *b = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
        for (int i = 0; i < n; i++) {
            // qin >> a[i];
        }
        memset(a + n, 0, (4 << lg) - 4 * n);
        for (int i = 0; i < m; i++) {
            // qin >> b[i];
        }
        memset(b + m, 0, (4 << lg) - 4 * m);
        ntt.convolve(lg, a, b);
        for (int i = 0; i < (n + m - 1); i++) {
            // qout << a[i] << ' ';  // " \n"[i + 1 == (n + m - 1)];
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
                    // qin >> a[i].ptr[j];
                }
                memset(a[i].ptr + d, 0, 4 * ((1 << L + 1) - d));
            }
        };
        read(a, n);
        read(b, m);

        // std::cerr << "input " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
        beg = clock();

        for (int i = 0; i < cnt; i++)
            ssa.convolve(lg, a, b);

        // std::cerr << "convolution " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
        beg = clock();

        for (int i = 0; i < (1 << B); i++) {
            int ind = i * (1 << L);
            int d = std::max(0, std::min(1 << L, sz - ind));
            if (d == 0) {
                break;
            }
            for (int j = 0; j < d; j++) {
                // qout << a[i].ptr[j] << ' ';
            }
        }
        // std::cerr << "output " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
        beg = clock();
    }
    return 0;
}
