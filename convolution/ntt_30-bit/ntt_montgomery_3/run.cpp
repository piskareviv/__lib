#include <cassert>
#include <iostream>

#include "IO.hpp"
#include "ntt.hpp"

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    auto __total = cum_timer("total");

    NTT fft(998'244'353);

    // int n, m;
    // n =m = 5e5;
    // qin >> n >> m;
    // int lg = std::__lg(std::max(1, n + m - 2)) + 1;

    int lg = 20;
    int n = 1 << lg - 1;
    int m = 1 << lg - 1;

    u32 *a = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
    u32 *b = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);

    {
        auto __ = cum_timer("input");

        for (int i = 0; i < n; i++) {
            // qin >> a[i];
        }
        memset(a + n, 0, (4 << lg) - 4 * n);
        for (int i = 0; i < m; i++) {
            // qin >> b[i];
        }
        memset(b + m, 0, (4 << lg) - 4 * m);
    }
    {
        auto __ = cum_timer("work");

        for (int i = 0; i < 100; i++)
        //
        {
            // fft.convolve(lg, a, b);
            fft.convolve2(lg, a, b);
        }
    }
    {
        auto __ = cum_timer("output");
        for (int i = 0; i < (n + m - 1); i++) {
            // qout << a[i] << " \n"[i + 1 == (n + m - 1)];
        }
    }

    return 0;
}
