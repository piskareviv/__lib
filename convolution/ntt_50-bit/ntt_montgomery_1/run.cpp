#include <cassert>
#include <iostream>

#include "IO.hpp"
#include "ntt.hpp"

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
