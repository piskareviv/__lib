#pragma GCC optimize("O3")
#pragma GCC target("avx2")
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "fft.hpp"

int32_t main() {
    std::cout.precision(4);
    std::cout << std::fixed;

    int n = 1 << 19;
    int m = 1 << 19;

    std::mt19937 rnd;
    float min = 1e9;
    constexpr int ITER = 100;
    clock_t beg = clock();

    u32 *a = (u32 *)aligned_alloc(64, 4 << 20);
    u32 *b = (u32 *)aligned_alloc(64, 4 << 20);

    std::cout << "cum  \n";
    for (int i = 0; i < ITER; i++) {
        clock_t beg = clock();

        for (int i = 0; i < n; i++) {
            a[i] = rnd() >> 3;
        }
        for (int i = 0; i < m; i++) {
            b[i] = rnd() >> 3;
        }

        FFT::convolve(n + m - 1, a, b);

        FFT::free_mem();
        // free(a);
        // free(b);

        float tm = (clock() - beg) * 1.0f / CLOCKS_PER_SEC;
        min = std::min(min, tm);
        std::cout << tm << " \n"[i + 1 == ITER];
        std::cout.flush();
    }

    std::cout << "min time: " << min << "\n";
    std::cout << "avg time: " << (clock() - beg) * 1.0f / CLOCKS_PER_SEC / ITER << "\n";
}