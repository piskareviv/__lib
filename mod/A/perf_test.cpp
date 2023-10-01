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

    int n = 5e5;
    int m = 5e5;

    std::mt19937 rnd;
    float min = 1e9;
    constexpr int ITER = 20;
    clock_t beg = clock();
    for (int i = 0; i < ITER; i++) {
        clock_t beg = clock();

        std::vector<u32> a(n), b(m);
        for (u32& i : a) {
            i = rnd() >> 3;
        }
        for (u32& i : b) {
            i = rnd() >> 3;
        }
        auto c = FFT::convolve(a, b);
        FFT::w.clear();
        FFT::w_rb.clear();

        float tm = (clock() - beg) * 1.0f / CLOCKS_PER_SEC;
        min = std::min(min, tm);
        std::cout << tm << " \n"[i + 1 == ITER];
        std::cout.flush();
    }

    std::cout << "min time: " << min << "\n";
    std::cout << "avg time: " << (clock() - beg) * 1.0f / CLOCKS_PER_SEC / ITER << "\n";
}