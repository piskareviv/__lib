#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "fft.hpp"

int32_t main() {
    std::mt19937 rnd;
    std::mt19937_64 rnd_64;

    auto test = [&](std::vector<u32> a, std::vector<u32> b) {
        auto c = FFT::convolve(a, b);
        auto c2 = FFT::convolve_slow(a, b);
        assert(c == c2);
    };

    auto rand_vec = [&](int n, u32 mod = FFT::mod) {
        std::vector<u32> a(n);
        for (u32& i : a) {
            i = rnd() % mod;
        }
        return a;
    };

    for (int i = 0; i < 1e3; i++) {
        int n = 3;
        int m = 3;
        test(rand_vec(n, 10), rand_vec(m, 10));
    }

    for (int i = 0; i < 1e5; i++) {
        int n = exp2(rnd() % int(1e9) / 1e9 * 10);
        int m = exp2(rnd() % int(1e9) / 1e9 * 10);
        test(rand_vec(n, 10), rand_vec(m, 10));
    }
}
