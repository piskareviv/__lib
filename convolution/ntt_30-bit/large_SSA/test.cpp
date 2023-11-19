// #pragma GCC optimize("O3")
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "ssa.hpp"

int32_t main() {
    std::mt19937 rnd;
    std::mt19937_64 rnd_64;

    SSA ssa;  //((1 << 16) + 1);

    auto test = [&](std::vector<u32> a, std::vector<u32> b) {
        auto c = ssa.convolve(a, b);
        auto c2 = ssa.ntt.convolve(a, b);
        if (c != c2) {
            for (auto& vec : std::array{a, c, c2}) {
                for (int i = 0; i < std::min<int>(vec.size(), 200); i++) {
                    std::cout << vec[i] << " ";
                }
                std::cout << "\n";
            }
        }
        assert(c == c2);
    };

    auto rand_vec = [&](int n, u32 mod) {
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

    for (int i = 1; i < 1e3; i++) {
        int n = i * 10;
        int m = i * 10;
        test(rand_vec(n, i + 1), rand_vec(m, i + 1));
    }
    for (int i = 1; i < 1e4; i++) {
        int n = exp2(rnd() % int(1e9) / 1e9 * 10);
        int m = exp2(rnd() % int(1e9) / 1e9 * 10);
        test(rand_vec(n, i + 1), rand_vec(m, i + 1));
    }
    for (int i = 0; i < 1; i++) {
        int n = 1 << 19;
        int m = 1 << 19;
        test(rand_vec(n, 0.9e9), rand_vec(m, 0.9e9));
    }
}
