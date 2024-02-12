
// #ifdef LOCAL
// #define optimize cum
// #endif

// #pragma GCC optimize("O3")
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "ntt.hpp"

int32_t main() {
    std::mt19937_64 rnd;
    // std::mt19937_64 rnd_64;

    for (u64 mod : std::vector<u64>{1108307720798209, 998'244'353, (1 << 16) + 1, (1 << 8) + 1, (1 << 4) + 1, 167772161, 469762049, 754974721}) {
        NTT ntt(mod);  //((1 << 16) + 1);

        auto test = [&](std::vector<u64> a, std::vector<u64> b) {
            int sz = std::max(0, (int)a.size() + (int)b.size() - 1);
            int lg = std::__lg(std::max(1, sz - 1)) + 1;
            if (((mod - 1) % (1 << lg)) != 0) {
                return;
            }

            auto c = ntt.convolve(a, b);
            auto c2 = ntt.convolve_slow(a, b);
            // auto c2 = ntt.convolve_slow(a, b);
            if (c != c2) {
                std::cout << a.size() << " " << b.size() << " " << mod << std::endl;
                for (auto& vec : std::array{a, b, c2, c}) {
                    for (int i = 0; i < std::min<int>(vec.size(), 100); i++) {
                        std::cout << vec[i] << " ";
                    }
                    std::cout << "\n";
                    std::cout << "\n";
                }
            }
            assert(c == c2);
        };

        auto rand_vec = [&](int n, u64 mod) {
            std::vector<u64> a(n);
            for (u64& i : a) {
                i = rnd() % mod;
            }
            return a;
        };

        for (int i = 0; i < 1e3; i++) {
            int n = 3;
            int m = 3;
            test(rand_vec(n, 10), rand_vec(m, 10));
        }

        for (int i = 1; i < 1e2; i++) {
            int n = rnd() % (i + 5);
            int m = rnd() % (i + 5);
            test(rand_vec(n, 5), rand_vec(m, 5));
        }
        // for (int i = 1; i < 1e4; i++) {
        //     int n = rnd() % (i + 5);
        //     int m = rnd() % (i + 5);
        //     test(rand_vec(n, mod), rand_vec(m, mod));
        // }
        for (int i = 0; i <= 10; i++) {
            int n = 1 << i;
            int m = 1 << i;
            test(rand_vec(n, mod), rand_vec(m, mod));
        }
    }
}
