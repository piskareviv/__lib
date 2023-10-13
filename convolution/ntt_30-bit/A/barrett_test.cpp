#include "barrett.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

int32_t main() {
    std::mt19937 rnd;
    std::mt19937_64 rnd_64;

    if (1) {
        int max = 0;
        for (int i = 0; i < 10; i++) {
            constexpr uint32_t mod = 998244353;
            // uint32_t mod = rnd() % (1 << 30) + 1;
            // mod = 1 << 30 - i;

            Barrett cum(mod);
            // if (cum.q == 0) {
            //     i--;
            //     continue;
            // }

            auto test = [&](uint64_t val) {
                uint32_t r = cum.mod_42(val);
                uint32_t r0 = val % mod;
                if (r % mod != r0) {
                    std::cerr << val << "\n";
                    cum.mod_21(val);
                    assert(r % mod == r0);
                }
                max = std::max<int>(max, (r - r0) / mod);
            };

            for (uint64_t j = 0; j < 1e7; j++) {
                uint64_t val = rnd_64() % (1 * uint64_t(mod) * uint64_t(mod));
                test(val);
            }
            // for (uint64_t j = 0; j < 1e5; j++) {
            //     uint64_t val = rnd_64() % (2 * uint64_t(mod) * uint64_t(mod));
            //     test(val);
            // }

            // for (uint64_t j = 1; j < (1LL << cum.s + 3); j++) {
            //     test(j * (1ULL << cum.s) - 1);
            // }
            // for (uint64_t j = 1; j < (1LL << cum.s + 3); j++) {
            //     test2(j * (1ULL << cum.s + 1) - 1);
            // }
            std::cerr << i << " " << mod << " " << max << "\n";
            max = 0;
        }

        std::cout << "max: " << max << "\n";
        return 0;
    } else {
        int max = 0;
        for (int i = 0; i < 1; i++) {
            constexpr uint32_t mod = 998244353;
            // uint32_t mod = rnd() % (1 << 30) + 1;

            Barrett_simd cum(mod);
            if (cum.bt.q == 0) {
                i--;
                continue;
            }

            for (int i = 0; i < 1e7; i++) {
                u32x8 a, b;
                for (int j = 0; j < 8; j++) {
                    a[j] = rnd_64() % mod;
                    b[j] = rnd_64() % (4 * mod);
                }
                u32x8 c = cum.mul_mod_42(a, b);
                for (int j = 0; j < 8; j++) {
                    u32 r0 = a[j] * (u64)b[j] % mod;
                    assert(c[j] == r0 || c[j] == r0 + mod);
                }
            }

            std::cerr << i << " " << mod << " " << max << "\n";
        }

        std::cout << "max: " << max << "\n";
        return 0;
    }
}