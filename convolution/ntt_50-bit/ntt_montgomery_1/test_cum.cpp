
#ifdef LOCAL
// #define optimize cum
#endif

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

// #include "ntt.hpp"
#include "aux.hpp"

int32_t main() {
    std::mt19937_64 rnd;

    // const u64 mod = 998'244'353;
    const u64 mod = 1'125'211'998'244'353;
    Montgomery mt(mod);
    Montgomery_simd mtg(mod);

    for (int i = 0; i < 1e5; i++) {
        u64 a = rnd() % (4 * mod);
        u64 b = rnd() % mod;
        // a = b = 1;
        u64 c = mt.mul<true>(a, b);
        u64 c2 = mtg.mul<true>(set1_u64x8(a), set1_u64x8(b))[0];
        assert(c == c2);
    }
}
