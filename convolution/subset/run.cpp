#include <immintrin.h>

#include <algorithm>
#include <iostream>

#include "subset.hpp"

int32_t main(int argc, char** argv) {
    int mod = 998'244'353;
    WTF wtf(mod);

    assert(argc >= 3);
    int lg = std::atoi(argv[1]);
    int cnt = std::atoi(argv[2]);

    u32* a = (u32*)_mm_malloc(4 << lg, 64);
    u32* b = (u32*)_mm_malloc(4 << lg, 64);

    std::fill(a, a + (1 << lg), 2);
    std::fill(b, b + (1 << lg), 3);

    wtf.convolve_subset(lg, a, b, a);
    // wtf.SOS(lg, a), wtf.SOS(lg, b);

    clock_t beg = clock();
    for (int i = 0; i < cnt; i++) {
        wtf.convolve_subset(lg, a, b, a);
        // wtf.SOS(lg, a), wtf.SOS(lg, b), wtf.SOS(lg, a);
    }
    std::cout << std::fixed;
    std::cout.precision(5);
    std::cout << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "" << std::endl;
    // std::cout << (clock() - beg) * 1.0 / CLOCKS_PER_SEC / cnt * 1000 << "ms" << std::endl;

    _mm_free(a);
    _mm_free(b);
}
