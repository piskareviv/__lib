#include <cassert>
#include <iostream>

// #include "IO.hpp"
#include "ntt.hpp"

int32_t main(int argc, char **argv) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int lg = 20;
    int cnt = 1000;
    if (argc > 1) {
        lg = std::atoi(argv[1]);
        cnt = std::atoi(argv[2]);
    }

    NTT fft(998'244'353);

    u32 *a = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
    u32 *b = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);

    for (int i = 0; i < cnt; i++) {
        // fft.convolve(lg, a, b);
        fft.convolve2(lg, a, b);
    }

    return 0;
}
