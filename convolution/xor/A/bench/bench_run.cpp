#pragma GCC optimize("O3")
#include <iostream>
#include <vector>

#include "IO.hpp"
#include "wht.hpp"

int32_t main(int argc, char** argv) {
    int lg;
    int cnt = 1000;
    lg = std::atoi(argv[1]);
    if (argc > 2) {
        cnt = std::atoi(argv[2]);
    }
    // cnt = 1;
    // lg = 29;

    // qin >> lg;
    int64_t n = 1LL << lg;
    u32* a = (u32*)_mm_malloc(n * 4, std::min<int64_t>(n * 4, 64));
    u32* b = (u32*)_mm_malloc(n * 4, std::min<int64_t>(n * 4, 64));
    // for (int i = 0; i < n; i++) {
    //     qin >> a[i];
    // }
    // for (int i = 0; i < n; i++) {
    //     qin >> b[i];
    // }

    Cum cum(998'244'353);

    for (int i = 0; i < cnt; i++)
        cum.convolve_xor(lg, a, b);

    // for (int i = 0; i < n; i++) {
    //     qout << a[i] << " \n"[i == n - 1];
    // }

    return 0;
}
