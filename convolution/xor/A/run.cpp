#pragma GCC optimize("O3")
#include <iostream>
#include <vector>

#include "IO.hpp"
#include "wht.hpp"

int32_t main(int argc, char** argv) {
    int lg = 20;
    int cnt = 1000;
    if (argc > 1) {
        lg = std::atoi(argv[1]);
        cnt = std::atoi(argv[2]);
    }
    // cnt = 1;
    // lg = 29;

    // qin >> lg;
    int64_t n = 1LL << lg;
    u32* a = (u32*)_mm_malloc(n * 4, std::min<int64_t>(n * 4, 64));
    u32* b = (u32*)_mm_malloc(n * 4, std::min<int64_t>(n * 4, 64));
    for (int64_t i = 0; i < n; i++) {
        a[i] = b[i] = i;
    }
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
