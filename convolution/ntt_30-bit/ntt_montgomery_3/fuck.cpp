#include <bits/stdc++.h>

#include "ntt.hpp"

int32_t main(int argc, char** argv) {
    // int cnt = std::atoi(argv[1]);
    const int cnt = 1e7;
    constexpr int C = 4;
    constexpr int LG = 4;

    alignas(64) u32 a[C << LG], b[C << LG];
    memset(a, 3, sizeof(a));
    memset(b, 3, sizeof(b));
    NTT ntt;
    auto mts = ntt.mts;

    for (int i = 0; i < cnt / C; i++) {
        // for (int j = 0; j < C; j++) {
        //     ntt.mul_mod<0>(LG, a + j * (1 << LG), b + j * (1 << LG),
        //                            992'312'321, 111232, mts);
        // }
        // for (int j = 0; j < C; j++) {
        //     ntt.aux_mod_mul_8(a + j * (1 << LG), b + j * (1 << LG),
        //                       992'312'321, mts);
        // }
        ntt.aux_mod_mul_x<4, LG, false>(a, b,
                                        {992'312'321, 992'312'321, 992'312'321, 992'312'321},
                                        111232, mts);
    }
    // asm volatile("# LLVM-MCA-BEGIN");
    // asm volatile("# LLVM-MCA-END");

    double tm = clock() * 1.0 / CLOCKS_PER_SEC;
    std::cerr << tm << " " << tm / cnt * 1e9 << "ns"
              << "\n";
    std::cerr << a[(1 << LG) * 3] << "\n";

    return 0;
}