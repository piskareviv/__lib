#include <cassert>
#include <iostream>

#include "IO.hpp"
// #define qin std::cin
// #define qout std::cout

#include "ssa.hpp"

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    n = m = 1 << 19;
    qin >> n >> m;

    int sz = std::max(0, n + m - 1);

    SSA_20 ssa(1e9 + 7);
    // auto [L, B] = ssa.get_LB(lg);
    const auto L = 9;
    const auto B = 11;

    clock_t beg = clock();

    std::vector<SSA_20::Cum_10> a(1 << B), b(1 << B);
    auto read = [L, B](auto &a, int n) {
        u32 *ptr = (u32 *)_mm_malloc(4 << L + B + 1, 64);

        for (int i = 0; i < (1 << B); i++) {
            a[i] = SSA_20::Cum_10{ptr + i * (1 << L + 1), 0};

            int ind = i * (1 << L);
            int d = std::max(0, std::min(1 << L, n - ind));
            for (int j = 0; j < d; j++) {
                qin >> a[i].ptr[j];
            }
            memset(a[i].ptr + d, 0, 4 * ((1 << L + 1) - d));
        }
    };
    read(a, n);
    read(b, m);

    std::cerr << "input " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
    beg = clock();

    ssa.convolve_20(a, b);

    std::cerr << "convolution " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
    beg = clock();

    for (int i = 0; i < (1 << B); i++) {
        int ind = i * (1 << L);
        int d = std::max(0, std::min(1 << L, sz - ind));
        if (d == 0) {
            break;
        }
        for (int j = 0; j < d; j++) {
            assert(a[i].ptr[j] < 1e9 + 7);
            qout << a[i].ptr[j] << ' ';
        }
    }
    qout << '\n';

    std::cerr << "output " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
    beg = clock();

    return 0;
}
