
#include <iostream>

// int total = 0;
#include "IO.hpp"
// #include "sqrt.hpp"
#include "sqrt_mtg.hpp"

int32_t main() {
    // std::ios_base::sync_with_stdio(false);
    // std::cin.tie(nullptr);

    int n;
    qin >> n;
    std::vector<int> val(n), mod(n);
    for (int i = 0; i < n; i++) {
        qin >> val[i] >> mod[i];
    }
    auto res = sqrt(val, mod);
    for (int i = 0; i < n; i++) {
        qout << res[i] << " \n"[i == n - 1];
    }

    // std::cerr << total / double(n) << "\n";

    return 0;
}
