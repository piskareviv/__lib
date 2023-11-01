#pragma GCC target("avx2")
#include <iostream>

#include "IO.hpp"
#include "sqrt.hpp"

int32_t main() {
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

    return 0;
}
