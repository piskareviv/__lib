#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "bin.hpp"

int32_t main() {
    const int TESTS = 10000;
    std::mt19937 rnd;

    // for (int i = 0; i < TESTS; i++) {
    //     int n = rnd() % (i + 1);
    //     if (i == 0) {
    //         n = 1e6;
    //     }
    //     std::vector<int> vec(n + 16);
    //     for (int i = 0; i < n; i++) {
    //         vec[i] = rnd() % int(1e9);
    //     }

    //     std::sort(vec.begin(), vec.begin() + n);
    //     Cum cum(n, vec.data());

    //     for (int j = 0; j < 1e4; j++) {
    //         int val = rnd() >> 1;
    //         int it = cum.lower_bound(val);
    //         int it0 = std::lower_bound(vec.begin(), vec.begin() + n, val) - vec.begin();
    //         assert(it == it0);
    //     }
    // }

    {
        // int n = 4.9e3;
        int n = 5e5;
        // int n = 280;
        // n = 290 * 15;
        std::vector<int> vec(n + 32);
        for (int i = 0; i < n; i++) {
            vec[i] = rnd();
        }

        std::sort(vec.begin(), vec.begin() + n);
        Cum cum(n, vec.data());

        clock_t beg = clock();
        int sum = 0;
        for (int j = 0; j < 1e8 + 10; j += 2) {
            int a = rnd();
            int b = rnd();
            int it1 = cum.lower_bound(a);
            int it2 = cum.lower_bound(b);
            sum ^= it1 ^ it2;

            // int it0 = std::lower_bound(vec.begin(), vec.begin() + n, val ^ sum) - vec.begin();
            // sum ^= it0;

            // std::cerr << it << " " << it0 << "\n";
            // assert(it == it0);
        }
        std::cout << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << " ";
        std::cout << sum << "\n";
    }
}