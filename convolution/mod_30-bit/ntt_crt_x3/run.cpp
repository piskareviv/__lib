#include <cassert>
#include <iostream>

#define qin std::cin
#define qout std::cout
// #include "IO.hpp"

#include "ntt.hpp"

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // 469762049 26 { 2 : 26, 7 : 1 }
    // 167772161 25 { 2 : 25, 5 : 1 }
    // 754974721 24 { 2 : 24, 3 : 2, 5 : 1 }
    // 897581057 23 {2: 23, 107: 1}
    // 998244353 23 {2: 23, 7: 1, 17: 1}

    using u128 = __uint128_t;

    constexpr u32 mod = 1e9 + 7;
    Montgomery mt(mod);
    Montgomery mts(mod);

    // std::array<u32, 3> mods = {167772161, 469762049, 754974721};
    const std::array<u32, 3> mods = {998'244'353, 897581057, 754974721};
    // ! using these primes because ntt input should be in [0, 2 *mod)

    std::array<u32, 3> mod_inv;
    std::array<u64, 3> crt;
    std::array<NTT, 3> ntt;
    for (int i = 0; i < 3; i++) {
        ntt[i] = NTT(mods[i]);

        u32 m0 = mods[i];
        u32 m1 = mods[(i + 1) % 3];
        u32 m2 = mods[(i + 2) % 3];

        mod_inv[i] = ntt[i].mt.power((m1 * 1ULL * m2) % m0, m0 - 2);
        crt[i] = m1 * 1ULL * m2;
        mod_inv[i] = ntt[i].mt.mul<true>(mod_inv[i], ntt[i].mt.r2);
    }

    int n, m;
    n = m = 5e5;
    qin >> n >> m;

    int lg = std::__lg(std::max(1, n + m - 2)) + 1;

    u32 *a = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
    u32 *b = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);

    for (int i = 0; i < n; i++) {
        qin >> a[i];
    }
    memset(a + n, 0, (4 << lg) - 4 * n);
    for (int i = 0; i < m; i++) {
        qin >> b[i];
    }
    memset(b + m, 0, (4 << lg) - 4 * m);

    u32 *a1 = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
    u32 *a2 = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
    u32 *b1 = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);
    u32 *b2 = (u32 *)_mm_malloc(std::max(32, (1 << lg) * 4), 32);

    auto beg = clock();

    // for (int i = 0; i < 100; i++)
    //
    {
        memcpy(a1, a, 4 << lg);
        memcpy(a2, a, 4 << lg);
        memcpy(b1, b, 4 << lg);
        memcpy(b2, b, 4 << lg);

        ntt[0].convolve(lg, a, b);
        ntt[1].convolve(lg, a1, b1);
        ntt[2].convolve(lg, a2, b2);

        std::cerr << "conv " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
        beg = clock();

        u128 m012 = u128(mods[0]) * mods[1] * mods[2];
        for (int i = 0; i < (1 << lg); i++) {
            u128 x = (ntt[0].mt.mul<true>(a[i], mod_inv[0])) * u128(crt[0]);
            u128 y = (ntt[1].mt.mul<true>(a1[i], mod_inv[1])) * u128(crt[1]);
            u128 z = (ntt[2].mt.mul<true>(a2[i], mod_inv[2])) * u128(crt[2]);

            u128 s = x + y + z;

            s = (s >= m012) ? s - m012 : s;
            s = (s >= m012) ? s - m012 : s;

            assert(s < m012);

            u32 res = s % mod;
            a[i] = res;
        }

        std::cerr << "crt " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "\n";
        beg = clock();
    }

    for (int i = 0; i < (n + m - 1); i++) {
        qout << a[i] << " \n"[i + 1 == (n + m - 1)];
    }

    return 0;
}
