#include <cassert>
#include <iostream>

#include "IO.hpp"
#include "fft.hpp"

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    FFT fft(998'244'353);

    int n, m;
    n = m = 5e5;
    // qin >> n >> m;

    int f;

    u32 *a = (u32 *)new u32x8[std::max(1, (1 << 20) / 8)];
    u32 *b = (u32 *)new u32x8[std::max(1, (1 << 20) / 8)];

    for (int i = 0; i < n; i++) {
        // qin >> a[i];
    }
    memset(a + n, 0, (4 << 20) - 4 * n);
    for (int i = 0; i < m; i++) {
        // qin >> b[i];
    }
    memset(b + m, 0, (4 << 20) - 4 * m);

    for (int i = 0; i < 100; i++)
        fft.convolve(n + m - 1, a, b);

    for (int i = 0; i < (n + m - 1); i++) {
        // qout << a[i] << " \n"[i + 1 == (n + m - 1)];
    }

    return 0;
}
