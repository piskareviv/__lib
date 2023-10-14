#pragma GCC optimize("O3")
#include <cassert>
#include <iostream>

// constexpr int MEM = 1e8;
// size_t _ptr = 0;
// alignas(64) char _data[MEM];
// void *operator new(size_t s) {
//     _ptr += s;
//     assert(_ptr <= MEM);
//     return _data + _ptr - s;
// }
// void operator delete(void *) { ; }

#include "IO.hpp"
#include "fft.hpp"

// alignas(64) u32 a[1 << 20];
// alignas(64) u32 b[1 << 20];

int32_t main() {
    int n, m;
    n = m = 5e5;
    // qin >> n >> m;

    u32 *a = (u32 *)std::aligned_alloc(64, (4 << 20));
    u32 *b = (u32 *)std::aligned_alloc(64, (4 << 20));

    // u32 *a = (u32 *)malloc((4 << 20) + 64);
    // u32 *b = (u32 *)malloc((4 << 20) + 64);
    // a += 64 - (RC(u64, a) & 63) & 63;
    // b += 64 - (RC(u64, b) & 63) & 63;

    for (int i = 0; i < n; i++) {
        // qin >> a[i];
    }
    for (int i = 0; i < m; i++) {
        // qin >> b[i];
    }

    FFT::convolve(n + m - 1, a, b);
    // FFT::free_mem();

    for (int i = 0; i < (n + m - 1); i++) {
        // qout << a[i] << " \n"[i + 1 == (n + m - 1)];
    }

    return 0;
}
