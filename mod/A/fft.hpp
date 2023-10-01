
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

#include "barrett.hpp"

namespace FFT {
    constexpr u32 mod = 998'244'353;
    constexpr u32 pr_root = 3;

    Barrett bt(mod);
    Barrett_simd bts(mod);

    u32 power(u32 base, u32 exp) {
        u32 res = 1;
        for (; exp > 0; exp >>= 1) {
            if (exp & 1) {
                res = bt.mod_42(u64(res) * base);
            }
            base = bt.mod_42(u64(base) * base);
        }
        res = bt.shrink_4(res);
        return res;
    }

    std::vector<std::vector<u32>> w, w_rb;
    std::vector<std::vector<int>> rev_bits;

    void expand_w(int k) {
        while (w.size() < k) {
            int r = w.size();
            w.emplace_back(1 << r);
            if (r == 0) {
                w.back()[0] = 1;
            } else {
                u32 f = power(pr_root, (mod - 1) >> r + 1);
                for (int j = 0; j < (1 << r - 1); j++) {
                    w[r][2 * j] = w[r - 1][j];
                    w[r][2 * j + 1] = bt.shrink(bt.mod_22(u64(f) * w[r - 1][j]));
                }
            }
        }
    }

    void expand_w_rb(int k) {
        while (w_rb.size() < k) {
            int r = w_rb.size();
            w_rb.emplace_back(1 << r);
            if (r == 0) {
                w_rb.back()[0] = 1;
            } else {
                u32 f = power(pr_root, (mod - 1) >> r + 1);
                for (int j = 0; j < (1 << r - 1); j++) {
                    w_rb[r][j] = w_rb[r - 1][j];
                    w_rb[r][j + (1 << r - 1)] = bt.shrink(bt.mod_22(u64(f) * w_rb[r - 1][j]));
                }
            }
        }
    }

    void expand_rb(int k) {
        while (rev_bits.size() <= k) {
            int r = rev_bits.size();
            rev_bits.emplace_back(1 << r);
            for (int j = 1; j < (1 << r); j++) {
                rev_bits[r][j] = (rev_bits[r][j >> 1] >> 1) | ((j & 1) << r - 1);
            }
        }
    }

    void fft(int lg, u32* data) {
        expand_w(lg);
        expand_rb(lg);
        int n = 1 << lg;
        for (int i = 0; i < n; i++) {
            if (rev_bits[lg][i] < i) {
                std::swap(data[i], data[rev_bits[lg][i]]);
            }
        }
        for (int k = 0; k < lg; k++) {
            for (int i = 0; i < n; i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    u32 a = data[i + j], b = data[i + (1 << k) + j];
                    u32 wj = w[k][j];

                    a = bt.shrink_2(a);
                    u32 c = bt.mod_42(u64(b) * wj);
                    data[i + j] = a + c;
                    data[i + (1 << k) + j] = a + bt.mod2 - c;
                }
            }
        }
    }

    void ifft(int lg, u32* data) {
        int n = 1 << lg;
        fft(lg, data);
        std::reverse(data + 1, data + n);
        u32 rv = power(n, (mod - 2));

        for (int i = 0; i < n; i++) {
            assert(data[i] < 4 * mod);
            data[i] = bt.shrink(bt.mod_42(u64(rv) * data[i]));
        }
    }

    void fft2(int lg, u32* data) {
        expand_w_rb(lg);
        int n = 1 << lg;

        for (int k = lg - 1; k >= 0; k--) {
            for (int i = 0; i < n; i += (1 << k + 1)) {
                u32 wj = w_rb[lg - k - 1][i >> k + 1];
                for (int j = 0; j < (1 << k); j++) {
                    u32 a = data[i + j], b = data[i + (1 << k) + j];

                    a = bt.shrink_2(a);
                    u32 c = bt.mod_42(u64(b) * wj);
                    data[i + j] = a + c;
                    data[i + (1 << k) + j] = a + bt.mod2 - c;
                }
            }
        }
    }

    void ifft2(int lg, u32* data) {
        expand_w(lg);
        int n = 1 << lg;

        for (int k = 0; k < lg; k++) {
            for (int i = 0; i < n; i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    u32 a = data[i + j], b = data[i + (1 << k) + j];
                    u32 wj = w[k][j];

                    a = bt.shrink_2(a);
                    u32 c = bt.mod_42(u64(b) * wj);
                    data[i + j] = a + c;
                    data[i + (1 << k) + j] = a + bt.mod2 - c;
                }
            }
        }

        std::reverse(data + 1, data + n);
        u32 rv = power(n, (mod - 2));

        for (int i = 0; i < n; i++) {
            assert(data[i] < 4 * mod);
            data[i] = bt.shrink(bt.mod_42(u64(rv) * data[i]));
        }
    }

    std::vector<u32> convolve_slow(std::vector<u32> a, std::vector<u32> b) {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        std::vector<u32> c(sz);
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                // c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % mod;
                c[i + j] = bt.shrink(c[i + j] + bt.mod_21(u64(a[i]) * b[j]));
            }
        }

        return c;
    }

    std::vector<u32> convolve(std::vector<u32> a, std::vector<u32> b) {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);

        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        assert(sz <= (1 << lg));
        a.resize(1 << lg, 0u);
        b.resize(1 << lg, 0u);

        fft2(lg, a.data());
        fft2(lg, b.data());
        for (int i = 0; i < (1 << lg); i++) {
            a[i] = bt.mod_42(u64(bt.shrink_2(a[i])) * bt.shrink_2(b[i]));
        }
        ifft2(lg, a.data());
        for (u32 val : a) {
            assert(val < mod);
        }
        a.resize(sz);
        return a;
    }

};  // namespace FFT
