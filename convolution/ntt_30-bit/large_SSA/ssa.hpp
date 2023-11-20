#include <iostream>
#include <string>
#include <vector>

#include "ntt.hpp"

struct SSA {
    // ring modulo X^(2^(L+1)) + 1
    struct Cum {
        u32* ptr;
        size_t sh;

        u32& operator()(int L, size_t ind) {
            return ptr[ind];
        };
    };

    NTT ntt;

    SSA(u32 mod = 998'244'353) : ntt(mod) {
        sizeof(SSA);
    }

    std::pair<int, int> get_LB(int lg) const {
        int L = (lg - 1) / 2;
        int B = lg - L;
        return {L, B};
    }

    // [a, b] -> [a + b, a - b]
    [[gnu::noinline]] __attribute__((optimize("O3"))) void add_sub(int L, Cum& a, Cum& b) const {
        const auto mt = ntt.mt;
        const auto mts = ntt.mts;
        size_t sz = 1ULL << L + 1;
        for (size_t i = 0; i < sz; i += 8) {
            u32x8 va = load_u32x8(a.ptr + i);
            u32x8 vb = load_u32x8(b.ptr + i);
            store_u32x8(a.ptr + i, mts.shrink(va + vb));
            store_u32x8(b.ptr + i, mts.shrink_n(va - vb));
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void mul(int L, size_t w, Cum& a) const {
        const auto mt = ntt.mt;
        size_t sz = 1ULL << L + 1;
        assert(w < 2 * sz);
        if (w >= sz) {
            w -= sz;
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (size_t i = w; i < sz; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        } else {
            std::rotate(a.ptr, a.ptr + sz - w, a.ptr + sz);
            for (size_t i = 0; i < w; i++) {
                a.ptr[i] = mt.mod - a.ptr[i];
            }
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void ssa_ntt_all_rec(int L, int B, int k, size_t w, Cum* cum_a, Cum* cum_b) const {
        if (k == 0) {
            auto mt = ntt.mt;
            u32 rv = mt.power(mt.mod + 1 >> 1, B);
            ntt.convolve_neg(L + 1, cum_a[0].ptr, cum_b[0].ptr, rv);
        } else {
            int dt = B - k;
            size_t sz = 1ULL << L + 1;

            size_t tf = (sz >> dt) * w;
            size_t tf_r = (sz * 2 - tf) & (sz * 2 - 1);

            for (auto cum : std::array{cum_a, cum_b}) {
                for (size_t i = 0; i < (1 << k - 1); i++) {
                    mul(L, tf, cum[i + (1 << k - 1)]);
                    add_sub(L, cum[i], cum[i + (1 << k - 1)]);
                }
            }

            ssa_ntt_all_rec(L, B, k - 1, w, cum_a, cum_b);
            ssa_ntt_all_rec(L, B, k - 1, w + (1ULL << dt), cum_a + (1 << k - 1), cum_b + (1 << k - 1));

            for (size_t i = 0; i < (1 << k - 1); i++) {
                add_sub(L, cum_a[i], cum_a[i + (1 << k - 1)]);
                mul(L, tf_r, cum_a[i + (1 << k - 1)]);
            }
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void ssa_ntt_rec(int L, int B, int k, size_t w, Cum* cum) const {
        if (k == 0) {
            return;
        }
        int dt = B - k;
        size_t sz = 1ULL << L + 1;
        size_t tf = (sz >> dt) * w;
        for (size_t i = 0; i < (1 << k - 1); i++) {
            mul(L, tf, cum[i + (1 << k - 1)]);
            add_sub(L, cum[i], cum[i + (1 << k - 1)]);
        }

        ssa_ntt_rec(L, B, k - 1, w, cum);
        ssa_ntt_rec(L, B, k - 1, w + (1ULL << dt), cum + (1 << k - 1));
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void ssa_intt_rec(int L, int B, int k, size_t w, Cum* cum) const {
        if (k == 0) {
            return;
        }
        int dt = B - k;
        size_t sz = 1ULL << L + 1;
        size_t tf = (sz >> dt) * w;
        size_t tf_r = (sz * 2 - tf) & (sz * 2 - 1);

        ssa_intt_rec(L, B, k - 1, w, cum);
        ssa_intt_rec(L, B, k - 1, w + (1ULL << dt), cum + (1 << k - 1));

        for (size_t i = 0; i < (1 << k - 1); i++) {
            add_sub(L, cum[i], cum[i + (1 << k - 1)]);
            mul(L, tf_r, cum[i + (1 << k - 1)]);
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) void convolve(int lg, std::vector<Cum>& a, std::vector<Cum>& b) const {
        size_t sz = 1ULL << lg;
        assert(lg > 6);

        auto [L, B] = get_LB(lg);
        assert(a.size() == (1 << B));
        assert(b.size() == (1 << B));

        if (!"CUM") {
            ssa_ntt_rec(L, B, B, 0, a.data());
            ssa_ntt_rec(L, B, B, 0, b.data());

            auto mt = ntt.mt;
            u32 rv = mt.power(mt.mod + 1 >> 1, B);
            for (int i = 0; i < (1 << B); i++) {
                ntt.convolve_neg(L + 1, a[i].ptr, b[i].ptr, rv);
            }

            ssa_intt_rec(L, B, B, 0, a.data());
        } else {
            ssa_ntt_all_rec(L, B, B, 0, a.data(), b.data());
        }

        const auto mt = ntt.mt;
        const auto mts = ntt.mts;

        for (size_t i = 0; i < (1ULL << B); i++) {
            if (i + 1 < (1ULL << B)) {
                for (size_t j = 0; j < (1ULL << L); j++) {
                    a[i + 1].ptr[j] = mt.shrink(a[i + 1].ptr[j] + a[i].ptr[j + (1ULL << L)]);
                    a[i].ptr[j + (1ULL << L)] = 0;
                }
            }
        }
    }

    [[gnu::noinline]] __attribute__((optimize("O3"))) std::vector<u32> convolve(const std::vector<u32>& a, const std::vector<u32>& b) const {
        int sz = std::max(0, (int)a.size() + (int)b.size() - 1);
        int lg = std::__lg(std::max(1, sz - 1)) + 1;
        if (lg <= 6) {
            return ntt.convolve(a, b);
        }

        auto [L, B] = get_LB(lg);
        std::vector<Cum> cum_a(1 << B), cum_b(1 << B);
        auto fill_cum = [&](std::vector<Cum>& cum, const std::vector<u32>& vec) {
            for (size_t i = 0; i < (1ULL << B); i++) {
                cum[i] = Cum{(u32*)_mm_malloc(4ULL << L + 1, 64), 0};
                memset(cum[i].ptr, 0, 4ULL << L + 1);
                for (size_t j = 0; j < (1ULL << L); j++) {
                    size_t ind = i * (1 << L) + j;
                    if (ind < vec.size())
                        cum[i].ptr[j] = vec[ind];
                }
            }
        };
        fill_cum(cum_b, b);
        fill_cum(cum_a, a);

        convolve(lg, cum_a, cum_b);
        std::vector<u32> res(sz);
        for (size_t i = 0; i < (1ULL << B); i++) {
            for (size_t j = 0; j < (1ULL << L); j++) {
                size_t ind = i * (1ULL << L) + j;
                if (ind < sz)
                    res[ind] = cum_a[i].ptr[j];
            }
        }
        return res;
    }
};
