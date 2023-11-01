#include <assert.h>
#include <stdint.h>

#include <iostream>
#include <random>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

// Mongtomery32
struct Montgomery {
    u32 mod;
    u32 mod2;
    u32 n_inv;
    u32 r;
    u32 r2;

    Montgomery(u32 mod) : mod(mod) {
        assert(mod < (1 << 30));
        assert(mod % 2);
        mod2 = 2 * mod;
        n_inv = 1;
        for (int i = 0; i < 5; i++)
            n_inv *= 2 + n_inv * mod;
        assert(n_inv * mod == u32(-1));
        r = (1ULL << 32) % mod;
        r2 = r * 1ULL * r % mod;
    }

    u32 shrink(u32 val) const {
        return std::min(val, val - mod);
    }
    u32 shrink_n(u32 val) const {
        return std::min(val, val + mod);
    }
    u32 shrink2(u32 val) const {
        return std::min(val, val - mod2);
    }
    u32 shrink2_n(u32 val) const {
        return std::min(val, val + mod2);
    }

    template <bool strict = false>
    u32 reduce(u64 val) const {
        u32 ans = val + u32(val) * n_inv * u64(mod) >> 32;
        if constexpr (strict)
            ans = shrink(ans);
        return ans;
    }

    template <bool strict = false>
    u32 mul(u32 a, u32 b) const {
        u64 res = u64(a) * b;
        return reduce<strict>(res);
    }

    [[gnu::noinline]] u32 power(u32 b, u32 e) const {
        b = mul(b, r2);
        u32 r = 1;
        for (; e > 0; e >>= 1) {
            if (e & 1) {
                r = mul(r, b);
            }
            b = mul(b, b);
        }
        r = shrink(r);
        return r;
    }
};

std::mt19937 rnd(2086);

int sqrt(u32 val, u32 mod) {
    if (val <= 1) {
        return val;
    }
    if (mod == 2) {
        return 1;
    }
    assert(mod % 2);
    Montgomery mt(mod);

    if (mt.power(val, (mod - 1) / 2) == mod - 1) {
        return -1;
    }

    const u32 val_mt = mt.mul<true>(val, mt.r2);
    auto mul_cum = [&](std::pair<u32, u32> a, std::pair<u32, u32> b) {
        return std::pair<u32, u32>{mt.reduce<true>(u64(a.first) * b.first + u64(val_mt) * mt.mul(a.second, b.second)), mt.reduce<true>(u64(a.first) * b.second + u64(a.second) * b.first)};
    };
    auto power_cum = [&](std::pair<u32, u32> b, u32 e) {
        std::pair<u32, u32> r(1, 0);
        b.first = mt.mul<true>(b.first, mt.r2);
        b.second = mt.mul<true>(b.second, mt.r2);
        for (; e > 0; e >>= 1) {
            if (e & 1) {
                r = mul_cum(r, b);
            }
            b = mul_cum(b, b);
        }
        r.first = mt.shrink(r.first);
        r.second = mt.shrink(r.second);
        return r;
    };

    while (true) {
        u32 d = rnd() % mod;
        if (mt.power(mt.shrink((d * 1ULL * d) % mod + mod - val), (mod - 1) / 2) == 1) {
            continue;
        }

        auto [x0, x1] = power_cum({d, 1}, (mod - 1) / 2);
        if (x1 != 0) {
            u32 res = mt.mul<true>(mt.r2, mt.mul(x0 + mod - 1, mt.power(x1, mod - 2)));
            if (mt.mul<true>(mt.r2, mt.mul(res, res)) != val) {
                continue;
            }
            return res;
        }
    }
}

std::vector<int> sqrt(const std::vector<int>& val, const std::vector<int>& mod) {
    int n = val.size();
    std::vector<int> res(n);
    for (int i = 0; i < n; i++) {
        res[i] = sqrt(val[i], mod[i]);
        if (res[i] != -1) {
            assert(res[i] * 1ULL * res[i] % mod[i] == val[i]);
        }
    }
    return res;
}
