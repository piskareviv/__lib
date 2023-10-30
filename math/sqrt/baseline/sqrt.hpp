#include <assert.h>
#include <stdint.h>

#include <random>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

std::mt19937 rnd(2086);

int sqrt(u32 val, u32 mod) {
    if (val == 0) {
        return 0;
    }
    if (mod == 2) {
        return 1;
    }
    assert(mod % 2);
    auto mul = [&](u32 a, u32 b) -> int {
        return a * 1ULL * b % mod;
    };
    auto add = [&](u32 a, u32 b) {
        return a + b - mod * (a + b >= mod);
    };
    auto power = [&](u32 b, u32 e) {
        u32 r = 1;
        for (; e > 0; e >>= 1) {
            if (e & 1) {
                r = mul(r, b);
            }
            b = mul(b, b);
        }
        return r;
    };
    if (power(val, (mod - 1) / 2) == mod - 1) {
        return -1;
    }

    auto mul_cum = [&](std::pair<u32, u32> a, std::pair<u32, u32> b) {
        return std::pair<u32, u32>{add(mul(a.first, b.first), mul(val, mul(a.second, b.second))), add(mul(a.first, b.second), mul(a.second, b.first))};
    };
    auto power_cum = [&](std::pair<u32, u32> b, u32 e) {
        std::pair<u32, u32> r(1, 0);
        for (; e > 0; e >>= 1) {
            if (e & 1) {
                r = mul_cum(r, b);
            }
            b = mul_cum(b, b);
        }
        return r;
    };
    while (true) {
        int d = rnd() % mod;
        d = std::max(d, 1);
        auto [x0, x1] = power_cum({d, 1}, (mod - 1) / 2);
        if (x1 != 0) {
            int res = mul(x0 + mod - 1, power(x1, mod - 2));
            if (mul(res, res) != val) {
                continue;
            }
            assert(res * 1ULL * res % mod == val);
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
