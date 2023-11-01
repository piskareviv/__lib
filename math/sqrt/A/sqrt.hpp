#include <assert.h>
#include <stdint.h>

#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "montgomery.hpp"

using u32 = uint32_t;
using u64 = uint64_t;

int sqrt_single(u32 val, u32 mod) {
    static std::mt19937 rnd(1329);

    if (val <= 1) {
        return val;
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

struct Cum {
    u32x8 val, mod, ind;
    u32x8 n_inv, r2;  // for Montgomery
    u32x8 aux1, aux2;
    u32 cnt;

    Cum() : cnt(0) { ; }
    Cum(u32x8 val, u32x8 mod, u32x8 ind) {
        Montgomery_simd mts(mod);
        this->val = val;
        this->mod = mod;
        this->ind = ind;
        this->n_inv = mts.n_inv;
        this->r2 = mts.r2;
        this->cnt = 8;
        this->aux1 = this->aux2 = set1_u32x8(0);
    }

    Cum& permute(u32x8 p) {
        val = permute_u32x8(val, p);
        mod = permute_u32x8(mod, p);
        ind = permute_u32x8(ind, p);
        n_inv = permute_u32x8(n_inv, p);
        r2 = permute_u32x8(r2, p);
        aux1 = permute_u32x8(aux1, p);
        aux2 = permute_u32x8(aux2, p);
        return *this;
    }

    Cum& blendv(const Cum& cum, u32x8 mask) {
        val = blendv_u32x8(val, cum.val, mask);
        mod = blendv_u32x8(mod, cum.mod, mask);
        ind = blendv_u32x8(ind, cum.ind, mask);
        n_inv = blendv_u32x8(n_inv, cum.n_inv, mask);
        r2 = blendv_u32x8(r2, cum.r2, mask);
        aux1 = blendv_u32x8(aux1, cum.aux1, mask);
        aux2 = blendv_u32x8(aux2, cum.aux2, mask);
        return *this;
    }

    Cum& set_cnt(u32 cnt) {
        this->cnt = cnt;
        return *this;
    }

    Montgomery_simd get_mts() const {
        Montgomery_simd mts;
        mts.mod = mod;
        mts.mod2 = mod << 1;
        mts.mod_sh = shift_right_u32x8_epi64(mod, 32);
        mts.n_inv = n_inv;
        mts.n_inv_sh = shift_right_u32x8_epi64(n_inv, 32);
        mts.r2 = r2;
        mts.r = mts.mul<true>(set1_u32x8(1), r2);
        return mts;
    }
};

struct Shit {
    Cum cum;

    std::function<void(Cum&, u32x8&)> proc;
    std::function<void(Cum&)> feed_0;
    std::function<void(Cum&)> feed_1;

    Shit() = default;

    void fuck(std::function<void(Cum&, u32x8&)> proc, std::function<void(Cum&)> feed_0, std::function<void(Cum&)> feed_1) {
        this->proc = proc;
        this->feed_0 = feed_0;
        this->feed_1 = feed_1;
    }

    void feed(Cum& cum2) {
        const u32x8 a = setr_u32x8(0, 1, 2, 3, 4, 5, 6, 7);
        if (cum.cnt + cum2.cnt < 8) {
            cum.blendv(cum2.permute(a - cum.cnt), (u32x8)(cum.cnt <= a)).set_cnt(cum.cnt + cum2.cnt);

            return;
        }
        cum.blendv(Cum(cum2).permute(a - cum.cnt), (u32x8)(cum.cnt <= a));

        u32x8 mask;
        proc(cum, mask);
        u32 msk = _mm256_movemask_epi8((i256)mask);
        u32 pcnt = __builtin_popcount(msk) / 4;
        u32x8 perm0 = get_compress_perm_epi32(~mask);
        u32x8 perm1 = get_compress_perm_epi32(mask);

        Cum cum_f0 = Cum(cum).permute(perm0).set_cnt(8 - pcnt);
        Cum cum_f1 = Cum(cum).permute(perm1).set_cnt(pcnt);

        cum = cum2.permute(a + (8 - cum.cnt)).set_cnt(cum.cnt + cum2.cnt - 8);

        feed_0(cum_f0);
        feed_1(cum_f1);
    }

    std::function<void(Cum&)> get_feed_func() {
        return std::function([&](Cum& cum) { feed(cum); });
    }

    void flush(int* ans) {
        for (int i = 0; i < cum.cnt; i++) {
            ans[cum.ind[i]] = sqrt_single(cum.val[i], cum.mod[i]);
        }
    }
};

std::vector<int> sqrt(const std::vector<int>& val, const std::vector<int>& mod) {
    constexpr u32 LG = 30;
    int n = val.size();
    std::mt19937 rnd(2086);
    std::vector<int> ans(n);

    Shit check_cum;
    Shit power_cum;
    Shit final_cum;

    std::function feed_ans = [&](Cum cum) {
        for (int i = 0; i < cum.cnt; i++) {
            ans[cum.ind[i]] = cum.aux1[i];
        }
    };
    std::function proc_check_cum = [&](Cum& cum, u32x8& mask) {
        Montgomery_simd mts = cum.get_mts();
        u32x8 mask_0 = (u32x8)(cum.val <= 1);
        u32x8 mask_1 = (u32x8)(mts.power<LG - 1>(cum.val, cum.mod >> 1) == cum.mod - 1);
        mask = mask_0 | mask_1;
        cum.aux1 = mask_0 ? cum.val : set1_u32x8(-1u);
    };

    static int cnt_cum0 = 0;

    std::function proc_power_cum = [&](Cum& cum, u32x8& mask) {
        cnt_cum0++;
        Montgomery_simd mts = cum.get_mts();

        u32x8 d;
        for (int i = 0; i < 8; i++) {
            // d[i] = rnd() % mod[i];  // ! cum
            d[i] = rnd();  // ! cum
        }

        u32x8 val_mt = mts.mul<true>(cum.val, mts.r2);
        u32x8 exp = cum.mod >> 1;
        u32x8 r0 = set1_u32x8(1), r1 = set1_u32x8(0);
        u32x8 b0 = mts.mul<true>(mts.r2, d);  // ! cum
        u32x8 b1 = mts.r;

        for (u32 i = 0; i < LG; i++, exp >>= 1) {
            auto [p0, p1] = mts.mul_mod(r0, r1, b0, b1, val_mt);

            r0 = (exp & 1) ? p0 : r0;
            r1 = (exp & 1) ? p1 : r1;

            std::tie(b0, b1) = mts.sq_mod(b0, b1, val_mt);
        }
        r0 = mts.shrink(r0);
        r1 = mts.shrink(r1);

        mask = (u32x8)(r1 != 0);
        cum.aux1 = r0;
        cum.aux2 = r1;
    };
    static int cnt_cum1 = 0;
    std::function proc_final_cum = [&](Cum& cum, u32x8& mask) {
        cnt_cum1++;

        Montgomery_simd mts = cum.get_mts();

        u32x8 res = mts.mul<true>(mts.r2, mts.mul(cum.aux1 + cum.mod - 1, mts.power<LG>(cum.aux2, cum.mod - 2)));
        u32x8 sq = mts.mul<true>(mts.r2, mts.mul(res, res));
        cum.aux1 = res;
        cum.aux2 = set1_u32x8(0);
        mask = (u32x8)(sq == cum.val);
    };

    check_cum.fuck(proc_check_cum, power_cum.get_feed_func(), feed_ans);
    power_cum.fuck(proc_power_cum, power_cum.get_feed_func(), final_cum.get_feed_func());
    final_cum.fuck(proc_final_cum, power_cum.get_feed_func(), feed_ans);

    u32x8 ind = setr_u32x8(0, 1, 2, 3, 4, 5, 6, 7);
    for (int i = 0; i + 8 <= n; i += 8) {
        Cum cum(loadu_u32x8((u32*)val.data() + i), loadu_u32x8((u32*)mod.data() + i), ind);
        check_cum.feed(cum);
        ind += 8;
    }
    check_cum.flush(ans.data());
    power_cum.flush(ans.data());
    final_cum.flush(ans.data());

    for (int i = n / 8 * 8; i < n; i++) {
        ans[i] = sqrt_single(val[i], mod[i]);
    }

#ifdef LOCAL
    for (int i = 0; i < n; i++) {
        if (ans[i] != -1) {
            assert(val[i] == 0 || ans[i] != 0);
            assert(ans[i] * 1ULL * ans[i] % mod[i] == val[i]);
        }
    }
#endif

    // std::cerr << cnt_cum0 * 8.0 / n << " " << cnt_cum1 * 8.0 / n << "\n";

    return ans;
}
