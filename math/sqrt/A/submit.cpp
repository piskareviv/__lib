#pragma GCC target("avx2,bmi,bmi2")
#include <immintrin.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iostream>

using u32 = uint32_t;
using u64 = uint64_t;

namespace simd {
    using i128 = __m128i;
    using i256 = __m256i;
    using u32x8 = u32 __attribute__((vector_size(32)));
    using u64x4 = u64 __attribute__((vector_size(32)));

    u32x8 load_u32x8(u32* ptr) {
        return (u32x8)(_mm256_load_si256((i256*)ptr));
    }
    u32x8 loadu_u32x8(u32* ptr) {
        return (u32x8)(_mm256_loadu_si256((i256*)ptr));
    }
    void store_u32x8(u32* ptr, u32x8 val) {
        _mm256_store_si256((i256*)ptr, (i256)(val));
    }
    void storeu_u32x8(u32* ptr, u32x8 val) {
        _mm256_storeu_si256((i256*)ptr, (i256)(val));
    }

    u32x8 set1_u32x8(u32 val) {
        return (u32x8)(_mm256_set1_epi32(val));
    }
    u64x4 set1_u64x4(u64 val) {
        return (u64x4)(_mm256_set1_epi64x(val));
    }

    u32x8 setr_u32x8(u32 a0, u32 a1, u32 a2, u32 a3, u32 a4, u32 a5, u32 a6, u32 a7) {
        return (u32x8)(_mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7));
    }
    u64x4 setr_u64x4(u64 a0, u64 a1, u64 a2, u64 a3) {
        return (u64x4)(_mm256_setr_epi64x(a0, a1, a2, a3));
    }

    template <int imm8>
    u32x8 shuffle_u32x8(u32x8 val) {
        return (u32x8)(_mm256_shuffle_epi32((i256)(val), imm8));
    }
    u32x8 permute_u32x8(u32x8 val, u32x8 p) {
        return (u32x8)(_mm256_permutevar8x32_epi32((i256)(val), (i256)(p)));
    }

    template <int imm8>
    u32x8 permute_u32x8_epi128(u32x8 a, u32x8 b) {
        return (u32x8)(_mm256_permute2x128_si256((i256)(a), (i256)(b), imm8));
    }

    template <int imm8>
    u32x8 blend_u32x8(u32x8 a, u32x8 b) {
        return (u32x8)(_mm256_blend_epi32((i256)(a), (i256)(b), imm8));
    }
    u32x8 blendv_u32x8(u32x8 a, u32x8 b, u32x8 mask) {
        return (u32x8)(_mm256_blendv_epi8((i256)(a), (i256)(b), (i256)mask));
    }

    u32x8 shift_left_u32x8_epi64(u32x8 val, int imm8) {
        return (u32x8)(_mm256_slli_epi64((i256)(val), imm8));
    }
    u32x8 shift_right_u32x8_epi64(u32x8 val, int imm8) {
        return (u32x8)(_mm256_srli_epi64((i256)(val), imm8));
    }

    u32x8 min_u32x8(u32x8 a, u32x8 b) {
        return (u32x8)(_mm256_min_epu32((i256)(a), (i256)(b)));
    }
    u32x8 mul64_u32x8(u32x8 a, u32x8 b) {
        return (u32x8)(_mm256_mul_epu32((i256)(a), (i256)(b)));
    }

    u32x8 get_compress_perm_epi32(u32x8 mask) {
        u32 msk = _mm256_movemask_epi8((i256)mask);
        u32 cum = _pext_u32(0x76543210, msk);
        u64 cum64 = _pdep_u64(cum, 0x0F'0F'0F'0F'0F'0F'0F'0F);
        return (u32x8)_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(cum64));
    }
};  // namespace simd
using namespace simd;

// Montgomery32
struct Montgomery {
    u32 mod;
    u32 mod2;   // 2 * mod
    u32 n_inv;  // n_inv * mod == -1 (mod 2^32)
    u32 r;      // 2^32 % mod
    u32 r2;     // (2^32) ^ 2 % mod

    Montgomery() = default;
    Montgomery(u32 mod) : mod(mod) {
        assert(mod % 2);
        assert(mod < (1 << 30));
        n_inv = -mod & 3;
        for (int i = 0; i < 4; i++) {
            n_inv *= 2u + n_inv * mod;
        }
        assert(n_inv * mod == -1u);

        mod2 = 2 * mod;
        r = (1ULL << 32) % mod;
        r2 = r * u64(r) % mod;
    }

    u32 shrink(u32 val) const {
        return std::min(val, val - mod);
    }
    u32 shrink2(u32 val) const {
        return std::min(val, val - mod2);
    }
    u32 shrink_n(u32 val) const {
        return std::min(val, val + mod);
    }
    u32 shrink2_n(u32 val) const {
        return std::min(val, val + mod2);
    }

    // val should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u32 reduce(u64 val) const {
        u32 res = (val + u32(val) * n_inv * u64(mod)) >> 32;
        if constexpr (strict)
            res = shrink(res);
        return res;
    }

    // a * b should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u32 mul(u32 a, u32 b) const {
        u64 val = u64(a) * b;
        u32 res = (val + u32(val) * n_inv * u64(mod)) >> 32;
        if constexpr (strict)
            res = shrink(res);
        return res;
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

// Montgomery32
struct Montgomery_simd {
    alignas(32) u32x8 mod, mod_sh;
    alignas(32) u32x8 mod2;             // 2 * mod
    alignas(32) u32x8 n_inv, n_inv_sh;  // n_inv * mod == -1 (mod 2^32)
    alignas(32) u32x8 r;                // 2^32 % mod
    alignas(32) u32x8 r2;               // (2^32) ^ 2 % mod

    Montgomery_simd() = default;
    Montgomery_simd(u32x8 md) {
        n_inv = -md & 3;
        mod = md;
        mod2 = 2 * md;
        for (int i = 0; i < 4; i++) {
            n_inv *= 2 + n_inv * mod;
        }
        n_inv_sh = shift_right_u32x8_epi64(n_inv, 32);
        mod_sh = shift_right_u32x8_epi64(mod, 32);
        for (int i = 0; i < 8; i++) {
            r2[i] = -1ULL % mod[i] + 1;  // !!!!!!!!! CUM
        }
        r = mul<true>(set1_u32x8(1), r2);

        // mod = set1_u32x8(mt.mod);
        // mod2 = set1_u32x8(mt.mod2);
        // n_inv = set1_u32x8(mt.n_inv);
        // r = set1_u32x8(mt.r);
        // r2 = set1_u32x8(mt.r2);
    }

    u32x8 shrink(u32x8 val) const {
        return min_u32x8(val, val - mod);
    }
    u32x8 shrink2(u32x8 val) const {
        return min_u32x8(val, val - mod2);
    }
    u32x8 shrink_n(u32x8 val) const {
        return min_u32x8(val, val + mod);
    }
    u32x8 shrink2_n(u32x8 val) const {
        return min_u32x8(val, val + mod2);
    }

    // val should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u32x8 reduce(u32x8 x0246, u32x8 x1357) const {
        u32x8 x0246_ninv = mul64_u32x8(x0246, n_inv);
        u32x8 x1357_ninv = mul64_u32x8(x1357, n_inv_sh);
        u32x8 res = blend_u32x8<0b10'10'10'10>(shift_right_u32x8_epi64(u32x8((u64x4)x0246 + (u64x4)mul64_u32x8(x0246_ninv, mod)), 32),
                                               u32x8((u64x4)x1357 + (u64x4)mul64_u32x8(x1357_ninv, mod_sh)));
        if constexpr (strict)
            res = shrink(res);
        return res;
    }

    // a * b should be in [0, 2**32 * mod)
    // result in [0, 2 * mod)   <false>
    // result in [0, mod)       <true>
    template <bool strict = false>
    u32x8 mul(u32x8 a, u32x8 b) const {
        u32x8 x0246 = mul64_u32x8(a, b);
        u32x8 x1357 = mul64_u32x8(shift_right_u32x8_epi64(a, 32), shift_right_u32x8_epi64(b, 32));
        u32x8 res = reduce<strict>(x0246, x1357);
        return res;
    }

    // multiplies mod x^2 - d
    std::pair<u32x8, u32x8> mul_mod(u32x8 a0, u32x8 a1, u32x8 b0, u32x8 b1, u32x8 d) {
        u32x8 a0_sh = shift_right_u32x8_epi64(a0, 32);
        u32x8 a1_sh = shift_right_u32x8_epi64(a1, 32);
        u32x8 b0_sh = shift_right_u32x8_epi64(b0, 32);
        u32x8 b1_sh = shift_right_u32x8_epi64(b1, 32);

        u32x8 c0 = mul(a1, d);
        u32x8 c0_sh = shift_right_u32x8_epi64(c0, 32);
        u32x8 res0 = reduce<true>(u32x8((u64x4)mul64_u32x8(a0, b0) + (u64x4)mul64_u32x8(c0, b1)),
                                  u32x8((u64x4)mul64_u32x8(a0_sh, b0_sh) + (u64x4)mul64_u32x8(c0_sh, b1_sh)));
        u32x8 res1 = reduce<true>(u32x8((u64x4)mul64_u32x8(a0, b1) + (u64x4)mul64_u32x8(a1, b0)),
                                  u32x8((u64x4)mul64_u32x8(a0_sh, b1_sh) + (u64x4)mul64_u32x8(a1_sh, b0_sh)));
        return {res0, res1};
    }

    // multiplies mod x^2 - d
    std::pair<u32x8, u32x8> sq_mod(u32x8 a0, u32x8 a1, u32x8 d) {
        u32x8 a0_sh = shift_right_u32x8_epi64(a0, 32);
        u32x8 a1_sh = shift_right_u32x8_epi64(a1, 32);

        u32x8 c0 = mul(a1, d);
        u32x8 c0_sh = shift_right_u32x8_epi64(c0, 32);
        u32x8 res0 = reduce<true>(u32x8((u64x4)mul64_u32x8(a0, a0) + (u64x4)mul64_u32x8(c0, a1)),
                                  u32x8((u64x4)mul64_u32x8(a0_sh, a0_sh) + (u64x4)mul64_u32x8(c0_sh, a1_sh)));
        u32x8 res1 = reduce<true>(u32x8((u64x4)mul64_u32x8(a0, a1) << 1),
                                  u32x8((u64x4)mul64_u32x8(a0_sh, a1_sh) << 1));
        return {res0, res1};
    }

    // lg - number of bits in exp
    template <u32 lg = 30>
    u32x8 power(u32x8 base, u32x8 exp) const {
        u32x8 res = set1_u32x8(1);
        base = mul(base, r2);
        for (u32 i = 0; i < lg; i++, exp >>= 1) {
            res = mul(res, (exp & 1) ? base : r);
            base = mul(base, base);
        }
        res = shrink(res);

        return res;
    }
};
#include <assert.h>
#include <stdint.h>

#include <functional>
#include <iostream>
#include <random>
#include <vector>


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
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstring>
#include <iostream>

using u32 = uint32_t;
using u64 = uint64_t;

// io from https://judge.yosupo.jp/submission/142782

namespace QIO_base {
    constexpr int O_buffer_default_size = 1 << 18;
    constexpr int O_buffer_default_flush_threshold = 40;
    struct _int_to_char_tab {
        char tab[40000];
        constexpr _int_to_char_tab() : tab() {
            for (int i = 0; i != 10000; ++i) {
                for (int j = 3, n = i; ~j; --j) {
                    tab[i * 4 + j] = n % 10 + 48, n /= 10;
                }
            }
        }
    } constexpr _otab;
}  // namespace QIO_base
namespace QIO_I {
    using namespace QIO_base;
    struct Qinf {
        FILE *f;
        char *bg, *ed, *p;
        struct stat Fl;
        Qinf(FILE *fi) : f(fi) {
            int fd = fileno(f);
            fstat(fd, &Fl);
            bg = (char *)mmap(0, Fl.st_size + 1, PROT_READ, MAP_PRIVATE, fd, 0);
            p = bg, ed = bg + Fl.st_size;
            madvise(p, Fl.st_size + 1, MADV_SEQUENTIAL);
        }
        ~Qinf() { munmap(bg, Fl.st_size + 1); }
        void skip_space() {
            while (*p <= ' ') {
                ++p;
            }
        }
        char get() { return *p++; }
        char seek() { return *p; }
        bool eof() { return p == ed; }
        Qinf &read(char *s, size_t count) { return memcpy(s, p, count), p += count, *this; }
        Qinf &operator>>(u32 &x) {
            skip_space(), x = 0;
            for (; *p > ' '; ++p) {
                x = x * 10 + (*p & 0xf);
            }
            return *this;
        }
        Qinf &operator>>(int &x) {
            skip_space();
            if (*p == '-') {
                for (++p, x = 48 - *p++; *p > ' '; ++p) {
                    x = x * 10 - (*p ^ 48);
                }
            } else {
                for (x = *p++ ^ 48; *p > ' '; ++p) {
                    x = x * 10 + (*p ^ 48);
                }
            }
            return *this;
        }
    } qin(stdin);
}  // namespace QIO_I
namespace QIO_O {
    using namespace QIO_base;
    struct Qoutf {
        FILE *f;
        char *bg, *ed, *p;
        char *ed_thre;
        int fp;
        u64 _fpi;
        Qoutf(FILE *fo, size_t sz = O_buffer_default_size) : f(fo), bg(new char[sz]), ed(bg + sz), p(bg), ed_thre(ed - O_buffer_default_flush_threshold), fp(6), _fpi(1000000ull) {}
        void flush() { fwrite_unlocked(bg, 1, p - bg, f), p = bg; }
        void chk() {
            if (__builtin_expect(p > ed_thre, 0)) {
                flush();
            }
        }
        ~Qoutf() {
            flush();
            delete[] bg;
        }
        void put4(u32 x) {
            if (x > 99u) {
                if (x > 999u) {
                    memcpy(p, _otab.tab + (x << 2) + 0, 4), p += 4;
                } else {
                    memcpy(p, _otab.tab + (x << 2) + 1, 3), p += 3;
                }
            } else {
                if (x > 9u) {
                    memcpy(p, _otab.tab + (x << 2) + 2, 2), p += 2;
                } else {
                    *p++ = x ^ 48;
                }
            }
        }
        void put2(u32 x) {
            if (x > 9u) {
                memcpy(p, _otab.tab + (x << 2) + 2, 2), p += 2;
            } else {
                *p++ = x ^ 48;
            }
        }
        Qoutf &write(const char *s, size_t count) {
            if (count > 1024 || p + count > ed_thre) {
                flush(), fwrite_unlocked(s, 1, count, f);
            } else {
                memcpy(p, s, count), p += count, chk();
            }
            return *this;
        }
        Qoutf &operator<<(char ch) { return *p++ = ch, *this; }
        Qoutf &operator<<(u32 x) {
            if (x > 99999999u) {
                put2(x / 100000000u), x %= 100000000u;
                memcpy(p, _otab.tab + ((x / 10000u) << 2), 4), p += 4;
                memcpy(p, _otab.tab + ((x % 10000u) << 2), 4), p += 4;
            } else if (x > 9999u) {
                put4(x / 10000u);
                memcpy(p, _otab.tab + ((x % 10000u) << 2), 4), p += 4;
            } else {
                put4(x);
            }
            return chk(), *this;
        }
        Qoutf &operator<<(int x) {
            if (x < 0) {
                *p++ = '-', x = -x;
            }
            return *this << static_cast<u32>(x);
        }
    } qout(stdout);
}  // namespace QIO_O
namespace QIO {
    using QIO_I::qin;
    using QIO_I::Qinf;
    using QIO_O::qout;
    using QIO_O::Qoutf;
}  // namespace QIO
using namespace QIO;
#pragma GCC target("avx2")
#include <iostream>


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
