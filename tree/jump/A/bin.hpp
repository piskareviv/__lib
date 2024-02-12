#pragma GCC target("avx2,bmi,bmi2")
#include <immintrin.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <numeric>

using i256 = __m256i;

int lower_bound_epi32(i256 vec, i256 val) {
    i256 cmp = _mm256_cmpgt_epi32(val, vec);
    uint32_t mask = _mm256_movemask_ps((__m256)cmp);
    return __builtin_ctz(~mask);
}

int lower_bound_epi32(i256 vec0, i256 vec1, i256 val) {
    i256 cmp0 = _mm256_cmpgt_epi32(val, vec0), cmp1 = _mm256_cmpgt_epi32(val, vec1);
    uint32_t mask = _mm256_movemask_ps((__m256)cmp0) | _mm256_movemask_ps((__m256)cmp1) << 8;
    return __builtin_ctz(~mask);
}
int div_up(int a, int b) {
    return (a - 1) / b + 1;
}

struct Cum {
    static constexpr int K = 16;

    int n;
    int lg;
    const int* data;
    const int** cum;

    Cum() = default;

    // data should be padded with at least 64 bytes
    [[gnu::noinline]] Cum(int n, const int* data) : n(n), data(data) {
        if (n <= K) {
            lg = 0;
            return;
        }
        int m = div_up(n + 1, K + 1);
        for (lg = 0; m > 1; m = div_up(m, K + 1), lg++) {
            ;
        }
        cum = new const int*[lg];
        m = div_up(n + 1, K + 1);
        for (int64_t i = lg - 1, f = 1; i >= 0; i--) {
            int b_cnt = div_up(m, K + 1);
            f *= (K + 1);
            int* ptr = (int*)_mm_malloc(4 * K * b_cnt, 64);

            for (int j = 0; j < b_cnt; j++) {
                for (int t = 0; t < K; t++) {
                    int64_t ind = j * f * (K + 1) + f * t + f - 1;
                    if (ind >= n) {
                        ptr[j * K + t] = std::numeric_limits<int>::max();
                    } else {
                        ptr[j * K + t] = data[ind];
                    }
                }
            }
            cum[i] = ptr;
            m = b_cnt;
        }
        assert(m == 1);
    }

    // [[gnu::noinline]]
    int lower_bound(int val) {
        int ind = 0;
        i256 vec_val = _mm256_set1_epi32(val);
        for (int i = 0; i < lg; i++) {
            i256 vec0 = _mm256_load_si256((__m256i*)(cum[i] + ind * K));
            i256 vec1 = _mm256_load_si256((__m256i*)(cum[i] + ind * K + 8));
            int dt = lower_bound_epi32(vec0, vec1, vec_val);
            ind = ind * (K + 1) + dt;
        }
        {
            // assert(0 <= ind && ind * (K + 1) <= n);
            i256 vec0 = _mm256_loadu_si256((__m256i*)(data + ind * (K + 1)));
            i256 vec1 = _mm256_loadu_si256((__m256i*)(data + ind * (K + 1) + 8));
            int dt = lower_bound_epi32(vec0, vec1, vec_val);
            ind = ind * (K + 1) + dt;

            ind = std::min(ind, n);
        }
        return ind;
    }
};
