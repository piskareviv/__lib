#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "bin.hpp"

namespace WTF {

    class ST {
       private:
        int size, layer_cnt, **data;

       public:
        ST(int n = 0, int *base = nullptr) : size(n), layer_cnt(32 - __builtin_clz(n)), data(new int *[32 - __builtin_clz(n)]) {
            for (int i = 0; i < layer_cnt; i++) {
                data[i] = new int[size];
            }
            if (base != nullptr) {
                config(n, base);
            }
        }

        void config(int n, int *base) {
            if (n > size) {
                for (int i = 0; i < layer_cnt; i++) {
                    // delete[] data[i];
                }
                // delete[] data;

                layer_cnt = 32 - __builtin_clz(n);
                data = new int *[layer_cnt];
                for (int i = 0; i < layer_cnt; i++) {
                    data[i] = new int[n];
                }
            }
            std::copy(base, base + n, data[0]);
            for (int k = 1; k < layer_cnt; k++) {
                for (int i = 0; i < (n - (1 << (k - 1))); i++) {
                    data[k][i] = std::min(data[k - 1][i], data[k - 1][i + (1 << (k - 1))]);
                }
            }
        }

        //[l, r)
        int get(int l, int r) {
            if (l == r) {
                return std::numeric_limits<int>::max();
            }
            int lg = 31 - __builtin_clz(r - l);
            int res = std::min(data[lg][l], data[lg][r - (1 << lg)]);
            return res;
        }
        ~ST() {
            for (int i = 0; i < layer_cnt; i++) {
                // delete[] data[i];
            }
            // delete[] data;
        }
    };

    class DS {
       private:
        static constexpr int b = 32;
        ST st;
        int size;
        const int *input;
        uint32_t *data;

        //[l, r)
        int get2(int l, int r) {
            if (l == r) {
                return std::numeric_limits<int>::max();
            }
            if ((r - l) > b) {
                exit(1);
            }
            uint32_t val = data[l];
            if ((r - l) != 32) {
                val &= (1ull << uint32_t(r * 1ull - l * 1ull)) - 1ull;
            }
            int x = 31 - __builtin_clz(val);
            if (x == -1) {
                exit(2);
            }
            return input[l + x];
        }

       public:
        DS() = default;
        DS(int n, const int *base) : size(n) {
            input = base;

            int *help = new int[n];
            {
                std::vector<int> stack(2 * b + 10);
                stack.clear();
                for (int i = 0; i < n; i++) {
                    if (stack.size() > (2 * b + 5)) {
                        for (int j = 0; j < b; j++) {
                            help[stack[j]] = n;
                        }
                        stack.erase(stack.begin(), stack.begin() + b);
                    }
                    while ((!stack.empty()) && (base[stack.back()] > base[i])) {
                        help[stack.back()] = i;
                        stack.pop_back();
                    }
                    stack.push_back(i);
                }
                for (int i : stack) {
                    help[i] = n;
                }
            }

            data = new uint32_t[n];
            for (int i = n - 1; i >= 0; i--) {
                data[i] = 1ull;
                if ((help[i] < n) && ((help[i] - i) < b)) {
                    data[i] |= data[help[i]] << ((uint32_t)(help[i] - i));
                }
            }
            // delete[] help;

            help = nullptr;
            help = new int[(n - 1ll) / b + 1];
            std::fill(help, help + (n - 1ll) / b + 1, std::numeric_limits<int>::max());
            for (int i = 0; i < n; i++) {
                help[i / b] = std::min(help[i / b], base[i]);
            }
            st.config((n - 1ll) / b + 1, help);
            // delete[] help;
        }

        //[l, r)
        int get(int l, int r) {
            if ((r - l) <= b) {
                return get2(l, r);
            }
            int ll = ((l != 0) ? ((l - 1) / b + 1) : 0), rr = r / b;
            return std::min({st.get(ll, rr), get2(l, ll * b), get2(rr * b, r)});
        }

        ~DS() {
            // delete[] data;
        }
    };

    class DS2 {
       private:
        static constexpr int b = 14;
        ST st;
        int size;
        std::pair<int, uint16_t> *data;

        //[l, r)
        int get2(int l, int r) {
            if (l == r) {
                return std::numeric_limits<int>::max();
            }
            if ((r - l) > b) {
                exit(1);
            }
            uint32_t val = data[l].second;
            if ((r - l) != 32) {
                val &= (1ull << uint32_t(r * 1ull - l * 1ull)) - 1ull;
            }
            int x = 31 - __builtin_clz(val);
            if (x == -1) {
                exit(2);
            }
            return data[l + x].first;
        }

       public:
        DS2() = default;
        DS2(int n, const int *base) : size(n) {
            data = new std::pair<int, uint16_t>[n];
            for (int i = 0; i < n; i++) {
                data[i].first = base[i];
            }
            // // delete[] base;

            int *help = new int[n];
            {
                std::vector<int> stack(2 * b + 10);
                stack.clear();
                for (int i = 0; i < n; i++) {
                    if (stack.size() > (2 * b + 5)) {
                        for (int j = 0; j < b; j++) {
                            help[stack[j]] = n;
                        }
                        stack.erase(stack.begin(), stack.begin() + b);
                    }
                    while ((!stack.empty()) && (data[stack.back()].first > data[i].first)) {
                        help[stack.back()] = i;
                        stack.pop_back();
                    }
                    stack.push_back(i);
                }
                for (int i : stack) {
                    help[i] = n;
                }
            }

            for (int i = n - 1; i >= 0; i--) {
                data[i].second = 1ull;
                if ((help[i] < n) && ((help[i] - i) < b)) {
                    data[i].second |= data[help[i]].second << ((uint32_t)(help[i] - i));
                }
            }
            // delete[] help;

            help = new int[(n - 1ll) / b + 1];
            std::fill(help, help + (n - 1ll) / b + 1, std::numeric_limits<int>::max());
            for (int i = 0; i < n; i++) {
                help[i / b] = std::min(help[i / b], data[i].first);
            }
            st.config((n - 1ll) / b + 1, help);
            // delete[] help;
        }

        //[l, r)
        int get(int l, int r) {
            if ((r - l) <= b) {
                return get2(l, r);
            }
            int ll = ((l != 0) ? ((l - 1) / b + 1) : 0), rr = r / b;
            return std::min({st.get(ll, rr), get2(l, ll * b), get2(rr * b, r)});
        }

        ~DS2() {
            // delete[] data;
        }
    };

};  // namespace WTF

struct ST {
    std::vector<std::vector<int>> data;
    int lg;

    ST() = default;

    [[gnu::noinline]] __attribute__((optimize("O3"))) ST(const std::vector<int> &vec) {
        lg = std::__lg(vec.size()) + 1;
        data.resize(lg);
        data[0] = vec;
        for (int k = 1; k < lg; k++) {
            data[k].resize(vec.size() - (1 << k) + 1);
#pragma GCC ivdep
            for (int i = 0; i < vec.size() - (1 << k) + 1; i++) {
                data[k][i] = std::min(data[k - 1][i], data[k - 1][i + (1 << k - 1)]);
            }
        }
    }

    int get_min(int l, int r) {
        int k = std::__lg(r - l);
        return std::min(data[k][l], data[k][r - (1 << k)]);
    }
};

struct Jump {
    ST st;
    // WTF::DS2 ds;

    std::vector<int> depth, tin, depth_eul;
    std::vector<int> eul_ind, depth_ind, eul;
    std::vector<Cum> cum;

    Jump() = default;

    [[gnu::noinline]] Jump(const std::vector<std::pair<int, int>> &edg) {
        int n = edg.size() + 1;
        std::vector<int> ind(n + 1);
        std::vector<int> nxt(2 * n);  // 2 * n instead of 2 * n - 2 for later reuse

        for (int i = 0; i < n - 1; i++) {
            ind[edg[i].first]++;
            ind[edg[i].second]++;
        }
        for (int i = 0, c = 0; i <= n; i++) {
            ind[i] = c += ind[i];
        }
        for (int i = 0; i < n - 1; i++) {
            nxt[--ind[edg[i].first]] = edg[i].second;
            nxt[--ind[edg[i].second]] = edg[i].first;
        }

        int e = 0;
        depth.assign(n, 0);
        tin.assign(n, 0);
        depth_eul.assign(2 * n - 1, 0);
        eul.assign(2 * n, 0);
        auto dfs = [&](auto dfs, int v, int f, int d) -> void {
            const int l0 = ind[v], r0 = ind[v + 1];
            tin[v] = e;
            depth_eul[e] = d;
            depth[v] = d;
            eul[e] = v;
            e++;

            for (int t_id = l0; t_id < r0; t_id++) {
                int t = nxt[t_id];
                if (t != f) {
                    dfs(dfs, t, v, d + 1);
                    eul[e] = v;
                    depth_eul[e] = d;
                    e++;
                }
            }
        };
        dfs(dfs, 0, -1, 0);
        assert(e == 2 * n - 1);

        depth_ind = std::move(ind);
        depth_ind.assign(n + 1, 0);
        for (int i = 0; i < e; i++) {
            depth_ind[depth_eul[i]]++;
        }
        for (int i = 0, c = 0; i <= n; i++) {
            depth_ind[i] = c += depth_ind[i];
        }
        eul_ind.resize(e + 32);
        for (int i = 0; i < e; i++) {
            eul_ind[--depth_ind[depth_eul[i]]] = i;
        }

        st = ST(depth_eul);
        // ds = WTF::DS2(depth_eul.size(), depth_eul.data());
        cum.resize(n);
        for (int i = 0; i < n; i++) {
            std::reverse(eul_ind.begin() + depth_ind[i], eul_ind.begin() + depth_ind[i + 1]);
            cum[i] = Cum(depth_ind[i + 1] - depth_ind[i], eul_ind.data() + depth_ind[i]);
        }
    }

    [[gnu::noinline]] int get_lca_depth(int u, int v) {
        int a = tin[u];
        int b = tin[v];
        int res = st.get_min(std::min(a, b), std::max(a, b) + 1);
        // int res = ds.get(std::min(a, b), std::max(a, b) + 1);
        return res;
    }

    // from vertex v to depth d
    [[gnu::noinline]] int jump(int v, int d) {
        int it = cum[d].lower_bound(tin[v]);
        int res = eul[cum[d].data[it]];
        return res;
    }

    [[gnu::noinline]] int jump_path(int s, int t, int i) {
        int d = get_lca_depth(s, t);
        int len = depth[s] + depth[t] - 2 * d;
        int ans = -1;
        if (i <= len) {
            if (i <= depth[s] - d) {
                ans = jump(s, depth[s] - i);
            } else {
                ans = jump(t, depth[t] - (len - i));
            }
        }
        return ans;
    }
};
