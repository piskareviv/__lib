#include <bits/stdc++.h>

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    std::cin >> n;
    std::vector<std::pair<int, int>> edg(n - 1);
    for (auto& [u, v] : edg) {
        std::cin >> u >> v;
        u--, v--;
    }

    std::vector<int> dfs_order;
    std::vector<int> last_occ, max_dist;
    {
        std::vector<int> xor_link(n), deg(n), sz(n, 1);
        for (auto [u, v] : edg) {
            deg[u] += 1, deg[v] += 1;
            xor_link[u] ^= v;
            xor_link[v] ^= u;
        }

        const int root = 0;
        std::vector<int> order;
        order.reserve(n), deg[root] = -1;
        for (int i = 0; i < n; i++) {
            for (int j = i; deg[j] == 1;) {
                int t = xor_link[j];
                order.push_back(j);
                deg[t]--, deg[j]--, xor_link[t] ^= j, sz[t] += sz[j], j = t;
            }
        }
        order.push_back(root);

        dfs_order.resize(2 * n - 2);
        sz[root] = 0;
        for (int i = 1; i < n; i++) {
            int v = order.rbegin()[i];
            int t = xor_link[v];
            int s = sz[v], s2 = sz[t];
            sz[v] = s2 + 1;
            sz[t] += s * 2;
            dfs_order[s2] = v, dfs_order[s2 + s * 2 - 1] = t;
        }

        last_occ.swap(xor_link), max_dist.swap(sz);
    }

    std::vector<int> ctr_depth(n);

    auto build = [&](auto build, int v, std::span<int> order, int f) -> void {
        if (!order.empty()) {
            for (int i = 0; i < order.size(); i++) {
                int v = order[i];
                max_dist[v] = -1, last_occ[v] = i;
            }
            for (int i = 0; i < order.size(); i++) {
                int v = order[i];
                int l = last_occ[v];
                int d = (int)order.size() + i - l;
                max_dist[v] = std::max(max_dist[v], d);
                last_occ[v] = (int)order.size() + i;
            }
            int it = std::min_element(order.begin(), order.end(), [&](int a, int b) { return max_dist[a] < max_dist[b]; }) - order.begin();
            v = order[it];
            std::rotate(order.begin(), order.begin() + it + 1, order.end());
        }

        ctr_depth[v] = f == -1 ? 0 : ctr_depth[f] + 1;

        for (int i = 0, l = 0; i < order.size(); i++) {
            if (order[i] == v) {
                if (l != i) {
                    build(build, order[l], order.subspan(l + 1, i - (l + 1)), v);
                }
                l = i + 1;
            }
        }
    };

    build(build, 0, std::span<int>(dfs_order.data(), 2 * n - 2), -1);

    for (int i = 0; i < n; i++) {
        std::cout << char('A' + ctr_depth[i]) << " \n"[i + 1 == n];
    }

    return 0;
}
