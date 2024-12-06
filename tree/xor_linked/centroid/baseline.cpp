#include <bits/stdc++.h>

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    std::cin >> n;
    std::vector<std::vector<int>> gr(n);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        std::cin >> u >> v;
        u--, v--;
        gr[u].push_back(v);
        gr[v].push_back(u);
    }

    std::vector<int> sz(n), depth(n, -1);
    auto find = [&](int v) {
        auto dfs = [&](auto dfs, int v, int f) -> void {
            sz[v] = 1;
            for (auto t : gr[v]) {
                if (t != f && depth[t] == -1) {
                    dfs(dfs, t, v);
                    sz[v] += sz[t];
                }
            }
        };
        dfs(dfs, v, -1);
        int f = -1, s = sz[v];
        while (true) {
            auto it = std::find_if(gr[v].begin(), gr[v].end(), [&](auto t) { return t != f && depth[t] == -1 && sz[t] > s / 2; });
            if (it != gr[v].end()) {
                f = v, v = *it;
            } else {
                break;
            }
        }
        return std::pair{v, s};
    };

    auto build = [&](auto build, int vvv, int dt, int f) -> void {
        auto [v, s] = find(vvv);
        depth[v] = dt;
        for (auto t : gr[v]) {
            if (depth[t] == -1) {
                build(build, t, dt + 1, v);
            }
        }
    };

    build(build, 0, 0, -1);

    for (int i = 0; i < n; i++) {
        std::cout << char('A' + depth[i]) << " \n"[i == n - 1];
    }

    return 0;
}
