#pragma GCC target("avx2")

// #define qin std::cin
// #define qout std::cout
#include "IO.hpp"
#include "jump.hpp"
//

#include <array>
#include <iostream>

int32_t main() {
    int n, q;
    qin >> n >> q;
    std::vector<std::pair<int, int>> edg(n - 1);
    for (auto& [u, v] : edg) {
        qin >> u >> v;
    }
    std::vector<std::array<int, 3>> quer(q);
    for (auto& [s, t, i] : quer) {
        qin >> s >> t >> i;
    }

    Jump jmp(edg);

    for (auto& [s, t, i] : quer) {
        // int d = jmp.get_lca_depth(s, t);
        // int len = jmp.depth[s] + jmp.depth[t] - 2 * d;
        // int ans = -1;
        // if (i <= len) {
        //     if (i <= jmp.depth[s] - d) {
        //         ans = jmp.jump(s, jmp.depth[s] - i);
        //     } else {
        //         ans = jmp.jump(t, jmp.depth[t] - (len - i));
        //     }
        // }
        int ans = jmp.jump_path(s, t, i);
        qout << ans << '\n';
    }

    return 0;
}
