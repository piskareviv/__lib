from random import randint, shuffle, choice


with open("test.in", 'w') as f:
    N = 5 * 10**5
    edg = []
    for i in range(1, N):
        p = randint(max(0, 0), i - 1)
        edg.append([p, i])

    p = list(range(N))
    shuffle(p)
    for i in range(N - 1):
        edg[i][0] = p[edg[i][0]]
        edg[i][1] = p[edg[i][1]]
    shuffle(edg)

    Q = 5 * 10**5

    print(N, Q, file=f)
    for i in range(N - 1):
        print(*edg[i], file=f)

    for i in range(Q):
        s, t = sorted([randint(0, N - 1), randint(0, N - 1)])
        s, t = p[s], p[t]
        # i = randint(0, abs(s - t) // 5)
        i = randint(1, 1)
        print(s, t, i, file=f)
