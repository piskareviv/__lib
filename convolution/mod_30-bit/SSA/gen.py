from random import randint

with open("test.in", 'w') as f:
    N = M = 2**19
    print(N, M, file=f)
    # print(*[randint(0, 10**9 + 7 - 1) for i in range(N)], file=f)
    # print(*[randint(0, 10**9 + 7 - 1) for i in range(M)], file=f)
    print(*[randint(1, 1) for i in range(N)], file=f)
    print(*[randint(1, 1) for i in range(M)], file=f)
