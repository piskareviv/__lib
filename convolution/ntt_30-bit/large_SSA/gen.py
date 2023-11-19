from random import randint

with open("test.in", 'w') as f:
    N = M = 2**24
    print(N, M, file=f)
    print(*[randint(0, 998_244_253 - 1) for i in range(N)], file=f)
    print(*[randint(0, 998_244_253 - 1) for i in range(M)], file=f)
