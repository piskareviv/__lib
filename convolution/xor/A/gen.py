from random import randint

with open("test.in", 'w') as f:
    N = 20
    print(N, file=f)
    print(*[randint(0, 998_244_253 - 1) for i in range(2 ** N)], file=f)
    print(*[randint(0, 998_244_253 - 1) for i in range(2 ** N)], file=f)
